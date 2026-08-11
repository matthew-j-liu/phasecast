"""Single-rock inference cards for the Andrews XRD sediments.

Mirrors `explorations/real_rock_fewshot/run_cards.py` -- best and median
test-slice rocks by adapted joint NLL, each rendered as a zero-shot-over-
adapted card plus a scenario view -- so the sediment set can be judged at
the level a user would interact with it. Three departures, all forced by
the source: the correction is fit on the full 363-rock train slice (the
protocol of `run_recalibrate.py`) rather than a 25-rock draw, the T/P
panel is dropped because the conditions head models magmatic cooling, and
picks rank per-rock rather than per-unit because no unit labels exist.

    conda run -n alignn_fix python explorations/sediment_transfer/run_inference_cards.py
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnmineral.eval import real_rock
from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import (
    CARD_INTERVAL_Q,
    GROUPS,
    real_rock_card_comparison,
    real_rock_scenario_figure,
    real_rock_scenarios,
    share_interval,
)
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    draw_units,
    fit_recalibrator,
    lift_group_presence_to_phases,
    score_recalibrator,
    select_weight_decay,
)

from load_andrews import load_andrews, write_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_CHECKPOINT = "checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=REPO / DEFAULT_CHECKPOINT)
    # single-Dirichlet tiers only: the card intervals and the scenario-view
    # correction read the Beta marginal directly, which the tailed tier's
    # mixture output breaks (same constraint as the volcanic cards)
    parser.add_argument("--tier", choices=("vector", "matrix", "mlp"),
                        default="matrix")
    parser.add_argument("--budget", type=int, default=None,
                        help="condition on this many train rocks instead of all "
                             "363; drawn per-rock (no units exist to draw whole)")
    parser.add_argument("--draw-seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=1024,
                        help="assemblage samples for scenario extraction")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    benchmark, _info = load_andrews()
    split_path = write_split(benchmark, HERE / "split.csv")
    real_rock.SPLIT_PATH = split_path
    subset = pd.read_csv(split_path)["subset"].to_numpy()
    train_mask, val_mask, test_mask = (subset == n for n in ("train", "val", "test"))

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    latents = bulk_latents(encoder.to(device), benchmark, device)
    mode_head = ckpt.build_mode_head().to(device)
    gp = build_group_predictions(mode_head, benchmark, latents)
    train, val, test = (gp.subset(m) for m in (train_mask, val_mask, test_mask))

    recal_json = HERE / "results" / "recalibrate.json"
    if recal_json.exists() and args.tier in json.loads(recal_json.read_text())["tiers"]:
        wd = json.loads(recal_json.read_text())["tiers"][args.tier]["weight_decay"]
    else:
        wd, _ = select_weight_decay(train, val, tier=args.tier)

    if args.budget is None:
        fit_slice, n_rocks = train, len(train)
        fit_desc = f"the {n_rocks}-rock train slice"
    else:
        rng = np.random.default_rng(args.draw_seed)
        drawn = draw_units(train, args.budget, rng)
        fit_slice, n_rocks = train.subset(drawn), int(drawn.sum())
        fit_desc = f"{n_rocks} train rocks (draw seed {args.draw_seed})"
    recalibrator = fit_recalibrator(fit_slice, tier=args.tier, weight_decay=wd)

    before, after = score_recalibrator(None, test), score_recalibrator(recalibrator, test)
    print(f"test slice ({len(test)} rocks): joint {before['joint'].mean():.3f} -> "
          f"{after['joint'].mean():.3f}   tier {args.tier} (wd={wd:g}), "
          f"fit on {n_rocks}")

    order = np.argsort(after["joint"])
    picks = {"best": order[0], "median": order[len(order) // 2]}
    test_rows = np.flatnonzero(test_mask)

    out_dir = HERE / "results" / f"cards_{args.tier}_{n_rocks}rocks"
    out_dir.mkdir(parents=True, exist_ok=True)
    subtitle_base = (f"Andrews XRD sediments; {args.tier} correction fit on "
                     f"{fit_desc}")

    # when a budget re-pick lands on different rocks than the full-train fit,
    # also render the full-fit picks under this correction, so the two folders
    # compare the same rocks side by side
    if args.budget is not None:
        sibling = HERE / "results" / f"cards_{args.tier}_{len(train)}rocks" / "cards.json"
        if sibling.exists():
            test_bulk_ids = benchmark.bulk_ids[np.flatnonzero(test_mask)]
            picked_ids = {str(test_bulk_ids[i]) for i in picks.values()}
            for entry in json.loads(sibling.read_text())["rocks"]:
                if entry["bulk_id"] not in picked_ids:
                    hits = np.flatnonzero(test_bulk_ids == entry["bulk_id"])
                    if len(hits):
                        picks[f"{entry['pick']}-of-full-fit"] = int(hits[0])

    with torch.no_grad():
        p21_all = torch.sigmoid(mode_head.presence_logits(latents)).cpu().numpy()
    manifest = []

    for label, local in picks.items():
        row = int(test_rows[local])
        measured = (benchmark.numeric | benchmark.censored).astype(np.float64)
        subtitle = (
            f"{subtitle_base}  |  held out: {label} of {len(test)}, "
            f"NLL {before['joint'][local]:.2f} -> {after['joint'][local]:.2f}, "
            f"{int(benchmark.numeric[row].sum())} measured groups"
        )

        with torch.no_grad():
            adj_logits, adj_alpha, _ = recalibrator(
                test.logits7[local:local + 1], test.alpha7[local:local + 1],
                test.sio2[local:local + 1],
            )
        arms = {}
        for name, alpha in (("zero", test.alpha7[local:local + 1].numpy()),
                            ("adapted", adj_alpha.numpy())):
            lo, hi = share_interval(alpha, measured[row:row + 1],
                                    q=CARD_INTERVAL_Q)
            arms[name] = (conditional_mean(alpha, measured[row:row + 1]), lo, hi)

        # the card panels index by benchmark row, so scatter the single-row
        # arrays back onto the full benchmark shape rather than reindexing them
        def full(vec, fill=0.0):
            out = np.full((len(benchmark), len(GROUPS)), fill)
            out[row] = vec[0]
            return out

        real_rock_card_comparison(
            (full(torch.sigmoid(test.logits7[local:local + 1]).numpy()),
             *(full(v, np.nan) for v in arms["zero"])),
            (full(torch.sigmoid(adj_logits).numpy()),
             *(full(v, np.nan) for v in arms["adapted"])),
            benchmark, row, out_dir / f"card_{label}.png", subtitle,
        )

        p21 = lift_group_presence_to_phases(
            recalibrator, test.logits7[local:local + 1], p21_all[row:row + 1],
            test.sio2[local:local + 1],
        )

        def correct(alpha7, _row=local):
            with torch.no_grad():
                _, out, _ = recalibrator(
                    test.logits7[_row:_row + 1].expand(len(alpha7), -1),
                    torch.as_tensor(alpha7, dtype=torch.float64),
                    test.sio2[_row:_row + 1].expand(len(alpha7)),
                )
            return out.numpy()

        scenarios = real_rock_scenarios(
            mode_head, latents[row], p21, correct_alpha7=correct,
            n_samples=args.n_samples, seed=args.seed,
        )
        real_rock_scenario_figure(
            scenarios, benchmark, row, out_dir / f"scenarios_{label}.png",
            subtitle, tp_samples=None,
        )
        manifest.append({
            "pick": label,
            "bulk_id": str(benchmark.bulk_ids[row]),
            "sio2": float(benchmark.sio2[row]),
            "joint_zero_shot": float(before["joint"][local]),
            "joint_adapted": float(after["joint"][local]),
            "scenarios": [
                {"groups": list(sc["groups"]), "prob": sc["prob"]}
                for sc in scenarios
            ],
        })

    (out_dir / "cards.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "tier": args.tier,
        "weight_decay": float(wd),
        "fit_rocks": n_rocks,
        "draw_seed": args.draw_seed if args.budget is not None else None,
        "test_joint_zero_shot": float(before["joint"].mean()),
        "test_joint_adapted": float(after["joint"].mean()),
        "conditions_panel": "dropped: the conditions head models magmatic cooling",
        "rocks": manifest,
    }, indent=2))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()

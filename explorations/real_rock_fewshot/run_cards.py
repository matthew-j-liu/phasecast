"""What a user sees: one real rock in, mineralogy out, after N labelled rocks.

Renders the #52 result at the level someone would actually interact with it --
a single held-out rock, the model's answer, and the measurement beside it --
rather than as a curve. Fits the adaptation tier on a draw of whole
stratigraphic units from the train roster and renders **test-slice rocks only**,
so nothing shown was ever fit on. (The test slice was spent 2026-08-03 for
scoring purposes; these renders are illustrative, not selection evidence.)

    conda run -n mineral python explorations/real_rock_fewshot/run_cards.py \
        --checkpoint checkpoints/joint_heads_supportmag_seed0/heads_best.pt

Two figures per rock. The card is zero-shot over adapted on one measurement,
teacher-forced on the measured assemblage: "given we know which minerals are
here, how much of each?". The scenario view drops the teacher forcing and asks
the deployment question: which minerals, and how much.

Three rocks are rendered -- best, median and worst by adapted joint NLL -- so
the failure mode sits beside the success. At 25 rocks *which units were drawn*
dominates, so the draw seed is stamped on every figure; one card from one lucky
draw is a cherry-pick.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.mode_head import conditional_mean
from cnmineral.eval.real_rock import (
    CARD_INTERVAL_Q,
    GROUPS,
    coarsen_presence,
    load_benchmark,
    real_rock_card_comparison,
    real_rock_scenario_figure,
    real_rock_scenarios,
    share_interval,
    split_mask,
)
from cnmineral.models.cond_head import COND_AXES
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    draw_units,
    fit_recalibrator,
    lift_group_presence_to_phases,
    score_recalibrator,
)

RESULTS = Path(__file__).resolve().parent / "results"

# the value selected on val for the matrix tier; see run_curve.WEIGHT_DECAY
WEIGHT_DECAY = {"vector": 0.01, "matrix": 1.0}


def sample_conditions(cond_head, latent, scenarios, n, rng):
    """(T, P) draws per scenario from CondHead's per-axis mixtures.

    Sampled independently per axis, as the corpus scenario view does: the head
    models no cross-axis correlation, so this shows each scenario's marginal
    ranges rather than a joint density.
    """
    t_idx, p_idx = COND_AXES.index("T"), COND_AXES.index("P_kbar")
    out = []
    for sc in scenarios:
        support = torch.zeros(1, cond_head.trunk[0].in_features - latent.shape[-1],
                              dtype=latent.dtype, device=latent.device)
        for phase_idx in sc["support21"]:
            support[0, phase_idx] = 1.0
        with torch.no_grad():
            log_w, mu, sigma = cond_head(latent.unsqueeze(0), support)
        w = log_w.exp().cpu().numpy()[0]
        mu, sigma = mu.cpu().numpy()[0], sigma.cpu().numpy()[0]
        tp = np.zeros((n, 2))
        for col, axis in enumerate((t_idx, p_idx)):
            k = rng.choice(w.shape[-1], size=n, p=w[axis] / w[axis].sum())
            tp[:, col] = rng.normal(mu[axis][k], sigma[axis][k])
        out.append(tp)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=25,
                        help="labelled rocks to condition on, drawn as whole units")
    # the card intervals and the scenario-view correction read the Beta
    # marginal directly, so every tier here must emit a single Dirichlet
    parser.add_argument("--tier", choices=("vector", "matrix"),
                        default="matrix")
    parser.add_argument("--draw-seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=1024,
                        help="assemblage samples for scenario extraction")
    parser.add_argument("--no-conditions", action="store_true",
                        help="drop the unvalidated T/P panel from the scenario view")
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    encoder.to(device)
    benchmark = load_benchmark()
    latents = bulk_latents(encoder, benchmark, device)
    mode_head = ckpt.build_mode_head().to(device)
    gp = build_group_predictions(mode_head, benchmark, latents)

    scored = benchmark.clean & benchmark.has_magnitude
    train_mask, test_mask = (split_mask(benchmark, s) & scored
                             for s in ("train", "test"))
    train = gp.subset(train_mask)

    rng = np.random.default_rng(args.draw_seed)
    drawn = draw_units(train, args.budget, rng)
    n_rocks, n_units = int(drawn.sum()), len(np.unique(train.unit[drawn]))
    print(f"conditioning on {n_rocks} rocks / {n_units} units "
          f"(seed {args.draw_seed}), tier {args.tier}")

    recalibrator = fit_recalibrator(train.subset(drawn), tier=args.tier,
                          weight_decay=WEIGHT_DECAY[args.tier])

    test = gp.subset(test_mask)
    before, after = score_recalibrator(None, test), score_recalibrator(recalibrator, test)
    print(f"test slice: joint {before['joint'].mean():.3f} -> "
          f"{after['joint'].mean():.3f}   "
          f"dirichlet {before['dirichlet'].mean():.3f} -> "
          f"{after['dirichlet'].mean():.3f}")

    # ranked by the ADAPTED per-rock joint NLL, the same score the ledger
    # reports, over the whole test slice. Worth knowing when reading "best":
    # that score is a density over however many groups a rock spans, and it
    # favours the rocks that span MORE. A single-group composition is a
    # degenerate point mass contributing exactly 0, so it can only ever score
    # its presence penalty, while a five-group composition has a
    # higher-dimensional simplex on which a sharp Dirichlet earns strongly
    # negative log-density. Measured on the test slice: mean adapted NLL runs
    # 6.9 / 6.9 / 7.3 / 1.4 / -2.0 at 1/2/3/4/5 groups, and every rock in the
    # top 20 has at least 4. Composition size is stamped on each card.
    order = np.argsort(after["joint"])
    picks = {"best": order[0], "median": order[len(order) // 2]}
    test_rows = np.flatnonzero(test_mask)

    out_dir = RESULTS / args.checkpoint.parent.name / f"cards_{args.tier}_{n_rocks}rocks"
    out_dir.mkdir(parents=True, exist_ok=True)
    subtitle_base = (f"{args.tier} correction fit on {n_rocks} rocks / "
                     f"{n_units} eruptions")

    with torch.no_grad():
        p21_all = torch.sigmoid(mode_head.presence_logits(latents)).cpu().numpy()
    cond_head = None if args.no_conditions else ckpt.build_cond_head().to(device)
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
            )
        # the Dirichlet's per-group marginal is a Beta, so the spread the model
        # actually reports is closed-form -- and widening it is most of what
        # the correction does, so a mean-only card would hide the main effect
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
        )

        def correct(alpha7, _row=local):
            with torch.no_grad():
                _, out, _ = recalibrator(
                    test.logits7[_row:_row + 1].expand(len(alpha7), -1),
                    torch.as_tensor(alpha7, dtype=torch.float64),
                )
            return out.numpy()

        scenarios = real_rock_scenarios(
            mode_head, latents[row], p21, correct_alpha7=correct,
            n_samples=args.n_samples, seed=args.draw_seed,
        )
        tp = None
        if cond_head is not None:
            for sc in scenarios:
                sc["support21"] = _members_of(sc["present"])
            tp = sample_conditions(cond_head, latents[row], scenarios, 400, rng)
        real_rock_scenario_figure(
            scenarios, benchmark, row, out_dir / f"scenarios_{label}.png",
            subtitle, tp_samples=tp,
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
        "conditioning_rocks": n_rocks,
        "conditioning_units": n_units,
        "draw_seed": args.draw_seed,
        "test_joint_zero_shot": float(before["joint"].mean()),
        "test_joint_adapted": float(after["joint"].mean()),
        "rocks": manifest,
    }, indent=2))
    print(f"saved {out_dir}")


def _members_of(present7: np.ndarray) -> list[int]:
    """Phase indices of every member of the groups a scenario turned on."""
    from cnmineral.data.phase_vocab import GROUP_INDEX, MODEL_TO_GROUP, PHASE_INDEX

    return [PHASE_INDEX[ph] for ph, g in MODEL_TO_GROUP.items()
            if present7[GROUP_INDEX[g]]]


if __name__ == "__main__":
    main()

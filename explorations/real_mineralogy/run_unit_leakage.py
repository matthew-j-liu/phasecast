"""Does the benchmark's fragmented unit key leak eruptions into val?

Splits group by stratigraphic unit so one eruption's near-identical rocks cannot
straddle a boundary.  The key is the raw source string, which fragments: 148
spellings for 141 units after case and whitespace alone, plus `?` qualifiers and
sub-facies suffixes (`, pumice`, `, basal vitrophyre`, `, upper part`) that name
the same eruption a different way.

The obvious repair is a curated eruption table, but building one means
adjudicating which of du Bray's two recorded labelings is authoritative where
they disagree, on 91 rocks.  This measures the damage instead, which needs no
such adjudication: flag every val rock that might share an eruption with a train
rock, then score val with and without them under the same fitted layer.  If the
two agree, the fragmentation costs nothing and the key can be left alone.

The merge rule is deliberately over-permissive.  Over-flagging costs a few rocks
of the leak-free subset; under-flagging leaves a leak in the number, so ties go
to "suspect" and the result is an upper bound on the exposure.

Dropping rocks rather than reassigning them is what keeps this cheap and
honest: the train roster is untouched, so the recalibrator is byte-identical and
every surviving rock keeps the score it already had.  Reassigning would change
the fit set and move every val score, including rocks that never went anywhere.

    conda run -n alignn_fix python explorations/real_mineralogy/run_unit_leakage.py \
        --checkpoint checkpoints/joint_heads_allrock_tuned_seed0/heads_best.pt
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from cnmineral.eval.latent import bulk_latents, load_encoder_from_checkpoint
from cnmineral.eval.real_rock import load_benchmark, split_mask
from cnmineral.training import pick_device
from cnmineral.training.joint_heads import JointHeadCheckpoint
from cnmineral.training.real_rock_recalibrator import (
    build_group_predictions,
    fit_recalibrator,
    score_recalibrator,
)

RESULTS = Path(__file__).resolve().parent / "results"

# selected on val for the production recalibrator; see run_curve.WEIGHT_DECAY
WEIGHT_DECAY_TAILED = 1.0


def normalize_unit(unit: str) -> str:
    """A unit string reduced to the eruption it names.

    Drops the `?` and `(?)` qualifiers, which record the mapper's uncertainty
    about the assignment rather than a different unit, and repairs the one
    missing-space typo.  Sub-facies suffixes are left in place -- they are
    handled by `same_eruption`, which needs to see the comma.
    """
    text = str(unit).lower().replace("(?)", "").replace("?", "")
    text = re.sub(r"\s+", " ", text).strip().strip(",").strip()
    return text.replace("fractiontuff", "fraction tuff")


def same_eruption(a: str, b: str) -> bool:
    """Whether two normalized unit strings name one eruption.

    Equal, or one is a prefix of the other ending at a comma -- which is how
    sub-facies are spelled (`Heller Tuff` against `Heller Tuff, pumice`).

    The comma boundary is what makes this safe.  Plain substring containment
    would let a bare `Rhyolite` swallow `Rhyolite of Bodie Hills`, and cutting
    at the first comma would collapse all nineteen `Plio-Pleistocene, ...` units
    into one, since that prefix is a chronostratigraphic era rather than a unit
    name.  Both are worse errors than the fragmentation being measured.
    """
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return long.startswith(short) and long[len(short):].lstrip().startswith(",")


def leaking_mask(units: np.ndarray, val: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Val rocks that may share an eruption with some train rock."""
    normalized = np.array([normalize_unit(u) for u in units])
    train_units = sorted(set(normalized[train]))
    leaking = np.zeros(len(units), dtype=bool)
    for i in np.where(val)[0]:
        leaking[i] = any(same_eruption(normalized[i], t) for t in train_units)
    return leaking


def merge_report(units: np.ndarray, leaking: np.ndarray, train: np.ndarray) -> list[dict]:
    """Every merge the rule makes, so it can be checked by eye rather than trusted."""
    normalized = np.array([normalize_unit(u) for u in units])
    train_units = sorted(set(normalized[train]))
    seen, rows = set(), []
    for i in np.where(leaking)[0]:
        match = next(t for t in train_units if same_eruption(normalized[i], t))
        if (units[i], match) not in seen:
            seen.add((units[i], match))
            rows.append({"val_unit": str(units[i]), "train_unit": match})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    ckpt = JointHeadCheckpoint.load(args.checkpoint)
    device = pick_device()
    encoder, _ = load_encoder_from_checkpoint(ckpt.encoder_checkpoint)
    head = ckpt.build_mode_head().to(device)
    encoder = encoder.to(device)

    benchmark = load_benchmark()
    latents = bulk_latents(encoder, benchmark, device)
    scored = benchmark.clean & benchmark.has_magnitude
    train = scored & split_mask(benchmark, "train")
    val = scored & split_mask(benchmark, "val")

    leaking = leaking_mask(benchmark.unit, val, train)
    merges = merge_report(benchmark.unit, leaking, train)
    print(f"scored: train {int(train.sum())}, val {int(val.sum())}")
    print(f"val rocks sharing an eruption with train: {int(leaking.sum())}")
    print("\nmerges the rule makes (val string -> train string):")
    for row in merges:
        print(f"  {row['val_unit']!r}  ->  {row['train_unit']!r}")

    recalibrator = fit_recalibrator(
        build_group_predictions(head, benchmark, latents).subset(train),
        tier="tailed", weight_decay=WEIGHT_DECAY_TAILED,
    )

    arms = {}
    for name, mask in (("all", val), ("leak_free", val & ~leaking)):
        gp = build_group_predictions(head, benchmark, latents).subset(mask)
        zero = score_recalibrator(None, gp)
        tailed = score_recalibrator(recalibrator, gp)
        arms[name] = {
            "n": int(mask.sum()),
            "zero_shot_joint": float(zero["joint"].mean()),
            "tailed_joint": float(tailed["joint"].mean()),
        }
        print(f"\nval {name} (n={arms[name]['n']})")
        print(f"  zero-shot joint  {arms[name]['zero_shot_joint']:.4f}")
        print(f"  tailed recalib.  {arms[name]['tailed_joint']:.4f}")

    delta = arms["leak_free"]["tailed_joint"] - arms["all"]["tailed_joint"]
    # A leak inflates the rocks it touches, so removing them would RAISE the
    # score.  A fall means the flagged rocks were harder than the rest, which is
    # the opposite signature and leaves nothing to repair.
    verdict = ("flagged rocks are harder than the rest -- no leak signature"
               if delta < 0 else
               "flagged rocks score better than the rest -- leak plausible")
    print(f"\nleak-free minus all, tailed: {delta:+.4f} nats -- {verdict}")

    tag = args.checkpoint.parent.name
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "unit_leakage.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "n_flagged": int(leaking.sum()),
        "merges": merges,
        "val": arms,
        "leak_free_minus_all_tailed": delta,
        "verdict": verdict,
    }, indent=2))
    print(f"\nwrote {out_dir}/unit_leakage.json")


if __name__ == "__main__":
    main()

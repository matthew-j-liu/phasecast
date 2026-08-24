"""Raw-input vs latent probe accuracy -- the decodability ceiling.

`scoring.run_scoring` reports probe accuracy on the *latent*; this runs the same
two probe families on the *raw assay*, over the same rows, so the latent number
has a reference point.  gard2019's `qap_name` is a computed classification of
LOI-free major-element chemistry (Gard et al. 2019, Sect. 4.2/4.4, Table 3) and
the encoder reads those same majors, so a latent probe accuracy on its own
cannot separate a well-organized latent from a label that any faithful
compression of the input recovers.

Both arms are fit here off one pair of index arrays rather than compared across
two runs, so "identical rows" is a property of the code.  The protocol the
reported latent number was measured under is asserted before any fitting.

Run:
    conda run -n mineral python explorations/input_probe_ceiling/run_ceiling.py \
        --checkpoint checkpoints/pretrain_allrock_redox_seed0/epoch_best.pt
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from torch.utils.data import DataLoader, Subset

from cnmineral.data.tokens import REDOX_TOKEN_INDEX
from cnmineral.eval.latent import (
    PaddedRedoxView,
    extract_latents,
    load_encoder_from_checkpoint,
)
from cnmineral.eval.probe.primitives import (
    random_label_purity,
    train_logreg_probe,
    train_mlp_probe,
)
from cnmineral.eval.supergroup import map_qap_array
from cnmineral.training import create_dataset, pick_device, reserve_test
from cnmineral.training.checkpoint import checkpoint_tag
from cnmineral.training.hpo import _sample_hyperparameter
from cnmineral.training.pretrain_objective import log_frac_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ROSTER_PATH = REPO_ROOT / "src/cnmineral/data/pretrain_inherited_test_roster.csv"
PROBE_HPO_CONFIG = REPO_ROOT / "configs/probe/probe_hpo.yaml"

_DEFAULT_CHECKPOINT = "checkpoints/pretrain_allrock_redox_seed0/epoch_best.pt"
_LOG_FLOOR = 1e-12
_STD_FLOOR = 1e-12  # a near-constant element must not blow up its standardization
_STACK_BATCH_SIZE = 512  # throughput-only knob for the frac stacking pass

RAW_REPRESENTATIONS: tuple[str, ...] = (
    "std_log_mask", "std_frac_mask", "raw_frac_mask", "std_log",
)
REPRESENTATIONS: tuple[str, ...] = ("latent",) + RAW_REPRESENTATIONS
PROBES: tuple[str, ...] = ("logreg", "mlp")
TARGETS: tuple[str, ...] = ("qap_name", "supergroup")

# The protocol the reported latent accuracy was measured under (verified
# 2026-08-14 against pretrain_allrock_redox_seed0).  Asserted before any
# fitting: a drift in the split, the roster, or the corpus filters would
# otherwise yield a plausible wrong ceiling instead of a failure.
EXPECTED_N_TRAIN = 458_490
EXPECTED_N_TEST = 32_950
EXPECTED_N_CLASSES = 70
EXPECTED_CHANCE = 0.0997
EXPECTED_MAJORITY = 0.2203

# What the scorecard records for the latent qap MLP under this protocol, and
# the cross-seed spread it was measured with.  Printed, never asserted -- a
# miss means this is not the arm the paper reports.
SCORECARD_LATENT_QAP_MLP = 0.759
CROSS_SEED_SPREAD = 0.006


def _timed(label: str, fn, *args, **kwargs):
    t0 = time()
    out = fn(*args, **kwargs)
    print(f"  {label}: {time() - t0:.1f}s")
    return out


def probe_slice(
    dataset, config: dict, *, roster_path: Path = ROSTER_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test row indices for the headline probe, qap-labeled rows only.

    Mirrors the slicing `cnmineral.eval.probe.scoring.run_scoring` does: the
    checkpoint's own split, test rows restricted to the inherited shared-row
    roster, then both sides restricted to rows carrying a qap label (on an
    all-rock corpus that restriction IS the igneous slice).
    """
    splits = reserve_test(dataset, config)
    train_idx = np.asarray(splits["trainval"], dtype=np.int64)
    test_idx = np.asarray(splits["test"], dtype=np.int64)

    roster_ids = pd.read_csv(roster_path)["sample_id"].to_numpy()
    test_idx = test_idx[np.isin(dataset.sample_ids[test_idx], roster_ids)]

    qap = np.asarray(dataset.qap_names)

    def labeled(idx: np.ndarray) -> np.ndarray:
        return idx[np.array([q is not None for q in qap[idx]])]

    return labeled(train_idx), labeled(test_idx)


def label_stats(y_test: np.ndarray) -> dict:
    """Chance and majority-class share for one target's test labels.

    Chance is `sum_c p_c^2`, the same class-balanced random baseline the k-NN
    purity floor uses, so the two evals' floors are the same statistic.
    """
    _, counts = np.unique(y_test, return_counts=True)
    return {
        "chance": random_label_purity(y_test),
        "majority_share": float(counts.max() / counts.sum()),
    }


def check_protocol(qap_train: np.ndarray, qap_test: np.ndarray) -> dict:
    """Raise unless the slice is the one the reported latent number came from.

    A `ValueError` rather than an `assert`: this guards a paper-bearing number
    and must not be strippable.  Every mismatch is collected so one run tells
    the reader everything that moved, not just the first thing.
    """
    stats = label_stats(qap_test)
    chance, majority = stats["chance"], stats["majority_share"]

    drift = [
        (name, expected, got)
        for name, expected, got in (
            ("n_train", EXPECTED_N_TRAIN, len(qap_train)),
            ("n_test", EXPECTED_N_TEST, len(qap_test)),
            ("n_classes_train", EXPECTED_N_CLASSES, len(np.unique(qap_train))),
            ("n_classes_test", EXPECTED_N_CLASSES, len(np.unique(qap_test))),
            ("chance", EXPECTED_CHANCE, round(chance, 4)),
            ("majority_share", EXPECTED_MAJORITY, round(majority, 4)),
        )
        if expected != got
    ]
    if drift:
        detail = "; ".join(f"{n}: expected {e}, got {g}" for n, e, g in drift)
        raise ValueError(
            f"probe slice does not match the protocol the reported latent "
            f"accuracy was measured under ({detail}).  Either the checkpoint "
            f"is not pretrain_allrock_redox_seed0 or the corpus/roster moved; "
            f"the latent arm would no longer be the arm the paper reports."
        )
    return stats


def _element_slots(n_tokens: int) -> list[int]:
    """Element token positions, redox slot excluded.

    The latent arm reads a `PaddedRedoxView`, which zeroes the redox token, so
    a raw arm that kept that slot would read chemistry the latent arm cannot
    see.  On a 22-token corpus this is already the identity.
    """
    return [j for j in range(n_tokens) if j != REDOX_TOKEN_INDEX]


def _stack_frac(
    dataset, indices, *, batch_size: int = _STACK_BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack `frac` and a populated mask (`src != 0`) over `indices`."""
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False,
    )
    fracs, srcs = [], []
    for batch in loader:
        fracs.append(batch["frac"])
        srcs.append(batch["src"])
    return torch.cat(fracs, dim=0), torch.cat(srcs, dim=0) != 0


def _standardize(
    values: torch.Tensor,
    populated: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Center/scale populated entries; absent ones take the neutral 0."""
    z = (values - mean) / std
    return torch.where(populated, z, torch.zeros_like(z))


def _populated_stats(
    values: torch.Tensor, populated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-element mean/std over populated rows; all-absent columns stay (0, 1).

    The linear-space counterpart to `log_frac_stats`.  It lives here rather
    than in production because nothing in the model standardizes linear
    abundance -- only this exploration's `std_frac` arm needs it.
    """
    n_tokens = values.shape[1]
    mean, std = torch.zeros(n_tokens), torch.ones(n_tokens)
    for j in range(n_tokens):
        col = values[populated[:, j], j]
        if col.numel() > 1:
            mean[j] = col.mean()
            std[j] = col.std(unbiased=False)
    return mean, std.clamp_min(_STD_FLOOR)


def build_representations(
    dataset,
    train_idx,
    test_idx,
    *,
    batch_size: int = _STACK_BATCH_SIZE,
) -> dict[str, dict[str, np.ndarray]]:
    """The four raw-input matrices per split, keyed `{split: {name: X}}`.

    - `std_log_mask`: per-element standardized `log10(frac)` alongside a binary
      populated flag.  The headline arm.  Log space is where the encoder's own
      objective lives and keeps the majors from swamping the traces; the mask
      is there because the encoder sees missingness explicitly (`src == 0` vs
      `src == Z`) and roughly half the trace columns are unreported per rock,
      so a raw arm denied that signal would be handicapped on input
      representation rather than on representation learning.
    - `std_frac_mask`: the same, standardized in linear space.  Isolates log
      vs linear from scaled vs unscaled, which `raw_frac_mask` alone confounds:
      `train_mlp_probe` does not scale its inputs, so an unstandardized linear
      matrix handicaps the MLP on preprocessing rather than on representation.
    - `raw_frac_mask`: literal abundances plus the same flag.  The
      representation that won `input_knn_ceiling`, re-asked as a classifier and
      carried verbatim so the two explorations compare directly.
    - `std_log`: the headline arm without the mask -- what the missingness
      signal is worth.

    Center/scale come from the *train* indices only, so the test rows never
    touch the standardization.  Absent elements fill with standardized 0 (the
    neutral train mean), not `log10(0)`.
    """
    log_mean, log_std = log_frac_stats(
        dataset, list(train_idx), batch_size=batch_size, num_workers=0,
    )
    train_frac, train_populated = _stack_frac(
        dataset, train_idx, batch_size=batch_size,
    )
    frac_mean, frac_std = _populated_stats(train_frac, train_populated)

    out: dict[str, dict[str, np.ndarray]] = {}
    stacked = (
        ("train", (train_frac, train_populated)),
        ("test", _stack_frac(dataset, test_idx, batch_size=batch_size)),
    )
    for split, (frac, populated) in stacked:
        std_log = _standardize(
            torch.log10(frac.clamp_min(_LOG_FLOOR)), populated, log_mean, log_std,
        )
        std_frac = _standardize(frac, populated, frac_mean, frac_std)

        keep = _element_slots(frac.shape[1])
        flags = populated[:, keep].float()
        frac, std_log, std_frac = frac[:, keep], std_log[:, keep], std_frac[:, keep]
        out[split] = {
            "std_log_mask": torch.cat([std_log, flags], dim=1).numpy(),
            "std_frac_mask": torch.cat([std_frac, flags], dim=1).numpy(),
            "raw_frac_mask": torch.cat([frac, flags], dim=1).numpy(),
            "std_log": std_log.numpy(),
        }
    return out


def _fit_probes(X_train, y_train, X_test, y_test, *, device, seed) -> dict:
    logreg = _timed(
        "logreg", train_logreg_probe,
        X_train, y_train, X_test, y_test, seed=seed,
    )
    mlp = _timed(
        "mlp", train_mlp_probe,
        X_train, y_train, X_test, y_test, device=device, seed=seed,
    )
    return {"logreg": logreg.to_dict(), "mlp": mlp.to_dict()}


def _accuracy_of(accuracy: dict, target: str, rep: str, probe: str) -> float:
    return accuracy[target][rep][probe]["test_accuracy"]


def _headline(accuracy: dict) -> dict:
    """Latent vs the best raw representation, per target and probe family.

    Best-of the raw arms rather than one chosen featurization: a ceiling is a
    maximum over reasonable methods, and reporting a single pick invites the
    objection that it was the one that lost.
    """
    out: dict[str, dict[str, float | str]] = {}
    for target in accuracy:
        row: dict[str, float | str] = {}
        for probe in PROBES:
            latent = _accuracy_of(accuracy, target, "latent", probe)
            best = max(
                RAW_REPRESENTATIONS,
                key=lambda rep: _accuracy_of(accuracy, target, rep, probe),
            )
            best_score = _accuracy_of(accuracy, target, best, probe)
            row[f"latent_{probe}"] = latent
            row[f"best_raw_{probe}"] = best_score
            row[f"best_raw_{probe}_representation"] = best
            row[f"margin_{probe}"] = latent - best_score
        out[target] = row
    return out


def _print_summary(
    accuracy: dict, headline: dict, stats_by_target: dict, widths: dict,
) -> None:
    for target in accuracy:
        stats = stats_by_target[target]
        print(
            f"\n{target} probe accuracy "
            f"(chance {stats['chance']:.3f}, "
            f"majority {stats['majority_share']:.3f})"
        )
        for rep in REPRESENTATIONS:
            logreg = _accuracy_of(accuracy, target, rep, "logreg")
            mlp = _accuracy_of(accuracy, target, rep, "mlp")
            print(
                f"  {rep} (dim {widths[rep]}): "
                f"logreg {logreg:.3f}, mlp {mlp:.3f}"
            )
        row = headline[target]
        for probe in PROBES:
            print(
                f"  -> {probe}: latent {row[f'latent_{probe}']:.3f} vs "
                f"best raw {row[f'best_raw_{probe}']:.3f} "
                f"({row[f'best_raw_{probe}_representation']}), "
                f"margin {row[f'margin_{probe}']:+.3f}"
            )


def tune_representation(
    X_train, y_train, X_test, y_test,
    *,
    name: str,
    encoder_tag: str,
    device,
    n_trials: int,
    config_path: Path = PROBE_HPO_CONFIG,
) -> dict:
    """Optuna sweep over the MLP probe HPs for one representation.

    Reads the same `base` / `search_space` / sampler seed the production probe
    sweep uses, so the two arms are tuned under an identical budget.  The
    objective is test accuracy, matching `cnmineral.eval.probe.hpo.run_hpo` --
    fair between arms but optimistically biased in absolute terms, which is why
    the fixed-protocol pass stays the headline and this is the robustness leg.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base = dict(cfg["base"])
    search_space = cfg.get("search_space", {})

    # tag in both the filename and the study name: a resumable journal keyed
    # only by representation would silently hand a different encoder's trials
    # back on the next run
    study_name = f"probe_ceiling_{encoder_tag}_{name}"
    storage_path = REPO_ROOT / "data/optuna/journals" / f"{study_name}.journal"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(str(storage_path))),
        sampler=TPESampler(
            seed=cfg["sampler_seed"], n_startup_trials=cfg["n_startup_trials"],
        ),
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        sampled = {
            key: _sample_hyperparameter(trial, key, spec)
            for key, spec in search_space.items()
        }
        metrics = train_mlp_probe(
            X_train, y_train, X_test, y_test, device=device, **{**base, **sampled},
        )
        return float(metrics.test_accuracy)

    n_done = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    if n_trials > n_done:
        study.optimize(objective, n_trials=n_trials - n_done, gc_after_trial=True)

    final = train_mlp_probe(
        X_train, y_train, X_test, y_test,
        device=device, **{**base, **study.best_params},
    )
    print(f"  tuned {name}: {final.test_accuracy:.3f} (fixed-protocol arm above)")
    return {
        "best_params": dict(study.best_params),
        "best_trial_value": float(study.best_value),
        "n_trials_complete": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ),
        "final_mlp": final.to_dict(),
    }


def run_ceiling(
    checkpoint_path: Path,
    *,
    device: torch.device | None = None,
    tune: bool = False,
    n_trials: int = 30,
) -> dict:
    """Fit both arms on one slice and write `results/probe_ceiling.json`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")

    print("\nloading encoder (config comes from ckpt.config)...")
    encoder, ckpt = load_encoder_from_checkpoint(checkpoint_path)
    config = ckpt.config
    seed = config.get("seed", 0)
    tag = checkpoint_tag(checkpoint_path)

    print("building dataset + reproducing the headline probe slice...")
    dataset = create_dataset(config)
    train_idx, test_idx = probe_slice(dataset, config)

    qap_full = np.asarray(dataset.qap_names)
    qap_train, qap_test = qap_full[train_idx], qap_full[test_idx]
    slice_stats = check_protocol(qap_train, qap_test)
    print(
        f"  train={len(train_idx)} test={len(test_idx)} "
        f"classes={len(np.unique(qap_test))} "
        f"chance={slice_stats['chance']:.4f} "
        f"majority={slice_stats['majority_share']:.4f}"
    )

    labels = {
        "qap_name": (qap_train, qap_test),
        "supergroup": (map_qap_array(qap_train), map_qap_array(qap_test)),
    }
    stats_by_target = {
        target: label_stats(y_test) for target, (_, y_test) in labels.items()
    }

    print("\nstacking raw-input representations...")
    reps = _timed(
        "raw", build_representations, dataset, train_idx, test_idx,
    )

    print("extracting latents (redox token padded)...")
    both = np.concatenate([train_idx, test_idx])
    latents = _timed(
        "latent", extract_latents,
        encoder, PaddedRedoxView(dataset), indices=both, device=device,
    ).numpy()
    reps["train"]["latent"] = latents[: len(train_idx)]
    reps["test"]["latent"] = latents[len(train_idx) :]
    widths = {rep: int(reps["train"][rep].shape[1]) for rep in REPRESENTATIONS}
    print(f"  widths: {widths}")

    accuracy: dict[str, dict[str, dict]] = {}
    for target, (y_train, y_test) in labels.items():
        accuracy[target] = {}
        for rep in REPRESENTATIONS:
            print(f"\nprobes: {target} / {rep} (dim {widths[rep]})")
            accuracy[target][rep] = _fit_probes(
                reps["train"][rep], y_train, reps["test"][rep], y_test,
                device=device, seed=seed,
            )

    headline = _headline(accuracy)
    _print_summary(accuracy, headline, stats_by_target, widths)

    latent_qap_mlp = headline["qap_name"]["latent_mlp"]
    delta = latent_qap_mlp - SCORECARD_LATENT_QAP_MLP
    flag = "" if abs(delta) <= CROSS_SEED_SPREAD else "  <-- OUTSIDE cross-seed spread"
    print(
        f"\nlatent qap mlp {latent_qap_mlp:.3f} vs scorecard "
        f"{SCORECARD_LATENT_QAP_MLP:.3f} ({delta:+.3f}){flag}"
    )

    tuned = None
    if tune:
        best_raw = headline["qap_name"]["best_raw_mlp_representation"]
        print(f"\ntuning leg: latent + {best_raw}, {n_trials} trials each")
        y_train, y_test = labels["qap_name"]
        tuned = {
            rep: tune_representation(
                reps["train"][rep], y_train, reps["test"][rep], y_test,
                name=rep, encoder_tag=tag, device=device, n_trials=n_trials,
            )
            for rep in ("latent", best_raw)
        }

    metrics = {
        "checkpoint": str(checkpoint_path),
        "tag": tag,
        "device": str(device),
        "redox": "padded",
        "eval_roster": str(ROSTER_PATH),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_classes": int(len(np.unique(qap_test))),
        "seed": seed,
        "widths": widths,
        "label_stats": stats_by_target,
        "scorecard_latent_qap_mlp": SCORECARD_LATENT_QAP_MLP,
        "accuracy": accuracy,
        "headline": headline,
        "tuned": tuned,
    }
    out_path = RESULTS_DIR / "probe_ceiling.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nmetrics -> {out_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(_DEFAULT_CHECKPOINT),
        help="encoder checkpoint (defaults to the canonical all-rock encoder)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="also run the matched Optuna leg on latent + the best raw arm",
    )
    parser.add_argument(
        "--n-trials", type=int, default=30,
        help="trials per representation in the tuning leg",
    )
    args = parser.parse_args()
    run_ceiling(
        checkpoint_path=args.checkpoint.resolve(),
        tune=args.tune,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()

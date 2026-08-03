"""Raw-input vs latent kNN purity -- the geometry data ceiling.

Settles whether a stronger SSL pretext could meaningfully raise kNN qap_name
purity in the latent (issue #15), by measuring how separable qap is from the
raw chemistry at all.  `scoring.run_scoring` reports kNN purity on the *latent*;
this runs the identical metric on the *raw 22-element input*.  Some qap splits
are textural (basalt vs gabbro share a chemistry the encoder never sees), so the
raw-input purity is a ceiling no encoder can pass -- the geometry analogue of the
imputation work's raw-input XGBoost ceiling.

Reuses production primitives (`knn_label_purity`, `log_frac_stats`,
`extract_latents`, `reserve_test`); the raw-matrix construction is the only
exploration-specific scaffolding and stays here.  Two raw representations:

- `raw_frac`: the literal 22 abundances (zeros for absent) -- naive input,
  scale-dominated by the majors.
- `std_log`: per-element standardized `log10(frac)` over the train split, absent
  elements filled with a neutral 0 -- the model's own input space, all elements
  weighted comparably.

Run:
    conda run -n alignn_fix python explorations/input_knn_ceiling/run_ceiling.py \
        --checkpoint checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cnmineral.eval.latent import extract_latents, load_encoder_from_checkpoint
from cnmineral.eval.probe.primitives import knn_label_purity, random_label_purity
from cnmineral.eval.supergroup import map_qap_array
from cnmineral.training import create_dataset, pick_device, reserve_test
from cnmineral.training.checkpoint import checkpoint_tag
from cnmineral.training.pretrain_objective import log_frac_stats

RESULTS_DIR = Path(__file__).resolve().parent / "results"

_LOG_FLOOR = 1e-12
_STACK_BATCH_SIZE = 512  # throughput-only knob for the frac stacking pass
K_VALUES: tuple[int, ...] = (5, 10, 50)
SPACES: tuple[str, ...] = ("raw_frac", "std_log", "latent")
METRICS: tuple[str, ...] = ("cosine", "euclidean")
_RAW_SPACES: tuple[str, ...] = ("raw_frac", "std_log")

_DEFAULT_CHECKPOINT = "checkpoints/pretrain_v3_logmse_std_seed0/epoch_best.pt"


def _stack_test_frac(
    dataset, indices: list[int], *, batch_size: int = _STACK_BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack `frac` and a populated mask (`src != 0`) over `indices`."""
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False,
    )
    fracs, srcs = [], []
    for batch in loader:
        fracs.append(batch["frac"])
        srcs.append(batch["src"])
    frac = torch.cat(fracs, dim=0)
    populated = torch.cat(srcs, dim=0) != 0
    return frac, populated


def build_raw_representations(
    dataset,
    test_indices: list[int],
    train_indices: list[int],
    *,
    batch_size: int = _STACK_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """The two raw-input test matrices, `[n_test, N_TOKENS]` each.

    `std_log` standardizes with train-split center/scale (`log_frac_stats`) so
    val/test never leak into the standardization, mirroring the model's input
    pipeline.  Absent elements (`src == 0`) are filled with standardized 0, the
    neutral value, rather than `log10(0)`.
    """
    frac, populated = _stack_test_frac(dataset, test_indices, batch_size=batch_size)
    mean, std = log_frac_stats(
        dataset, train_indices, batch_size=batch_size, num_workers=0,
    )
    log_frac = torch.log10(frac.clamp_min(_LOG_FLOOR))
    std_log = (log_frac - mean) / std
    std_log = torch.where(populated, std_log, torch.zeros_like(std_log))
    return {"raw_frac": frac.numpy(), "std_log": std_log.numpy()}


def run_ceiling(
    checkpoint_path: Path, *, device: torch.device | None = None,
) -> dict:
    """Compute raw-input vs latent kNN purity and write `results/input_ceiling.json`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")

    print("\nloading encoder (config comes from ckpt.config)...")
    encoder, ckpt = load_encoder_from_checkpoint(checkpoint_path)
    config = ckpt.config

    print("building dataset + reproducing test split...")
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    train_idx = list(splits["trainval"])
    test_idx = list(splits["test"])
    print(f"  n_test={len(test_idx)} (trainval={len(train_idx)})")

    print("stacking raw-input representations...")
    reps = build_raw_representations(dataset, test_idx, train_idx)

    print("extracting latents over the test split...")
    t0 = time()
    reps["latent"] = extract_latents(
        encoder, dataset, indices=test_idx, device=device,
    ).numpy()
    print(f"  latent {reps['latent'].shape} in {time() - t0:.1f}s")

    qap_full = np.asarray(dataset.qap_names)
    qap_test = qap_full[np.asarray(test_idx)]
    labels = {"qap_name": qap_test, "supergroup": map_qap_array(qap_test)}
    baselines = {name: random_label_purity(y) for name, y in labels.items()}

    purity: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        lname: {space: {} for space in SPACES} for lname in labels
    }
    for space in SPACES:
        X = reps[space]
        for metric in METRICS:
            t0 = time()
            for lname, y in labels.items():
                by_k = knn_label_purity(X, y, k_values=K_VALUES, metric=metric)
                purity[lname][space][metric] = {str(k): v for k, v in by_k.items()}
            print(f"  knn {space}/{metric}: {time() - t0:.1f}s")

    _print_tables(purity, baselines)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "tag": checkpoint_tag(checkpoint_path),
        "device": str(device),
        "n_test": len(test_idx),
        "latent_dim": int(reps["latent"].shape[1]),
        "k_values": list(K_VALUES),
        "random_baseline": baselines,
        "purity": purity,
    }
    out_path = RESULTS_DIR / "input_ceiling.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nmetrics -> {out_path}")
    return metrics


def _print_tables(purity: dict, baselines: dict) -> None:
    """One block per label: a space/metric x k grid, then the headline gap."""
    for lname, baseline in baselines.items():
        print(f"\n{lname} kNN purity (random baseline {baseline:.3f})")
        print(f"  {'space/metric':<22}" + "".join(f"k={k:<6}" for k in K_VALUES))
        for space in SPACES:
            for metric in METRICS:
                row = purity[lname][space][metric]
                cells = "".join(f"{row[str(k)]:<8.3f}" for k in K_VALUES)
                print(f"  {space + '/' + metric:<22}{cells}")
        latent5 = purity[lname]["latent"]["cosine"]["5"]
        ceiling5 = max(
            purity[lname][space][metric]["5"]
            for space in _RAW_SPACES for metric in METRICS
        )
        print(
            f"  -> k=5: latent(cosine) {latent5:.3f} | "
            f"best raw-input {ceiling5:.3f} | headroom {ceiling5 - latent5:+.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(_DEFAULT_CHECKPOINT),
        help="encoder checkpoint (defaults to the canonical log-MSE encoder)",
    )
    args = parser.parse_args()
    run_ceiling(checkpoint_path=args.checkpoint.resolve())


if __name__ == "__main__":
    main()

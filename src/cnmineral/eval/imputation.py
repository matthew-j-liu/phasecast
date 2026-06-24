"""
Element imputation against a frozen pretrain checkpoint.

`run_imputation` runs the SSL `PretrainHead` at inference: for each element
in turn, mask it out of every held-out test bulk where it was measured,
predict it from the rest of the rock, and compare to the withheld truth.
No training happens here -- this is a standalone use of the encoder + head
that shows how well the pretrained representation fills missing analytes.

It sits beside the probe eval (`cnmineral.eval.probe`): same checkpoint,
same Gard test split reconstructed from `ckpt.config`, but it loads the head
(probes discard it) and reports regression error per element rather than
classification accuracy.  Checkpoint loading lives in
`cnmineral.eval.latent`; the test split in `cnmineral.training.splits`.
"""

import json
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from cnmineral.data.bulk_chemistry import MAJOR_TOKEN_TO_COL, OXIDE_TO_CATION_FACTOR
from cnmineral.data.tokens import MASK_TOKEN_ID, MAJORS, VOCAB
from cnmineral.eval.latent import load_encoder_and_head_from_checkpoint
from cnmineral.eval.supergroup import SUPERGROUP_COLORS, map_qap_array
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead
from cnmineral.training import create_dataset, pick_device, reserve_test
from cnmineral.training.checkpoint import checkpoint_tag

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"


def _back_transform(token_pos: int, frac: np.ndarray) -> tuple[np.ndarray, str]:
    """Invert the dataset's frac normalization for one token position.

    Returns the values in the unit a scientist reads -- oxide wt% for
    majors, ppm for traces -- plus a short axis label.  Fe is a blend of
    FeO + Fe2O3 cation contributions, so its frac inverts to cation wt%
    (`frac * 100`), not to a single oxide.
    """
    sym, _z = VOCAB[token_pos]
    if token_pos < len(MAJORS):
        if sym == "Fe":
            return frac * 100.0, "Fe (cation wt%)"
        factor = OXIDE_TO_CATION_FACTOR[MAJOR_TOKEN_TO_COL[sym]]
        return frac * 100.0 / factor, f"{sym} (oxide wt%)"
    return frac * 1e6, f"{sym} (ppm)"


def _regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    ss_res = float(np.sum(err**2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


@torch.no_grad()
def _impute_token(
    encoder: Encoder,
    head: PretrainHead,
    dataset: Dataset,
    indices: np.ndarray,
    token_pos: int,
    *,
    device: torch.device | str,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-out impute `token_pos` over `Subset(dataset, indices)`.

    Masks exactly `token_pos` (`src = MASK_TOKEN_ID`, `frac = 0`) in every
    row, predicts it, and keeps only rows where the element was originally
    populated (those have a truth value to compare against).

    Returns `(kept_dataset_idx, true_frac, pred_frac)`, each aligned, where
    `kept_dataset_idx` are absolute indices into `dataset`.
    """
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    kept_idx: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    offset = 0
    for batch in loader:
        src = batch["src"]
        frac = batch["frac"]
        b = src.shape[0]
        populated = (src[:, token_pos] != 0).numpy()

        src_m = src.clone().to(device)
        frac_m = frac.clone().to(device)
        src_m[:, token_pos] = MASK_TOKEN_ID
        frac_m[:, token_pos] = 0.0
        pred = head(encoder(src=src_m, frac=frac_m))[:, token_pos].cpu().numpy()

        local = np.nonzero(populated)[0]
        kept_idx.append(indices[offset + local])
        true_chunks.append(frac[local, token_pos].numpy())
        pred_chunks.append(pred[local])
        offset += b

    if not any(c.size for c in true_chunks):
        empty = np.empty(0)
        return empty.astype(np.int64), empty.astype(np.float32), empty.astype(np.float32)
    return (
        np.concatenate(kept_idx),
        np.concatenate(true_chunks),
        np.concatenate(pred_chunks),
    )


def _plot_group(
    elements: list[dict],
    title: str,
    out_path: Path,
    *,
    log_scale: bool,
) -> None:
    """One scatter panel per element, predicted vs true, colored by supergroup."""
    ncols = 4
    nrows = ceil(len(elements) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False,
    )
    seen_groups: dict[str, None] = {}
    for ax, el in zip(axes.flat, elements):
        true, pred, sg = el["true_bt"], el["pred_bt"], el["supergroup"]
        colors = [SUPERGROUP_COLORS.get(g, SUPERGROUP_COLORS["other"]) for g in sg]
        ax.scatter(true, pred, c=colors, s=8, alpha=0.5, edgecolors="none")
        for g in sg:
            seen_groups.setdefault(g, None)
        # head output is unconstrained: only log-scale a panel when both
        # axes carry positive data, else matplotlib warns and blanks it.
        span = np.concatenate([true, pred])
        if log_scale and (true > 0).any() and (pred > 0).any():
            ax.set_xscale("log")
            ax.set_yscale("log")
            span = span[span > 0]
        if span.size:
            lo, hi = float(span.min()), float(span.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"true {el['label']}")
        ax.set_ylabel(f"predicted {el['label']}")
        m = el["metrics_bt"]
        ax.set_title(f"{el['label']}  MAE={m['mae']:.3g} R2={m['r2']:.2f} n={el['n']}")
    for ax in axes.flat[len(elements):]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=SUPERGROUP_COLORS[g], label=g)
        for g in seen_groups
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_imputation(
    checkpoint_path: Path,
    *,
    device: torch.device | None = None,
    batch_size: int = 512,
) -> dict:
    """Leave-one-element-out imputation over a checkpoint's Gard test split.

    Writes per-element metrics (back-transformed wt%/ppm space, primary; plus
    frac space for troubleshooting) and predicted-vs-true scatter grids to
    `results/imputation/<tag>/`, and returns the metrics dict.
    """
    tag = checkpoint_tag(checkpoint_path)
    output_dir = RESULTS_ROOT / "imputation" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    print(f"device:     {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output:     {output_dir}")

    encoder, head, ckpt = load_encoder_and_head_from_checkpoint(checkpoint_path)
    encoder.to(device)
    head.to(device)

    dataset = create_dataset(ckpt.config)
    test_idx = np.asarray(reserve_test(dataset, ckpt.config)["test"], dtype=np.int64)
    sg_all = map_qap_array(dataset.qap_names)
    print(f"dataset n={len(dataset)} test={len(test_idx)}")

    per_element: dict[str, dict] = {}
    major_panels: list[dict] = []
    trace_panels: list[dict] = []
    for token_pos, (sym, _z) in enumerate(VOCAB):
        kept, true_frac, pred_frac = _impute_token(
            encoder, head, dataset, test_idx, token_pos,
            device=device, batch_size=batch_size,
        )
        if kept.size == 0:
            continue
        true_bt, label = _back_transform(token_pos, true_frac)
        pred_bt, _ = _back_transform(token_pos, pred_frac)
        metrics_bt = _regression_metrics(true_bt, pred_bt)
        metrics_frac = _regression_metrics(true_frac, pred_frac)
        per_element[sym] = {
            "token_pos": token_pos,
            "n": int(kept.size),
            "unit": label,
            "backtransformed": metrics_bt,
            "frac": metrics_frac,
        }
        panel = {
            "label": label,
            "n": int(kept.size),
            "true_bt": true_bt,
            "pred_bt": pred_bt,
            "supergroup": sg_all[kept],
            "metrics_bt": metrics_bt,
        }
        (major_panels if token_pos < len(MAJORS) else trace_panels).append(panel)
        print(f"  {sym:<3s} n={kept.size:<6d} MAE={metrics_bt['mae']:.3g} R2={metrics_bt['r2']:.3f}")

    if major_panels:
        _plot_group(
            major_panels, f"major imputation -- {tag}",
            output_dir / "scatter_majors.png", log_scale=False,
        )
    if trace_panels:
        _plot_group(
            trace_panels, f"trace imputation -- {tag}",
            output_dir / "scatter_traces.png", log_scale=True,
        )

    metrics = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "checkpoint_epoch": ckpt.epoch,
        "checkpoint_val_loss": ckpt.val_loss,
        "n_test": int(len(test_idx)),
        "latent_dim": int(ckpt.config["d_model"]),
        "per_element": per_element,
    }
    metrics_path = output_dir / "imputation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"metrics -> {metrics_path}")
    return metrics

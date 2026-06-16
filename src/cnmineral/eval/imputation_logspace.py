"""
Log-space evaluation of the frozen-encoder imputation head.

Scores a trained `LogImputationHead` the way the trace-imputation gate scores
its predictors: leave-one-element-out on the encoder's Gard test split, in
log space.  For each element it masks the target, encodes with the frozen
encoder, reads the head's de-standardized `log10(frac)` prediction, and
reports `log_r2` / `log_mae` / `log_rmse` against the withheld truth.

The gate's two reference lines are drawn for context, never recomputed here:
the input-element XGBoost ceiling (the best any predictor of these inputs can
do) and the frozen-latent XGBoost target (a strong nonlinear readout's reach
on this latent).  The latter is a reference, not a hard ceiling -- it is a
tuned gradient-boosted forest, so a linear or one-layer head can trail it on
readout capacity alone even when the latent holds the signal.  The head should
approach it and stay under the input ceiling.  Both are read from the gate's
`diagnostic_metrics.json` if present.

Artifacts land under `results/imputation/<encoder_tag>/readout/`, beside (not
over) the SSL head's `results/imputation/<encoder_tag>/` outputs.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cnmineral.data.tokens import MAJORS, MASK_TOKEN_ID, VOCAB
from cnmineral.eval.imputation import _back_transform, _plot_group, _regression_metrics
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.eval.supergroup import map_qap_array
from cnmineral.models.encoder import Encoder
from cnmineral.models.imputation_head import LogImputationHead
from cnmineral.training import checkpoint_tag, create_dataset, pick_device, reserve_test
from cnmineral.training.imputation_head import ImputationHeadCheckpoint

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
GATE_RESULTS_ROOT = (
    REPO_ROOT / "explorations" / "trace_imputation_loss_gate" / "results"
)

_N_MAJORS = len(MAJORS)
_LOG_FLOOR = 1e-12


@torch.no_grad()
def _impute_token_logspace(
    encoder: Encoder,
    head: LogImputationHead,
    dataset,
    indices: np.ndarray,
    token_pos: int,
    *,
    device: torch.device | str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-out impute `token_pos`, returning predictions in log space.

    Masks exactly `token_pos` (`src=MASK_TOKEN_ID, frac=0`), frozen-encodes,
    and reads `head.predict_log_frac` at that position.  Returns
    `(kept_dataset_idx, true_frac, pred_log_frac)` for the originally-populated
    rows, each aligned; `pred_log_frac` is `log10(frac)`, `true_frac` is linear
    (the caller logs it).
    """
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=batch_size,
        shuffle=False, num_workers=0,
    )
    kept_idx, true_chunks, pred_chunks = [], [], []
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
        pred_log = head.predict_log_frac(encoder(src=src_m, frac=frac_m))
        pred_log = pred_log[:, token_pos].cpu().numpy()

        local = np.nonzero(populated)[0]
        kept_idx.append(indices[offset + local])
        true_chunks.append(frac[local, token_pos].numpy())
        pred_chunks.append(pred_log[local])
        offset += b

    if not any(c.size for c in true_chunks):
        empty = np.empty(0)
        return empty.astype(np.int64), empty.astype(np.float32), empty.astype(np.float32)
    return (
        np.concatenate(kept_idx),
        np.concatenate(true_chunks),
        np.concatenate(pred_chunks),
    )


def _load_gate_references(encoder_tag: str) -> dict[str, dict[str, float]]:
    """Per-element `{input_xgb_ceiling_r2, frozen_latent_xgb_r2}` from the gate.

    Empty dict when the gate has not been run for this encoder -- the eval
    still produces the head's own numbers, just without the reference lines.
    """
    path = GATE_RESULTS_ROOT / encoder_tag / "diagnostic_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        gate = json.load(f)
    refs: dict[str, dict[str, float]] = {}
    for sym, rec in gate.get("per_element", {}).items():
        refs[sym] = {
            "input_xgb_ceiling_r2": rec["run_1a"]["r2"],
            "frozen_latent_xgb_r2": rec["run_1b"]["r2"],
        }
    return refs


def _plot_logspace_r2(
    per_element: dict, tag: str, out_path: Path,
) -> None:
    """Per-element head log-R2 bars with the gate's two reference markers."""
    syms = [s for s, _z in VOCAB if s in per_element]
    majors = [s for s in syms if per_element[s]["token_pos"] < _N_MAJORS]
    traces = [s for s in syms if per_element[s]["token_pos"] >= _N_MAJORS]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(syms) * 0.5), 8))
    for ax, group, gtitle in ((axes[0], majors, "majors"), (axes[1], traces, "traces")):
        x = np.arange(len(group))
        vals = [per_element[s]["log_r2"] for s in group]
        ax.bar(x, vals, 0.6, color="#d95f02", label="head log-R2")
        ceil = [per_element[s].get("input_xgb_ceiling_r2") for s in group]
        latent = [per_element[s].get("frozen_latent_xgb_r2") for s in group]
        if any(v is not None for v in ceil):
            ax.scatter(x, ceil, marker="_", s=400, color="#1b9e77",
                       label="input XGB ceiling")
        if any(v is not None for v in latent):
            ax.scatter(x, latent, marker="_", s=400, color="#7570b3",
                       label="frozen-latent XGB target")
        ax.set_xticks(x)
        ax.set_xticklabels(group)
        ax.set_ylabel("log-space R2")
        ax.set_title(gtitle)
        # drop the floor below 0 only when a bar is actually negative, so a
        # failing element renders as a negative bar instead of clipping to 0.
        ax.set_ylim(min(0.0, *vals) if vals else 0.0, 1.0)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"imputation head log-space R2 -- {tag}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_imputation_logspace(
    head_checkpoint_path: Path,
    *,
    device: torch.device | None = None,
    batch_size: int = 512,
) -> dict:
    """Score a trained imputation head in log space over its encoder's test split.

    Writes per-element log-space metrics (with the gate's reference lines when
    available), an R2 bar chart, and trace parity grids to
    `results/imputation/<encoder_tag>/readout/`, and returns the metrics dict.
    """
    head_ckpt = ImputationHeadCheckpoint.load(head_checkpoint_path)
    encoder_ckpt_path = Path(head_ckpt.encoder_checkpoint)
    tag = checkpoint_tag(encoder_ckpt_path)
    output_dir = RESULTS_ROOT / "imputation" / tag / "readout"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    print(f"device:          {device}")
    print(f"head checkpoint: {head_checkpoint_path}")
    print(f"encoder:         {encoder_ckpt_path}")
    print(f"output:          {output_dir}")

    encoder, enc_ckpt = load_encoder_from_checkpoint(encoder_ckpt_path)
    encoder.to(device)
    head = head_ckpt.build_head().to(device)

    dataset = create_dataset(enc_ckpt.config)
    test_idx = np.asarray(reserve_test(dataset, enc_ckpt.config)["test"], dtype=np.int64)
    sg_all = map_qap_array(dataset.qap_names)
    references = _load_gate_references(tag)
    print(f"dataset n={len(dataset)} test={len(test_idx)} "
          f"gate_refs={'yes' if references else 'no'}")

    per_element: dict[str, dict] = {}
    trace_panels: list[dict] = []
    for token_pos, (sym, _z) in enumerate(VOCAB):
        kept, true_frac, pred_log = _impute_token_logspace(
            encoder, head, dataset, test_idx, token_pos,
            device=device, batch_size=batch_size,
        )
        if kept.size == 0:
            continue
        true_log = np.log10(np.clip(true_frac, _LOG_FLOOR, None))
        m = _regression_metrics(true_log, pred_log)
        rec = {
            "token_pos": token_pos,
            "n": int(kept.size),
            "unit": _back_transform(token_pos, true_frac[:1])[1],
            "log_r2": m["r2"],
            "log_mae": m["mae"],
            "log_rmse": m["rmse"],
        }
        ref = references.get(sym)
        if ref is not None:
            rec.update(ref)
        per_element[sym] = rec

        if token_pos >= _N_MAJORS:
            bt_true, label = _back_transform(token_pos, true_frac)
            bt_pred, _ = _back_transform(token_pos, np.power(10.0, pred_log))
            trace_panels.append({
                "label": label, "n": int(kept.size), "true_bt": bt_true,
                "pred_bt": bt_pred, "supergroup": sg_all[kept],
                "metrics_bt": {"mae": m["mae"], "r2": m["r2"]},
            })
        print(f"  {sym:<3s} n={kept.size:<6d} log_R2={m['r2']:.3f}")

    _plot_logspace_r2(per_element, tag, output_dir / "r2_by_element.png")
    if trace_panels:
        _plot_group(trace_panels, f"imputation head traces (log) -- {tag}",
                    output_dir / "trace_parity.png", log_scale=True)

    summary = _summarize(per_element)
    metrics = {
        "head_checkpoint": str(head_checkpoint_path),
        "encoder_checkpoint": str(encoder_ckpt_path),
        "device": str(device),
        "space": "log10",
        "head_config": head_ckpt.head_config,
        "n_test": int(len(test_idx)),
        "latent_dim": int(enc_ckpt.config["d_model"]),
        "per_element": per_element,
        "trace_summary": summary,
    }
    metrics_path = output_dir / "readout_logspace_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    _print_summary(summary)
    print(f"metrics -> {metrics_path}")
    return metrics


def _summarize(per_element: dict) -> dict:
    """Median trace head log-R2 and, when present, the two reference medians."""
    head, ceil, latent = [], [], []
    for sym, rec in per_element.items():
        if rec["token_pos"] < _N_MAJORS:
            continue
        head.append(rec["log_r2"])
        if "input_xgb_ceiling_r2" in rec:
            ceil.append(rec["input_xgb_ceiling_r2"])
        if "frozen_latent_xgb_r2" in rec:
            latent.append(rec["frozen_latent_xgb_r2"])
    return {
        "n_traces": len(head),
        "median_head_log_r2": float(np.median(head)) if head else float("nan"),
        "median_input_xgb_ceiling_r2": float(np.median(ceil)) if ceil else None,
        "median_frozen_latent_xgb_r2": float(np.median(latent)) if latent else None,
    }


def _print_summary(summary: dict) -> None:
    head = summary["median_head_log_r2"]
    latent = summary["median_frozen_latent_xgb_r2"]
    ceil = summary["median_input_xgb_ceiling_r2"]
    print(f"\nimputation head -- {summary['n_traces']} traces, log-space R2:")
    print(f"  head: median {head:.3f}")
    if latent is not None:
        frac = head / latent if latent > 0 else float("nan")
        print(f"  frozen-latent XGBoost target: median {latent:.3f} "
              f"(head recovers {frac:.0%} of it)")
    if ceil is not None:
        frac = head / ceil if ceil > 0 else float("nan")
        print(f"  input-element XGBoost ceiling: median {ceil:.3f} "
              f"(head reaches {frac:.0%} of it)")
    if latent is None and ceil is None:
        print("  (no gate references found for this encoder; run the trace "
              "gate to populate them)")

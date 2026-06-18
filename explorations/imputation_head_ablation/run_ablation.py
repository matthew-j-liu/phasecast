"""Read-out head design ablation: capacity, standardization, masking scheme.

The trace-imputation gate (`explorations/trace_imputation_loss_gate/`) settled
*where* to fix imputation (a head on the frozen latent, not a re-pretrain).
This ablation settles *how* to build that head, by training the shipped
`LogImputationHead` four ways on the frozen encoder and scoring each
leave-one-out in log space on the Gard test split, against the gate's
yardsticks (frozen-latent XGBoost 0.839, input-element XGBoost ceiling 0.892
medians over the 13 traces).

  L-std    linear  + standardize   (does per-element std matter for a linear head?)
  L-raw    linear  + no standardize (isolates what log space alone buys)
  MLP-std  MLP(256) + standardize  (does readout capacity matter?)
  L-1tok   linear  + standardize, single-token masking (does matching the LOO
           eval's one-masked-token regime close the gap to the yardstick?)

Read-only on the encoder: it is frozen (eval, no grad) and only produces
pooled latents.  Writes `results/<encoder_tag>/ablation_metrics.json`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from cnmineral.data.tokens import MAJORS, MASK_TOKEN_ID, VOCAB, mask_for_ssl
from cnmineral.eval.imputation import _regression_metrics
from cnmineral.eval.imputation_logspace import _impute_token_logspace
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.losses.pretrain_loss import masked_element_log_mse
from cnmineral.models.imputation_head import LogImputationHead
from cnmineral.training import checkpoint_tag, create_dataset, pick_device, reserve_test
from cnmineral.training.pretrain_objective import log_frac_stats
from cnmineral.training.splits import _qap_labels_for_indices

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
_N_MAJORS = len(MAJORS)
_LOG_FLOOR = 1e-12


def single_token_mask(
    src: torch.Tensor, frac: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask exactly one populated token per sample (matches the LOO eval)."""
    populated = src > 0
    scores = torch.rand(src.shape, device=src.device)
    scores[~populated] = -1.0
    choice = scores.argmax(dim=1)
    has_pop = populated.any(dim=1)
    mask_positions = torch.zeros_like(src, dtype=torch.bool)
    rows = torch.arange(src.shape[0], device=src.device)
    mask_positions[rows[has_pop], choice[has_pop]] = True
    src_c, frac_c = src.clone(), frac.clone()
    src_c[mask_positions] = MASK_TOKEN_ID
    frac_c[mask_positions] = 0.0
    return src_c, frac_c, mask_positions


def train_head(encoder, dataset, train_idx, mean, std, cfg, *, hidden, standardize,
               masker, mask_prob, device, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    head = LogImputationHead(d_model=cfg["d_model"], hidden=hidden, standardize=standardize).to(device)
    if standardize:
        head.fit_standardization(mean, std)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    for _ep in range(epochs):
        head.train()
        for batch in loader:
            src = batch["src"].to(device)
            frac = batch["frac"].to(device)
            if masker == "ssl":
                m = mask_for_ssl(src=src, frac=frac, mask_prob=mask_prob)
                sc, fc, mp, tf = m.src_corrupted, m.frac_corrupted, m.mask_positions, m.target_frac
            else:
                sc, fc, mp = masker(src, frac)
                tf = frac
            with torch.no_grad():
                pooled = encoder(src=sc, frac=fc)
            loss = masked_element_log_mse(
                head(pooled), tf, mp, target_mean=head.target_mean, target_std=head.target_std,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
    return head


def eval_head(encoder, head, dataset, test_idx, device):
    head.eval()
    out = {}
    for tp, (sym, _z) in enumerate(VOCAB):
        kept, true_frac, pred_log = _impute_token_logspace(
            encoder, head, dataset, test_idx, tp, device=device, batch_size=512)
        if kept.size == 0:
            continue
        true_log = np.log10(np.clip(true_frac, _LOG_FLOOR, None))
        out[sym] = {"token_pos": tp, "log_r2": _regression_metrics(true_log, pred_log)["r2"]}
    return out


def run(encoder_ckpt: str, *, epochs: int, batch_size: int, lr: float) -> dict:
    device = pick_device()
    encoder, enc = load_encoder_from_checkpoint(encoder_ckpt)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    cfg = enc.config
    seed = int(cfg.get("seed", 0))
    mask_prob = cfg["mask_prob"]

    dataset = create_dataset(cfg)
    splits = reserve_test(dataset, cfg)
    trainval, test_idx = splits["trainval"], np.asarray(splits["test"], dtype=np.int64)
    train_idx, _val = train_test_split(
        trainval, test_size=0.20, random_state=seed,
        stratify=_qap_labels_for_indices(dataset, trainval),
    )
    mean, std = log_frac_stats(dataset, train_idx, batch_size=batch_size, num_workers=0)
    print(f"device={device} encoder={Path(encoder_ckpt).name} n_train={len(train_idx)} "
          f"n_test={len(test_idx)} mask_prob={mask_prob:.3f} epochs={epochs}")

    variants = [
        ("L-std",   dict(hidden=None, standardize=True,  masker="ssl")),
        ("L-raw",   dict(hidden=None, standardize=False, masker="ssl")),
        ("MLP-std", dict(hidden=256,  standardize=True,  masker="ssl")),
        ("L-1tok",  dict(hidden=None, standardize=True,  masker=single_token_mask)),
    ]
    per_variant = {}
    for name, v in variants:
        t = time.time()
        head = train_head(
            encoder, dataset, train_idx, mean, std, cfg,
            hidden=v["hidden"], standardize=v["standardize"], masker=v["masker"],
            mask_prob=mask_prob, device=device, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
        )
        res = eval_head(encoder, head, dataset, test_idx, device)
        traces = [r["log_r2"] for r in res.values() if r["token_pos"] >= _N_MAJORS]
        majors = [r["log_r2"] for r in res.values() if r["token_pos"] < _N_MAJORS]
        per_variant[name] = {
            "config": {"hidden": v["hidden"], "standardize": v["standardize"],
                       "masking": "single_token" if v["masker"] != "ssl" else "ssl"},
            "median_trace_log_r2": float(np.median(traces)),
            "min_trace_log_r2": float(np.min(traces)),
            "median_major_log_r2": float(np.median(majors)),
            "per_element": {s: r["log_r2"] for s, r in res.items()},
        }
        print(f"[{name}] {time.time()-t:.0f}s  median trace={np.median(traces):.3f}  "
              f"min trace={np.min(traces):.3f}  median major={np.median(majors):.3f}", flush=True)

    return {
        "encoder_checkpoint": str(encoder_ckpt),
        "space": "log10",
        "epochs": epochs,
        "yardsticks": {"frozen_latent_xgb_median": 0.839, "input_xgb_ceiling_median": 0.892},
        "variants": per_variant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=Path("checkpoints/pretrain_v3_seed0/epoch_best.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    metrics = run(str(args.encoder_checkpoint.resolve()),
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    out_dir = RESULTS_ROOT / checkpoint_tag(args.encoder_checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics -> {path}")


if __name__ == "__main__":
    main()

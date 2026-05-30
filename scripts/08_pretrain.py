"""SSL pretrain (v2): masked-element modeling on unlabeled igneous geochemistry.

Trains `cnmineral.models.encoder.Encoder` end-to-end on ~474k unlabeled bulks
filtered from complete.csv (excluding the v1 supervised train/val/test bulks
so the SSL pretrain stays genuinely unsupervised w.r.t. our eval). Pretext:
mask a random fraction of each rock's populated tokens by zeroing their
`frac`, predict the original `frac` from context via a per-token MLP, MSE
loss on masked positions only.

Per-class oversampling (lifted from mineralML's §2.1 practice) keeps rare
petrogenetic supergroups (ultramafic ~1%, felsic-plutonic ~7%, foid-bearing
~9%) from being drowned out by mafic-volcanic (~28%).

Saves encoder.state_dict() + a manifest of the cation vocabulary and config
to checkpoints/pretrain_v2/. The PretrainHead is discarded.

Run:
    python scripts/08_pretrain.py --config configs/pretrain_v2.yaml
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnmineral.data.unlabeled import (
    SUPERGROUPS, UnlabeledV2Dataset, V2_CATIONS_ORDER, V2_N_TOKENS,
)
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead
from cnmineral.models.pretrain_losses import (
    apply_mask, masked_element_mse, sample_mask,
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()):
        return torch.device("mps")
    return torch.device("cpu")


def make_supergroup_weights(supergroups: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights per row, normalized to sum to len(supergroups).

    Effect: WeightedRandomSampler draws each supergroup equally often in
    expectation, regardless of its raw count in the corpus.
    """
    unique, inv = np.unique(supergroups, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    per_group_w = 1.0 / counts
    per_row_w = per_group_w[inv]
    per_row_w = per_row_w * (len(supergroups) / per_row_w.sum())
    return per_row_w


def build_loader(
    dataset, indices: np.ndarray, batch_size: int, num_workers: int,
    weights: np.ndarray | None = None, shuffle: bool = False, seed: int = 42,
) -> DataLoader:
    """Wrap a Subset in a DataLoader, optionally with WeightedRandomSampler."""
    subset = Subset(dataset, indices.tolist())
    if weights is not None:
        sub_weights = torch.as_tensor(weights[indices], dtype=torch.float64)
        gen = torch.Generator()
        gen.manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=sub_weights, num_samples=len(indices),
            replacement=True, generator=gen,
        )
        return DataLoader(subset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, drop_last=False)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False)


def run_epoch(
    encoder: Encoder, head: PretrainHead, loader: DataLoader,
    optimizer: AdamW | None, device: torch.device,
    mask_p: float, grad_clip: float | None,
    desc: str, progress: bool,
    per_token_std: torch.Tensor | None = None,
) -> dict[str, float]:
    """One pass over the loader. `optimizer=None` means eval (no grad)."""
    training = optimizer is not None
    encoder.train(training)
    head.train(training)
    accum = defaultdict(list)

    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if progress else loader
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for batch in iterator:
            src = batch["src"].to(device)
            frac = batch["frac"].to(device)

            mask = sample_mask(src, frac, mask_p)
            # Zero BOTH src and frac at masked positions so kingcrab pads them
            # out cleanly via its frac>0 logic. Otherwise (the original v0 of
            # this code) the encoder treated masked tokens as padding via frac
            # alone but the pretrain head expected per-token predictions back,
            # and gradients never reached the encoder. Now: encoder pools only
            # the surviving tokens; head reconstructs the full vocab from that
            # pooled latent.
            src_masked = torch.where(mask, torch.zeros_like(src), src)
            frac_masked = apply_mask(frac, mask)

            pooled = encoder(src_masked, frac_masked)             # [B, d_model]
            pred = head(pooled)                                    # [B, n_tokens]
            loss = masked_element_mse(pred, frac, mask, per_token_std=per_token_std)

            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(head.parameters()),
                        grad_clip,
                    )
                optimizer.step()

            n_masked = int(mask.sum().item())
            accum["loss"].append(float(loss.item()))
            accum["n_masked"].append(n_masked)
            if progress:
                iterator.set_postfix(loss=f"{loss.item():.4e}", n_mask=n_masked)
    return {k: float(np.mean(v)) for k, v in accum.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/pretrain_v2.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override pretrain.epochs from config (for smoke runs).")
    parser.add_argument("--limit-bulks", type=int, default=None,
                        help="Subsample dataset to N bulks after filtering (smoke runs).")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Appended to checkpoint_dir for smoke runs (e.g. '_smoke').")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["pretrain"]["epochs"] = args.epochs
    if args.output_suffix:
        cfg["output"]["checkpoint_dir"] = cfg["output"]["checkpoint_dir"] + args.output_suffix
        cfg["output"]["metrics_log"] = (
            cfg["output"]["metrics_log"].replace(
                "/metrics.jsonl", f"{args.output_suffix}/metrics.jsonl")
            if "/metrics.jsonl" in cfg["output"]["metrics_log"]
            else cfg["output"]["metrics_log"] + args.output_suffix)

    device = pick_device()
    print(f"device: {device}")

    # Resolve exclusion list from v1 splits manifest, if configured.
    splits_path = cfg["data"].get("splits_path_for_exclusion")
    exclude_ids = None
    if splits_path:
        with open(REPO_ROOT / splits_path) as f:
            splits = json.load(f)
        exclude_ids = (splits["train_sample_ids"]
                       + splits["val_sample_ids"]
                       + splits["test_sample_ids"]
                       + splits.get("dropped_class_sample_ids", []))
        print(f"excluding {len(exclude_ids)} supervised bulks from SSL corpus")

    t0 = time.time()
    dataset = UnlabeledV2Dataset(
        REPO_ROOT / cfg["data"]["complete_csv_path"],
        exclude_sample_ids=exclude_ids,
        min_sum=cfg["data"].get("min_sum", 80.0),
        max_sum=cfg["data"].get("max_sum", 120.0),
    )
    print(f"loaded {len(dataset):,} unlabeled bulks in {time.time()-t0:.1f}s")
    if args.limit_bulks is not None and args.limit_bulks < len(dataset):
        rng_lim = np.random.default_rng(cfg["pretrain"]["seed"])
        keep_idx = np.sort(rng_lim.choice(len(dataset), args.limit_bulks, replace=False))
        # rebuild via direct subset (preserve dataset interface)
        dataset._src = dataset._src[keep_idx]
        dataset._frac = dataset._frac[keep_idx]
        dataset._sample_ids = dataset._sample_ids[keep_idx]
        dataset._qap_names = dataset._qap_names[keep_idx]
        dataset._supergroups = dataset._supergroups[keep_idx]
        dataset.sample_ids = dataset._sample_ids
        dataset.qap_names = dataset._qap_names
        dataset.supergroups = dataset._supergroups
        print(f"limited to {args.limit_bulks:,} bulks for smoke run")
    print("supergroup counts:")
    counts = dataset.supergroup_counts()
    for sg in SUPERGROUPS:
        n = counts.get(sg, 0)
        if n > 0:
            print(f"  {sg:>22s}: {n:>7,}")

    # Train/val split: stratified-ish by supergroup via shuffle on indices.
    seed = cfg["pretrain"]["seed"]
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(dataset))
    rng.shuffle(all_idx)
    n_val = int(len(all_idx) * cfg["data"]["val_frac"])
    val_idx = all_idx[:n_val]
    train_idx = all_idx[n_val:]
    print(f"split: train={len(train_idx):,}  val={len(val_idx):,}")

    weights = None
    if cfg.get("sampling", {}).get("supergroup_balanced", True):
        weights = make_supergroup_weights(dataset.supergroups)
        print("using supergroup-balanced sampling on the train loader")

    # Per-token std over the populated values in the train pool. Used by the
    # per-element-normalized MSE loss (if cfg.pretrain.per_token_norm is on)
    # so traces contribute proportionally to the loss budget instead of being
    # drowned by the major-element MSE magnitudes.
    per_token_std = None
    if cfg.get("pretrain", {}).get("per_token_norm", False):
        train_frac = dataset._frac[train_idx]   # [N, T]
        # Std over POPULATED values per column (zeros = absent, exclude).
        std_per = np.zeros(train_frac.shape[1], dtype=np.float32)
        for t in range(train_frac.shape[1]):
            v = train_frac[:, t]
            v_pop = v[v > 0]
            std_per[t] = float(v_pop.std()) if len(v_pop) > 10 else 1.0
        std_per = np.maximum(std_per, 1e-6)
        per_token_std = torch.tensor(std_per, dtype=torch.float32, device=device)
        print(f"per-token-normalized MSE: std range [{std_per.min():.2e}, {std_per.max():.2e}]")

    train_loader = build_loader(
        dataset, train_idx,
        batch_size=cfg["pretrain"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        weights=weights, shuffle=(weights is None), seed=seed,
    )
    val_loader = build_loader(
        dataset, val_idx,
        batch_size=cfg["pretrain"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        weights=None, shuffle=False,
    )

    enc_cfg = cfg["encoder"]
    encoder = Encoder(
        d_model=enc_cfg["d_model"], N=enc_cfg["N"], heads=enc_cfg["heads"],
        dropout=enc_cfg["dropout"], elem_prop=enc_cfg["elem_prop"],
    ).to(device)
    head_cfg = cfg.get("pretrain_head", {})
    head = PretrainHead(
        d_model=enc_cfg["d_model"],
        n_tokens=V2_N_TOKENS,
        hidden=head_cfg.get("hidden"),
        dropout=head_cfg.get("dropout", 0.1),
    ).to(device)

    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_head = sum(p.numel() for p in head.parameters())
    print(f"encoder params: {n_params_enc:,}  head params: {n_params_head:,}")

    optimizer = AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg["pretrain"]["lr"],
        weight_decay=cfg["pretrain"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg["pretrain"]["epochs"],
        eta_min=cfg["pretrain"]["cosine_eta_min"],
    )

    out_dir = REPO_ROOT / cfg["output"]["checkpoint_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = REPO_ROOT / cfg["output"]["metrics_log"]
    metrics_log.parent.mkdir(parents=True, exist_ok=True)

    # Seed the default RNG once; mask sampling uses the per-device default
    # generator (torch.Generator can't cross CPU/MPS).
    torch.manual_seed(seed)

    best_val = float("inf")
    progress = cfg["output"].get("progress_bar", True)
    for epoch in range(cfg["pretrain"]["epochs"]):
        t_epoch = time.time()
        train_metrics = run_epoch(
            encoder, head, train_loader, optimizer, device,
            mask_p=cfg["pretrain"]["mask_p"],
            grad_clip=cfg["pretrain"].get("grad_clip"),
            desc=f"epoch {epoch} train", progress=progress, per_token_std=per_token_std,
        )
        val_metrics = run_epoch(
            encoder, head, val_loader, optimizer=None, device=device,
            mask_p=cfg["pretrain"]["mask_p"],
            grad_clip=None,
            desc=f"epoch {epoch} val", progress=progress,
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_n_masked": train_metrics["n_masked"],
            "val_n_masked": val_metrics["n_masked"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "wall_s": time.time() - t_epoch,
        }
        with open(metrics_log, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"epoch {epoch:3d}  train_loss={record['train_loss']:.4e}  "
              f"val_loss={record['val_loss']:.4e}  wall={record['wall_s']:.1f}s")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "config": cfg,
                "v2_cations_order": V2_CATIONS_ORDER,
                "v2_n_tokens": V2_N_TOKENS,
            }, out_dir / "encoder_best.pt")
            print(f"           ^ saved best to {out_dir / 'encoder_best.pt'}")

    # Always also save the last (post-cosine) state for record.
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "epoch": cfg["pretrain"]["epochs"] - 1,
        "val_loss": val_metrics["loss"],
        "config": cfg,
        "v2_cations_order": V2_CATIONS_ORDER,
        "v2_n_tokens": V2_N_TOKENS,
    }, out_dir / "encoder_last.pt")
    print(f"\nbest val_loss: {best_val:.4e}")
    print(f"checkpoints -> {out_dir / 'encoder_best.pt'} (best), "
          f"{out_dir / 'encoder_last.pt'} (last)")
    print(f"metrics log -> {metrics_log}")


if __name__ == "__main__":
    main()

"""
Train the log-space imputation head on a frozen SSL encoder.

The encoder checkpoint named in the config is loaded, frozen (eval mode, no
gradients, no dropout), and used only to produce pooled latents.  A fresh
`LogImputationHead` trains on those latents with the per-element standardized
log-space objective (`masked_element_log_mse`).  Dataset and splits are
reconstructed from the *encoder's* stamped config so the head's train / val /
test partitions line up with the probe suite and the trace-imputation gate.

This is the head-gap fix from the 2026-06-15 gate: the frozen latent already
carries the trace signal, so no encoder retraining happens here.  The result
is an `ImputationHeadCheckpoint` -- head weights + de-standardization buffers +
a pointer back to the encoder checkpoint it was trained against.
"""

import datetime
from dataclasses import dataclass
from pathlib import Path
from time import time

import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from cnmineral.data.tokens import mask_for_ssl
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.losses.pretrain_loss import masked_element_log_mse
from cnmineral.models.encoder import Encoder
from cnmineral.models.imputation_head import LogImputationHead
from cnmineral.training.build_from_config import create_dataset
from cnmineral.training.checkpoint import _git_sha
from cnmineral.training.pretrain_objective import log_frac_stats
from cnmineral.training.splits import _qap_labels_for_indices, reserve_test


@dataclass(frozen=True, slots=True)
class ImputationHeadCheckpoint:
    """One serialized log-space imputation head, bound to its frozen encoder.

    `head_state` carries the weights and the de-standardization buffers;
    `head_config` carries what `build_head` needs to reconstruct the module.
    `encoder_checkpoint` records which frozen encoder produced the latents --
    re-running the head against a re-pretrained encoder reads this.  Dataset
    and splits are *not* stored: they reconstruct from the encoder checkpoint's
    own config.
    """

    head_state: dict
    head_config: dict
    encoder_checkpoint: str
    train_loss: float
    val_loss: float
    monitor: dict
    git_sha: str | None = None
    saved_at: str | None = None

    @classmethod
    def from_head(
        cls,
        head: LogImputationHead,
        encoder_checkpoint: str,
        train_loss: float,
        val_loss: float,
        monitor: dict,
    ) -> "ImputationHeadCheckpoint":
        return cls(
            head_state=head.state_dict(),
            head_config={
                "d_model": head.d_model,
                "hidden": head.hidden,
                "dropout": head.dropout,
            },
            encoder_checkpoint=str(encoder_checkpoint),
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            monitor=monitor,
            git_sha=_git_sha(),
            saved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    @classmethod
    def load(cls, path: Path | str) -> "ImputationHeadCheckpoint":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            head_state=payload["head"],
            head_config=payload["head_config"],
            encoder_checkpoint=payload["encoder_checkpoint"],
            train_loss=payload["train_loss"],
            val_loss=payload["val_loss"],
            monitor=payload["monitor"],
            git_sha=payload.get("git_sha"),
            saved_at=payload.get("saved_at"),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head": self.head_state,
                "head_config": self.head_config,
                "encoder_checkpoint": self.encoder_checkpoint,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "monitor": self.monitor,
                "git_sha": self.git_sha,
                "saved_at": self.saved_at,
            },
            path,
        )

    def build_head(self) -> LogImputationHead:
        """Reconstruct the head (eval mode, CPU) with its weights + buffers.

        A stale `standardize` key on an old checkpoint's `head_config` is
        ignored -- the head is always standardized now.
        """
        head = LogImputationHead(
            d_model=self.head_config["d_model"],
            hidden=self.head_config["hidden"],
            dropout=self.head_config.get("dropout", 0.0),
        )
        head.load_state_dict(self.head_state)
        head.eval()
        return head


def _head_forward_loss(
    batch: dict,
    encoder: Encoder,
    head: LogImputationHead,
    mask_prob: float,
    device: torch.device,
) -> torch.Tensor:
    """mask -> frozen encode (no grad) -> head -> log-space MSE."""
    src = batch["src"].to(device)
    frac = batch["frac"].to(device)
    masked = mask_for_ssl(src=src, frac=frac, mask_prob=mask_prob)
    with torch.no_grad():
        pooled = encoder(src=masked.src_corrupted, frac=masked.frac_corrupted)
    predicted = head(pooled)
    return masked_element_log_mse(
        predicted, masked.target_frac, masked.mask_positions,
        target_mean=head.target_mean, target_std=head.target_std,
    )


def _train_val_split(
    dataset, trainval: list[int], val_split: float, seed: int,
) -> tuple[list[int], list[int]]:
    """Stratified (by `qap_name`) inner split of the trainval indices.

    Shared by the production trainer and the head HPO so both carve the same
    val split from the same encoder's trainval -- the tuned numbers and the
    shipped head are then scored on identical held-out rows.
    """
    if val_split <= 0:
        raise ValueError(f"val_split must be > 0; got {val_split}")
    train_indices, val_indices = train_test_split(
        trainval, test_size=val_split, random_state=seed,
        stratify=_qap_labels_for_indices(dataset, trainval),
    )
    if not train_indices or not val_indices:
        raise ValueError(
            f"train/val split produced empty subset: "
            f"trainval={len(trainval)}, val_split={val_split}"
        )
    return train_indices, val_indices


def train_head_on_split(
    encoder: Encoder,
    enc_config: dict,
    dataset,
    train_indices: list[int],
    val_indices: list[int],
    config: dict,
    device: torch.device,
    *,
    verbose: bool = True,
) -> tuple[LogImputationHead, dict[str, list[float]], float, float]:
    """Train one head on a fixed train/val split; return it at its best val.

    Builds the head from `config` (`hidden`, `dropout`), fits standardization
    on `train_indices`, and trains up to `epochs` with val-loss
    early stopping: the best-val weights are restored before returning, and an
    `early_stop_patience` (when set) halts once val has not improved for that
    many epochs.  Returns `(head, monitor, train_loss, val_loss)` where the two
    losses are the best-val epoch's, not the last epoch's.  The encoder is
    assumed already frozen + on `device`.
    """
    mask_prob = config.get("mask_prob", enc_config["mask_prob"])
    num_workers = config.get("num_workers", 0)
    batch_size = config["batch_size"]
    patience = config.get("early_stop_patience")

    head = LogImputationHead(
        d_model=enc_config["d_model"],
        hidden=config.get("hidden"),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    mean, std = log_frac_stats(
        dataset, train_indices, batch_size=batch_size, num_workers=num_workers,
    )
    head.fit_standardization(mean, std)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )

    monitor: dict[str, list[float]] = {"train": [], "val": []}
    best_val = float("inf")
    best_train = float("nan")
    best_state: dict | None = None
    epochs_no_improve = 0
    for epoch in range(config["epochs"]):
        head.train()
        train_total, n_train = 0.0, 0
        epoch_start = time()
        for batch in train_loader:
            loss = _head_forward_loss(batch, encoder, head, mask_prob, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += loss.item()
            n_train += 1
        assert n_train > 0, "no train batches processed"
        train_mean = train_total / n_train
        monitor["train"].append(train_mean)

        head.eval()
        val_total, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                val_total += _head_forward_loss(
                    batch, encoder, head, mask_prob, device,
                ).item()
                n_val += 1
        assert n_val > 0, "no val batches processed"
        val_mean = val_total / n_val
        monitor["val"].append(val_mean)
        if verbose:
            print(
                f"epoch {epoch + 1}/{config['epochs']} "
                f"train={train_mean:.4f} val={val_mean:.4f} "
                f"elapsed={(time() - epoch_start) * 1000:.0f}ms"
            )

        if val_mean < best_val:
            best_val, best_train = val_mean, train_mean
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch + 1} (best val={best_val:.4f})")
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, monitor, best_train, best_val


def train_imputation_head(
    config: dict, device: torch.device,
) -> ImputationHeadCheckpoint:
    """Train the log-space imputation head on the configured frozen encoder.

    Required config keys: `encoder_checkpoint`, `epochs`, `lr`, `batch_size`.
    Optional: `hidden` (None -> linear),
    `dropout` (default 0.0, MLP only), `mask_prob` (default: the encoder's
    pretrain `mask_prob`), `weight_decay`, `early_stop_patience` (None -> run
    all epochs), `val_split` (default 0.20), `num_workers`,
    `head_checkpoint_dir` + `head_checkpoint_filename` (saves the checkpoint
    when both are set).  `seed` defaults to the encoder checkpoint's seed so
    the inner split matches the rest of the pipeline.
    """
    encoder_ckpt_path = config["encoder_checkpoint"]
    encoder, enc_ckpt = load_encoder_from_checkpoint(encoder_ckpt_path)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    seed = int(config.get("seed", enc_ckpt.config.get("seed", 0)))
    torch.manual_seed(seed)

    dataset = create_dataset(enc_ckpt.config)
    trainval = reserve_test(dataset, enc_ckpt.config)["trainval"]
    train_indices, val_indices = _train_val_split(
        dataset, trainval, config.get("val_split", 0.20), seed,
    )

    print(
        f"train_imputation_head: device={device} encoder={Path(encoder_ckpt_path).name} "
        f"n_train={len(train_indices)} n_val={len(val_indices)} "
        f"hidden={config.get('hidden')} "
        f"dropout={config.get('dropout', 0.0)} epochs={config['epochs']} "
        f"batch_size={config['batch_size']} lr={config['lr']} "
        f"mask_prob={config.get('mask_prob', enc_ckpt.config['mask_prob'])}"
    )

    head, monitor, train_loss, val_loss = train_head_on_split(
        encoder, enc_ckpt.config, dataset, train_indices, val_indices, config, device,
    )

    ckpt = ImputationHeadCheckpoint.from_head(
        head, str(encoder_ckpt_path), train_loss, val_loss, monitor,
    )
    if config.get("head_checkpoint_dir"):
        out_path = (
            Path(config["head_checkpoint_dir"]) / config["head_checkpoint_filename"]
        )
        ckpt.save(out_path)
        print(f"head checkpoint -> {out_path}")
    return ckpt


def run_single(config_path: Path, device: torch.device) -> dict:
    """Train the head, then score the saved checkpoint on the test split once.

    Mirrors the encoder's single-run entry: one pass trains and reports its
    held-out number.  Test scoring reads a saved checkpoint, so it runs only
    when `head_checkpoint_dir` / `head_checkpoint_filename` are set; otherwise
    the head still trains and the scorecard is skipped with a note.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ckpt = train_imputation_head(config, device)
    result = {"train_loss": ckpt.train_loss, "val_loss": ckpt.val_loss}

    if not config.get("head_checkpoint_dir"):
        print("no head_checkpoint_dir set -- skipping test scoring")
        return result

    # local import: cnmineral.eval.imputation imports back from this module
    # (ImputationHeadCheckpoint), so a top-level import would cycle.
    from cnmineral.eval.imputation import run_imputation

    saved_path = Path(config["head_checkpoint_dir"]) / config["head_checkpoint_filename"]
    metrics = run_imputation(saved_path, device=device)
    result["trace_summary"] = metrics["trace_summary"]
    return result

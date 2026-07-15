"""Train the hurdle ModeHead on a frozen SSL encoder.

The encoder checkpoint named in the config is loaded, frozen, and used once
to produce one pooled latent per corpus bulk (all draws/rungs of a bulk share
one input vector, so the latent table is tiny and lives on the device). The
head trains on those latents against the `ModeHeadDataset` hurdle targets
with `mode_head_loss`, early-stopping on val total loss over the provisional
bulk-level split. Evaluation assembles the joint hurdle NLL explicitly (the
training loss is two surrogates; the joint likelihood of a modal vector is
the summed Bernoulli log-likelihoods plus the support-Dirichlet NLL) and
reports per-phase presence calibration and recall.
"""

import datetime
from dataclasses import dataclass
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset

from cnmineral.data.mineralogy_dataset import ModeHeadDataset, provisional_bulk_split
from cnmineral.data.phase_vocab import SOLID_PHASES
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.losses.mode_loss import masked_dirichlet_nll, mode_head_loss
from cnmineral.models.encoder import Encoder
from cnmineral.models.mode_head import ModeHead
from cnmineral.training.checkpoint import _git_sha


@dataclass(frozen=True, slots=True)
class ModeHeadCheckpoint:
    """One serialized ModeHead, bound to its frozen encoder and label semantics.

    `label_config` records what the targets were built from (`corpus_dir`,
    `rungs`, `presence_cutoff`) -- scoring a checkpoint against differently
    built labels is a category error, so the semantics travel with the
    weights. Dataset rows are not stored; they rebuild from `label_config`.
    """

    head_state: dict
    head_config: dict
    encoder_checkpoint: str
    label_config: dict
    train_loss: float
    val_loss: float
    monitor: dict
    git_sha: str | None = None
    saved_at: str | None = None

    @classmethod
    def from_head(
        cls,
        head: ModeHead,
        head_config: dict,
        encoder_checkpoint: str,
        label_config: dict,
        train_loss: float,
        val_loss: float,
        monitor: dict,
    ) -> "ModeHeadCheckpoint":
        return cls(
            head_state=head.state_dict(),
            head_config=head_config,
            encoder_checkpoint=str(encoder_checkpoint),
            label_config=label_config,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            monitor=monitor,
            git_sha=_git_sha(),
            saved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    @classmethod
    def load(cls, path: Path | str) -> "ModeHeadCheckpoint":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            head_state=payload["head"],
            head_config=payload["head_config"],
            encoder_checkpoint=payload["encoder_checkpoint"],
            label_config=payload["label_config"],
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
                "label_config": self.label_config,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "monitor": self.monitor,
                "git_sha": self.git_sha,
                "saved_at": self.saved_at,
            },
            path,
        )

    def build_head(self) -> ModeHead:
        head = ModeHead(
            d_model=self.head_config["d_model"],
            hidden=self.head_config.get("hidden"),
            dropout=self.head_config.get("dropout", 0.0),
            alpha_eps=self.head_config.get("alpha_eps", 1e-6),
        )
        head.load_state_dict(self.head_state)
        head.eval()
        return head


@torch.no_grad()
def _bulk_latents(
    encoder: Encoder,
    dataset: ModeHeadDataset,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """One pooled latent per dataset bulk, on `device`."""
    outs = []
    for start in range(0, len(dataset.bulk_ids), batch_size):
        src = dataset.bulk_src[start : start + batch_size].to(device)
        frac = dataset.bulk_frac[start : start + batch_size].to(device)
        outs.append(encoder(src=src, frac=frac))
    return torch.cat(outs)


def _row_split_indices(
    dataset: ModeHeadDataset, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bulk-level split mapped down to row indices."""
    _, val_ids = provisional_bulk_split(dataset.bulk_ids, val_fraction, seed)
    is_val_bulk = np.zeros(len(dataset.bulk_ids), dtype=bool)
    is_val_bulk[np.searchsorted(dataset.bulk_ids, val_ids)] = True
    row_is_val = is_val_bulk[dataset.row_bulk_idx.numpy()]
    return np.flatnonzero(~row_is_val), np.flatnonzero(row_is_val)


def evaluate_mode_head(
    head: ModeHead,
    latents: torch.Tensor,
    dataset: ModeHeadDataset,
    row_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> dict:
    """Joint hurdle NLL + per-phase presence calibration / recall / magnitude MAE.

    Returns `joint_nll` (mean nats per row) and a `per_phase` list of dicts
    (empirical frequency, mean predicted presence probability, recall at 0.5,
    conditional-magnitude MAE on rows where the phase is truly present).
    """
    head.eval()
    n_phases = len(SOLID_PHASES)
    joint_total = 0.0
    freq = torch.zeros(n_phases)
    mean_p = torch.zeros(n_phases)
    tp = torch.zeros(n_phases)
    mag_abs_err = torch.zeros(n_phases)
    n_rows = len(row_indices)

    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            idx = row_indices[start : start + batch_size]
            presence = dataset.presence[idx].to(device)
            magnitude = dataset.magnitude[idx].to(device)
            h = latents[dataset.row_bulk_idx[idx].to(device)]
            logits, alpha = head(h)

            bern = F.binary_cross_entropy_with_logits(
                logits, presence, reduction="none"
            ).sum(-1)
            dir_nll = masked_dirichlet_nll(alpha, magnitude, presence, reduction="none")
            joint_total += (bern + dir_nll).sum().item()

            p = torch.sigmoid(logits)
            freq += presence.sum(0).cpu()
            mean_p += p.sum(0).cpu()
            tp += ((p > 0.5) & (presence > 0)).sum(0).cpu()
            cond_mean = (presence * alpha) / (presence * alpha).sum(
                -1, keepdim=True
            ).clamp_min(1e-12)
            mag_abs_err += (presence * (cond_mean - magnitude).abs()).sum(0).cpu()

    per_phase = []
    for i, name in enumerate(SOLID_PHASES):
        n_present = freq[i].item()
        per_phase.append({
            "phase": name,
            "frequency": n_present / n_rows,
            "mean_p": mean_p[i].item() / n_rows,
            "recall": tp[i].item() / n_present if n_present else float("nan"),
            "magnitude_mae": mag_abs_err[i].item() / n_present if n_present else float("nan"),
        })
    return {"joint_nll": joint_total / n_rows, "per_phase": per_phase}


def train_mode_head(config: dict, device: torch.device) -> ModeHeadCheckpoint:
    """Train the ModeHead per `config`; return the best-val checkpoint.

    Required config keys: `encoder_checkpoint`, `corpus_dir`, `epochs`, `lr`,
    `batch_size`. Optional: `rungs`, `presence_cutoff`, `hidden`, `dropout`,
    `alpha_eps`, `magnitude_weight`, `weight_decay`, `early_stop_patience`,
    `val_fraction`, `seed`, `data_kwargs`, `head_checkpoint_dir` +
    `head_checkpoint_filename`.
    """
    encoder_ckpt_path = config["encoder_checkpoint"]
    encoder, enc_ckpt = load_encoder_from_checkpoint(encoder_ckpt_path)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)

    label_config = {
        "corpus_dir": str(config["corpus_dir"]),
        "rungs": tuple(config.get("rungs", (50, 30))),
        "presence_cutoff": config.get("presence_cutoff"),
    }
    dataset_kwargs = {k: v for k, v in label_config.items() if v is not None}
    dataset = ModeHeadDataset(
        **dataset_kwargs, data_kwargs=config.get("data_kwargs")
    )
    label_config["presence_cutoff"] = float(dataset.presence_cutoff)

    latents = _bulk_latents(encoder, dataset, device)
    train_rows, val_rows = _row_split_indices(
        dataset, config.get("val_fraction", 0.20), seed
    )
    print(
        f"train_mode_head: device={device} encoder={Path(encoder_ckpt_path).name} "
        f"bulks={len(dataset.bulk_ids)} rows={len(dataset)} "
        f"n_train={len(train_rows)} n_val={len(val_rows)} (provisional split)"
    )

    head_config = {
        "d_model": enc_ckpt.config["d_model"],
        "hidden": config.get("hidden"),
        "dropout": config.get("dropout", 0.0),
        "alpha_eps": config.get("alpha_eps", 1e-6),
    }
    head = ModeHead(
        d_model=head_config["d_model"],
        hidden=head_config["hidden"],
        dropout=head_config["dropout"],
        alpha_eps=head_config["alpha_eps"],
    ).to(device)
    magnitude_weight = config.get("magnitude_weight", 1.0)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    rows = TensorDataset(dataset.row_bulk_idx, dataset.presence, dataset.magnitude)
    train_loader = DataLoader(
        Subset(rows, train_rows.tolist()), batch_size=config["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        Subset(rows, val_rows.tolist()), batch_size=config["batch_size"], shuffle=False,
    )

    def _batch_loss(batch):
        bidx, presence, magnitude = (t.to(device) for t in batch)
        logits, alpha = head(latents[bidx])
        return mode_head_loss(
            logits, alpha, presence, magnitude, magnitude_weight=magnitude_weight
        )

    monitor: dict[str, list[float]] = {
        k: [] for k in ("train", "val", "train_bce", "val_bce", "train_nll", "val_nll")
    }
    patience = config.get("early_stop_patience")
    best_val = float("inf")
    best_train = float("nan")
    best_state: dict | None = None
    epochs_no_improve = 0
    for epoch in range(config["epochs"]):
        head.train()
        sums = {"total": 0.0, "presence_bce": 0.0, "dirichlet_nll": 0.0}
        n_batches = 0
        epoch_start = time()
        for batch in train_loader:
            terms = _batch_loss(batch)
            optimizer.zero_grad()
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            for k in sums:
                sums[k] += terms[k].item()
            n_batches += 1
        monitor["train"].append(sums["total"] / n_batches)
        monitor["train_bce"].append(sums["presence_bce"] / n_batches)
        monitor["train_nll"].append(sums["dirichlet_nll"] / n_batches)

        head.eval()
        sums = {"total": 0.0, "presence_bce": 0.0, "dirichlet_nll": 0.0}
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                terms = _batch_loss(batch)
                for k in sums:
                    sums[k] += terms[k].item()
                n_batches += 1
        val_mean = sums["total"] / n_batches
        monitor["val"].append(val_mean)
        monitor["val_bce"].append(sums["presence_bce"] / n_batches)
        monitor["val_nll"].append(sums["dirichlet_nll"] / n_batches)
        print(
            f"epoch {epoch + 1}/{config['epochs']} "
            f"train={monitor['train'][-1]:.4f} val={val_mean:.4f} "
            f"(bce={monitor['val_bce'][-1]:.4f} nll={monitor['val_nll'][-1]:.4f}) "
            f"elapsed={time() - epoch_start:.1f}s"
        )

        if val_mean < best_val:
            best_val, best_train = val_mean, monitor["train"][-1]
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                print(f"early stop at epoch {epoch + 1} (best val={best_val:.4f})")
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    metrics = evaluate_mode_head(head, latents, dataset, val_rows, device)
    print(f"\nval joint hurdle NLL: {metrics['joint_nll']:.4f} nats/row")
    table = pd.DataFrame(metrics["per_phase"]).sort_values(
        "frequency", ascending=False
    )
    print(table.round(4).to_string(index=False))

    ckpt = ModeHeadCheckpoint.from_head(
        head, head_config, str(encoder_ckpt_path), label_config,
        best_train, best_val, monitor,
    )
    if config.get("head_checkpoint_dir"):
        out_path = (
            Path(config["head_checkpoint_dir"]) / config["head_checkpoint_filename"]
        )
        ckpt.save(out_path)
        print(f"head checkpoint -> {out_path}")
    return ckpt


def run_single(config_path: Path, device: torch.device) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ckpt = train_mode_head(config, device)
    return {"train_loss": ckpt.train_loss, "val_loss": ckpt.val_loss}

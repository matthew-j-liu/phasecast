"""Train the three supervised heads jointly on a frozen SSL encoder.

Same frozen-encoder scheme as `training.mode_head` (one pooled latent per
bulk, cached on device), but one optimizer over ModeHead + CompHead +
CondHead under `joint_heads_loss`: the mode hurdle terms, the present-masked
comp Dirichlet NLL, the support-conditioned per-axis GMM NLL (teacher-forced
true presence), and the mass-balance closure penalty with corpus-pinned
melt/fluid fixings. Each head is snapshotted at its own best raw val epoch
(`selection_series`); training stops once every head is `patience` epochs
past its own best. Loss weights come from the config; `joint_heads_hpo`
sweeps them (ADR 006).
"""

import datetime
from dataclasses import dataclass
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from cnmineral.data.mineralogy_dataset import JointHeadDataset, load_valtest_split
from cnmineral.eval.latent import load_encoder_from_checkpoint
from cnmineral.eval.training_curves import plot_joint_run_loss
from cnmineral.losses.joint_loss import joint_heads_loss
from cnmineral.models.comp_head import CompHead
from cnmineral.models.cond_head import CondHead
from cnmineral.models.mode_head import ModeHead
from cnmineral.training.checkpoint import _git_sha
from cnmineral.training.mode_head import _bulk_latents, evaluate_mode_head

_TERMS = ("presence_bce", "mode_nll", "comp_nll", "cond_nll", "closure")

HEAD_NAMES = ("mode", "comp", "cond")


def selection_series(monitor: dict, split: str = "val") -> dict[str, list[float]]:
    """Per-head selection curves from a run monitor, in raw (unweighted) units.

    Mode selects on the assemblage NLL `presence_bce + mode_nll` --
    deliberately NOT scaled by `magnitude_weight`, so selection (and the HPO
    objective built on it) cannot be gamed by the weight.  Comp and cond
    select on their own NLL terms.
    """
    bce = monitor[f"{split}_presence_bce"]
    nll = monitor[f"{split}_mode_nll"]
    return {
        "mode": [b + n for b, n in zip(bce, nll)],
        "comp": list(monitor[f"{split}_comp_nll"]),
        "cond": list(monitor[f"{split}_cond_nll"]),
    }


@dataclass(frozen=True, slots=True)
class JointHeadCheckpoint:
    """The three serialized heads, each at its own best-val epoch.

    Selection is per head on raw (unweighted) val terms -- mode on
    `presence_bce + mode_nll`, comp on `comp_nll`, cond on `cond_nll` -- so
    the stamped states generally come from different epochs.  Best epochs and
    values are not stamped: derive them from `monitor` via
    `selection_series`.  `label_config` records what the targets were built
    from (`corpus_dir`, `valtest_corpus_dir`, `rungs`, `presence_cutoff`,
    `comp_eps`); `loss_config` the term weights the heads were trained
    under.  Dataset rows rebuild from `label_config` plus the committed
    valtest roster, as with `ModeHeadCheckpoint`.
    """

    mode_state: dict
    comp_state: dict
    cond_state: dict
    head_config: dict
    loss_config: dict
    encoder_checkpoint: str
    label_config: dict
    monitor: dict
    git_sha: str | None = None
    saved_at: str | None = None

    @classmethod
    def from_heads(
        cls,
        mode_head: ModeHead,
        comp_head: CompHead,
        cond_head: CondHead,
        head_config: dict,
        loss_config: dict,
        encoder_checkpoint: str,
        label_config: dict,
        monitor: dict,
    ) -> "JointHeadCheckpoint":
        return cls(
            mode_state=mode_head.state_dict(),
            comp_state=comp_head.state_dict(),
            cond_state=cond_head.state_dict(),
            head_config=head_config,
            loss_config=loss_config,
            encoder_checkpoint=str(encoder_checkpoint),
            label_config=label_config,
            monitor=monitor,
            git_sha=_git_sha(),
            saved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    @classmethod
    def load(cls, path: Path | str) -> "JointHeadCheckpoint":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            mode_state=payload["mode"],
            comp_state=payload["comp"],
            cond_state=payload["cond"],
            head_config=payload["head_config"],
            loss_config=payload["loss_config"],
            encoder_checkpoint=payload["encoder_checkpoint"],
            label_config=payload["label_config"],
            monitor=payload["monitor"],
            git_sha=payload.get("git_sha"),
            saved_at=payload.get("saved_at"),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": self.mode_state,
                "comp": self.comp_state,
                "cond": self.cond_state,
                "head_config": self.head_config,
                "loss_config": self.loss_config,
                "encoder_checkpoint": self.encoder_checkpoint,
                "label_config": self.label_config,
                "monitor": self.monitor,
                "git_sha": self.git_sha,
                "saved_at": self.saved_at,
            },
            path,
        )

    def build_mode_head(self) -> ModeHead:
        head = ModeHead(
            d_model=self.head_config["d_model"],
            hidden=self.head_config.get("mode_hidden"),
            layers=self.head_config.get("mode_layers", 1),
            dropout=self.head_config.get("dropout", 0.0),
            alpha_eps=self.head_config.get("alpha_eps", 1e-6),
        )
        head.load_state_dict(self.mode_state)
        head.eval()
        return head

    def build_comp_head(self) -> CompHead:
        head = CompHead(
            d_model=self.head_config["d_model"],
            hidden=self.head_config.get("comp_hidden"),
            layers=self.head_config.get("comp_layers", 1),
            dropout=self.head_config.get("dropout", 0.0),
            alpha_eps=self.head_config.get("alpha_eps", 1e-6),
        )
        head.load_state_dict(self.comp_state)
        head.eval()
        return head

    def build_cond_head(self) -> CondHead:
        head = CondHead(
            d_model=self.head_config["d_model"],
            n_components=self.head_config["cond_n_components"],
            hidden=self.head_config.get("cond_hidden"),
            layers=self.head_config.get("cond_layers", 1),
            dropout=self.head_config.get("dropout", 0.0),
            sigma_eps=self.head_config.get("cond_sigma_eps", 1e-3),
        )
        head.load_state_dict(self.cond_state)
        head.eval()
        return head


def _row_tensors(ds: JointHeadDataset) -> TensorDataset:
    return TensorDataset(
        ds.row_bulk_idx, ds.presence, ds.magnitude, ds.comp_target,
        ds.cond_target, ds.f_liq, ds.f_fl, ds.c_liq, ds.c_fl, ds.bulk_11,
    )


def train_joint_heads(config: dict, device: torch.device) -> JointHeadCheckpoint:
    """Train all three heads per `config`; return the best-val checkpoint.

    Required config keys: `encoder_checkpoint`, `corpus_dir`,
    `valtest_corpus_dir`, `epochs`, `lr`, `batch_size`. Optional: `rungs`,
    `presence_cutoff`, `comp_eps`, per-head `{mode,comp,cond}_hidden` +
    `{mode,comp,cond}_layers`, `dropout`, `alpha_eps`, `cond_n_components`,
    `cond_sigma_eps`, the four loss weights, `weight_decay`,
    `early_stop_patience` (per-head: stop once every head is this many
    epochs past its own best), `seed`, `data_kwargs`,
    `head_checkpoint_dir` + `head_checkpoint_filename`.
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
        "valtest_corpus_dir": str(config["valtest_corpus_dir"]),
        "rungs": tuple(config.get("rungs", (50, 30))),
        "presence_cutoff": config.get("presence_cutoff"),
        "comp_eps": float(config.get("comp_eps", 1e-4)),
    }
    dataset_kwargs = {
        k: v for k, v in label_config.items()
        if k != "valtest_corpus_dir" and v is not None
    }
    train_ds = JointHeadDataset(
        **dataset_kwargs, data_kwargs=config.get("data_kwargs")
    )
    label_config["presence_cutoff"] = float(train_ds.presence_cutoff)
    val_ds = JointHeadDataset(
        corpus_dir=config["valtest_corpus_dir"],
        rungs=label_config["rungs"],
        presence_cutoff=label_config["presence_cutoff"],
        comp_eps=label_config["comp_eps"],
        data_kwargs=config.get("data_kwargs"),
        bulk_ids=load_valtest_split("val"),
    )

    train_latents = _bulk_latents(encoder, train_ds, device)
    val_latents = _bulk_latents(encoder, val_ds, device)
    print(
        f"train_joint_heads: device={device} encoder={Path(encoder_ckpt_path).name} "
        f"train: bulks={len(train_ds.bulk_ids)} rows={len(train_ds)} "
        f"val: bulks={len(val_ds.bulk_ids)} rows={len(val_ds)} (valtest val roster)"
    )

    head_config = {
        "d_model": enc_ckpt.config["d_model"],
        "mode_hidden": config.get("mode_hidden"),
        "mode_layers": int(config.get("mode_layers", 1)),
        "comp_hidden": config.get("comp_hidden"),
        "comp_layers": int(config.get("comp_layers", 1)),
        "cond_hidden": config.get("cond_hidden"),
        "cond_layers": int(config.get("cond_layers", 1)),
        "dropout": config.get("dropout", 0.0),
        "alpha_eps": config.get("alpha_eps", 1e-6),
        "cond_n_components": int(config.get("cond_n_components", 3)),
        "cond_sigma_eps": config.get("cond_sigma_eps", 1e-3),
    }
    loss_config = {
        "magnitude_weight": config.get("magnitude_weight", 1.0),
        "lambda_comp": config.get("lambda_comp", 1.0),
        "lambda_cond": config.get("lambda_cond", 1.0),
        "lambda_balance": config.get("lambda_balance", 1.0),
    }
    mode_head = ModeHead(
        d_model=head_config["d_model"], hidden=head_config["mode_hidden"],
        layers=head_config["mode_layers"], dropout=head_config["dropout"],
        alpha_eps=head_config["alpha_eps"],
    ).to(device)
    comp_head = CompHead(
        d_model=head_config["d_model"], hidden=head_config["comp_hidden"],
        layers=head_config["comp_layers"], dropout=head_config["dropout"],
        alpha_eps=head_config["alpha_eps"],
    ).to(device)
    cond_head = CondHead(
        d_model=head_config["d_model"],
        n_components=head_config["cond_n_components"],
        hidden=head_config["cond_hidden"], layers=head_config["cond_layers"],
        dropout=head_config["dropout"], sigma_eps=head_config["cond_sigma_eps"],
    ).to(device)
    cond_head.init_output_bias(
        train_ds.cond_target.mean(0).to(device),
        train_ds.cond_target.std(0).to(device),
    )
    heads = (mode_head, comp_head, cond_head)
    params = [p for head in heads for p in head.parameters()]

    optimizer = torch.optim.AdamW(
        params, lr=config["lr"], weight_decay=config.get("weight_decay", 0.0)
    )
    train_loader = DataLoader(
        _row_tensors(train_ds), batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        _row_tensors(val_ds), batch_size=config["batch_size"], shuffle=False
    )

    def _batch_loss(batch, latents):
        (bidx, presence, magnitude, comp_target, cond_target,
         f_liq, f_fl, c_liq, c_fl, bulk_11) = (t.to(device) for t in batch)
        h = latents[bidx]
        mode_logits, mode_alpha = mode_head(h)
        comp_alpha = comp_head(h)
        cond_params = cond_head(h, presence)
        return joint_heads_loss(
            mode_logits, mode_alpha, comp_alpha, cond_params,
            presence, magnitude, comp_target, cond_target,
            f_liq, f_fl, c_liq, c_fl, bulk_11,
            **loss_config,
        )

    monitor: dict[str, list[float]] = {
        k: []
        for k in ("train", "val", *(f"train_{t}" for t in _TERMS),
                  *(f"val_{t}" for t in _TERMS))
    }
    patience = config.get("early_stop_patience")
    heads_by_name = dict(zip(HEAD_NAMES, heads))
    best = {name: float("inf") for name in HEAD_NAMES}
    best_epoch = {name: 0 for name in HEAD_NAMES}
    best_states: dict[str, dict] = {}
    since_improve = {name: 0 for name in HEAD_NAMES}
    for epoch in range(config["epochs"]):
        for head in heads:
            head.train()
        sums = {k: 0.0 for k in ("total", *_TERMS)}
        n_batches = 0
        epoch_start = time()
        for batch in train_loader:
            terms = _batch_loss(batch, train_latents)
            optimizer.zero_grad()
            terms["total"].backward()
            # per-head clipping: the cond head's raw-unit NLL produces
            # gradients orders larger than the simplex heads'; one joint norm
            # would spend the whole clip budget on it and starve the others.
            for head in heads:
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            for t in ("total", *_TERMS):
                sums[t] += terms[t].item()
            n_batches += 1
        monitor["train"].append(sums["total"] / n_batches)
        for t in _TERMS:
            monitor[f"train_{t}"].append(sums[t] / n_batches)

        for head in heads:
            head.eval()
        sums = {k: 0.0 for k in ("total", *_TERMS)}
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                terms = _batch_loss(batch, val_latents)
                for t in ("total", *_TERMS):
                    sums[t] += terms[t].item()
                n_batches += 1
        val_mean = sums["total"] / n_batches
        monitor["val"].append(val_mean)
        for t in _TERMS:
            monitor[f"val_{t}"].append(sums[t] / n_batches)
        print(
            f"epoch {epoch + 1}/{config['epochs']} "
            f"train={monitor['train'][-1]:.4f} val={val_mean:.4f} "
            f"(bce={monitor['val_presence_bce'][-1]:.4f} mode={monitor['val_mode_nll'][-1]:.4f} "
            f"comp={monitor['val_comp_nll'][-1]:.4f} cond={monitor['val_cond_nll'][-1]:.4f} "
            f"closure={monitor['val_closure'][-1]:.5f}) "
            f"elapsed={time() - epoch_start:.1f}s"
        )

        sel = {name: series[-1] for name, series in selection_series(monitor).items()}
        for name, value in sel.items():
            if value < best[name]:
                best[name], best_epoch[name] = value, epoch + 1
                best_states[name] = {
                    k: v.clone()
                    for k, v in heads_by_name[name].state_dict().items()
                }
                since_improve[name] = 0
            else:
                since_improve[name] += 1
        if patience is not None and all(
            v >= patience for v in since_improve.values()
        ):
            print(
                f"early stop at epoch {epoch + 1}: every head is "
                f"{patience}+ epochs past its own best"
            )
            break

    for name, head in heads_by_name.items():
        if name in best_states:
            head.load_state_dict(best_states[name])
    print(
        "per-head bests: "
        + " ".join(
            f"{name}={best[name]:.4f}@ep{best_epoch[name]}" for name in HEAD_NAMES
        )
    )

    metrics = evaluate_mode_head(
        mode_head, val_latents, val_ds, np.arange(len(val_ds)), device
    )
    print(f"\nval joint hurdle NLL: {metrics['joint_nll']:.4f} nats/row")
    table = pd.DataFrame(metrics["per_phase"]).sort_values(
        "frequency", ascending=False
    )
    print(table.round(4).to_string(index=False))

    ckpt = JointHeadCheckpoint.from_heads(
        mode_head, comp_head, cond_head, head_config, loss_config,
        str(encoder_ckpt_path), label_config, monitor,
    )
    if config.get("head_checkpoint_dir"):
        out_path = (
            Path(config["head_checkpoint_dir"]) / config["head_checkpoint_filename"]
        )
        ckpt.save(out_path)
        print(f"heads checkpoint -> {out_path}")
        fig = plot_joint_run_loss(
            monitor, selection=selection_series(monitor), title=out_path.stem
        )
        curve_path = out_path.with_name("loss_curves.png")
        fig.savefig(curve_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"loss curves -> {curve_path}")
    return ckpt


def run_single(config_path: Path, device: torch.device) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ckpt = train_joint_heads(config, device)
    return {
        name: min(vals)
        for name, vals in selection_series(ckpt.monitor).items()
    }

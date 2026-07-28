"""Probe + UMAP report orchestration against a frozen encoder checkpoint.

`run_scoring()` produces the canonical headline-probe artifact: a
`probe_metrics.json` plus 7 PNGs (UMAP supergroup, UMAP qap, UMAP
chem axes, 3D UMAP, k-NN qap purity, and the two MLP-probe loss
curves).  Drives both probe families (LR + MLP) for both probe
targets (qap_name + supergroup), plus the k-NN-purity diagnostic.
"""

import json
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cnmineral.eval.latent import (
    PaddedRedoxView,
    extract_latents,
    load_encoder_from_checkpoint,
)
from cnmineral.eval.probe.primitives import (
    knn_label_purity,
    random_label_purity,
    train_logreg_probe,
    train_mlp_probe,
)
from cnmineral.eval.supergroup import map_qap_array
from cnmineral.eval.training_curves import plot_run_loss
from cnmineral.eval.umap_views import (
    fit_umap_2d,
    fit_umap_3d,
    plot_knn_qap_purity,
    plot_umap_3d_supergroup,
    plot_umap_chem_axes,
    plot_umap_qap,
    plot_umap_supergroup,
)
from cnmineral.training import create_dataset, pick_device, reserve_test
from cnmineral.training.checkpoint import checkpoint_tag


REPO_ROOT = Path(__file__).resolve().parents[4]


def _timed(label: str, fn, *args, **kwargs):
    t0 = time()
    out = fn(*args, **kwargs)
    print(f"  {label}: {time() - t0:.1f}s")
    return out


def run_scoring(
    checkpoint_path: Path,
    *,
    device: torch.device | None = None,
    eval_roster: Path | None = None,
    redox: str = "populated",
) -> dict:
    """Fixed-protocol probe pass; see the module docstring.

    `eval_roster` restricts the SCORED test rows to the roster's
    sample_ids intersected with the checkpoint's own test split (probes
    still train on the checkpoint's own trainval) -- the shared-row
    protocol for comparing encoders across corpora.  `redox="padded"`
    zeroes the redox token before latent extraction (legal only on
    redox corpora): the head-to-head condition against a 22-token
    encoder, and the deployment regime.
    """
    if redox not in ("populated", "padded"):
        raise ValueError(f"redox must be populated|padded, got {redox!r}")

    # Single-mode output is derived from the checkpoint path (plus any
    # protocol variant) so runs against different encoders or variants
    # land in different directories without an --output-dir flag.
    tag = checkpoint_tag(checkpoint_path)
    if eval_roster is not None:
        tag += "_inherited"
    if redox == "padded":
        tag += "_redoxpadded"
    output_dir = REPO_ROOT / "runs" / "probe" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output:     {output_dir}")

    print("\nloading encoder (pretrain config comes from ckpt.config)...")
    encoder, ckpt = load_encoder_from_checkpoint(checkpoint_path)
    config = ckpt.config
    print(
        f"  checkpoint epoch={ckpt.epoch} "
        f"d_model={config.get('d_model')} "
        f"val_loss={ckpt.val_loss}"
    )

    print("\nbuilding dataset + reproducing test split...")
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    n_train = len(splits["trainval"])
    n_test = len(splits["test"])
    print(f"  dataset n={len(dataset)} trainval={n_train} test={n_test}")

    if redox == "padded" and not config.get("data_kwargs", {}).get(
        "include_redox", False,
    ):
        raise ValueError(
            "redox='padded' only applies to a redox (23-token) corpus; "
            "this checkpoint's corpus has no redox token"
        )
    extract_view = (
        PaddedRedoxView(dataset) if redox == "padded" else dataset
    )

    print("\nextracting latents over the full corpus...")
    latents = _timed(
        "extract_latents",
        extract_latents,
        encoder,
        extract_view,
        device=device,
    ).numpy()
    print(f"  latents: {latents.shape}")

    if not hasattr(dataset, "qap_names"):
        raise SystemExit(
            "dataset has no qap_names attribute; probes need qap labels"
        )
    qap_full = np.asarray(dataset.qap_names)
    sg_full = map_qap_array(qap_full)

    train_idx = np.asarray(splits["trainval"], dtype=np.int64)
    test_idx = np.asarray(splits["test"], dtype=np.int64)

    n_roster_scored = None
    if eval_roster is not None:
        roster_ids = pd.read_csv(eval_roster)["sample_id"].to_numpy()
        keep = np.isin(dataset.sample_ids[test_idx], roster_ids)
        if not keep.any():
            raise ValueError(
                f"eval roster {eval_roster} shares no row with this "
                f"checkpoint's test split"
            )
        test_idx = test_idx[keep]
        n_roster_scored = int(len(test_idx))
        print(f"  scored test rows restricted to roster: {n_roster_scored}")

    # qap probes (and the supergroup mapping derived from qap) restrict
    # to labeled rows: on an all-rock corpus, qap_name is null outside
    # the igneous slice, so this IS the igneous-slice restriction.
    qap_train_mask = np.array([qap_full[i] is not None for i in train_idx])
    qap_test_mask = np.array([qap_full[i] is not None for i in test_idx])
    qap_train_idx = train_idx[qap_train_mask]
    qap_test_idx = test_idx[qap_test_mask]
    if len(qap_test_idx) < len(test_idx):
        print(
            f"  qap probes on the labeled slice: "
            f"train {len(qap_train_idx)}/{len(train_idx)} "
            f"test {len(qap_test_idx)}/{len(test_idx)}"
        )

    X_train = latents[qap_train_idx]
    X_test = latents[qap_test_idx]
    qap_train = qap_full[qap_train_idx]
    qap_test = qap_full[qap_test_idx]
    sg_train = sg_full[qap_train_idx]
    sg_test = sg_full[qap_test_idx]

    print("\ntraining qap_name probes...")
    qap_logreg = _timed(
        "logreg",
        train_logreg_probe,
        X_train, qap_train, X_test, qap_test,
        seed=config.get("seed", 0),
    )
    qap_mlp = _timed(
        "mlp",
        train_mlp_probe,
        X_train, qap_train, X_test, qap_test,
        device=device, seed=config.get("seed", 0),
    )
    print(
        f"  qap_logreg test_acc={qap_logreg.test_accuracy:.3f}  "
        f"qap_mlp test_acc={qap_mlp.test_accuracy:.3f}  "
        f"n_classes={qap_mlp.n_classes}"
    )

    print("\ntraining supergroup probes...")
    sg_logreg = _timed(
        "logreg",
        train_logreg_probe,
        X_train, sg_train, X_test, sg_test,
        seed=config.get("seed", 0),
    )
    sg_mlp = _timed(
        "mlp",
        train_mlp_probe,
        X_train, sg_train, X_test, sg_test,
        device=device, seed=config.get("seed", 0),
    )
    print(
        f"  sg_logreg test_acc={sg_logreg.test_accuracy:.3f}  "
        f"sg_mlp test_acc={sg_mlp.test_accuracy:.3f}  "
        f"n_classes={sg_mlp.n_classes}"
    )

    print("\ncomputing k-NN qap_name purity (test latents)...")
    purity = _timed(
        "knn",
        knn_label_purity,
        X_test, qap_test, k_values=(5, 10, 50),
    )
    random_baseline = random_label_purity(qap_test)
    print(
        f"  k=5: {purity[5]:.3f}  k=10: {purity[10]:.3f}  "
        f"k=50: {purity[50]:.3f}  baseline: {random_baseline:.3f}"
    )

    # rock_group probe: the coarse axis an all-rock corpus can measure
    # on every row.  "unknown" rows are a data-availability artifact,
    # not a class, so they sit out.
    rg_logreg = rg_mlp = None
    rock_groups = getattr(dataset, "rock_groups", None)
    if rock_groups is not None:
        rg_full = np.asarray(rock_groups)
        rg_train_idx = train_idx[rg_full[train_idx] != "unknown"]
        rg_test_idx = test_idx[rg_full[test_idx] != "unknown"]
        if len(np.unique(rg_full[rg_train_idx])) > 1:
            print("\ntraining rock_group probes...")
            rg_logreg = _timed(
                "logreg",
                train_logreg_probe,
                latents[rg_train_idx], rg_full[rg_train_idx],
                latents[rg_test_idx], rg_full[rg_test_idx],
                seed=config.get("seed", 0),
            )
            rg_mlp = _timed(
                "mlp",
                train_mlp_probe,
                latents[rg_train_idx], rg_full[rg_train_idx],
                latents[rg_test_idx], rg_full[rg_test_idx],
                device=device, seed=config.get("seed", 0),
            )
            print(
                f"  rock_group_logreg test_acc={rg_logreg.test_accuracy:.3f}  "
                f"rock_group_mlp test_acc={rg_mlp.test_accuracy:.3f}"
            )

    print("\nfitting UMAP-2D over the full latent...")
    emb2 = _timed("umap_2d", fit_umap_2d, latents)
    print("fitting UMAP-3D over the full latent...")
    emb3 = _timed("umap_3d", fit_umap_3d, latents)

    print("rendering plots...")
    plot_umap_supergroup(
        emb2, sg_full, output_dir / "umap_supergroup.png",
    )
    qap_plot = np.array(
        [q if q is not None else "unlabeled" for q in qap_full],
        dtype=object,
    )
    plot_umap_qap(
        emb2, qap_plot, output_dir / "umap_qap.png",
    )
    chem_axes = dataset.petrological_axes()
    plot_umap_chem_axes(
        emb2, chem_axes, output_dir / "umap_chem_axes.png",
    )
    plot_umap_3d_supergroup(
        emb3, sg_full, output_dir / "umap_3d_supergroup.png",
    )
    plot_knn_qap_purity(
        purity, random_baseline, output_dir / "knn_qap_purity.png",
    )
    for mlp, target_name, filename in (
        (qap_mlp, "qap_name", "mlp_loss_qap.png"),
        (sg_mlp, "supergroup", "mlp_loss_supergroup.png"),
    ):
        fig, ax = plt.subplots()
        plot_run_loss(
            mlp.monitor, ax=ax, title=f"MLP probe ({target_name}) -- {tag}",
            ylabel="cross-entropy",
        )
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "checkpoint_epoch": ckpt.epoch,
        "checkpoint_val_loss": ckpt.val_loss,
        "checkpoint_test_loss": ckpt.test_loss,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "eval_roster": str(eval_roster) if eval_roster else None,
        "n_roster_scored": n_roster_scored,
        "redox": redox,
        "latent_dim": int(latents.shape[1]),
        "probes": {
            "qap_logreg": qap_logreg.to_dict(),
            "qap_mlp": qap_mlp.to_dict(),
            "supergroup_logreg": sg_logreg.to_dict(),
            "supergroup_mlp": sg_mlp.to_dict(),
            **(
                {
                    "rock_group_logreg": rg_logreg.to_dict(),
                    "rock_group_mlp": rg_mlp.to_dict(),
                }
                if rg_logreg is not None else {}
            ),
        },
        "knn_qap_purity": {
            "by_k": purity,
            "random_baseline": random_baseline,
        },
    }
    metrics_path = output_dir / "probe_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nmetrics -> {metrics_path}")

    print("\nsummary")
    print(f"  qap_name LR test acc:    {qap_logreg.test_accuracy:.3f}")
    print(f"  qap_name MLP test acc:   {qap_mlp.test_accuracy:.3f}")
    print(f"  supergroup LR test acc:  {sg_logreg.test_accuracy:.3f}")
    print(f"  supergroup MLP test acc: {sg_mlp.test_accuracy:.3f}")
    if rg_logreg is not None:
        print(f"  rock_group LR test acc:  {rg_logreg.test_accuracy:.3f}")
        print(f"  rock_group MLP test acc: {rg_mlp.test_accuracy:.3f}")
    print(
        f"  k-NN qap purity (k=5/10/50): "
        f"{purity[5]:.3f}/{purity[10]:.3f}/{purity[50]:.3f}  "
        f"random {random_baseline:.3f}"
    )

    return metrics

"""Linear probe: how well does the SSL-pretrained encoder linearly classify
rock type, without any supervised mineralogy training?

Standard SSL evaluation protocol. For each encoder variant (random init,
SSL-pretrained, cell-c fine-tuned), pool the supervised corpus through it,
de-duplicate to per-bulk latents, fit logistic regression on the v1
train-split labels (supergroup = 9 classes, qap_name = ~60 fine classes),
score on the v1 test split.

Tests the user's hypothesis: if the pretrained encoder already classifies
rock type well, a staged "classify-first, predict-mineralogy-conditional-
on-class" architecture becomes a viable design path.

Run:
    python scripts/11_linear_probe.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Load 09_visualize_latent for build_encoder + pooled_latents helpers
# (filename starts with a digit so import-by-path).
_spec = importlib.util.spec_from_file_location(
    "vis_latent", REPO_ROOT / "scripts" / "09_visualize_latent.py")
_vl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vl)

from cnmineral.data.unlabeled import _supergroup_of
from cnmineral.models.dataset import TrainFCv2Dataset


def per_bulk_latents(encoder, dataset, indices, batch_size, device):
    """Pool latents and dedupe to per-bulk (since v2-c majors-only model emits
    identical latent for the 8 paths of one bulk anyway)."""
    sub = Subset(dataset, indices.tolist())
    lat = _vl.pooled_latents(encoder, sub, batch_size, device)
    sample_ids = dataset.sample_ids[indices]
    # First-occurrence dedupe (latents are identical across rows of the same
    # bulk for any encoder that takes only the bulk-shared input).
    _, first_idx = np.unique(sample_ids, return_index=True)
    first_idx = np.sort(first_idx)
    return lat[first_idx], sample_ids[first_idx]


def probe_classification(train_X, train_y, test_X, test_y, max_iter=2000):
    """Multinomial logistic regression train+score (linear probe)."""
    clf = LogisticRegression(
        max_iter=max_iter, multi_class="multinomial",
        solver="lbfgs", C=1.0, n_jobs=1,
    )
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    return {
        "accuracy": float(accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_y, pred, average="weighted", zero_division=0)),
        "n_classes_train": int(len(np.unique(train_y))),
        "n_classes_test": int(len(np.unique(test_y))),
    }


def probe_classification_mlp(train_X, train_y, test_X, test_y,
                              hidden_layer_sizes=(128,), max_iter=200, seed=42):
    """Nonlinear MLP probe: 1 hidden layer of 128 ReLU units.

    Tests whether SSL added class structure that's nonlinearly accessible from
    the latent but invisible to logistic regression. Early-stopping disabled
    (incompatible with sklearn string-label scoring); 200 iters is plenty for
    this scale of data.
    """
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu", solver="adam",
        alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
        max_iter=max_iter, early_stopping=False,
        random_state=seed, verbose=False,
    )
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    return {
        "accuracy": float(accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_y, pred, average="weighted", zero_division=0)),
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "n_iter": int(clf.n_iter_),
    }


def main():
    device = _vl.pick_device()
    print(f"device: {device}")

    splits = json.load(open(REPO_ROOT / "data/splits_v1.json"))
    train_ids = np.asarray(splits["train_sample_ids"], dtype=np.int64)
    test_ids = np.asarray(splits["test_sample_ids"], dtype=np.int64)

    print("loading TrainFCv2Dataset (majors+traces, full corpus)...")
    sup = TrainFCv2Dataset(
        parquet_path=REPO_ROOT / "data/train_fc_v1.parquet",
        complete_csv_path=REPO_ROOT / "data/complete.csv",
        liq_threshold=0.10,
    )
    train_mask = np.isin(sup.sample_ids, train_ids)
    test_mask = np.isin(sup.sample_ids, test_ids)
    train_row_idx = np.where(train_mask)[0]
    test_row_idx = np.where(test_mask)[0]
    print(f"  supervised: train={len(train_row_idx)} test={len(test_row_idx)}")

    encoders = [
        ("random-init",  None),
        ("SSL-pretrained", _vl.PRETRAIN_CKPT),
        ("cell-c fine-tuned", REPO_ROOT / "checkpoints/base_v2_c_pretrain_majorsonly/best.pt"),
    ]

    results = {}
    for label, ckpt in encoders:
        print(f"\n=== {label} ===")
        if ckpt is not None and not Path(ckpt).exists():
            print(f"  (skipping: {ckpt} not present)")
            continue
        enc = _vl.build_encoder(ckpt, device)
        train_lat, train_sids = per_bulk_latents(enc, sup, train_row_idx, 512, device)
        test_lat, test_sids = per_bulk_latents(enc, sup, test_row_idx, 512, device)
        print(f"  per-bulk latents: train={train_lat.shape} test={test_lat.shape}")

        # Map sample_id -> qap_name (one per bulk, since qap_name is bulk-constant)
        sid_to_qap = {}
        for sid, qap in zip(sup.sample_ids, sup.qap_names):
            sid_to_qap[int(sid)] = str(qap)
        train_qap = np.array([sid_to_qap[int(s)] for s in train_sids])
        test_qap = np.array([sid_to_qap[int(s)] for s in test_sids])
        train_sg = np.array([_supergroup_of(q) for q in train_qap])
        test_sg = np.array([_supergroup_of(q) for q in test_qap])

        # Supergroup probe (9 classes, easier). Both LR and MLP.
        sg_lr = probe_classification(train_lat, train_sg, test_lat, test_sg)
        sg_mlp = probe_classification_mlp(train_lat, train_sg, test_lat, test_sg)
        # qap_name probe (60+ classes, hard).
        seen = set(np.unique(train_qap).tolist())
        keep_test = np.array([q in seen for q in test_qap])
        qap_lr = probe_classification(
            train_lat, train_qap, test_lat[keep_test], test_qap[keep_test],
        )
        qap_mlp = probe_classification_mlp(
            train_lat, train_qap, test_lat[keep_test], test_qap[keep_test],
        )

        results[label] = {
            "supergroup_lr": sg_lr, "supergroup_mlp": sg_mlp,
            "qap_name_lr": qap_lr, "qap_name_mlp": qap_mlp,
        }
        print(f"  supergroup LR : acc={sg_lr['accuracy']:.3f}  macro_F1={sg_lr['macro_f1']:.3f}")
        print(f"  supergroup MLP: acc={sg_mlp['accuracy']:.3f}  macro_F1={sg_mlp['macro_f1']:.3f}  "
              f"(MLP-LR delta: {sg_mlp['accuracy']-sg_lr['accuracy']:+.3f})")
        print(f"  qap_name LR  ({qap_lr['n_classes_train']}): "
              f"acc={qap_lr['accuracy']:.3f}  macro_F1={qap_lr['macro_f1']:.3f}")
        print(f"  qap_name MLP : "
              f"acc={qap_mlp['accuracy']:.3f}  macro_F1={qap_mlp['macro_f1']:.3f}  "
              f"(MLP-LR delta: {qap_mlp['accuracy']-qap_lr['accuracy']:+.3f})")

    print("\n=== summary (LR = linear / MLP = 1-hidden 128) ===")
    print(f"{'encoder':>22s}  {'sg LR acc':>10s}  {'sg MLP acc':>11s}  "
          f"{'qap LR acc':>11s}  {'qap MLP acc':>12s}")
    for lbl, r in results.items():
        print(f"{lbl:>22s}  {r['supergroup_lr']['accuracy']:>10.3f}  "
              f"{r['supergroup_mlp']['accuracy']:>11.3f}  "
              f"{r['qap_name_lr']['accuracy']:>11.3f}  "
              f"{r['qap_name_mlp']['accuracy']:>12.3f}")

    out_path = REPO_ROOT / "data/diagnostics/latent_v2/linear_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()

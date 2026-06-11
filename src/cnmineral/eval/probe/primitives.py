"""Linear + MLP probes on a frozen latent.

Two probe families, scored on the same train/test split:

- `train_logreg_probe` -- multinomial sklearn `LogisticRegression`.
  Fixed-protocol baseline (no `C` or `solver` knob) per standard SSL
  eval (SimCLR, MoCo, CLIP).  The point of the linear probe is
  "no model knobs, just a linear readout" -- sweeping its HPs makes
  the baseline less interpretable.  Defaults: `StandardScaler` on
  the train latent (needed for lbfgs convergence at many-class
  targets), no class weighting.
- `train_mlp_probe` -- a small torch MLP.  This is the probe whose
  HPs we sweep with probe-HPO; defaults match standard SSL-eval
  protocols, kwargs expose the knobs an Optuna sweep can vary.

Both return `ProbeMetrics`, a JSON-friendly frozen dataclass
suitable for dumping straight into `probe_metrics.json`.  Each
`ProbeMetrics` instance carries the hyperparameters that produced
it so a reader of the metrics JSON can tell what was actually run.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True, slots=True)
class ProbeMetrics:
    """Serialized once into the run-level JSON.

    `per_class_accuracy` is a `{label: accuracy}` dict over labels seen
    in the test split; absent test classes are omitted rather than
    reported as nan, so a reader can spot coverage gaps directly.

    `hyperparameters` is the dict of kwargs that produced this result,
    so the metrics JSON tells a reader what was actually run.

    `monitor` is the per-epoch `{"train": [...], "val": [...]}`
    cross-entropy curve.  Populated by the MLP probe (which has an
    epoch loop); empty for probes that don't train iteratively (LR).
    """

    probe: str
    train_accuracy: float
    test_accuracy: float
    n_classes: int
    n_train: int
    n_test: int
    per_class_accuracy: dict[str, float]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    monitor: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "probe": self.probe,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "n_classes": self.n_classes,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "per_class_accuracy": self.per_class_accuracy,
            "hyperparameters": dict(self.hyperparameters),
            "monitor": self.monitor,
        }


def _per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: np.ndarray,
) -> dict[str, float]:
    """`{label: fraction correctly classified}` over labels present in `y_true`."""
    out: dict[str, float] = {}
    for cls in np.unique(y_true):
        mask = y_true == cls
        out[str(label_names[cls])] = float(
            (y_pred[mask] == cls).mean()
        )
    return out


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Unit-norm each row.  Standard SSL-eval preprocessing."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def train_logreg_probe(
    train_latents: np.ndarray,
    train_labels,
    test_latents: np.ndarray,
    test_labels,
    *,
    standardize: bool = True,
    class_weight: str | None = None,
    max_iter: int = 5000,
    seed: int = 0,
) -> ProbeMetrics:
    """Multinomial logistic-regression probe -- fixed-protocol baseline.

    `C` and `solver` are deliberately not exposed: this is the linear
    readout, not a tuned classifier.  `standardize=True` is the real
    fix for the lbfgs convergence problem at many-class scales (the
    pooled latent's per-dim scale is unconstrained; L2 reg is feature-
    blind).  `class_weight="balanced"` lifts rare-class accuracy at
    some cost to majority -- kept as an opt-in diagnostic switch
    rather than the default, so headline numbers stay comparable.
    """
    enc = LabelEncoder().fit(np.concatenate([train_labels, test_labels]))
    y_train = enc.transform(train_labels)
    y_test = enc.transform(test_labels)

    X_train, X_test = train_latents, test_latents
    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=max_iter,
        C=1.0,
        solver="lbfgs",
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    return ProbeMetrics(
        probe="logreg",
        train_accuracy=float(accuracy_score(y_train, train_pred)),
        test_accuracy=float(accuracy_score(y_test, test_pred)),
        n_classes=int(len(enc.classes_)),
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        per_class_accuracy=_per_class_accuracy(
            y_test, test_pred, enc.classes_,
        ),
        hyperparameters={
            "standardize": standardize,
            "class_weight": class_weight,
            "max_iter": max_iter,
            "seed": seed,
            "C": 1.0,
            "solver": "lbfgs",
        },
    )


class _MLPProbe(nn.Module):
    """1 or 2 hidden layers with LayerNorm + ReLU + Dropout per layer.

    `n_hidden_layers=1` recovers the simplest sufficient classifier
    (Linear -> ReLU -> Dropout -> Linear).  `n_hidden_layers=2` adds
    LayerNorm between layers; the intermediate activations also
    benefit from scale normalization (same reason LR needs
    StandardScaler).  Beyond 2 hidden layers is almost always
    overpowered for a probe.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float,
        n_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        if n_hidden_layers not in (1, 2):
            raise ValueError(
                f"n_hidden_layers must be 1 or 2; got {n_hidden_layers}"
            )

        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if n_hidden_layers == 2 else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if n_hidden_layers == 2:
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_probe(
    train_latents: np.ndarray,
    train_labels,
    test_latents: np.ndarray,
    test_labels,
    *,
    hidden_dim: int = 512,
    dropout: float = 0.2,
    n_hidden_layers: int = 1,
    label_smoothing: float = 0.0,
    l2_normalize_inputs: bool = False,
    use_cosine_schedule: bool = True,
    class_weight: str | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.1,
    early_stop_patience: int = 5,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> ProbeMetrics:
    """Small-MLP probe.  This is the probe whose HPs probe-HPO sweeps.

    Carves a small val slice off train for early stopping, then
    reports test accuracy at the best-val epoch.  Defaults match
    a sensible SSL-eval baseline; kwargs expose every knob a TPE
    sampler would want to vary.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    enc = LabelEncoder().fit(np.concatenate([train_labels, test_labels]))
    y_train_full = enc.transform(train_labels)
    y_test = enc.transform(test_labels)

    X_train_arr = train_latents
    X_test_arr = test_latents
    if l2_normalize_inputs:
        X_train_arr = _l2_normalize_rows(X_train_arr)
        X_test_arr = _l2_normalize_rows(X_test_arr)

    # train/val carve-out for early stopping
    n_train_full = len(y_train_full)
    perm = rng.permutation(n_train_full)
    n_val = max(1, int(n_train_full * val_frac))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    X_tr = torch.from_numpy(X_train_arr[tr_idx]).float()
    y_tr = torch.from_numpy(y_train_full[tr_idx]).long()
    X_val = torch.from_numpy(X_train_arr[val_idx]).float().to(device)
    y_val = torch.from_numpy(y_train_full[val_idx]).long().to(device)
    X_test_t = torch.from_numpy(X_test_arr).float().to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)
    X_train_all = torch.from_numpy(X_train_arr).float().to(device)
    y_train_all = torch.from_numpy(y_train_full).long().to(device)

    in_dim = X_train_arr.shape[1]
    n_classes = int(len(enc.classes_))
    model = _MLPProbe(
        in_dim, hidden_dim, n_classes, dropout,
        n_hidden_layers=n_hidden_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_cosine_schedule
        else None
    )

    class_weight_tensor: torch.Tensor | None = None
    if class_weight == "balanced":
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(n_classes),
            y=y_train_full,
        )
        class_weight_tensor = torch.from_numpy(weights).float().to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=label_smoothing,
    )

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
    )

    best_val_loss = float("inf")
    best_state: dict | None = None
    bad_epochs = 0
    monitor: dict[str, list[float]] = {"train": [], "val": []}
    for _epoch in range(epochs):
        model.train()
        train_total, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_total += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        monitor["train"].append(train_total / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        monitor["val"].append(val_loss)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_all).argmax(dim=1).cpu().numpy()
        test_pred = model(X_test_t).argmax(dim=1).cpu().numpy()

    return ProbeMetrics(
        probe="mlp",
        train_accuracy=float(accuracy_score(y_train_full, train_pred)),
        test_accuracy=float(accuracy_score(y_test, test_pred)),
        n_classes=n_classes,
        n_train=int(len(y_train_full)),
        n_test=int(len(y_test)),
        per_class_accuracy=_per_class_accuracy(
            y_test, test_pred, enc.classes_,
        ),
        hyperparameters={
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "n_hidden_layers": n_hidden_layers,
            "label_smoothing": label_smoothing,
            "l2_normalize_inputs": l2_normalize_inputs,
            "use_cosine_schedule": use_cosine_schedule,
            "class_weight": class_weight,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "val_frac": val_frac,
            "early_stop_patience": early_stop_patience,
            "seed": seed,
        },
        monitor=monitor,
    )


def knn_label_purity(
    latents: np.ndarray,
    labels,
    *,
    k_values: tuple[int, ...] = (5, 10, 50),
    metric: str = "cosine",
) -> dict[int, float]:
    """For each k, mean fraction of k-NN that share the query's label.

    Excludes the query point itself from its own neighborhood (the
    n_neighbors=k+1 trick).  Cosine because the encoder's pool is
    unnormalized and rocks differ more in latent direction than in
    magnitude.  Random baseline for a label distribution is
    `sum(p_i^2)` where `p_i` is the class share; the caller can
    compute and report that for context.
    """
    from sklearn.neighbors import NearestNeighbors

    labels_arr = np.asarray(labels)
    max_k = max(k_values)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric=metric).fit(latents)
    _, idx = nn.kneighbors(latents)
    nbr_idx = idx[:, 1:]  # drop self at column 0
    nbr_labels = labels_arr[nbr_idx]
    same = nbr_labels == labels_arr[:, None]
    return {
        int(k): float(same[:, :k].mean()) for k in k_values
    }


def random_label_purity(labels) -> float:
    """Expected k-NN purity for a class-balanced random baseline.

    `sum_c p_c^2` where `p_c = n_c / N`.  Independent of k.
    """
    labels_arr = np.asarray(labels)
    _, counts = np.unique(labels_arr, return_counts=True)
    p = counts / counts.sum()
    return float((p ** 2).sum())

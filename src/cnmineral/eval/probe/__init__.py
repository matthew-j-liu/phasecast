"""Probe machinery -- the public probe API.

- `primitives` -- `train_logreg_probe`, `train_mlp_probe`,
  `knn_label_purity`, the `ProbeMetrics` dataclass.  No file I/O.
- `scoring` -- single-mode orchestrator (`run_scoring`); LR + MLP
  probes at a fixed protocol + UMAP figures.
- `hpo` -- Optuna sweep over MLP probe HPs (`run_hpo`); emits a
  per-encoder summary JSON.

CLI entry: `scripts/eval/probe_encoder.py`.
"""

from cnmineral.eval.probe.hpo import run_hpo
from cnmineral.eval.probe.primitives import (
    ProbeMetrics,
    knn_label_purity,
    random_label_purity,
    train_logreg_probe,
    train_mlp_probe,
)
from cnmineral.eval.probe.scoring import run_scoring

__all__ = [
    "ProbeMetrics",
    "knn_label_purity",
    "random_label_purity",
    "run_hpo",
    "run_scoring",
    "train_logreg_probe",
    "train_mlp_probe",
]

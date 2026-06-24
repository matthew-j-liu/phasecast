"""SSL pretrain machinery -- the public training API.

- `device` -- `pick_device()` (cuda / mps / cpu fallback)
- `build_from_config` -- `create_dataset`, `create_encoder_and_head`
- `splits` -- `reserve_test` (stratified outer split by qap_name)
- `checkpoint` -- `Checkpoint` (payload), `CheckpointStore` (best/last
  dir), `checkpoint_tag` (per-checkpoint artifact namespacing)
- `reconstruction_mse` -- `evaluate_on_test`
- `ssl_pretrain` -- `run_pretraining`, `run_single`
- `hpo` -- `run_hpo`

CLI entry: `scripts/training/pretrain.py`.
"""

from cnmineral.training.checkpoint import (
    Checkpoint,
    CheckpointStore,
    checkpoint_tag,
)
from cnmineral.training.device import pick_device
from cnmineral.training.build_from_config import (
    create_dataset,
    create_encoder_and_head,
)
from cnmineral.training.splits import reserve_test
from cnmineral.training.reconstruction_mse import evaluate_on_test
from cnmineral.training.ssl_pretrain import run_pretraining, run_single
from cnmineral.training.hpo import run_hpo

__all__ = [
    "Checkpoint",
    "CheckpointStore",
    "checkpoint_tag",
    "pick_device",
    "create_dataset",
    "create_encoder_and_head",
    "reserve_test",
    "evaluate_on_test",
    "run_pretraining",
    "run_single",
    "run_hpo",
]

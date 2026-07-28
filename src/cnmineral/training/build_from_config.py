"""
Build the dataset + encoder + head from a config dict.

Two small builder functions called by `run_pretraining`.  Kept
separate from `ssl_pretrain.py` so a future ablation harness can call them
without going through the training loop.
"""

from torch.utils.data import Dataset

from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.data.tokens import N_TOKENS, N_TOKENS_REDOX
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead


def create_dataset(config: dict) -> Dataset:
    """Build the SSL pretrain dataset from `config['data_kwargs']`.

    `data_kwargs` is passed to `BulkChemistryDataset` verbatim; a
    missing key means the dataset's own defaults (the full Gard 2019
    corpus at `data/bulks/gard2019.csv`).
    """
    return BulkChemistryDataset(**config.get("data_kwargs", {}))


def create_encoder_and_head(config: dict) -> tuple[Encoder, PretrainHead]:
    """Construct the encoder + SSL pretrain head from a config dict.

    `config[X]` for keys whose absence is a bug (no safe default exists,
    e.g. `d_model`, `n_heads`).  `config.get(X, default)` for keys whose
    default is the common case (`dropout=0.0`, `frac_resolution=5000`).

    Token width follows `data_kwargs.include_redox`: a redox corpus gets
    the 121-row embedding table and a 23-wide head.  Checkpoints stamped
    before the knob existed default to the 22-token shapes and load
    strictly.
    """
    include_redox = config.get("data_kwargs", {}).get("include_redox", False)
    encoder = Encoder(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=config.get("dropout", 0.0),
        frac_resolution=config.get("frac_resolution", 5000),
        include_redox_token=include_redox,
    )
    head = PretrainHead(
        d_model=config["d_model"],
        n_tokens=N_TOKENS_REDOX if include_redox else N_TOKENS,
    )
    return encoder, head

"""Build the dataset + encoder + head from a config dict.

Two small builder functions called by `run_pretraining`.  Kept
separate from `loop.py` so a future ablation harness can call them
without going through the training loop.
"""

from torch.utils.data import Dataset

from cnmineral.data.synthetic import SyntheticBulkChemistryDataset
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead


def create_dataset(config: dict) -> Dataset:
    """Build the SSL pretrain dataset per `config['data_source']`.

    `synthetic` always works; `real` tries `cnmineral.data.bulk_chemistry`
    and falls back to synthetic with a printed warning if the module
    is unavailable.
    """
    source = config.get("data_source", "real")
    if source == "synthetic":
        return SyntheticBulkChemistryDataset(
            n=config.get("synthetic_n", 256),
            seed=config.get("synthetic_seed", 0),
        )
    try:
        from cnmineral.data.bulk_chemistry import BulkChemistryDataset
    except ImportError as exc:
        print(
            f"warning: BulkChemistryDataset unavailable ({exc}); "
            f"falling back to SyntheticBulkChemistryDataset"
        )
        return SyntheticBulkChemistryDataset(
            n=config.get("synthetic_n", 256),
            seed=config.get("synthetic_seed", 0),
        )
    return BulkChemistryDataset(**config.get("data_kwargs", {}))


def create_encoder_and_head(config: dict) -> tuple[Encoder, PretrainHead]:
    """Construct the encoder + SSL pretrain head from a config dict.

    `config[X]` for keys whose absence is a bug (no safe default exists,
    e.g. `d_model`, `n_heads`).  `config.get(X, default)` for keys whose
    default is the common case (`dropout=0.0`, `frac_resolution=5000`).
    """
    encoder = Encoder(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=config.get("dropout", 0.0),
        frac_resolution=config.get("frac_resolution", 5000),
    )
    head = PretrainHead(d_model=config["d_model"])
    return encoder, head

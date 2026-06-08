"""Synthetic stand-in for the real bulk-chemistry SSL pretrain dataset.

Used by `scripts/pretrain.py` (when `data_source: synthetic` is set in
the YAML) and by the pretrain script tests so neither path depends on
the real Gard 2019 parquet being present.  Emits batches that obey the v3 padding convention
(`src==0 iff frac==0`) so `assert_padding_convention` passes and the
encoder's masked-mean pool sees realistic input geometry.

The first 9 token slots (the majors) are populated with deterministic
atomic numbers and pseudo-random `frac` values in `(0.01, 0.5]`; the
13 trace slots stay padded.  Deterministic from `seed`; the same
`(seed, n)` always yields identical samples.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from cnmineral.data.tokens import ATOMIC_NUMBERS, N_TOKENS


@dataclass(frozen=True, slots=True)
class _Sample:
    src: torch.Tensor
    frac: torch.Tensor
    sample_id: int


class SyntheticBulkChemistryDataset(Dataset[dict[str, torch.Tensor | int]]):
    """Synthetic majors-only batches that match the real dataset's shape.

    Parameters
    ----------
    n
        Number of samples in the dataset.
    seed
        RNG seed for the per-sample `frac` draws.  Construction is
        deterministic given `(n, seed)`.
    """

    def __init__(self, n: int = 256, seed: int = 0) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.n = n
        self.seed = seed

        g = torch.Generator().manual_seed(seed)
        src = torch.zeros(n, N_TOKENS, dtype=torch.long)
        frac = torch.zeros(n, N_TOKENS, dtype=torch.float32)
        src[:, :9] = ATOMIC_NUMBERS[:9]
        frac[:, :9] = 0.01 + 0.49 * torch.rand((n, 9), generator=g)
        self._src = src
        self._frac = frac

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        return {
            "src": self._src[idx].clone(),
            "frac": self._frac[idx].clone(),
            "sample_id": idx,
        }

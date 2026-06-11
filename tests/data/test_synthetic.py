"""Tests for `cnmineral.data.synthetic.SyntheticBulkChemistryDataset`.

Three contracts:
- Shape + dtype of each yielded item match the 22-token convention.
- The padding convention (`src == 0` iff `frac == 0`) is satisfied.
- Reproducibility on `(n, seed)`.
"""

import torch
from torch.utils.data import DataLoader

from cnmineral.data.synthetic import SyntheticBulkChemistryDataset
from cnmineral.data.tokens import N_TOKENS, assert_padding_convention


class TestSyntheticDataset:
    def test_shape_and_dtype(self) -> None:
        ds = SyntheticBulkChemistryDataset(n=4, seed=0)
        assert len(ds) == 4
        item = ds[0]
        assert set(item.keys()) == {"src", "frac", "sample_id"}
        assert item["src"].shape == (N_TOKENS,)
        assert item["frac"].shape == (N_TOKENS,)
        assert item["src"].dtype == torch.long
        assert item["frac"].dtype == torch.float32
        assert isinstance(item["sample_id"], int)

    def test_obeys_padding_convention(self) -> None:
        ds = SyntheticBulkChemistryDataset(n=8, seed=1)
        loader = DataLoader(ds, batch_size=4)
        for batch in loader:
            assert_padding_convention(batch["src"], batch["frac"])

    def test_reproducible(self) -> None:
        a = SyntheticBulkChemistryDataset(n=4, seed=7)
        b = SyntheticBulkChemistryDataset(n=4, seed=7)
        torch.testing.assert_close(a[0]["frac"], b[0]["frac"])

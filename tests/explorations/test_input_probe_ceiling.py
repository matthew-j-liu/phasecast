"""Tripwire for the raw-input probe ceiling's featurization and row slicing.

The run's own protocol asserts cover the end-to-end numbers; these cover the
two contracts a synthetic fixture can hold: what the raw matrices contain, and
which rows both arms are fit on.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from cnmineral.data.tokens import N_TOKENS_REDOX, REDOX_TOKEN_INDEX
from explorations.input_probe_ceiling.run_ceiling import (
    RAW_REPRESENTATIONS,
    _element_slots,
    build_representations,
    probe_slice,
)

N_ROWS = 60
N_UNLABELED = 12


class FakeCorpus(torch.utils.data.Dataset):
    """Minimal stand-in for `BulkChemistryDataset` on a redox (23-token) corpus.

    Carries `strat_labels` on every row (as the real corpus does, so unlabeled
    rows can still be split) while leaving some `qap_names` None -- the shape
    that makes the qap-labeled restriction meaningful.
    """

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        src = np.zeros((N_ROWS, N_TOKENS_REDOX), dtype=np.int64)
        frac = np.zeros((N_ROWS, N_TOKENS_REDOX), dtype=np.float32)
        for row in range(N_ROWS):
            # majors always reported, traces and redox intermittently
            populated = list(range(9)) + [
                j for j in range(9, N_TOKENS_REDOX) if rng.random() < 0.5
            ]
            src[row, populated] = np.arange(1, len(populated) + 1)
            frac[row, populated] = rng.uniform(1e-6, 0.5, size=len(populated))
        self._src = torch.from_numpy(src)
        self._frac = torch.from_numpy(frac)
        self.sample_ids = np.arange(N_ROWS, dtype=np.int64)
        self.qap_names = np.array(
            [None] * N_UNLABELED
            + ["basalt" if i % 2 else "granite" for i in range(N_ROWS - N_UNLABELED)],
            dtype=object,
        )
        self.strat_labels = np.array(
            ["unlabeled" if q is None else q for q in self.qap_names], dtype=object,
        )

    def __len__(self) -> int:
        return N_ROWS

    def __getitem__(self, idx: int) -> dict:
        return {
            "src": self._src[idx].clone(),
            "frac": self._frac[idx].clone(),
            "sample_id": int(self.sample_ids[idx]),
        }


@pytest.fixture
def corpus():
    return FakeCorpus()


def test_representations_have_expected_widths_and_exclude_the_redox_slot(corpus):
    train, test = np.arange(40), np.arange(40, N_ROWS)
    reps = build_representations(corpus, train, test)

    n_elements = N_TOKENS_REDOX - 1
    assert REDOX_TOKEN_INDEX not in _element_slots(N_TOKENS_REDOX)
    assert len(_element_slots(N_TOKENS_REDOX)) == n_elements
    for split, n in (("train", len(train)), ("test", len(test))):
        assert set(reps[split]) == set(RAW_REPRESENTATIONS)
        assert reps[split]["std_log"].shape == (n, n_elements)
        for name in ("std_log_mask", "std_frac_mask", "raw_frac_mask"):
            assert reps[split][name].shape == (n, 2 * n_elements)


def test_absent_elements_are_neutral_and_populated_ones_track_abundance(corpus):
    train, test = np.arange(40), np.arange(40, N_ROWS)
    reps = build_representations(corpus, train, test)

    keep = _element_slots(N_TOKENS_REDOX)
    n_elements = len(keep)
    frac = corpus._frac[train][:, keep].numpy()
    populated = (corpus._src[train][:, keep] != 0).numpy()

    for name in ("std_log_mask", "std_frac_mask", "raw_frac_mask"):
        block = reps["train"][name]
        assert np.array_equal(block[:, n_elements:], populated.astype(np.float32))
        assert np.all(block[:, :n_elements][~populated] == 0.0)

    values = reps["train"]["std_log_mask"][:, :n_elements]
    assert np.array_equal(reps["train"]["std_log"], values)
    assert np.array_equal(reps["train"]["raw_frac_mask"][:, :n_elements], frac)

    # standardization is affine in log10(frac), so within an element the
    # transformed values must order exactly as the abundances do
    for j in range(n_elements):
        rows = populated[:, j]
        if rows.sum() > 1:
            assert np.array_equal(
                np.argsort(values[rows, j]), np.argsort(frac[rows, j]),
            )


def test_standardization_never_reads_the_test_rows(corpus):
    train, test = np.arange(40), np.arange(40, N_ROWS)
    before = build_representations(corpus, train, test)["train"]["std_log"]

    corpus._frac[test] = corpus._frac[test].clamp(max=1e-5) * 1e-4
    after = build_representations(corpus, train, test)["train"]["std_log"]

    assert np.array_equal(before, after)


def test_probe_slice_keeps_labeled_roster_rows_only(corpus, tmp_path):
    config = {"seed": 0, "test_split": 0.25}
    roster = tmp_path / "roster.csv"
    keep_ids = corpus.sample_ids[::2]
    pd.DataFrame({"sample_id": keep_ids}).to_csv(roster, index=False)

    train_idx, test_idx = probe_slice(corpus, config, roster_path=roster)

    assert not set(train_idx) & set(test_idx)
    assert len(test_idx) > 0
    assert all(corpus.qap_names[i] is not None for i in train_idx)
    assert all(corpus.qap_names[i] is not None for i in test_idx)
    assert set(corpus.sample_ids[test_idx]) <= set(keep_ids)

"""Tripwire test for the `input_knn_ceiling` exploration.

Locks the raw-input matrix contract (train-split standardization + neutral
absent-fill) and doubles as a tripwire: a `src/` change that breaks the
exploration's reuse of `log_frac_stats` / the dataset fails here.  The
orchestrator `run_ceiling` needs a real checkpoint, so it is exercised by the
CLI run, not here.
"""

from pathlib import Path

import numpy as np
import torch

from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.data.tokens import N_TOKENS
from cnmineral.training.pretrain_objective import log_frac_stats
from explorations.input_knn_ceiling.run_ceiling import (
    _stack_test_frac,
    build_raw_representations,
)


def _fixture_dataset(path: Path) -> BulkChemistryDataset:
    return BulkChemistryDataset(
        csv_path=path,
        rock_group="igneous",
        require_all_majors=True,
        require_qap_name=True,
        min_class_count=2,
    )


def _split(n: int) -> tuple[list[int], list[int]]:
    """Deterministic non-overlapping test/train index lists over `range(n)`."""
    test_idx = list(range(0, n, 3))
    test_set = set(test_idx)
    train_idx = [i for i in range(n) if i not in test_set]
    return test_idx, train_idx


def test_build_raw_representations_shapes_and_absent_fill(
    corpus_subset_path: Path,
) -> None:
    ds = _fixture_dataset(corpus_subset_path)
    test_idx, train_idx = _split(len(ds))

    reps = build_raw_representations(ds, test_idx, train_idx, batch_size=64)

    assert set(reps) == {"raw_frac", "std_log"}
    for X in reps.values():
        assert X.shape == (len(test_idx), N_TOKENS)
        assert np.isfinite(X).all()

    # absent elements (raw_frac == 0) are filled with the neutral standardized 0;
    # the fixture has missing traces, so this path is actually exercised.
    absent = reps["raw_frac"] == 0.0
    assert absent.any()
    assert np.allclose(reps["std_log"][absent], 0.0)
    assert not np.allclose(reps["std_log"][~absent], 0.0)


def test_build_raw_representations_standardization_matches_train_stats(
    corpus_subset_path: Path,
) -> None:
    """std_log on populated positions == (log10(frac) - mean) / std, train stats."""
    ds = _fixture_dataset(corpus_subset_path)
    test_idx, train_idx = _split(len(ds))

    reps = build_raw_representations(ds, test_idx, train_idx, batch_size=64)

    mean, std = log_frac_stats(ds, train_idx, batch_size=64, num_workers=0)
    frac, populated = _stack_test_frac(ds, test_idx, batch_size=64)
    expected = (torch.log10(frac.clamp_min(1e-12)) - mean) / std
    expected = torch.where(populated, expected, torch.zeros_like(expected)).numpy()

    assert np.allclose(reps["std_log"], expected, atol=1e-5)
    assert np.array_equal(reps["raw_frac"], frac.numpy())

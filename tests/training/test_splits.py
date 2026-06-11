"""Tests for `cnmineral.training.splits.reserve_test`.

Three contracts:
- Deterministic split sizes for a given seed.
- Synthetic datasets (no qap_names) fall through to unstratified.
- Real datasets (with qap_names) stratify proportionally.
"""

from pathlib import Path

from conftest import _tiny_config

from cnmineral.training import create_dataset, reserve_test
from cnmineral.training.splits import _qap_labels_for_indices


def test_reserve_test_split_sizes() -> None:
    """reserve_test splits to expected sizes, deterministic on seed."""
    config = _tiny_config()
    dataset = create_dataset(config)
    splits_a = reserve_test(dataset, config)
    splits_b = reserve_test(dataset, config)
    # n=16, test_frac=0.25 -> 4 test, 12 trainval
    assert len(splits_a["test"]) == 4
    assert len(splits_a["trainval"]) == 12
    # disjoint + cover everything
    assert set(splits_a["test"]).isdisjoint(set(splits_a["trainval"]))
    assert sorted(splits_a["test"] + splits_a["trainval"]) == list(range(16))
    # same seed -> same indices
    assert splits_a["test"] == splits_b["test"]
    assert splits_a["trainval"] == splits_b["trainval"]


def test_reserve_test_synthetic_falls_back_to_unstratified() -> None:
    """Synthetic datasets lack qap_names; _qap_labels_for_indices
    returns None so train_test_split runs unstratified.  This test
    pins that the synthetic code path still works end-to-end after
    the stratify wiring landed."""
    from cnmineral.data.synthetic import SyntheticBulkChemistryDataset
    dataset = SyntheticBulkChemistryDataset(n=16, seed=0)
    assert not hasattr(dataset, "qap_names")
    labels = _qap_labels_for_indices(dataset, range(len(dataset)))
    assert labels is None


def test_reserve_test_stratifies_when_qap_names_present(
    corpus_subset_path: Path,
) -> None:
    """On a dataset that carries qap_names, reserve_test preserves
    per-class proportions in train/test within tolerance.

    Uses min_class_count=5 so every surviving class is large enough
    that stratification can actually divide it; uses test_frac=0.25
    so smallest classes (>=5 samples) land >=1 row in test."""
    from cnmineral.data.bulk_chemistry import BulkChemistryDataset
    dataset = BulkChemistryDataset(
        csv_path=corpus_subset_path, min_class_count=5,
    )
    config = {"seed": 0, "test_frac": 0.25}
    splits = reserve_test(dataset, config)
    test_set = set(splits["test"])
    trainval_set = set(splits["trainval"])

    from collections import Counter
    total = Counter(dataset.qap_names)
    in_test = Counter(dataset.qap_names[i] for i in test_set)

    # every class with >= 5 samples must appear in BOTH splits
    for cls, n in total.items():
        assert in_test[cls] >= 1, f"class {cls!r} (n={n}) missing from test"
        assert in_test[cls] < n, f"class {cls!r} (n={n}) entirely in test"
        # proportional preservation: in_test/total approximately test_frac
        # tolerance is wide for small classes (5-sample class -> 1 in test
        # is 20%, ratio swings if class has 6 samples -> 1 in test = 16.7%)
        ratio = in_test[cls] / n
        assert 0.10 <= ratio <= 0.40, (
            f"class {cls!r} (n={n}) test ratio {ratio:.2f} outside "
            f"[0.10, 0.40]; expected near 0.25"
        )

    # train + test = full corpus, no overlap
    assert test_set.isdisjoint(trainval_set)
    assert len(test_set) + len(trainval_set) == len(dataset)

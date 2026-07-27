import numpy as np
import pandas as pd
import pytest

from cnmineral.data.mineralogy_dataset import VALTEST_SPLIT_PATH, load_valtest_split

DROPPED_BULK = 411132


ROSTER_SIZE = 1999


def test_roster_partitions_and_excludes_dropped_bulk():
    roster = pd.read_csv(VALTEST_SPLIT_PATH)
    assert len(roster) == ROSTER_SIZE
    assert roster["sample_id"].is_unique
    assert set(roster["subset"]) == {"val", "test"}
    assert DROPPED_BULK not in set(roster["sample_id"])


def test_roster_counts_and_stratification():
    roster = pd.read_csv(VALTEST_SPLIT_PATH)
    counts = roster["subset"].value_counts()
    assert counts["val"] == 887
    assert counts["test"] == 1112
    per_class = roster.groupby("qap_name")["subset"].nunique()
    assert (per_class == 2).all()


def test_load_valtest_split_sorted_disjoint():
    val = load_valtest_split("val")
    test = load_valtest_split("test")
    for ids in (val, test):
        assert ids.dtype == np.int64
        assert np.array_equal(ids, np.sort(ids))
    assert len(np.intersect1d(val, test)) == 0
    assert len(val) + len(test) == ROSTER_SIZE


def test_load_valtest_split_unknown_subset_raises():
    with pytest.raises(ValueError, match="unknown valtest subset"):
        load_valtest_split("trainval")

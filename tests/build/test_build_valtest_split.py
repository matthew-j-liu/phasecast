"""`--extend` on the sealed roster: new bulks assigned, held ones untouched."""
import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "scripts" / "build" / "build_valtest_split.py")
spec = importlib.util.spec_from_file_location("build_valtest_split", SCRIPT)
bvs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bvs)


def _roster(tmp_path, monkeypatch, n_held=40, n_new=60):
    ids = range(n_held + n_new)
    bulks = pd.DataFrame([dict(sample_id=i, qap_name="big" if i % 2 else "mid")
                          for i in ids])
    held = bulks.iloc[:n_held].copy()
    held["subset"] = ["val" if i % 3 else "test" for i in range(n_held)]
    bulks_csv, out_csv = tmp_path / "valtest_bulks.csv", tmp_path / "split.csv"
    bulks.to_csv(bulks_csv, index=False)
    held.to_csv(out_csv, index=False)
    monkeypatch.setattr(bvs, "BULKS_CSV", bulks_csv)
    monkeypatch.setattr(bvs, "OUTPUT_CSV", out_csv)
    monkeypatch.setattr(sys, "argv", ["build_valtest_split", "--extend"])
    return held, out_csv


def test_extend_freezes_committed_assignments(tmp_path, monkeypatch):
    held, out_csv = _roster(tmp_path, monkeypatch)
    bvs.main()
    grown = pd.read_csv(out_csv)
    merged = held.merge(grown, on="sample_id", suffixes=("_held", "_new"))
    assert len(merged) == len(held)
    assert (merged.subset_held == merged.subset_new).all()


def test_extend_assigns_every_new_bulk_at_the_split_ratio(tmp_path, monkeypatch):
    held, out_csv = _roster(tmp_path, monkeypatch, n_held=40, n_new=60)
    bvs.main()
    grown = pd.read_csv(out_csv)
    assert len(grown) == 100
    assert grown.sample_id.is_unique
    added = grown[~grown.sample_id.isin(held.sample_id)]
    assert (added.subset == "test").sum() == math.ceil(60 * bvs.TEST_FRACTION)


def test_extend_refuses_a_table_the_roster_already_covers(tmp_path, monkeypatch):
    _roster(tmp_path, monkeypatch, n_held=40, n_new=0)
    with pytest.raises(SystemExit, match="already covers"):
        bvs.main()


def test_extend_rejects_a_class_too_thin_to_stratify(tmp_path, monkeypatch):
    held, out_csv = _roster(tmp_path, monkeypatch)
    bulks = pd.read_csv(tmp_path / "valtest_bulks.csv")
    bulks.loc[len(bulks)] = dict(sample_id=999, qap_name="lonely")
    bulks.to_csv(tmp_path / "valtest_bulks.csv", index=False)
    with pytest.raises(SystemExit, match="cannot stratify"):
        bvs.main()

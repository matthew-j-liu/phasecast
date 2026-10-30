"""Tier expansion in build_corpus_bulk_tables: the contamination guard and the
exhausted-class respill."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (Path(__file__).resolve().parents[2]
          / "scripts" / "build" / "build_corpus_bulk_tables.py")
spec = importlib.util.spec_from_file_location("build_corpus_bulk_tables", SCRIPT)
bct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bct)

# `gone` is fully consumed by the two train tables plus valtest: it exercises
# the exhausted-class path, where the alpha-shape quota is unreachable.
SIZES = {"big": 600, "mid": 120, "rare": 18, "gone": 12}


def _subset(tmp_path):
    df = pd.DataFrame([
        dict(sample_id=i, qap_name=q, series="subalkaline", sio2=50.0,
             feo=8.0, fe2o3=np.nan, fe_measured=False,
             flag_oxidesum=False, flag_supersilicic=False, flag_ultramafic=False)
        for i, q in enumerate(q for q, n in SIZES.items() for _ in range(n))])
    train = df.groupby("qap_name", sort=False).head(8)
    rest = df[~df.sample_id.isin(train.sample_id)]
    expansion = rest.groupby("qap_name", sort=False).head(2)
    rest = rest[~rest.sample_id.isin(expansion.sample_id)]
    valtest = rest.groupby("qap_name", sort=False).head(2)
    out = tmp_path / "data" / "corpus"
    out.mkdir(parents=True)
    for name, tab in [("train_bulks", train), ("train_bulks_expansion", expansion),
                      ("valtest_bulks", valtest)]:
        tab[bct.KEEP_COLS].to_csv(out / f"{name}.csv", index=False)
    return df, train, expansion, valtest, out


def _used(*tiers):
    return set().union(*(set(t.sample_id) for t in tiers))


def test_expansion_never_draws_a_bulk_another_tier_holds(tmp_path):
    df, train, expansion, valtest, _ = _subset(tmp_path)
    have, used = valtest.qap_name.value_counts(), _used(train, expansion, valtest)
    out = bct.draw_expansion(df, 200, np.random.default_rng(3), have, used)
    assert not set(out.sample_id) & used


def test_exhausted_class_keeps_its_held_count_and_budget_respills(tmp_path):
    df, train, expansion, valtest, _ = _subset(tmp_path)
    have, used = valtest.qap_name.value_counts(), _used(train, expansion, valtest)
    out = bct.draw_expansion(df, 200, np.random.default_rng(3), have, used)
    assert "gone" not in set(out.qap_name)
    assert len(out) == 200


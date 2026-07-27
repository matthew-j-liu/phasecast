"""Tier expansion in build_corpus_bulk_tables: block emit and the valtest grow."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cnmineral.data.volcanic_subset import expansion_quotas

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


def _quotas(df, have, used, budget):
    rem = df[~df.sample_id.isin(used)]
    pool = df.qap_name.value_counts()
    have = have.reindex(pool.index).fillna(0).astype(int)
    avail = rem.qap_name.value_counts().reindex(pool.index).fillna(0).astype(int)
    return expansion_quotas(pool, have, avail, budget, 0.5)


def test_nested_block_prefix_realizes_inner_quotas(tmp_path):
    df, train, expansion, valtest, _ = _subset(tmp_path)
    have, used = train.qap_name.value_counts(), _used(train, expansion, valtest)
    q_full = _quotas(df, have, used, 200)
    q_inner = _quotas(df, have, used, 80)
    out = bct.draw_expansion(df, 200, 0.5, np.random.default_rng(3), have, used,
                             block=80)
    counts = out.qap_name.value_counts()
    assert counts.to_dict() == {q: int(k) for q, k in q_full.items() if k > 0}
    k1 = np.minimum(q_inner, q_full)
    prefix = out.iloc[: int(k1.sum())].qap_name.value_counts()
    assert prefix.to_dict() == {q: int(k) for q, k in k1.items() if k > 0}


def test_block_reorders_but_never_changes_the_selection(tmp_path):
    df, train, expansion, valtest, _ = _subset(tmp_path)
    have, used = train.qap_name.value_counts(), _used(train, expansion, valtest)
    flat = bct.draw_expansion(df, 200, 0.5, np.random.default_rng(3), have, used)
    nested = bct.draw_expansion(df, 200, 0.5, np.random.default_rng(3), have, used,
                                block=80)
    assert set(flat.sample_id) == set(nested.sample_id)


def test_expansion_never_draws_a_bulk_another_tier_holds(tmp_path):
    df, train, expansion, valtest, _ = _subset(tmp_path)
    have, used = valtest.qap_name.value_counts(), _used(train, expansion, valtest)
    out = bct.draw_expansion(df, 200, 0.5, np.random.default_rng(3), have, used)
    assert not set(out.sample_id) & used


def test_exhausted_class_keeps_its_held_count_and_budget_respills(tmp_path):
    df, train, expansion, valtest, _ = _subset(tmp_path)
    have, used = valtest.qap_name.value_counts(), _used(train, expansion, valtest)
    out = bct.draw_expansion(df, 200, 0.5, np.random.default_rng(3), have, used)
    assert "gone" not in set(out.qap_name)
    assert len(out) == 200


def _run(tmp_path, monkeypatch, df, *argv):
    monkeypatch.setattr(bct, "REPO", tmp_path)
    monkeypatch.setattr(bct, "build_subset", lambda: df.copy())
    monkeypatch.setattr(sys, "argv", ["build_corpus_bulk_tables", *argv])
    bct.main()


def test_valtest_expand_leads_with_the_held_roster(tmp_path, monkeypatch):
    df, train, expansion, valtest, out = _subset(tmp_path)
    _run(tmp_path, monkeypatch, df, "--valtest-expand", "200")
    grown = pd.read_csv(out / "valtest_bulks.csv")
    assert list(grown.columns) == bct.KEEP_COLS
    assert len(grown) == len(valtest) + 200
    assert grown.sample_id.is_unique
    assert set(grown.sample_id[:len(valtest)]) == set(valtest.sample_id)
    assert not set(grown.sample_id) & _used(train, expansion)


def test_valtest_expand_carries_the_alpha_shape_for_the_whole_tier(tmp_path,
                                                                  monkeypatch):
    df, train, expansion, valtest, out = _subset(tmp_path)
    have, used = valtest.qap_name.value_counts(), _used(train, expansion, valtest)
    quotas = _quotas(df, have, used, 200)
    _run(tmp_path, monkeypatch, df, "--valtest-expand", "200")
    grown = pd.read_csv(out / "valtest_bulks.csv").qap_name.value_counts()
    assert grown.to_dict() == (have + quotas).to_dict()

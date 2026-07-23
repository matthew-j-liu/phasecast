"""Nested-block expansion emit (`--expand-block`) in build_corpus_bulk_tables."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from cnmineral.data.volcanic_subset import expansion_quotas

SCRIPT = (Path(__file__).resolve().parents[2]
          / "scripts" / "build" / "build_corpus_bulk_tables.py")
spec = importlib.util.spec_from_file_location("build_corpus_bulk_tables", SCRIPT)
bct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bct)


def _subset(tmp_path):
    sizes = {"big": 600, "mid": 120, "rare": 18}
    df = pd.DataFrame([dict(sample_id=i, qap_name=q) for i, q in enumerate(
        q for q, n in sizes.items() for _ in range(n))])
    train = df.groupby("qap_name", sort=False).head(10)
    rest = df[~df.sample_id.isin(train.sample_id)]
    valtest = rest.groupby("qap_name", sort=False).head(2)
    train.to_csv(tmp_path / "train_bulks.csv", index=False)
    valtest.to_csv(tmp_path / "valtest_bulks.csv", index=False)
    return df, train, valtest


def _quotas(df, train, valtest, budget):
    used = set(train.sample_id) | set(valtest.sample_id)
    rem = df[~df.sample_id.isin(used)]
    pool = df.qap_name.value_counts()
    have = train.qap_name.value_counts().reindex(pool.index).fillna(0).astype(int)
    avail = rem.qap_name.value_counts().reindex(pool.index).fillna(0).astype(int)
    return expansion_quotas(pool, have, avail, budget, 0.5)


def test_nested_block_prefix_realizes_inner_quotas(tmp_path):
    df, train, valtest = _subset(tmp_path)
    q_full = _quotas(df, train, valtest, 200)
    q_inner = _quotas(df, train, valtest, 80)
    out = bct.draw_expansion(df, 200, 0.5, np.random.default_rng(3), tmp_path,
                             block=80)
    counts = out.qap_name.value_counts()
    assert counts.to_dict() == {q: int(k) for q, k in q_full.items() if k > 0}
    k1 = np.minimum(q_inner, q_full)
    prefix = out.iloc[: int(k1.sum())].qap_name.value_counts()
    assert prefix.to_dict() == {q: int(k) for q, k in k1.items() if k > 0}


def test_block_reorders_but_never_changes_the_selection(tmp_path):
    df, train, valtest = _subset(tmp_path)
    flat = bct.draw_expansion(df, 200, 0.5, np.random.default_rng(3), tmp_path)
    nested = bct.draw_expansion(df, 200, 0.5, np.random.default_rng(3), tmp_path,
                                block=80)
    assert set(flat.sample_id) == set(nested.sample_id)

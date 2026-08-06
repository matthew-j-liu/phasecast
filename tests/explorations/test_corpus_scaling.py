"""Tripwire test for the `corpus_scaling` exploration.

Locks the pure pieces: the md5 roster order (deterministic, input-order
independent, prefixes nest), the per-point config overrides, tag parsing into
axis coordinates, and the last-doubling delta table. The ladder orchestrator
needs real checkpoints, so it is exercised by the CLI run, not here.
"""

from explorations.corpus_scaling.aggregate import (
    PHASE_METRICS,
    last_doubling_deltas,
    point_record,
)
from explorations.corpus_scaling.run_ladder import hash_order, point_config


def test_hash_order_deterministic_and_input_order_independent() -> None:
    ids = list(range(100, 200))
    ordered = hash_order(ids)
    assert ordered == hash_order(list(reversed(ids)))
    assert sorted(ordered) == ids


def test_hash_order_prefixes_nest() -> None:
    ordered = hash_order(list(range(1000)))
    small, large = set(ordered[:100]), set(ordered[:400])
    assert small < large


def test_point_config_overrides_only_the_subset_keys() -> None:
    base = {"epochs": 120, "head_checkpoint_dir": "checkpoints/prod",
            "train_bulk_ids": "stale.csv", "seed": 0}
    config = point_config(base, "corpus_scaling_allrock_n625", None,
                          draw_limit=24, seed=1)
    assert config["epochs"] == 120
    assert config["head_checkpoint_dir"].endswith("corpus_scaling_allrock_n625")
    assert "train_bulk_ids" not in config
    assert config["train_draw_limit"] == 24
    assert config["seed"] == 1
    assert base["head_checkpoint_dir"] == "checkpoints/prod"
    assert base["seed"] == 0


def _metrics(joint_nll: float, pf_r: float) -> dict:
    phase = {k: 0.1 for k in PHASE_METRICS} | {"pf_r": pf_r}
    return {
        "joint_nll": joint_nll, "bernoulli_nll": 1.0, "dirichlet_nll": -2.0,
        "per_phase": [
            {"phase": name, **phase} for name in ("g", "spl", "amp", "ilm", "plag")
        ],
    }


def test_point_record_parses_both_axes() -> None:
    bests = {"mode": -1.0, "comp": -2.0, "cond": 3.0}
    n = point_record("corpus_scaling_n625", _metrics(-1.0, 0.5), bests)
    assert (n["axis"], n["n_bulks"], n["draws_per_bulk"]) == ("bulks", 625, 48)
    assert n["n_train_rows_nominal"] == 625 * 48 * 2
    d = point_record("corpus_scaling_d24", _metrics(-1.0, 0.5), bests)
    assert (d["axis"], d["n_bulks"], d["draws_per_bulk"]) == ("draws", 10000, 24)
    assert d["per_phase"]["g"]["pf_r"] == 0.5


def test_last_doubling_deltas() -> None:
    bests = {"mode": 0.0, "comp": 0.0, "cond": 0.0}
    points = [
        point_record("corpus_scaling_n5000", _metrics(-1.0, 0.5), bests),
        point_record("corpus_scaling_n10000", _metrics(-1.5, 0.7), bests),
        point_record("corpus_scaling_n625", _metrics(-0.2, 0.1), bests),
        point_record("corpus_scaling_d24", _metrics(-9.0, 0.9), bests),
    ]
    deltas = last_doubling_deltas(points)
    joint = next(r for r in deltas if r["metric"] == "joint_nll")
    assert (joint["at_prev"], joint["at_last"]) == (-1.0, -1.5)
    g_pf = next(r for r in deltas if r["metric"] == "pf_r" and r["phase"] == "g")
    assert abs(g_pf["delta"] - 0.2) < 1e-12
    assert last_doubling_deltas(points[:1]) == []

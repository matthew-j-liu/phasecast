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


from torch.utils.data import TensorDataset
import pandas as pd
import torch

from explorations.corpus_scaling.run_depth_grid import (
    aggregate,
    batches_per_epoch,
    line,
    selected_steps,
)
from explorations.corpus_scaling.run_variant import truncated_loader


def _monitor(mode_curve: list[float], probe_curve: list[float]) -> dict:
    n = len(mode_curve)
    return {"train": [0.0] * n, "val": [0.0] * n,
            "val_presence_bce": [0.0] * n, "val_mode_nll": mode_curve,
            "val_comp_nll": [0.0] * n, "val_cond_nll": [0.0] * n,
            "real_rock_train_share_mae": probe_curve}


def test_selected_steps_follows_the_cell_rule() -> None:
    monitor = _monitor(mode_curve=[3.0, 1.0, 2.0, 4.0], probe_curve=[0.9, 0.8, 0.7, 0.75])
    grid = dict(series="grid", n_bulks=78, rule="val")
    final = dict(series="final", n_bulks=78, rule="final")
    stop = dict(series="stop mae", n_bulks=78, rule="real_rock_train_share_mae")
    ladder = dict(series="corpus val", n_bulks=78, rule="val")
    assert selected_steps(grid, monitor) == 8 * 2
    assert selected_steps(final, monitor) == 8 * 4
    assert selected_steps(stop, monitor) == 8 * 3
    assert selected_steps(ladder, monitor) == batches_per_epoch(78) * 2
    assert batches_per_epoch(78) == 8
    assert batches_per_epoch(10000) == 932


def _rows() -> pd.DataFrame:
    def row(series, n, depth, seed, mae, steps):
        r = dict(series=series, n_bulks=n, depth=depth, seed=seed,
                 tag=f"{series}{n}s{seed}", rule="val", steps=steps, mae_mean=mae)
        r.update({m: 0.0 for m in ("headline_nll", "auroc_mean", "recal_val_joint",
                                   "recal_share_mae")})
        r.update({f"mae_{g}": 0.0 for g in ("fsp", "px", "amp", "bi", "ol", "opq", "q")})
        return r
    return pd.DataFrame([
        row("grid", 78, 224, 0, 0.30, 56), row("final", 78, 224, 0, 0.12, 224),
        row("grid", 312, 224, 0, 0.11, 224),
        row("grid", 10000, 224, 0, 0.10, 224), row("grid", 10000, 224, 1, 0.12, 224),
        row("grid", 10000, 224, 2, 0.11, 224),
    ])


def test_aggregate_spans_seeds_and_line_prefers_final_cells() -> None:
    agg = aggregate(_rows())
    big = agg[(agg.series == "grid") & (agg.n_bulks == 10000)].iloc[0]
    assert big.seeds == 3
    assert (big.mae_mean_min, big.mae_mean_max) == (0.10, 0.12)
    assert abs(big.mae_mean_mean - 0.11) < 1e-12
    curve = line(agg, "grid", 224)
    assert curve.n_bulks.tolist() == [78, 312, 10000]
    assert curve.series.tolist() == ["final", "grid", "grid"]
    assert curve.mae_mean_mean.tolist()[0] == 0.12


def test_truncated_loader_cuts_only_shuffled_epochs() -> None:
    data = TensorDataset(torch.arange(40.0))
    loader_cls = truncated_loader(3)
    assert len(list(loader_cls(data, batch_size=4, shuffle=True))) == 3
    assert len(list(loader_cls(data, batch_size=4))) == 10

"""Tripwire test for the `corpus_scaling` exploration.

Locks the pure pieces: the md5 roster order (deterministic, input-order
independent, prefixes nest), the per-point config overrides, tag parsing into
axis coordinates, the last-doubling delta table, and the depth grid's cell
rules, share metrics and share-error slopes; the zero-shot line's
per-group aggregation, pretraining match and membership rule, random
training subsets, paired bands and subset percentiles. The ladder orchestrator
needs real checkpoints, so it is exercised by the CLI run, not here.
"""

from explorations.corpus_scaling.aggregate import (
    PHASE_METRICS,
    last_doubling_deltas,
    point_record,
)
from explorations.corpus_scaling.run_ladder import hash_order, point_config, write_roster


def test_hash_order_deterministic_and_input_order_independent() -> None:
    ids = list(range(100, 200))
    ordered = hash_order(ids)
    assert ordered == hash_order(list(reversed(ids)))
    assert sorted(ordered) == ids


def test_hash_order_prefixes_nest() -> None:
    ordered = hash_order(list(range(1000)))
    small, large = set(ordered[:100]), set(ordered[:400])
    assert small < large


def test_write_roster_writes_the_sorted_md5_prefix_and_nests(tmp_path) -> None:
    ordered = hash_order(list(range(1000)))
    rosters = {n: pd.read_csv(write_roster(tmp_path / f"n{n}", ordered, n))["sample_id"].tolist()
               for n in (10, 20, 40)}
    assert rosters[10] == sorted(ordered[:10])
    assert set(rosters[10]) < set(rosters[20]) < set(rosters[40])
    assert (tmp_path / "n10" / "train_bulk_ids.csv").exists()


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
import numpy as np
import pytest
import pandas as pd
import torch

from explorations.corpus_scaling.run_depth_grid import (
    METRICS,
    aggregate,
    batches_per_epoch,
    line,
    selected_steps,
    share_metrics,
    share_slopes,
    with_means,
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
    # 10 compositions fill one 1,024-row batch, so an epoch is one step
    assert batches_per_epoch(10) == 1
    assert selected_steps(dict(series="final", n_bulks=10, rule="final"), monitor) == 4
    assert selected_steps(dict(series="final", n_bulks=40, rule="final"), monitor) == 4 * 4


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


def test_zero_shot_figure_aggregates_one_group_at_one_depth_per_size(tmp_path) -> None:
    from explorations.corpus_scaling.fig_zero_shot_scaling import PANELS, load_line
    cats = ("fsp", "px", "amp", "bi", "ol", "opq", "q")
    assert set(PANELS) == set(cats) | {"mean7"}

    def row(n, seed, depth=224, group="all"):
        r = dict(benchmark="plutonic", group=group, depth=depth, n_bulks=n, subset=0, seed=seed,
                 tag=f"final{depth}_allrock_n{n}", n_rocks=40)
        r.update({f"mae_{g}": v for g, v in zip(cats, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))})
        r["mae_mean7"] = 0.4
        return r
    random_subset = dict(row(78, 0), subset=3, mae_fsp=0.9)
    df = pd.DataFrame([row(78, 0), row(78, 1), row(312, 0), row(312, 0, depth=56),
                       row(312, 1, group="pp1793"), random_subset])
    df.loc[1, "mae_fsp"] = 0.3
    df.to_csv(tmp_path / "line.csv", index=False)
    agg = load_line(224, csv=tmp_path / "line.csv")
    assert agg.index.tolist() == [78, 312]
    assert agg.loc[78, "seeds"] == 2 and agg.loc[312, "seeds"] == 1
    assert abs(agg.loc[78, "mae_fsp_mean"] - 0.2) < 1e-12
    assert (agg.loc[78, "mae_fsp_min"], agg.loc[78, "mae_fsp_max"]) == (0.1, 0.3)
    assert abs(agg.loc[312, "mae_mean7_mean"] - 0.4) < 1e-12 and agg.loc[312, "n_rocks"] == 40
    assert load_line(224, group="pp1793", csv=tmp_path / "line.csv").index.tolist() == [312]


def test_subset_line_averages_random_subsets_and_keeps_the_full_corpus_from_its_seeds(tmp_path) -> None:
    from explorations.corpus_scaling.fig_zero_shot_scaling import PANELS, load_subset_line

    def row(n, subset, seed, v, group="all"):
        r = dict(group=group, depth=224, n_bulks=n, subset=subset, seed=seed)
        r.update({f"mae_{c}": v for c in PANELS})
        return r
    rows = [row(10, 0, 0, 0.9), row(10, 1, 0, 0.1), row(10, 2, 0, 0.2), row(10, 3, 0, 0.3), row(10, 4, 0, 0.4),
            row(10, 5, 0, 0.5), row(10, 6, 0, 0.7, group="pp1793"), row(10000, 0, 0, 0.1), row(10000, 0, 1, 0.3)]
    pd.DataFrame(rows).to_csv(tmp_path / "line.csv", index=False)
    agg = load_subset_line(224, csv=tmp_path / "line.csv")
    assert agg.index.tolist() == [10, 10000]
    assert agg.loc[10, "mae_fsp_mean"] == pytest.approx(0.3)
    assert (agg.loc[10, "mae_fsp_min"], agg.loc[10, "mae_fsp_max"]) == pytest.approx((0.2, 0.4))
    assert agg.loc[10000, "mae_mean7_mean"] == pytest.approx(0.2)
    assert (agg.loc[10000, "mae_fsp_min"], agg.loc[10000, "mae_fsp_max"]) == pytest.approx((0.1, 0.3))


def test_pretraining_match_renormalizes_iron_and_takes_every_copy_within_tolerance() -> None:
    from explorations.corpus_scaling import run_zero_shot_line as z
    base = dict(sio2=70.0, tio2=0.3, al2o3=15.0, mgo=1.0, cao=2.5, na2o=4.0, k2o=3.5)
    rock = pd.DataFrame([dict(base, feo_tot=np.nan, fe2o3_tot=2.0)])
    # the same analysis reported 2% higher (a raw total), with its iron split into FeO
    # and Fe2O3; a second copy; a near-miss about 0.06 wt% off in silica after renormalizing; an empty row
    fe = 2.0 * z.FE2O3_TO_FEO_MASS
    copy = {k: 1.02 * v for k, v in base.items()}
    pool = pd.DataFrame([
        dict(copy, feo=1.02 * fe / 2, fe2o3=1.02 / 2 * 2.0, feo_tot=np.nan, fe2o3_tot=np.nan),
        dict(copy, feo=np.nan, fe2o3=np.nan, feo_tot=1.02 * fe, fe2o3_tot=np.nan),
        dict(base, sio2=70.0 + 0.2, feo=np.nan, fe2o3=np.nan, feo_tot=fe, fe2o3_tot=np.nan),
        dict({k: np.nan for k in base}, feo=np.nan, fe2o3=np.nan, feo_tot=np.nan, fe2o3_tot=np.nan)])
    hits, dist = z.match(z.renormalized_majors(rock), z.renormalized_majors(pool))
    assert sorted(hits[0].tolist()) == [0, 1] and dist[0] < 1e-9
    ids = [np.array([11, 12]), np.array([13]), np.array([], dtype=int)]
    assert z.membership_status(ids, test_ids=[13, 12]).tolist() == ["seen", "held_out", "absent"]
    assert z.membership_status(ids, test_ids=[11, 12, 13]).tolist() == ["held_out", "held_out", "absent"]


def test_group_metrics_read_each_rock_group_on_its_own_rows() -> None:
    from explorations.corpus_scaling.run_zero_shot_line import group_metrics
    from explorations.corpus_scaling.run_depth_grid import share_metrics, with_means
    rng = np.random.default_rng(0)
    magnitude, share = rng.dirichlet(np.ones(7), 12), rng.dirichlet(np.ones(7), 12)
    numeric = rng.random((12, 7)) > 0.3
    groups = {"all": np.ones(12, dtype=bool), "first": np.arange(12) < 5}
    out = group_metrics(share, magnitude, numeric, groups)
    for name, m in groups.items():
        expected = with_means(share_metrics(share[m], magnitude[m], numeric[m])[0])
        assert np.allclose([out[name][f"mae_{k}"] for k in ("fsp", "px", "amp", "bi", "ol", "opq", "q",
                                                              "silicate", "other", "mean7")],
                           expected, equal_nan=True)
        assert out[name]["n_rocks"] == int(m.sum()) and out[name]["n_fsp"] == int(numeric[m, 0].sum())


def test_random_training_subsets_are_reproducible_distinct_and_drawn_from_the_corpus() -> None:
    from explorations.corpus_scaling.run_training_subsets import random_subset
    ids = list(range(1000, 1500))
    a, b = random_subset(ids, 40, 1), random_subset(ids, 40, 2)
    assert np.array_equal(a, random_subset(ids, 40, 1)) and not np.array_equal(a, b)
    assert len(np.unique(a)) == 40 and set(a) <= set(ids) and np.all(np.diff(a) > 0)


def test_paired_bands_are_zero_for_identical_sizes_and_contain_the_estimate() -> None:
    from explorations.corpus_scaling import run_zero_shot_line as z
    rng = np.random.default_rng(0)
    one = rng.random((60, 7))
    one[rng.random((60, 7)) < 0.3] = np.nan
    units = np.repeat(np.arange(20), 3).astype(str)
    flat = np.stack([one] * len(z.SIZES))
    out = z.paired_bands(flat, units, n_boot=200)
    for name, per_label in out["differences"].items():
        assert np.allclose(per_label["mean7"], 0.0)
    moved = flat.copy()
    moved[z.SIZES.index(10000)] += 0.05
    out = z.paired_bands(moved, units, n_boot=200)
    est, lo, hi = out["differences"]["net_10000_minus_10"]["mean7"]
    assert est == pytest.approx(0.05) and lo <= est <= hi
    for n, per_label in out["points"].items():
        e, a, b = per_label["mean7"]
        assert a <= e <= b


def test_subset_percentile_ranks_a_subset_among_random_unit_draws() -> None:
    from explorations.corpus_scaling import run_zero_shot_line as z
    rng = np.random.default_rng(1)
    units = np.repeat(np.arange(40), 4).astype(str)
    errors = np.stack([rng.random((160, 7)) for _ in z.SIZES])
    # a subset whose rocks carry a large extra rise sits at the top of the draws
    subset = np.isin(units, units[:40])
    steep = errors.copy()
    steep[z.SIZES.index(78)][subset] += 1.0
    out = z.subset_percentiles(steep, units, subset, n_draw=300)
    assert out["rise_78_minus_10"]["percentile"] > 95 and out["rise_78_minus_10"]["units_drawn"] == 10
    out = z.subset_percentiles(errors, units, subset, n_draw=300)
    assert 2 < out["net_10000_minus_10"]["percentile"] < 98


def test_distance_table_reads_each_basis_against_its_target() -> None:
    from explorations.corpus_scaling.fig_head_simulator_distance import distance_table
    rng = np.random.default_rng(0)
    cats = ("fsp", "px", "amp", "bi", "ol", "opq", "q")
    numeric = np.ones((4, 7), dtype=bool)
    numeric[0, 6] = False
    magnitude = rng.dirichlet(np.ones(7), size=4) * numeric
    sim = rng.dirichlet(np.ones(7), size=4) * numeric
    sim[3] = np.nan  # a rock no draw reaches
    marginal = np.where(np.isfinite(sim), sim, magnitude)
    d = distance_table(magnitude, marginal, sim, magnitude, numeric)
    # the marginal read copies the simulator where it exists: zero distance to it
    assert all(d[f"d_sim_{g}"] == 0.0 for g in cats) and d["d_sim_mean7"] == 0.0
    # the forced read copies the rocks: zero distance to them on every quantified row
    assert all(d[f"d_rock_{g}"] == 0.0 for g in cats)
    # the marginal read against the rocks counts every quantified row, unreached included
    assert abs(d["d_rock_marginal_fsp"] - np.abs(marginal[:, 0] - magnitude[:, 0]).mean()) < 1e-12
    assert abs(d["d_rock_marginal_q"] - np.abs(marginal[1:, 6] - magnitude[1:, 6]).mean()) < 1e-12
    assert abs(d["d_rock_marginal_mean7"] - np.mean([d[f"d_rock_marginal_{g}"] for g in cats])) < 1e-12
    # against the simulator only the reached rows count
    d = distance_table(magnitude, magnitude, sim, magnitude, numeric)
    assert abs(d["d_sim_fsp"] - np.abs(magnitude[:3, 0] - sim[:3, 0]).mean()) < 1e-12


def test_truncated_loader_cuts_only_shuffled_epochs() -> None:
    data = TensorDataset(torch.arange(40.0))
    loader_cls = truncated_loader(3)
    assert len(list(loader_cls(data, batch_size=4, shuffle=True))) == 3
    assert len(list(loader_cls(data, batch_size=4))) == 10


def test_share_metrics_match_numpy_on_the_quantified_rows() -> None:
    rng = np.random.default_rng(1)
    share, truth = rng.uniform(size=(50, 7)), rng.uniform(size=(50, 7))
    numeric = rng.uniform(size=(50, 7)) > 0.3
    share[3, 2] = np.nan
    mae, r, n = share_metrics(share, truth, numeric)
    for j in range(7):
        m = numeric[:, j] & np.isfinite(share[:, j])
        assert n[j] == m.sum()
        assert np.isclose(mae[j], np.abs(share[m, j] - truth[m, j]).mean())
        assert np.isclose(r[j], np.corrcoef(share[m, j], truth[m, j])[0, 1])
    means = with_means(mae)
    assert len(means) == 10
    assert np.isclose(means[7], mae[:4].mean()) and np.isclose(means[8], mae[4:].mean())
    assert np.isclose(means[9], mae.mean())


def test_share_slopes_recover_a_known_slope_per_doubling() -> None:
    magnitude = np.zeros((20, 7))
    magnitude[:, :2] = 0.5
    numeric = magnitude > 0
    units = np.repeat(np.arange(5), 4)
    rows, per_rock_share = [], {}
    for n in (78, 312, 1250):
        e = 0.2 - 0.01 * np.log2(n)
        # the two seeds err by e in opposite directions, so an ensemble of
        # their shares would read zero error and hide the slope
        for seed, sign in ((0, 1.0), (1, -1.0)):
            share = np.zeros((20, 7))
            share[:, 0], share[:, 1] = 0.5 + sign * e, 0.5 - sign * e
            tag = f"final224_allrock_n{n}" + (f"_seed{seed}" if seed else "")
            per_rock_share[tag] = {"zero": share, "recal": share}
            rows.append(dict(series="final", n_bulks=n, depth=224, seed=seed, tag=tag, rule="final",
                             steps=224, **{m: 0.0 for m in METRICS}))
    df = pd.DataFrame(rows)
    slopes = share_slopes(df, aggregate(df), per_rock_share, magnitude, numeric, units, n_boot=50)
    for name in ("zero_shot_val", "recal"):
        s = slopes["grid 224"][f"{name}_share_mae_per_doubling"]
        for label in ("fsp", "px", "silicate"):
            assert np.isclose(s[label]["mean"], -0.01)
            assert np.isclose(s[label]["lo"], -0.01) and np.isclose(s[label]["hi"], -0.01)
        assert np.isnan(s["amp"]["mean"])

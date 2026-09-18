"""Tripwire for `explorations/prior_search`: the scheme family (clipping,
density, draws), the importance reweighting of the #78 draws, the score
aggregation per subset, the nested rosters, and the screen end to end on a
synthetic run; the predictability read on the same synthetic run."""

import json

import numpy as np
import pandas as pd
import pytest

from cnmineral.data.condition_proposal import P_HI, P_LO, W_CAP, draw_conditions, q_density, w_max
from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.real_rock import load_benchmark
from explorations.corpus_scaling.run_ladder import hash_order
from explorations.generator_reach import run_read
from explorations.generator_reach.compositions import composition_table
from explorations.prior_search import run_predictability as predict
from explorations.prior_search import run_prior_screen as screen
from explorations.prior_search.schemes import GRID, Scheme, area_fraction, density, draw

SAMPLE_COLS = ["source", "sample_id", "unit", "total_crystals_pct", "alteration_extent",
               "mode_basis", "sio2", "tio2", "al2o3", "feo_tot", "fe2o3_tot", "mno", "mgo",
               "cao", "na2o", "k2o", "p2o5", "loi"]


def test_scheme_clips_to_the_broad_box_and_the_solubility_bound():
    edge = Scheme("edge", 11.0, 7.0, 3.0, 2.5)
    assert (edge.p_lo, edge.p_hi, edge.w_lo, edge.w_hi) == (8.0, 12.0, 4.5, 8.0)
    low = Scheme("low", 1.0, 1.0, 1.0, 1.0)
    d = density(low, [0.6, 1.5, 3.0], [1.0, 1.0, 1.0])
    assert d[2] == 0.0 and d[0] > 0 and d[1] > 0
    assert density(low, [1.0], [2.5])[0] == 0.0
    whole = Scheme("whole", (P_LO + P_HI) / 2, W_CAP / 2, (P_HI - P_LO) / 2, W_CAP / 2)
    p, w, q = draw(whole, 500, np.random.default_rng(0))
    assert np.allclose(q, q_density(p, w)) and (w <= w_max(p) + 1e-9).all()
    assert area_fraction(whole) == pytest.approx(1.0, abs=1e-3)
    for s in GRID:
        assert P_LO <= s.p_lo < s.p_hi <= P_HI and 0.0 <= s.w_lo < s.w_hi <= W_CAP
        assert 0.0 < area_fraction(s) < 0.5


def _rungs():
    rows = []
    for d, s in zip(range(4), (0.2, 0.4, 0.6, 0.8)):
        for T, rung in ((1000.0, 50), (900.0, 30)):
            row = dict(bulk=0, draw=d, T_celsius=T, rung=rung)
            row.update({f"s_{g}": 0.0 for g in GROUPS})
            row.update(s_fsp=s, s_px=1 - s)
            rows.append(row)
    return pd.DataFrame(rows)


def _weights(w):
    return pd.DataFrame(dict(bulk=0, draw=range(4), w=w))


def test_reweighting_is_the_plain_mean_over_the_draws_inside():
    rungs = _rungs()
    two = screen.reweighted_marginals(rungs, _weights([0.0, 1.0, 1.0, 0.0])).loc[0]
    assert two["mean_fsp"] == pytest.approx(0.5) and two["mean_px"] == pytest.approx(0.5)
    assert two["n_eff"] == pytest.approx(2.0)
    assert two["spread_fsp"] == pytest.approx(0.1)
    skew = screen.reweighted_marginals(rungs, _weights([0.0, 3.0, 1.0, 0.0])).loc[0]
    assert skew["mean_fsp"] == pytest.approx(0.45) and skew["n_eff"] == pytest.approx(1.6)
    none = screen.reweighted_marginals(rungs, _weights([0.0] * 4)).loc[0]
    assert np.isnan(none["mean_fsp"]) and none["n_eff"] == 0.0


def test_scores_average_per_category_then_over_the_silicates_per_subset():
    cols = [f"mean_{g}" for g in GROUPS] + [f"spread_{g}" for g in GROUPS]
    marg = pd.DataFrame(0.0, index=[0, 1, 2], columns=cols)
    marg.loc[0, ["mean_fsp", "mean_px"]] = [0.6, 0.4]
    marg.loc[1, "mean_fsp"] = 0.4
    marg.loc[2, ["mean_fsp", "mean_amp"]] = [0.5, 0.3]
    marg.loc[2, "spread_fsp"] = 0.2
    marg["n_eff"] = [20.0, 5.0, 40.0]
    comps = pd.DataFrame(dict(bulk=[0, 1, 2], subset=["train", "train", "val"]))
    magnitude = np.zeros((3, len(GROUPS)))
    numeric = np.zeros((3, len(GROUPS)), dtype=bool)
    for b, shares in enumerate(({"fsp": 0.5, "px": 0.5}, {"fsp": 0.2}, {"fsp": 0.5, "amp": 0.5})):
        for g, v in shares.items():
            magnitude[b, GROUPS.index(g)] = v
            numeric[b, GROUPS.index(g)] = True
    out = screen.score_scheme(marg, comps, magnitude, numeric)
    assert out["fsp_train"] == pytest.approx(0.15) and out["px_train"] == pytest.approx(0.1)
    assert np.isnan(out["amp_train"]) and out["silicate_train"] == pytest.approx(0.125)
    assert out["fsp_val"] == pytest.approx(0.0) and out["amp_val"] == pytest.approx(0.2)
    assert out["silicate_val"] == pytest.approx(0.1) and np.isnan(out["other_val"])
    assert out["spread_silicate_val"] == pytest.approx(0.1)
    assert out["n_rocks_train"] == 2 and out["n_empty_train"] == 0
    assert out["n_eff_median"] == 20.0 and out["frac_thin"] == pytest.approx(1 / 3)


def test_rosters_nest_and_prefix_the_hash_order(tmp_path):
    ids = list(range(1, 3001))
    screen.write_rosters(tmp_path, ids)
    ordered = hash_order(ids)
    prev = set()
    for n in screen.LADDER:
        got = pd.read_csv(tmp_path / f"n{n}.csv")["sample_id"].tolist()
        assert got == sorted(ordered[:n]) and prev < set(got)
        prev = set(got)
    screen.write_rosters(tmp_path, ids)
    (tmp_path / "n128.csv").write_text("sample_id\n1\n")
    with pytest.raises(SystemExit):
        screen.write_rosters(tmp_path, ids)


def _benchmark(tmp_path):
    samples = pd.DataFrame([
        ["src", "d1", "unit-a", 30.0, "1", "phenocryst_vol_visual",
         65.0, 0.6, 16.0, 4.0, np.nan, 0.1, 2.0, 4.0, 4.0, 3.0, 0.2, 1.0],
        ["src", "a2", "unit-b", 20.0, "1", "phenocryst_vol_visual",
         58.0, 0.9, 17.0, 7.0, np.nan, 0.15, 3.5, 6.5, 3.5, 1.5, 0.3, 1.5],
    ], columns=SAMPLE_COLS)
    modes = pd.DataFrame([
        ["src", "d1", "plagioclase", "fsp", 20.0],
        ["src", "d1", "augite", "cpx", 4.0],
        ["src", "d1", "quartz", "q", 3.0],
        ["src", "a2", "plagioclase", "fsp", 12.0],
        ["src", "a2", "hornblende", "amp", 5.0],
        ["src", "a2", "Fe-Ti oxides", "spl", 1.0],
    ], columns=["source", "sample_id", "phase", "magemin_phase", "abundance_pct"])
    samples.to_csv(tmp_path / "samples.csv", index=False)
    modes.to_csv(tmp_path / "modes.csv", index=False)
    return load_benchmark(tmp_path), samples


def _synthetic_run(n_draws=16):
    rng = np.random.default_rng(0)
    draws, points = [], []
    for bulk in (0, 1):
        p, w, q = draw_conditions(n_draws, rng)
        for d in range(n_draws):
            draws.append(dict(bulk=bulk, draw=d, P_kbar=p[d], H2O=w[d], q=q[d],
                              T_end_50=1000.0, T_end_30=900.0))
            for T, liq in ((1000.0, 0.45), (900.0, 0.25)):
                row = dict(bulk=bulk, draw=d, T_celsius=T, liq_frac=liq, fl_frac=0.0)
                row.update({c: 0.0 for c in run_read.MASS_COLS})
                row.update(m_fsp=(0.1 + 0.03 * p[d]) * (1 - liq), m_px=0.2 * (1 - liq),
                           m_amp=0.05 * w[d] * (1 - liq), m_q=0.05 * (1 - liq),
                           m_opq=0.02 * (1 - liq))
                points.append(row)
    return pd.DataFrame(points), pd.DataFrame(draws)


def test_screen_end_to_end_on_a_synthetic_run(tmp_path):
    benchmark, samples = _benchmark(tmp_path)
    comps = composition_table(benchmark, samples)
    comps["subset"] = ["train", "val"]
    samples["key"] = samples["source"] + "|" + samples["sample_id"]
    mask = benchmark.clean & benchmark.has_magnitude
    magnitude, numeric = benchmark.magnitude[mask], benchmark.numeric[mask]
    raw_points, draws = _synthetic_run()
    points = run_read.shares_and_distances(raw_points, magnitude, numeric)

    table = screen.screen(points, draws, comps, magnitude, numeric, samples, schemes=GRID[:2])
    assert table["name"].tolist()[:2] == ["broad", "traced"]
    assert set(table["name"][2:]) == {GRID[0].name, GRID[1].name}
    assert table["silicate_train"].iloc[2:].is_monotonic_increasing
    by = table.set_index("name")
    assert by.loc["broad", "n_eff_median"] == 16.0 and by.loc["broad", "frac_thin"] == 0.0
    assert (table["n_eff_median"] <= 16.0).all() and (table["n_eff_median"] > 0.0).all()
    assert np.isfinite(by.loc["traced", "silicate_train"])
    assert by.loc["broad", "n_rocks_train"] == 1 and by.loc["broad", "n_rocks_val"] == 1
    assert np.isnan(by.loc["broad", "amp_train"]) and np.isfinite(by.loc["broad", "amp_val"])
    assert by.loc[GRID[0].name, "area_frac"] == pytest.approx(area_fraction(GRID[0]))

    distances = pd.DataFrame(dict(subset=["train", "val"], category=["fsp", "fsp"],
                                  d_marginal=[by.loc["broad", "fsp_train"], by.loc["broad", "fsp_val"]]))
    assert screen.reproduction_check(table, distances) == pytest.approx(0.0)

    summary = screen.summarize(table, {"n_rocks": 2}, 0.0, top=1)
    assert summary["top"] == [table["name"].iloc[2]] and len(summary["ranking"]) == 2
    json.dumps(summary)
    screen.draw_figures(table, tmp_path / "res")
    for name in ("prior_ranking.png", "prior_grid.png"):
        assert (tmp_path / "res" / name).stat().st_size > 0


def test_hits_use_tau_then_fall_back_to_the_best_rows():
    rungs = _rungs()
    rungs["e_sil"] = [0.01, 0.02, 0.2, 0.3, 0.04, 0.5, 0.6, 0.7]
    draws = pd.DataFrame(dict(bulk=0, draw=range(4), P_kbar=[2.0, 5.0, 8.0, 11.0], H2O=[1.0, 2.0, 3.0, 4.0]))
    hits, summary = predict.hit_rows(rungs, draws, tau=0.05, fallback=2)
    assert len(hits) == 3 and not hits["fallback"].any()
    assert summary.loc[0, "n_hits"] == 3 and summary.loc[0, "best_e"] == pytest.approx(0.01)
    assert summary.loc[0, "hit_p"] == pytest.approx((2.0 + 2.0 + 8.0) / 3)
    weights = predict.hit_weights(draws, hits)
    assert weights["w"].tolist() == [1.0, 0.0, 0.5, 0.0]
    hits, summary = predict.hit_rows(rungs, draws, tau=0.001, fallback=2)
    assert len(hits) == 2 and hits["fallback"].all() and summary.loc[0, "fallback"]
    assert sorted(hits["e_sil"]) == [0.01, 0.02]


def test_window_weights_center_a_rectangle_on_each_rock():
    draws = pd.DataFrame(dict(bulk=[0, 0, 1, 1], draw=[0, 1, 0, 1],
                              P_kbar=[8.0, 2.0, 8.0, 2.0], H2O=[4.0, 4.0, 4.0, 4.0], q=0.01))
    centers = pd.DataFrame(dict(hit_p=[8.0, 2.0], hit_w=[4.0, 4.0]), index=[0, 1])
    w = predict.window_weights(draws, centers, 2.0, 1.5)["w"].to_numpy()
    assert w[0] > 0 and w[1] == 0 and w[2] == 0 and w[3] > 0
    centers.loc[1] = np.nan
    w = predict.window_weights(draws, centers, 2.0, 1.5)["w"].to_numpy()
    assert w[0] > 0 and (w[1:] == 0).all()


def test_predictability_read_end_to_end_on_a_synthetic_run(tmp_path):
    benchmark, samples = _benchmark(tmp_path)
    comps = composition_table(benchmark, samples)
    comps["subset"] = ["train", "val"]
    mask = benchmark.clean & benchmark.has_magnitude
    magnitude, numeric = benchmark.magnitude[mask], benchmark.numeric[mask]
    raw_points, draws = _synthetic_run()
    points = run_read.shares_and_distances(raw_points, magnitude, numeric)
    table, summary, preds, facts = predict.read(points, draws, comps, magnitude, numeric, tau=0.05, fallback=4)
    assert facts["n_rocks"] == 2 and set(preds) == {"train_mean", "class_mean", "ridge", "forest"}
    assert table["name"].tolist()[:3] == ["hit rows", "own hit center, narrow", "own hit center, wide"]
    assert (table["kind"].iloc[:3] == "ceiling").all() and (table["kind"].iloc[3:] == "predicted").all()
    wide = table["name"].str.endswith("wide") | (table["name"] == "hit rows")
    assert np.isfinite(table.loc[wide, "silicate_val"]).all()
    assert table.loc[0, "silicate_train"] <= table.loc[2, "silicate_train"] + 1e-9
    assert (table["n_empty_val"] <= 1).all()
    assert {"r2_hit_p_val", "r2_hit_w_train"} <= set(table.columns)
    predict.draw_figure(table, summary, preds, comps, {"broad": 0.2}, tmp_path)
    assert (tmp_path / "predictability.png").stat().st_size > 0

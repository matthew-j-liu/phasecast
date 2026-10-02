"""Tripwire for `explorations/generator_reach`: the benchmark composition
table and its TAS iron classes, the broad-box expansion, the whole-ladder
parser, the chunked runner against a fake Julia, the three-distance read,
and the signal read's pure pieces (renormalization, the line fit, the
bootstrap table, the ruler disagreement).
The real generator needs Julia and MAGEMin, so it is exercised by the smoke
run, not here."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cnmineral.data.condition_proposal import q_density, w_max
from cnmineral.data.fe_redox import MIDDLEMOST_1989
from cnmineral.data.phase_vocab import GROUPS
from cnmineral.eval.real_rock import load_benchmark
from explorations.generator_reach import run_generator, run_read
from explorations.generator_reach.compositions import (
    TAS_FIELDS,
    TAS_TO_CLASS,
    composition_table,
    rock_class,
    tas_field,
)

LADDER = [1100.0, 1080.0]

SAMPLE_COLS = ["source", "sample_id", "unit", "total_crystals_pct", "alteration_extent",
               "mode_basis", "sio2", "tio2", "al2o3", "feo_tot", "fe2o3_tot", "mno", "mgo",
               "cao", "na2o", "k2o", "p2o5", "loi"]


def _benchmark(tmp_path):
    samples = pd.DataFrame([
        ["src", "d1", "unit-a", 30.0, "1", "phenocryst_vol_visual",
         65.0, 0.6, 16.0, 4.0, np.nan, 0.1, 2.0, 4.0, 4.0, 3.0, 0.2, 1.0],
        ["src", "r2", "unit-b", 10.0, "fresh", "phenocryst_vol_visual",
         75.0, 0.2, 13.0, np.nan, 2.0, 0.05, 0.3, 1.0, 4.0, 4.0, 0.05, 0.5],
        ["src", "a3", "unit-c", 20.0, "3", "phenocryst_vol_visual",
         55.0, 1.0, 17.0, 8.0, np.nan, 0.15, 4.0, 7.0, 3.5, 1.5, 0.3, 1.5],
        ["src", "t4", "unit-d", 5.0, "1", "phenocryst_vol_visual",
         60.0, 0.8, 16.0, 6.0, np.nan, 0.1, 3.0, 6.0, 3.5, 2.0, 0.2, 1.0],
    ], columns=SAMPLE_COLS)
    modes = pd.DataFrame([
        ["src", "d1", "plagioclase", "fsp", 20.0],
        ["src", "d1", "augite", "cpx", 4.0],
        ["src", "d1", "quartz", "q", 3.0],
        ["src", "d1", "hornblende", "amp", np.nan],
        ["src", "r2", "quartz", "q", 10.0],
        ["src", "a3", "plagioclase", "fsp", 10.0],
        ["src", "t4", "plagioclase", "fsp", np.nan],
    ], columns=["source", "sample_id", "phase", "magemin_phase", "abundance_pct"])
    samples.to_csv(tmp_path / "samples.csv", index=False)
    modes.to_csv(tmp_path / "modes.csv", index=False)
    return load_benchmark(tmp_path), samples


def _comps(tmp_path):
    benchmark, samples = _benchmark(tmp_path)
    return composition_table(benchmark, samples), benchmark


def _rec(sid, liq, extra=None, status=0, G=-1000.0):
    phases = [dict(name="liq", mode_wt=liq)] if liq else []
    phases += extra or [dict(name="fsp", mode_wt=1.0 - liq)]
    return dict(sid=sid, P=5.0, T=1000.0, status=status, G=G, phases=phases, ss=[])


def _write(tmp_path, recs):
    path = tmp_path / "out.jsonl"
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return path


def _meta(n, bulk=1, draw=0):
    ts = [1100.0 - 20 * i for i in range(n)]
    return pd.DataFrame(dict(sid=range(1, n + 1), bulk=bulk, draw=draw, T_celsius=ts))


def _fake_julia(calls):
    # melt falls with temperature; the hotter step sits above rung 50, the
    # cooler one between the rungs, so rung 30 is attrition on every draw
    liq_of_T = {1100.0: 0.9, 1080.0: 0.4}

    def run(input_csv, output_jsonl, database, timeout=None):
        calls.append(Path(input_csv).name)
        df = pd.read_csv(input_csv)
        with open(output_jsonl, "w") as fh:
            for sid, T in zip(df.sample_id, df.T_celsius):
                liq = liq_of_T[T]
                solid = 1.0 - liq
                phases = [dict(name="liq", mode_wt=liq),
                          dict(name="fsp", mode_wt=0.6 * solid),
                          dict(name="cpx", mode_wt=0.2 * solid),
                          dict(name="q", mode_wt=0.15 * solid),
                          dict(name="spl", mode_wt=0.05 * solid)]
                fh.write(json.dumps(dict(sid=int(sid), P=1.0, T=T, status=0, G=-1000.0,
                                         phases=phases, ss=[])) + "\n")
        return SimpleNamespace(returncode=0, stderr="")
    return run


@pytest.fixture
def short_ladder(monkeypatch):
    monkeypatch.setattr(run_generator, "T_LADDER", LADDER)
    monkeypatch.setattr(run_generator.gmc, "T_LADDER", LADDER)


def _args(**kw):
    base = dict(draws=2, seed=0, bulks_per_call=25, database="ig", keep_raw=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_tas_fields_land_known_points_on_iron_ratio_keys():
    for vertices in TAS_FIELDS.values():
        v = np.asarray(vertices, dtype=float)
        area = 0.5 * np.dot(v[:, 0], np.roll(v[:, 1], -1)) - 0.5 * np.dot(v[:, 1], np.roll(v[:, 0], -1))
        assert abs(area) > 1.0
    points = {"picrobasalt": (43, 1.5), "basalt": (48, 3), "basaltic andesite": (54, 3),
              "andesite": (60, 3), "dacite": (66, 6), "rhyolite": (75, 8),
              "trachybasalt": (47, 5.8), "basaltic trachyandesite": (54, 7.5),
              "trachyandesite": (58, 9), "trachyte": (60, 10), "trachydacite": (66, 10),
              "tephrite": (43, 5), "phonotephrite": (48, 9.5), "tephriphonolite": (52, 12),
              "phonolite": (57, 15), "foidite": (45, 12)}
    assert set(points) == set(TAS_FIELDS)
    names, inside = tas_field([p[0] for p in points.values()], [p[1] for p in points.values()])
    assert list(names) == list(points) and inside.all()
    for name in TAS_FIELDS:
        assert rock_class(name) in MIDDLEMOST_1989
    assert rock_class("basaltic trachyandesite") == TAS_TO_CLASS["basaltic trachyandesite"]
    with pytest.raises(KeyError):
        rock_class("komatiite basalt")


def test_tas_outside_every_field_takes_the_nearest_and_flags_it():
    names, inside = tas_field([92.0, 52.0], [2.0, 3.0])
    assert list(names) == ["rhyolite", "basalt"] or list(names) == ["rhyolite", "basaltic andesite"]
    assert list(inside) == [False, True]


def test_composition_table_is_the_evaluated_rocks_in_benchmark_order(tmp_path):
    comps, benchmark = _comps(tmp_path)
    assert list(comps["bulk"]) == [0, 1]
    assert list(comps["key"]) == ["src|d1", "src|r2"]
    assert list(comps["fe_provenance"]) == ["assumed_class", "assumed_class"]
    assert list(comps["tas_field"]) == ["dacite", "rhyolite"]
    assert list(comps["fe3_ratio"]) == [MIDDLEMOST_1989["dacite"], MIDDLEMOST_1989["rhyolite"]]
    row = comps.iloc[0]
    anhydrous = row[["sio2", "tio2", "al2o3", "feo_tot", "mno", "mgo", "cao", "na2o",
                     "k2o", "p2o5"]].sum()
    assert row["tas_sio2"] == pytest.approx(100 * row["sio2"] / anhydrous)
    assert list(comps["subset"]) == ["none", "none"]


def test_build_inputs_draws_the_broad_box_without_routing_columns(tmp_path):
    comps, _ = _comps(tmp_path)
    inputs, meta, draws = run_generator.build_inputs(comps, n_draws=6, seed=3)
    oxides = inputs.drop(columns=["sample_id", "T_celsius", "P_kbar"])
    assert np.allclose(oxides.sum(axis=1), 100.0)
    assert len(inputs) == 2 * 6 * 31 and len(meta) == len(inputs)
    assert draws["P_kbar"].between(0.5, 12.0).all()
    assert (draws["H2O"] <= w_max(draws["P_kbar"].to_numpy()) + 1e-9).all()
    assert np.allclose(draws["q"], q_density(draws["P_kbar"], draws["H2O"]))
    assert not {"proposal_tier", "proposal_rung", "zr_ppm"} & set(draws.columns)
    _, _, alone = run_generator.build_inputs(comps.iloc[[1]], n_draws=6, seed=3)
    pd.testing.assert_frame_equal(alone.reset_index(drop=True),
                                  draws[draws["bulk"] == 1].reset_index(drop=True))


def test_parse_chunk_keeps_the_whole_ladder_with_liq_frac(tmp_path):
    liqs = [0.8, 0.5, 0.25, 0.1]
    path = _write(tmp_path, [_rec(i + 1, l) for i, l in enumerate(liqs)])
    ladder, stats = run_generator.parse_chunk(path, _meta(4))
    assert sorted(ladder["T_celsius"].unique(), reverse=True) == [1100.0, 1080.0, 1060.0, 1040.0]
    assert ladder.groupby("T_celsius")["liq_frac"].first().sort_index(ascending=False).tolist() == liqs
    st = stats.iloc[0]
    assert st["T_end_50"] == 1060.0 and st["T_end_30"] == 1060.0
    assert not st["attrition_50"] and not st["attrition_30"]
    assert st["n_points"] == 4 and st["n_failed"] == 0


def test_parse_chunk_counts_failures_attrition_and_oversaturation(tmp_path):
    recs = [_rec(1, 0.9, extra=[dict(name="fl", mode_wt=0.02), dict(name="fsp", mode_wt=0.08)]),
            _rec(2, 0.7, status=-1), _rec(3, 0.6)]
    ladder, stats = run_generator.parse_chunk(_write(tmp_path, recs), _meta(3))
    st = stats.iloc[0]
    assert st["n_failed"] == 1 and st["n_points"] == 2
    assert 1080.0 not in set(ladder["T_celsius"])
    assert st["oversaturated"]
    assert st["attrition_50"] and st["attrition_30"]
    assert np.isnan(st["T_end_50"]) and np.isnan(st["T_end_30"])


def test_run_chunk_writes_both_tables_and_the_manifest(tmp_path, monkeypatch, short_ladder):
    comps, _ = _comps(tmp_path)
    calls = []
    monkeypatch.setattr(run_generator.gmc, "run_julia", _fake_julia(calls))
    out = tmp_path / "out"
    out.mkdir()
    rec = run_generator.run_chunk(0, comps, _args(), out)
    draws = pd.read_parquet(out / "chunk_0000_draws.parquet")
    ladder = pd.read_parquet(out / "chunk_0000_ladder.parquet")
    assert calls == ["in_0.csv"]
    assert len(draws) == 4 and {"P_kbar", "H2O", "q", "in_sio2", "n_failed", "T_end_50",
                                "T_end_30", "attrition_50", "attrition_30",
                                "oversaturated"} <= set(draws.columns)
    assert (draws["T_end_50"] == 1080.0).all() and draws["attrition_30"].all()
    assert len(ladder) == 2 * 2 * 2 * 5 and list(ladder.columns) == run_generator.LADDER_COLS
    assert rec["bulk_range"] == [0, 1] and rec["n_points_run"] == 8 and "wall_s" in rec
    assert run_generator.gmc.read_manifest(out)[0]["n_ladder_rows"] == len(ladder)


def test_main_resumes_and_rejects_a_changed_draw_count(tmp_path, monkeypatch, short_ladder):
    comps, _ = _comps(tmp_path)
    calls = []
    monkeypatch.setattr(run_generator.gmc, "run_julia", _fake_julia(calls))
    monkeypatch.setattr(run_generator, "compositions", lambda: comps)
    out = tmp_path / "out"
    argv = ["run_generator.py", "--out", str(out), "--draws", "2", "--chunk-size", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    run_generator.main()
    assert calls == ["in_0.csv", "in_0.csv"]
    stray = out / "chunk_0007_draws.parquet"
    stray.write_bytes(b"")
    run_generator.main()
    assert calls == ["in_0.csv", "in_0.csv"] and not stray.exists()
    monkeypatch.setattr(sys, "argv", argv[:-4] + ["--draws", "3", "--chunk-size", "1"])
    with pytest.raises(SystemExit):
        run_generator.main()


def _points(rows):
    cols = ["bulk", "draw", "T_celsius", "liq_frac", "fl_frac", *run_read.MASS_COLS]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows]).fillna(0.0)


def _truth(numeric_shares):
    magnitude = np.zeros((1, len(GROUPS)))
    numeric = np.zeros((1, len(GROUPS)), dtype=bool)
    for g, v in numeric_shares.items():
        magnitude[0, GROUPS.index(g)] = v
        numeric[0, GROUPS.index(g)] = True
    return magnitude, numeric


def test_shares_renormalize_over_quantified_categories():
    magnitude, numeric = _truth({"fsp": 0.7, "px": 0.2, "q": 0.1})
    pts = _points([
        [0, 0, 1000.0, 0.4, 0.0, 0.3, 0.1, 0, 0, 0, 0, 0, 0.2],
        [0, 0, 900.0, 0.8, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.2],
    ])
    out = run_read.shares_and_distances(pts, magnitude, numeric)
    assert out.loc[0, ["s_fsp", "s_px", "s_q"]].tolist() == pytest.approx([0.75, 0.25, 0.0])
    assert out.loc[0, ["d_fsp", "d_px", "d_q"]].tolist() == pytest.approx([0.05, 0.05, 0.1])
    assert np.isnan(out.loc[0, "d_amp"]) and out.loc[0, "s_amp"] == 0.0
    assert out.loc[1, ["s_fsp", "s_px", "s_q"]].tolist() == [0.0, 0.0, 0.0]
    assert list(out["zero_mass"]) == [False, True]


def test_marginal_best_draw_and_best_step_on_a_synthetic_ladder():
    magnitude, numeric = _truth({"fsp": 0.9, "opq": 0.1})
    pts = _points([
        [0, 0, 1100.0, 0.9, 0.0, 0.09, 0, 0, 0, 0, 0.01, 0, 0],
        [0, 0, 1000.0, 0.45, 0.0, 0.5, 0, 0, 0, 0, 0.05, 0, 0],
        [0, 0, 900.0, 0.25, 0.0, 0.6, 0, 0, 0, 0, 0.15, 0, 0],
        [0, 1, 1100.0, 0.4, 0.0, 0.3, 0, 0, 0, 0, 0.3, 0, 0],
        [0, 1, 1000.0, 0.35, 0.0, 0.4, 0, 0, 0, 0, 0.25, 0, 0],
    ])
    draws = pd.DataFrame(dict(bulk=[0, 0], draw=[0, 1], T_end_50=[1000.0, 1100.0],
                              T_end_30=[900.0, np.nan]))
    comps = pd.DataFrame(dict(bulk=[0], key=["k"], subset=["train"], total_crystals_pct=[20.0]))
    points = run_read.shares_and_distances(pts, magnitude, numeric)
    table = run_read.reach_table(points, draws, comps, magnitude, numeric).set_index("category")
    fsp = table.loc["fsp"]
    assert fsp["n_draws_50"] == 2 and fsp["n_draws_30"] == 1
    assert fsp["mean_rungs"] == pytest.approx((0.5 / 0.55 + 0.8 + 0.5) / 3)
    assert fsp["d_marginal"] == pytest.approx(abs(fsp["mean_rungs"] - 0.9))
    assert fsp["best_draw"] == 0 and fsp["best_rung"] == 50
    assert fsp["d_best_draw"] == pytest.approx(abs(0.5 / 0.55 - 0.9))
    assert fsp["best_T"] == 1100.0 and fsp["best_liq"] == 0.9
    assert fsp["d_best_step"] == pytest.approx(0.0) and fsp["d_best_any_step"] == pytest.approx(0.0)
    assert (table["d_best_step"] <= table["d_best_draw"] + 1e-12).all()
    assert (table["d_q10_draw"] >= table["d_best_draw"] - 1e-12).all()
    assert list(table["n_numeric"]) == [2, 2] and list(table["key"]) == ["k", "k"]


def test_bins_partition_at_tau_and_a_missing_marginal_is_a_miss():
    table = pd.DataFrame(dict(
        d_marginal=[0.01, 0.2, 0.2, 0.3, np.nan],
        d_best_draw=[0.01, 0.03, 0.1, 0.2, np.nan],
        d_best_step=[0.0, 0.01, 0.04, 0.1, np.nan]))
    assert list(run_read.bin_outcomes(table, 0.05)) == [
        "reproduced", "buried", "wrong_stage", "wrong_recipe", "wrong_recipe"]
    assert list(run_read.bin_outcomes(table, 0.02)) == [
        "reproduced", "wrong_stage", "wrong_recipe", "wrong_recipe", "wrong_recipe"]
    fractions = run_read.bin_fractions(table, 0.05)
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_read_end_to_end_on_a_fake_run(tmp_path, monkeypatch, short_ladder):
    comps, benchmark = _comps(tmp_path)
    monkeypatch.setattr(run_generator.gmc, "run_julia", _fake_julia([]))
    work = tmp_path / "work"
    work.mkdir()
    run_generator.run_chunk(0, comps, _args(), work)

    mask = benchmark.clean & benchmark.has_magnitude
    magnitude, numeric = benchmark.magnitude[mask], benchmark.numeric[mask]
    points = run_read.shares_and_distances(run_read.build_points(work), magnitude, numeric)
    table = run_read.reach_table(points, run_read.load_draws(work), comps, magnitude, numeric)
    assert len(table) == int(numeric.sum())
    assert set(table["category"]) == {"fsp", "px", "q"}
    assert table["d_marginal"].notna().all() and (table["d_best_step"] <= table["d_best_draw"]).all()
    assert (table["n_draws_30"] == 0).all()

    table["bin"] = run_read.bin_outcomes(table, 0.05)
    summary = run_read.summarize(table, 0.05, [0.02, 0.05, 0.1], {"fsp": 0.14},
                                 run_read.run_facts(work))
    assert set(summary["per_category"]) == set(GROUPS)
    assert summary["per_category"]["q"]["n_rocks"] == 2
    assert summary["per_category"]["q"]["n_rocks_multi"] == 1
    assert summary["per_category"]["amp"]["bins"]["0.05"]["buried"] is None
    assert summary["run"]["n_rocks"] == 2 and summary["run"]["attrition_30"] == 1.0
    json.dumps(summary)

    run_read.draw_figures(table, summary, tmp_path / "read")
    for name in ("reach_cdf.png", "reach_bins.png", "reach_melt.png"):
        assert (tmp_path / "read" / name).stat().st_size > 0


from cnmineral.data.phase_vocab import GROUP_INDEX
from explorations.generator_reach import run_signal


def test_renormalize_clips_keeps_the_mineral_list_and_a_missing_row():
    share = np.zeros((3, 7))
    share[0, :3] = [1.4, -0.2, 0.3]
    share[1] = 0.2
    share[2] = np.nan
    numeric = np.zeros((3, 7), dtype=bool)
    numeric[0, :3] = True
    numeric[2, 0] = True
    out = run_signal.renormalize(share, numeric)
    assert np.allclose(out[0], [1 / 1.3, 0.0, 0.3 / 1.3, 0, 0, 0, 0])
    assert (out[1] == 0).all()
    assert np.isnan(out[2]).all()


def test_linear_fit_is_the_constant_for_a_flat_source_and_exact_for_a_linear_one():
    rng = np.random.default_rng(0)
    n = 40
    x = rng.uniform(0.1, 0.9, n)
    magnitude = np.zeros((n, 7))
    magnitude[:, 0], magnitude[:, 1] = x, 1 - x
    numeric = np.zeros((n, 7), dtype=bool)
    numeric[:, :2] = True
    train = np.arange(n) < 30

    fitted, coef = run_signal.linear_fit(np.full((n, 7), 0.3), magnitude, numeric, train)
    const = run_signal.constant_vector(magnitude, numeric, train)
    assert np.allclose(fitted, run_signal.renormalize(np.tile(const, (n, 1)), numeric))
    assert coef["fsp"][1] == 0.0

    source = np.zeros((n, 7))
    source[:, 0], source[:, 1] = 0.5 * x + 0.1, 0.5 * (1 - x) + 0.1
    fitted, coef = run_signal.linear_fit(source, magnitude, numeric, train)
    assert np.allclose(fitted, magnitude)
    assert np.allclose(coef["fsp"], (-0.2, 2.0)) and np.allclose(coef["px"], (-0.2, 2.0))


def test_bootstrap_table_bands_contain_the_estimate():
    rng = np.random.default_rng(2)
    n = 60
    units = np.repeat(np.arange(15), 4)
    truth = rng.dirichlet(np.ones(7), n)
    numeric = np.ones((n, 7), dtype=bool)
    noisy = np.clip(truth + rng.normal(0, 0.05, truth.shape), 0, 1)
    shares = {"simulator": noisy, "constant": np.tile(truth.mean(0), (n, 1))}
    table = run_signal.bootstrap_table(shares, {"simulator": noisy}, truth, numeric, units, n_boot=200)
    assert len(table) == 2 * len(run_signal.LABELS)
    for metric in ("mae", "pearson_r", "mae_linear_fit", "mae_minus_constant"):
        rows = table[table[metric].notna()]
        assert len(rows) > 0
        assert (rows[f"{metric}_lo"] <= rows[metric] + 1e-12).all()
        assert (rows[metric] <= rows[f"{metric}_hi"] + 1e-12).all()
    sim = table[(table.source == "simulator") & (table.category == "fsp")].iloc[0]
    assert sim["n_rocks"] == n and sim["pearson_r"] > 0.8


def test_ruler_disagreement_on_a_synthetic_pair():
    fsp, bi = GROUP_INDEX["fsp"], GROUP_INDEX["bi"]
    visual, counted = np.zeros((3, 7)), np.zeros((3, 7))
    visual[:, fsp], counted[:, fsp] = [0.5, 0.6, 0.7], [0.7, 0.6, 0.8]
    visual[:, bi], counted[:, bi] = 1 - visual[:, fsp], 1 - counted[:, fsp]
    visual_num, counted_num = visual > 0, counted > 0
    counted_num[2, bi] = False
    out = run_signal.ruler_disagreement(visual, visual_num, counted, counted_num)
    assert out["fsp"]["n_rocks"] == 3
    assert np.isclose(out["fsp"]["ruler_disagreement"], 0.1)
    assert np.isclose(out["fsp"]["signed_mean"], -0.1)
    assert out["bi"]["n_rocks"] == 2 and np.isclose(out["bi"]["signed_mean"], 0.1)
    assert out["amp"]["n_rocks"] == 0 and out["amp"]["ruler_disagreement"] is None

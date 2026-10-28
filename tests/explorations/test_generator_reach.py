"""Tripwire for `explorations/generator_reach`: the benchmark composition
table and its TAS iron classes, the broad-box expansion, the whole-ladder
parser, the chunked runner against a fake Julia, the three-distance read,
and the signal read's pure pieces (renormalization, the line fit, the
bootstrap table, the ruler disagreement, and for the plutonic read the rank
metrics, the CIPW norm and the packaging of a share source for the
recalibrator); the fractional runner's inputs, parser and chunk writer
against a fake Julia, the table-aware points builder, the ladder read's
pure pieces on a synthetic path, and the recipe read's rung ends, matched
draws, read set, bootstrap, norm input and benchmark alignment.
The real generators need Julia and MAGEMin, so they are exercised by the smoke
runs, not here."""

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
from explorations.generator_reach import run_fractional, run_generator, run_ladder, run_read
from explorations.generator_reach.compositions import (
    TAS_FIELDS,
    TAS_TO_CLASS,
    composition_table,
    plutonic_class,
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
    base = dict(draws=2, seed=0, bulks_per_call=25, database="ig", keep_raw=False,
                benchmark="eruption")
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
    ladder, stats = run_generator.parse_chunk(path, _meta(4), run_generator.RUNGS["eruption"])
    assert sorted(ladder["T_celsius"].unique(), reverse=True) == [1100.0, 1080.0, 1060.0, 1040.0]
    assert ladder.groupby("T_celsius")["liq_frac"].first().sort_index(ascending=False).tolist() == liqs
    st = stats.iloc[0]
    assert st["T_end_50"] == 1060.0 and st["T_end_30"] == 1060.0
    assert not st["attrition_50"] and not st["attrition_30"]
    assert st["n_points"] == 4 and st["n_failed"] == 0
    assert st["T_last"] == 1040.0 and st["liq_last"] == 0.1


def test_parse_chunk_counts_failures_attrition_and_oversaturation(tmp_path):
    recs = [_rec(1, 0.9, extra=[dict(name="fl", mode_wt=0.02), dict(name="fsp", mode_wt=0.08)]),
            _rec(2, 0.7, status=-1), _rec(3, 0.6)]
    ladder, stats = run_generator.parse_chunk(_write(tmp_path, recs), _meta(3),
                                              run_generator.RUNGS["eruption"])
    st = stats.iloc[0]
    assert st["n_failed"] == 1 and st["n_points"] == 2
    assert 1080.0 not in set(ladder["T_celsius"])
    assert st["oversaturated"]
    assert st["attrition_50"] and st["attrition_30"]
    assert np.isnan(st["T_end_50"]) and np.isnan(st["T_end_30"])
    assert st["T_last"] == 1060.0 and st["liq_last"] == 0.6


def test_parse_chunk_reads_the_plutonic_rungs_near_the_solidus(tmp_path):
    path = _write(tmp_path, [_rec(i + 1, l) for i, l in enumerate([0.5, 0.08, 0.0])])
    _, stats = run_generator.parse_chunk(path, _meta(3), run_generator.RUNGS["plutonic"])
    st = stats.iloc[0]
    assert st["T_end_10"] == 1080.0 and st["T_end_5"] == 1060.0 and st["T_last"] == 1060.0
    assert "T_end_50" not in stats.columns


def test_plutonic_class_maps_reported_names_onto_iron_ratio_keys():
    assert plutonic_class("Monzogranite") == "monzogranite"
    assert plutonic_class("Alkali-feldspar granite") == "alkali feldspar granite"
    assert plutonic_class("aplite") == plutonic_class("Alaskite") == "granite"
    assert plutonic_class("alkali-feldspar quartz syenite") == "quartz syenite"
    with pytest.raises(KeyError):
        plutonic_class("charnockite")


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


def test_rank_metrics_match_scipy_and_sklearn_over_the_quantified_rows():
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    n = 50
    share, magnitude = rng.uniform(size=(n, 7)), rng.uniform(size=(n, 7))
    magnitude[:, 0] = share[:, 0] + rng.normal(0, 0.1, n)
    numeric = rng.uniform(size=(n, 7)) > 0.2
    numeric[:, GROUP_INDEX["ol"]] = False
    rho = run_signal.spearman(share, magnitude, numeric)
    for j in range(7):
        m = numeric[:, j]
        expected = spearmanr(share[m, j], magnitude[m, j]).statistic if m.sum() >= 3 else np.nan
        assert np.allclose(rho[j], expected, equal_nan=True)
    assert np.isnan(rho[GROUP_INDEX["ol"]])

    presence = (rng.uniform(size=(n, 7)) > 0.4).astype(float)
    presence[:, GROUP_INDEX["q"]] = 1.0
    measured = np.ones((n, 7), dtype=bool)
    measured[:10, GROUP_INDEX["opq"]] = False
    auroc = run_signal.presence_auroc(share, presence, measured)
    for j in range(7):
        if j != GROUP_INDEX["q"]:
            m = measured[:, j]
            assert np.isclose(auroc[j], roc_auc_score(presence[m, j], share[m, j]))
    assert np.isnan(auroc[GROUP_INDEX["q"]])


def test_cipw_shares_close_on_the_seven_groups_with_no_hydrous_minerals():
    samples = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "real_mineralogy"
                          / "plutonic" / "samples.csv").dropna(subset=["sio2"])
    rows = pd.concat([samples.nlargest(3, "sio2"), samples.nsmallest(2, "sio2")])
    shares = run_signal.cipw_shares(rows)
    assert np.allclose(shares.sum(1), 1.0)
    assert (shares[:, GROUP_INDEX["amp"]] == 0).all() and (shares[:, GROUP_INDEX["bi"]] == 0).all()
    granite = shares[:3]
    assert (granite[:, GROUP_INDEX["q"]] + granite[:, GROUP_INDEX["fsp"]] > 0.9).all()
    assert (granite[:, GROUP_INDEX["q"]] > 0.2).all()
    mafic = shares[3:]
    assert (mafic[:, GROUP_INDEX["px"]] + mafic[:, GROUP_INDEX["ol"]]
            > granite[:, GROUP_INDEX["px"]].max()).all()


def test_packaged_source_reproduces_its_floored_shares_at_the_identity_correction():
    import torch

    from cnmineral.eval.mode_head import conditional_mean
    from cnmineral.models.real_rock_recalibrator import RealRockRecalibrator
    from cnmineral.training.real_rock_recalibrator import GroupPredictions

    rng = np.random.default_rng(4)
    n = 6
    share = rng.dirichlet(np.ones(7), n)
    share[:, GROUP_INDEX["amp"]] = 0.0
    numeric = rng.uniform(size=(n, 7)) > 0.3
    numeric[:, GROUP_INDEX["amp"]] = True
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    gp = GroupPredictions(logits7=t(np.zeros((n, 7))), alpha7=t(np.ones((n, 7))), sio2=t(np.zeros(n)),
                          presence=t(numeric), magnitude=t(share), numeric=t(numeric),
                          unit=np.arange(n))
    out = run_signal.packaged(gp, share, (share > 0).astype(float), kappa=40.0)
    floored = np.maximum(share, run_signal.PACKAGE_FLOOR)
    expected = conditional_mean(floored, numeric.astype(float))
    assert np.allclose(conditional_mean(out.alpha7.numpy(), numeric.astype(float)), expected)
    assert np.isclose(out.alpha7.numpy().sum(1), 40.0).all()
    assert (out.alpha7.numpy()[:, GROUP_INDEX["amp"]] > 0).all()
    p = torch.sigmoid(out.logits7).numpy()
    assert np.allclose(p, np.where(share > 0, 1 - run_signal.PACKAGE_FLOOR, run_signal.PACKAGE_FLOOR))
    assert out.magnitude is gp.magnitude and out.log_w is None
    with torch.no_grad():
        _, alpha, _ = RealRockRecalibrator("matrix").to(torch.float64)(out.logits7, out.alpha7)
    assert torch.allclose(alpha, out.alpha7)


def _two_source_data(n=24, seed=5):
    import torch

    from cnmineral.training.real_rock_recalibrator import GroupPredictions

    rng = np.random.default_rng(seed)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    numeric = rng.uniform(size=(n, 7)) > 0.3
    numeric[:, GROUP_INDEX["fsp"]] = True
    magnitude = np.where(numeric, rng.uniform(0.1, 1.0, (n, 7)), 0.0)
    magnitude /= magnitude.sum(1, keepdims=True)
    gp = GroupPredictions(logits7=t(rng.normal(size=(n, 7))), alpha7=t(rng.uniform(0.5, 5, (n, 7))),
                          sio2=t(np.zeros(n)), presence=t(numeric), magnitude=t(magnitude),
                          numeric=t(numeric), unit=np.arange(n) // 3)
    other = run_signal.packaged(gp, rng.dirichlet(np.ones(7), n), rng.uniform(size=(n, 7)), 30.0)
    return gp, other


def test_two_source_recalibrator_with_zero_b_terms_is_its_matrix_base():
    import torch

    gp, other = _two_source_data()
    model = run_signal.TwoSourceRecalibrator().double()
    with torch.no_grad():
        for p in model.base.parameters():
            p.add_(0.1 * torch.randn_like(p))
        logits, alpha = model(gp.logits7, gp.alpha7, other.logits7, other.alpha7)
        base_logits, base_alpha, _ = model.base(gp.logits7, gp.alpha7)
    assert torch.equal(logits, base_logits) and torch.equal(alpha, base_alpha)


def test_fit_two_source_moves_the_b_terms_and_stays_finite():
    import torch

    from cnmineral.training.real_rock_recalibrator import recalibrator_nll

    gp, other = _two_source_data()
    model = run_signal.fit_two_source(gp, other, epochs=50)
    with torch.no_grad():
        logits, alpha = model(gp.logits7, gp.alpha7, other.logits7, other.alpha7)
        nll = recalibrator_nll(logits, alpha, gp.presence, gp.magnitude, gp.numeric)["joint"]
    assert torch.isfinite(nll).all()
    assert model.magnitude_mix_b.abs().sum() > 0


def test_generator_reads_package_draw_fractions_and_fall_back_to_the_last_step(
        tmp_path, monkeypatch, short_ladder):
    comps, benchmark = _comps(tmp_path)
    monkeypatch.setattr(run_generator.gmc, "run_julia", _fake_julia([]))
    work = tmp_path / "work"
    work.mkdir()
    # the fake ladder's melt is 0.9 then 0.4, so every draw reaches 50% at its
    # last step and none reaches 30, 10 or 5%
    run_generator.run_chunk(0, comps, _args(benchmark="plutonic"), work)
    evaluated = benchmark.subset(benchmark.clean & benchmark.has_magnitude)
    reads, rows, attrition = run_signal.generator_reads(work, comps, evaluated)
    assert set(reads) == set(run_signal.STAGE_A_RUNGS)
    assert attrition["50"]["draws_unreached"] == 0
    for rung in ("30", "10", "5"):
        assert attrition[rung]["draws_unreached"] == attrition[rung]["draws"] == 4
    # every draw ends at 1080 C with fsp 0.6, cpx 0.2, q 0.15, spl 0.05 of the crystals
    expected = np.zeros(7)
    expected[[GROUP_INDEX["fsp"], GROUP_INDEX["px"], GROUP_INDEX["q"], GROUP_INDEX["opq"]]] = \
        [0.6, 0.2, 0.15, 0.05]
    for read in reads.values():
        assert np.allclose(read["share"], expected)
        assert np.array_equal(read["presence"], np.tile(expected > 0, (len(evaluated), 1)).astype(float))
        quantified = evaluated.numeric
        assert np.allclose(read["zero_shot"].sum(1), 1.0) and (read["zero_shot"][~quantified] == 0).all()
    assert set(rows["T_celsius"]) == {1080.0}


def test_marginal_shares_equal_the_conditional_mean_on_a_deterministic_support():
    import torch
    from cnmineral.data.phase_vocab import SOLID_PHASES
    from cnmineral.eval.mode_head import conditional_mean
    from cnmineral.eval.real_rock import coarsen_alpha
    from cnmineral.models.mode_head import ModeHead

    torch.manual_seed(0)
    n, head = 12, ModeHead(d_model=16).eval()
    latents = torch.randn(n, 16)
    # presence certain in both directions, so every sampled support is the same one
    support = (torch.rand(n, len(SOLID_PHASES)) > 0.5).double()
    support[:, 0] = 1.0
    head.presence_logits = lambda h: (support * 40.0 - 20.0).to(h.dtype)
    numeric = np.ones((n, 7), dtype=bool)
    numeric[:, GROUP_INDEX["ol"]] = False
    marginal = run_signal.marginal_shares(head, latents, numeric.astype(float), n_samples=4)
    with torch.no_grad():
        alpha = head.magnitude_alpha(latents, support.to(latents.dtype)).numpy()
    expected = conditional_mean(coarsen_alpha(alpha, support.numpy())[0], numeric.astype(float))
    assert np.allclose(marginal, expected, atol=1e-6)
    assert np.allclose(marginal.sum(1), 1.0) and (marginal[:, GROUP_INDEX["ol"]] == 0).all()


def test_cross_fit_reads_every_rock_once_out_of_fold_with_intrusions_whole(monkeypatch):
    n = 40
    units = np.array([f"u{i // 4}" for i in range(n)])
    rows = np.ones(n, dtype=bool)
    rows[-4:] = False
    calls = []

    def fake_fit(inputs, fit_rows):
        calls.append(fit_rows.copy())
        mark = np.tile(fit_rows[:, None].astype(float), (1, 7))
        return {"x": mark}, {"x": mark}, {"x": fit_rows.astype(float)}, 1.0

    monkeypatch.setattr(run_signal, "fit_sources", fake_fit)
    shares, presence, nll, folds = run_signal.cross_fit(None, rows, units, k=3)
    assert len(calls) == 3 and sum(f["rocks"] for f in folds) == rows.sum()
    assert (shares["x"] == 0).all() and (nll["x"] == 0).all()
    for fit_rows in calls:
        assert not fit_rows[~rows].any()
        read = rows & ~fit_rows
        assert not set(units[read]) & set(units[fit_rows])


def _fc_rec(sid, path_id, mass, phases, status=0, G=-900.0, end=None):
    for p in phases:
        p.setdefault("comp_wt", {"SiO2": 0.7, "FeO": 0.03, "O": 0.002, "H2O": 0.04})
    return dict(sid=sid, P=5.0, T=1000.0, path_id=path_id, mass=mass, status=status, G=G,
                end=end, phases=phases, ss=[])


def _fake_fc_julia(calls):
    # melt falls with temperature; the second step holds the melt below rung
    # 50 and the path ends there by melt-out, so the third ladder row is never
    # run
    liq_of_T = {1100.0: 0.6, 1080.0: 0.0}

    def run(input_csv, output_jsonl, database, melt_floor, timeout=None):
        calls.append(Path(input_csv).name)
        df = pd.read_csv(input_csv)
        with open(output_jsonl, "w") as fh:
            for path_id, g in df.groupby("path_id"):
                mass = 1.0
                for sid, T in zip(g.sample_id, g.T_celsius):
                    liq = liq_of_T[T]
                    solid = 1.0 - liq
                    phases = ([dict(name="liq", mode_wt=liq)] if liq else []) + [
                        dict(name="fsp", mode_wt=0.6 * solid), dict(name="q", mode_wt=0.4 * solid)]
                    fh.write(json.dumps(_fc_rec(int(sid), int(path_id), mass, phases,
                                                end=None if liq else "melt_out")) + "\n")
                    mass *= liq
                    if not liq:
                        break
        return SimpleNamespace(returncode=0, stderr="")
    return run


def test_fractional_inputs_carry_a_path_id_per_draw(tmp_path, short_ladder):
    comps, _ = _comps(tmp_path)
    inp, meta, draws = run_fractional.build_inputs(comps, n_draws=3, seed=0)
    base, _, _ = run_generator.build_inputs(comps, n_draws=3, seed=0)
    pd.testing.assert_frame_equal(inp.drop(columns="path_id"), base)
    assert (inp["path_id"] == meta["bulk"] * 3 + meta["draw"]).all()
    assert inp.groupby("path_id")["sample_id"].count().eq(len(LADDER)).all()
    assert inp["path_id"].nunique() == len(draws) == 3 * len(comps)


def test_fractional_parse_chunk_masses_on_the_starting_basis_and_the_melt_out_end(tmp_path):
    # step 1: half crystallizes; step 2 fails; step 3 runs on the remaining
    # half and freezes it; step 4 is never run
    recs = [
        _fc_rec(1, 7, 1.0, [dict(name="liq", mode_wt=0.5), dict(name="fsp", mode_wt=0.5)]),
        _fc_rec(2, 7, 0.5, [], status=-1, G=None),
        _fc_rec(3, 7, 0.5, [dict(name="fsp", mode_wt=0.6), dict(name="q", mode_wt=0.4)], end="melt_out"),
    ]
    step, pile, melt, stats = run_fractional.parse_chunk(_write(tmp_path, recs), _meta(4), run_fractional.RUNGS)
    assert list(step.columns) == run_generator.LADDER_COLS and list(pile.columns) == run_generator.LADDER_COLS
    s1, s3 = step[step.T_celsius == 1100.0], step[step.T_celsius == 1060.0]
    assert dict(zip(s1.phase, s1.mode_wt)) == {"liq": 0.5, "fsp": 0.5} and s1.liq_frac.eq(0.5).all()
    assert dict(zip(s3.phase, s3.mode_wt)) == pytest.approx({"fsp": 0.3, "q": 0.2}) and s3.liq_frac.eq(0.0).all()
    p3 = pile[pile.T_celsius == 1060.0]
    assert dict(zip(p3.phase, p3.mode_wt)) == pytest.approx({"fsp": 0.8, "q": 0.2, "liq": 0.0})
    assert 1080.0 not in set(step.T_celsius) and 1040.0 not in set(step.T_celsius)
    assert list(melt.columns) == run_fractional.MELT_COLS and list(melt.T_celsius) == [1100.0, 1060.0]
    assert list(melt["end"]) == [None, "melt_out"] and list(melt["mass"]) == [1.0, 0.5]
    st = stats.iloc[0]
    assert st.n_failed == 1 and st.n_steps_run == 3 and st.n_points == 2
    assert st["end"] == "melt_out" and st.T_liquid_out == 1060.0 and st.T_last == 1060.0
    assert st.T_end_50 == 1060.0 and st.T_end_5 == 1060.0 and not st.attrition_5


def test_fractional_melt_chemistry_reads_ferric_iron_from_the_excess_oxygen():
    liq = dict(name="liq", mode_wt=1.0, comp_wt={"FeO": 0.071844, "O": 0.0079995, "H2O": 0.05})
    fe3, h2o = run_fractional._melt_chemistry(_fc_rec(1, 0, 1.0, [liq]))
    assert fe3 == pytest.approx(1.0) and h2o == pytest.approx(5.0)
    assert np.isnan(run_fractional._melt_chemistry(_fc_rec(1, 0, 1.0, []))[0])


def test_fractional_complete_output_requires_every_path_ended(tmp_path):
    path = _write(tmp_path, [_fc_rec(1, 0, 1.0, [], end="ladder"), _fc_rec(2, 1, 1.0, [])])
    assert run_fractional._complete_output(path, [0])
    assert not run_fractional._complete_output(path, [0, 1])
    assert not run_fractional._complete_output(tmp_path / "missing.jsonl", [0])


def test_fractional_run_chunk_writes_four_tables_and_the_manifest(tmp_path, monkeypatch, short_ladder):
    comps, _ = _comps(tmp_path)
    calls = []
    monkeypatch.setattr(run_fractional, "run_julia", _fake_fc_julia(calls))
    out = tmp_path / "out"
    out.mkdir()
    args = _args(melt_floor=1e-4)
    rec = run_fractional.run_chunk(0, comps, args, out)
    tables = {t: pd.read_parquet(out / f"chunk_0000_{t}.parquet") for t in run_fractional.TABLES}
    assert calls == ["in_0.csv"]
    draws = tables["draws"]
    assert len(draws) == 4 and {"T_end_50", "T_end_5", "attrition_10", "end", "T_liquid_out",
                                "n_steps_run", "n_failed"} <= set(draws.columns)
    assert (draws["end"] == "melt_out").all() and (draws["T_liquid_out"] == 1080.0).all()
    assert (draws["T_end_50"] == 1080.0).all() and not draws["attrition_5"].any()
    # 4 draws x (3 phases at the first step + 2 at the melt-out step)
    assert len(tables["step"]) == 4 * 5 and len(tables["pile"]) == 4 * (3 + 3)
    assert list(tables["step"].columns) == run_generator.LADDER_COLS
    last = tables["pile"][(tables["pile"].draw == 0) & (tables["pile"].bulk == 0)]
    last = last[last.T_celsius == 1080.0].set_index("phase")["mode_wt"]
    assert last[["fsp", "q", "liq"]].tolist() == pytest.approx([0.6, 0.4, 0.0])
    assert rec["recipe"] == "fractional" and rec["draws_by_end"] == {"melt_out": 4}
    assert rec["n_points_run"] == 8 and rec["n_points_laddered"] == 8
    assert run_generator.gmc.read_manifest(out)[0]["n_pile_rows"] == len(tables["pile"])


def test_build_points_reads_a_named_table_into_its_own_cache(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    rows = pd.DataFrame([[0, 0, 1000.0, 0.4, "fsp", 0, 0.5], [0, 0, 1000.0, 0.4, "liq", 1, 0.4]],
                        columns=run_generator.LADDER_COLS)
    rows.to_parquet(work / "chunk_0000_pile.parquet", index=False)
    (work / "manifest.jsonl").write_text(json.dumps(dict(chunk=0)) + "\n")
    points = run_read.build_points(work, "pile")
    assert (work / "points_pile.parquet").exists() and not (work / "points.parquet").exists()
    assert points.loc[0, ["liq_frac", "m_fsp"]].tolist() == [0.4, 0.5]
    with pytest.raises(SystemExit):
        run_read.build_points(work)


def _ladder_points():
    # draw 0: amphibole peaks at the third step and is gone by the last while
    # biotite grows; draw 1 never carries amphibole
    rows = []
    for draw, amp, h2o in ((0, [0.0, 0.1, 0.2, 0.1, 0.0], 1.0), (1, [0.0] * 5, 5.0)):
        for k, (T, liq, bi) in enumerate(zip([1100.0, 1000.0, 900.0, 800.0, 700.0],
                                             [0.9, 0.7, 0.5, 0.2, 0.05], [0, 0, 0, 0.1, 0.2])):
            rows.append(dict(bulk=0, draw=draw, T_celsius=T, liq_frac=liq, fl_frac=0.0,
                             m_fsp=0.1 * (k + 1), m_px=0.0, m_amp=amp[k], m_bi=bi, m_ol=0.0,
                             m_opq=0.0, m_q=0.0, m_other=0.0, P_kbar=2.0 + 8.0 * draw, H2O=h2o))
    return pd.DataFrame(rows)


def test_ladder_peaks_remaining_gain_budget_and_proxy_on_a_synthetic_path():
    points = _ladder_points()
    ends = run_ladder.rung_ends(points)
    assert ends[50].loc[(0, 0)] == 800.0 and ends[10].loc[(0, 0)] == 700.0
    table = run_ladder.per_draw_table(points, ends).set_index(["category", "draw"])
    amp0, amp1 = table.loc[("amp", 0)], table.loc[("amp", 1)]
    assert amp0.carries and amp0.T_peak == 900.0 and amp0.peak == pytest.approx(0.2)
    assert amp0.remaining_50 == pytest.approx(0.5) and amp0.remaining_10 == pytest.approx(0.0)
    assert amp0.declined and amp0.loss == pytest.approx(0.2)
    assert amp0.gain_bi == pytest.approx(0.2) and amp0.gain_fsp == pytest.approx(0.2) and amp0.gain_q == 0.0
    assert not amp1.carries and np.isnan(amp1.remaining_10) and not amp1.declined
    fsp0 = table.loc[("fsp", 0)]
    assert fsp0.T_peak == 700.0 and not fsp0.declined and np.isnan(fsp0.loss)
    proxy = run_ladder.no_resorption_pile(points)
    at_end = proxy[(proxy.draw == 0) & (proxy.T_celsius == 700.0)].iloc[0]
    assert at_end.m_amp == pytest.approx(0.2) and at_end.m_fsp == pytest.approx(0.5)
    merged = proxy.merge(points, on=["bulk", "draw", "T_celsius"], suffixes=("_proxy", ""))
    assert (merged[[f"m_{g}_proxy" for g in GROUPS]].to_numpy()
            >= merged[[f"m_{g}" for g in GROUPS]].to_numpy() - 1e-12).all()
    medians = {"H2O": 3.0, "P_kbar": 6.0}
    masks = run_ladder.split_masks(table.reset_index(), medians)
    assert masks["wet"].sum() == masks["deep"].sum() == len(GROUPS)
    summary = run_ladder.summarize(table.reset_index(), medians, "synthetic").set_index(["category", "split"])
    amp_all, amp_dry = summary.loc[("amp", "all")], summary.loc[("amp", "dry")]
    assert amp_all.n_draws == 2 and amp_all.n_carrying == 1 and amp_all.frac_declining == 1.0
    assert amp_all.median_remaining_10 == 0.0 and amp_all.gain_bi == pytest.approx(1.0)
    assert amp_dry.n_carrying == 1 and summary.loc[("amp", "wet")].n_carrying == 0
    assert np.isnan(summary.loc[("px", "all")].median_T_peak)


from explorations.generator_reach import run_recipe_rungs


def _two_recipe_points():
    # rock 0: equilibrium brings both draws below 50% melt, fractional only draw 0;
    # only draw 0 of either recipe gets below 30%. Rock 1 has no fractional run.
    eq = _points([
        [0, 0, 1100.0, 0.9, 0.0, 0.1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1000.0, 0.45, 0.0, 0.3, 0.2, 0, 0, 0, 0, 0, 0],
        [0, 0, 900.0, 0.25, 0.0, 0.4, 0.3, 0, 0, 0, 0, 0, 0],
        [0, 1, 1100.0, 0.6, 0.0, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
        [0, 1, 1000.0, 0.4, 0.0, 0.3, 0.3, 0, 0, 0, 0, 0, 0],
        [1, 0, 1100.0, 0.3, 0.0, 0.5, 0, 0, 0, 0, 0, 0, 0],
    ])
    fc = _points([
        [0, 0, 1100.0, 0.9, 0.0, 0.1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1000.0, 0.45, 0.0, 0.2, 0.3, 0, 0, 0, 0, 0, 0],
        [0, 0, 900.0, 0.25, 0.0, 0.3, 0.4, 0, 0, 0, 0, 0, 0],
        [0, 1, 1100.0, 0.6, 0.0, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
        [0, 1, 1000.0, 0.55, 0.0, 0.3, 0.3, 0, 0, 0, 0, 0, 0],
    ])
    magnitude = np.zeros((2, 7))
    numeric = np.zeros((2, 7), dtype=bool)
    magnitude[:, :2], numeric[:, :2] = [0.5, 0.5], True
    return {"equilibrium": run_read.shares_and_distances(eq, magnitude, numeric),
            "fractional": run_read.shares_and_distances(fc, magnitude, numeric)}


def test_rung_ends_take_the_highest_t_point_below_a_scalar_or_per_rock_target():
    points = _two_recipe_points()["equilibrium"]
    ends = run_recipe_rungs.rung_ends(points, 0.5).set_index(["bulk", "draw"])["T_celsius"]
    assert ends.to_dict() == {(0, 0): 1000.0, (0, 1): 1000.0, (1, 0): 1100.0}
    ends = run_recipe_rungs.rung_ends(points, 0.3).set_index(["bulk", "draw"])["T_celsius"]
    assert ends.to_dict() == {(0, 0): 900.0}
    target = pd.Series({0: 0.95, 1: 0.2})
    ends = run_recipe_rungs.rung_ends(points, target).set_index(["bulk", "draw"])["T_celsius"]
    assert ends.to_dict() == {(0, 0): 1100.0, (0, 1): 1100.0}


def test_recipe_reads_use_only_draws_both_recipes_reach_and_pool_50_and_30():
    points = _two_recipe_points()
    reads, counts = run_recipe_rungs.recipe_reads(points, run_recipe_rungs.READS, pd.Series({0: 0.5, 1: 0.95}), 2)
    assert counts["50"] == {"equilibrium": 3, "fractional": 1, "matched": 1}
    # the last step is each draw's coolest converged point, on the draws both recipes ran
    assert counts["last"] == {"equilibrium": 3, "fractional": 2, "matched": 2}
    last, _ = reads["fractional", "last"]
    assert last[0, :2] == pytest.approx([(0.3 / 0.7 + 0.5) / 2, (0.4 / 0.7 + 0.5) / 2])
    assert counts["50+30"] == {"equilibrium": 1, "fractional": 1, "matched": 1}
    eq50, _ = reads["equilibrium", "50"]
    assert eq50[0, :2] == pytest.approx([0.6, 0.4])
    assert np.isnan(eq50[1]).all()
    fc50, presence = reads["fractional", "50"]
    assert fc50[0, :2] == pytest.approx([0.4, 0.6]) and presence[0, :2].tolist() == [1.0, 1.0]
    pooled, _ = reads["equilibrium", "50+30"]
    assert pooled[0, :2] == pytest.approx([(0.6 + 0.4 / 0.7) / 2, (0.4 + 0.3 / 0.7) / 2])
    own, _ = reads["equilibrium", "own_melt"]
    assert np.allclose(own[0], eq50[0])
    assert np.isnan(reads["equilibrium", "10"][0][0]).all()
    without = tuple(r for r in run_recipe_rungs.READS if r != "own_melt")
    reads, counts = run_recipe_rungs.recipe_reads(points, without, None, 2)
    assert "own_melt" not in counts and ("equilibrium", "own_melt") not in reads
    assert np.allclose(reads["equilibrium", "50+30"][0], pooled, equal_nan=True)


def test_align_places_benchmark_rows_on_the_composition_rows_by_key():
    comps = pd.DataFrame(dict(bulk=[0, 1, 2], key=["a", "b", "c"]))
    benchmark = SimpleNamespace(
        bulk_ids=np.array(["c", "a"]), magnitude=np.full((2, 7), 0.5), numeric=np.ones((2, 7), dtype=bool),
        presence=np.ones((2, 7)), measured=np.ones((2, 7), dtype=bool), unit=np.array(["u2", "u1"]))
    benchmark.magnitude[0, 0] = 0.9
    truth, at = run_recipe_rungs.align(benchmark, comps)
    assert at.tolist() == [2, 0]
    assert truth["evaluated"].tolist() == [True, False, True]
    assert truth["magnitude"][2, 0] == 0.9 and truth["magnitude"][0, 0] == 0.5
    assert not truth["numeric"][1].any() and truth["unit"].tolist() == ["u1", None, "u2"]


def test_read_set_keeps_only_rocks_read_everywhere():
    full = np.ones((3, 7))
    gap = full.copy()
    gap[1, 2] = np.nan
    reads = {("equilibrium", "50"): (full, full), ("fractional", "50"): (gap, full)}
    rows = np.array([True, True, False])
    assert run_recipe_rungs.read_set(reads, rows).tolist() == [True, False, False]


def test_bootstrap_bands_contain_the_estimate_and_lowest_fractions_sum_to_one():
    rng = np.random.default_rng(0)
    n = 30
    magnitude = rng.dirichlet(np.ones(7), n)
    numeric = np.ones((n, 7), dtype=bool)
    units = np.repeat(np.arange(10), 3)
    shares = {}
    for rec, exact in (("equilibrium", "30"), ("fractional", "own_melt")):
        for i, r in enumerate(run_recipe_rungs.READS):
            noisy = magnitude + rng.normal(0, 0.02 * (i + 1), magnitude.shape)
            shares[rec, r] = magnitude if r == exact else run_signal.renormalize(noisy, numeric)
    est, reps, wins = run_recipe_rungs.bootstrap(shares, magnitude, numeric, units, n_boot=50)
    for rec, exact in (("equilibrium", "30"), ("fractional", "own_melt")):
        assert sum(wins[rec].values()) == pytest.approx(1.0) and wins[rec][exact] == 1.0
    for name, e in est.items():
        lo, hi = run_recipe_rungs.band(reps[name])
        assert (lo <= e + 1e-12).all() and (e <= hi + 1e-12).all()
    presence = {k: (v > 0.1).astype(float) for k, v in shares.items()}
    tab = run_recipe_rungs.table(shares, presence, est, reps, magnitude, numeric,
                                 (magnitude > 0.1).astype(float), numeric)
    row = tab[(tab["source"] == "fractional") & (tab["read"] == "30") & (tab["category"] == "mean7")].iloc[0]
    m7 = run_recipe_rungs.OUT_LABELS.index("mean7")
    assert row["fc_minus_eq"] == pytest.approx(est["fractional", "30"][m7] - est["equilibrium", "30"][m7])
    assert row["fc_minus_eq_lo"] <= row["fc_minus_eq"] <= row["fc_minus_eq_hi"]


def test_norm_input_carries_total_iron_as_feo_on_every_row():
    comps = pd.DataFrame(dict(feo_tot=[5.0, np.nan], fe2o3_tot=[np.nan, 10.0]))
    out = run_recipe_rungs.norm_input(comps)
    assert out["feo_tot"].tolist() == pytest.approx([5.0, 10.0 * 2 * 71.844 / 159.69])
    assert comps["feo_tot"].isna().sum() == 1

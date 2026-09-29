import numpy as np
import pandas as pd
import pytest

from cnmineral.eval.real_rock import (
    BENCHMARK_DIR,
    GROUP_INDEX,
    POINTCOUNT_BASIS,
    VISUAL_BASIS,
    canonical_units,
    load_benchmark,
    split_mask,
)
from explorations.real_mineralogy import harvest
from explorations.real_mineralogy import run_composition_tables as tables_script
from explorations.real_mineralogy import run_scale_convention as scale_script
from explorations.real_mineralogy import run_unit_leakage as leakage_script
from explorations.real_mineralogy.harvest import (
    DATA,
    PLUTONIC_BASIS,
    POINTCOUNT_PHASES,
    _mode_cell,
    _pointcount_cell,
    parse_dubray_pointcounts,
    parse_ds942,
    parse_pp1793,
)

PLUTONIC_SOURCES = {"dubray_ds942", "dubray_pp1793"}
MAJORS = ["sio2", "tio2", "al2o3", "feo_tot", "mno", "mgo", "cao", "na2o",
          "k2o", "p2o5"]


@pytest.fixture(scope="module")
def plutonic():
    """Both plutonic workbooks parsed once, as (samples, modes) per source."""
    return {"dubray_ds942": parse_ds942(), "dubray_pp1793": parse_pp1793()}


@pytest.fixture(scope="module")
def visual_only(tmp_path_factory):
    """The benchmark on disk with every point-count row removed.

    Reconstructs what `data/real_mineralogy/` held before Appendix 10 was
    harvested, so the default load can be compared against it directly.
    """
    out = tmp_path_factory.mktemp("visual_only")
    samples = pd.read_csv(BENCHMARK_DIR / "samples.csv")
    modes = pd.read_csv(BENCHMARK_DIR / "modes.csv")
    key = modes["source"] + "|" + modes["sample_id"].astype(str)
    both = modes.groupby(key)["basis"].transform("nunique") > 1
    samples.to_csv(out / "samples.csv", index=False)
    modes[~(both & (modes["basis"] == POINTCOUNT_BASIS))].to_csv(
        out / "modes.csv", index=False
    )
    return out


def test_default_load_is_unchanged_by_the_pointcount_rows(visual_only):
    """Harvesting a second ruler moved no number that was already scored.

    Every real-rock verdict on record was measured against the visual
    estimates. Appendix 10 adds rows for 70 rocks that already had them, and
    the default load has to ignore those rows exactly -- not approximately.
    """
    before = load_benchmark(visual_only)
    after = load_benchmark()
    assert len(before) == len(after)
    for field in ("bulk_ids", "presence", "magnitude", "numeric", "censored",
                  "censor_bound", "has_magnitude", "grade", "clean", "sio2",
                  "unit"):
        assert np.array_equal(getattr(before, field), getattr(after, field)), field
    assert np.array_equal(before.bulk_src.numpy(), after.bulk_src.numpy())
    assert np.array_equal(before.bulk_frac.numpy(), after.bulk_frac.numpy())


def test_basis_preference_selects_one_ruler_per_rock():
    """A rock measured twice is read once, and the two readings differ.

    The rulers must not sum: both describe the same thin section, so adding
    them would double the crystal cargo.
    """
    visual = load_benchmark(prefer_basis=VISUAL_BASIS)
    counted = load_benchmark(prefer_basis=POINTCOUNT_BASIS)
    assert np.array_equal(visual.bulk_ids, counted.bulk_ids)

    moved = ~np.all(visual.magnitude == counted.magnitude, axis=1)
    assert moved.sum() == 70
    # rocks with a single ruler are untouched by the preference
    assert np.array_equal(visual.magnitude[~moved], counted.magnitude[~moved])
    for benchmark in (visual, counted):
        totals = benchmark.magnitude[benchmark.has_magnitude].sum(1)
        assert np.allclose(totals, 1.0)

    # a misspelled basis would otherwise drop every doubly-measured rock's
    # modes, quietly shrinking the benchmark instead of failing
    with pytest.raises(ValueError, match="prefer_basis"):
        load_benchmark(prefer_basis="pointcount")


def test_pointcount_cell_grammar():
    """The four cell forms Appendix 10 actually contains.

    The composite-feldspar form is the one that matters most: read naively it
    parses as a plagioclase count, silently recording total feldspar under one
    of its two members.
    """
    assert _pointcount_cell(19) == (19, None, None)
    assert _pointcount_cell("TR") == (None, None, "TR")
    assert _pointcount_cell("TR?") == (None, None, "TR?")
    assert _pointcount_cell("TR(?)") == (None, None, "TR(?)")
    assert _pointcount_cell("1(?)") == (1, None, "?")
    assert _pointcount_cell("87 (total feldspar)") == (87, "feldspar", None)
    assert _pointcount_cell("110 (total probable feldspar)") == (110, "feldspar", None)
    with pytest.raises(ValueError):
        _pointcount_cell("about half")


def test_derived_percent_reproduces_the_published_block(monkeypatch):
    """Percentages are computed from raw counts, then checked against the file.

    The workbook publishes both the counts and its own derived percentages;
    deriving ours and asserting agreement is what makes a revised download fail
    loudly instead of shifting the truth underneath a frozen benchmark.
    """
    samples = pd.read_csv(BENCHMARK_DIR / "samples.csv")
    known = set(samples[samples["source"] == "dubray_ds1099"]["sample_id"])
    modes, n_points, skipped = parse_dubray_pointcounts(known)
    assert len(n_points) == 70 and skipped == 25
    assert set(modes["phase"]) <= {name for name, _ in POINTCOUNT_PHASES.values()}

    workbook = pd.read_excel(DATA / "dubray_ds1099" / "Appendix10.xlsx", header=2)
    tampered = workbook.copy()
    tampered.loc[tampered.index[0], "Plagioclase"] += 1
    monkeypatch.setattr(harvest.pd, "read_excel", lambda *a, **k: tampered)
    with pytest.raises(ValueError, match="published"):
        parse_dubray_pointcounts(known)


def test_composition_tables_partition_the_scored_set():
    """Every appendix table counts the same rocks, exactly once.

    These tables go into the paper, so a row that double-counts a rock measured
    two ways -- or a cut computed on the wrong stage of the funnel -- is a
    published error. Each breakdown has to sum back to the scored total.
    """
    samples = pd.read_csv(BENCHMARK_DIR / "samples.csv")
    modes = pd.read_csv(BENCHMARK_DIR / "modes.csv")
    counts = tables_script.funnel(samples, modes, tables_script.published_counts())
    frame = tables_script.scored_frame(samples, counts["scored_keys"])
    data = tables_script.tables(frame, modes, counts)

    scored = data["funnel"]["scored"]
    assert scored == len(counts["scored_keys"])
    assert sum(r["rocks"] for r in data["by_survey"]) == scored
    assert sum(r["rocks"] for r in data["by_silica"]) == scored
    assert sum(r["rocks"] for r in data["unit_structure"]) == scored
    assert sum(r["units"] for r in data["unit_structure"]) == data["units_total"]
    assert sum(r["rocks"] for r in data["by_split"]) == scored
    assert sum(data["by_method"].values()) == scored
    assert all(r["trace_only"] >= 0 for r in data["minerals"])
    assert all(r["rocks"] <= scored for r in data["minerals"])


def test_pointcount_truth_is_a_crystal_cargo_like_the_visual_one():
    """Both rulers land on the same seven groups and the same conventions.

    Appendix 10 counts matrix explicitly where the visual estimates do not, so
    this guards that groundmass still drops out rather than entering the
    composition as an eighth component.
    """
    counted = load_benchmark(prefer_basis=POINTCOUNT_BASIS)
    visual = load_benchmark(prefer_basis=VISUAL_BASIS)
    moved = ~np.all(visual.magnitude == counted.magnitude, axis=1)

    # Appendix 10 never looks for pyroxene or olivine, so those groups carry no
    # counted presence on these rocks -- silence, which the loader records as
    # absence and the contrast must not read as a measured zero.
    for group in ("px", "ol"):
        assert counted.presence[moved, GROUP_INDEX[group]].sum() == 0
    assert np.allclose(counted.magnitude[moved].sum(1), 1.0)


def test_undoing_the_recalculation_lands_on_the_raw_analytical_total(tmp_path):
    """The scale-convention undo reproduces the total the lab actually reported.

    du Bray's published majors are recalculated to sum to 100 volatile-free, and
    the workbooks record the pre-recalculation analytical total.  Scaling by
    `(Total_I_pct - Volatile_pct)/100` must therefore put the full major set
    back on exactly that raw volatile-free total -- if it does not, the factor
    is being read against the wrong quantity and every number the contrast
    produces is measuring the wrong translation.
    """
    factors = scale_script.scale_factors()
    out, moved = scale_script.unrecalculated_dir(tmp_path, factors)
    samples = pd.read_csv(out / "samples.csv")
    keys = samples["source"] + "|" + samples["sample_id"].astype(str)

    cols = [c for c in scale_script.RECALCULATED_COLS if c in samples.columns]
    rescaled = keys.isin(set(moved))
    total = samples.loc[rescaled, cols].sum(axis=1, min_count=1)
    expected = keys[rescaled].map(factors) * 100.0
    ok = total.notna() & expected.notna()
    # a handful of rocks are missing a minor oxide, so the sum falls short of
    # the target by that oxide rather than by a scaling error
    assert (total[ok] - expected[ok]).abs().median() < 0.05

    # the truth side must be untouched: this changes chemistry only
    published = load_benchmark()
    undone = load_benchmark(out)
    assert np.array_equal(published.bulk_ids, undone.bulk_ids)
    assert np.array_equal(published.magnitude, undone.magnitude)
    assert np.array_equal(published.presence, undone.presence)


def test_eruption_merge_rule_splits_prefixes_it_must_not_join():
    """The leak rule may over-flag, but only along the sub-facies axis.

    Two failure modes would make its answer meaningless in opposite directions.
    Cutting at the first comma collapses every `Plio-Pleistocene, ...` unit into
    one era, and plain substring containment lets a bare `Rhyolite` swallow
    every rhyolite in the benchmark; either would flag most of val and turn the
    contrast into a comparison of nothing. The comma boundary is what prevents
    both, so it is asserted rather than assumed.
    """
    same = leakage_script.same_eruption
    norm = leakage_script.normalize_unit

    # sub-facies and qualifiers name the same eruption: these must merge
    assert same(norm("Heller Tuff"), norm("Heller Tuff ?"))
    assert same(norm("Fraction Tuff"), norm("FractionTuff"))
    assert same(norm("Fraction Tuff"), norm("Fraction Tuff, basal vitrophyre"))
    assert same(norm("Eureka Valley Tuff"),
                norm("Eureka Valley Tuff, middle, By-Day Member"))

    # distinct units sharing a prefix must NOT merge
    assert not same(norm("Plio-Pleistocene, rhyolite of Spring Peak"),
                    norm("Plio-Pleistocene, trachyandesite of Mud Spring"))
    assert not same(norm("Rhyolite"), norm("Rhyolite of Bodie Hills"))
    assert not same(norm("Heller Tuff"), norm("Fraction Tuff"))


def test_leak_flagging_is_confined_to_val_and_stays_a_minority():
    """The flag marks val rocks only, and does not swallow the slice.

    Guards the rule against silently degenerating: if a normalization change
    made it match too broadly, the leak-free arm would shrink toward nothing and
    the contrast would read as agreement for the wrong reason.
    """
    benchmark = load_benchmark()
    scored = benchmark.clean & benchmark.has_magnitude
    train = scored & split_mask(benchmark, "train")
    val = scored & split_mask(benchmark, "val")

    leaking = leakage_script.leaking_mask(benchmark.unit, val, train)
    assert not (leaking & ~val).any()
    assert 0 < leaking.sum() < 0.5 * val.sum()

    # every flag is explainable by a named train unit, so the report is complete
    merges = leakage_script.merge_report(benchmark.unit, leaking, train)
    assert {m["val_unit"] for m in merges} == set(benchmark.unit[leaking])


def test_plutonic_mode_cell_grammar():
    """The cell forms the two plutonic workbooks contain.

    A trace or `<x` cell is a present-but-unquantified mineral and must keep
    its qualifier, a whitespace-only cell is blank, and a reported zero stays a
    row so the loader can read it as looked-for-and-not-found.  Anything else
    must raise rather than parse: the known typos go through `MODE_TYPOS`, not
    through a lenient reader.
    """
    assert _mode_cell(np.nan) is None
    assert _mode_cell("   ") is None
    assert _mode_cell(12.5) == (12.5, None)
    assert _mode_cell(0) == (0.0, None)
    for text in ("TR", "tr", "<0.01", "<1"):
        value, qualifier = _mode_cell(text)
        assert np.isnan(value) and qualifier == text
    for text in ("l.0", "`0", "about half"):
        with pytest.raises(ValueError):
            _mode_cell(text)


def test_plutonic_harvest_emits_each_rock_once_from_its_own_screens(plutonic):
    """Each source contributes the rocks its own primary dataset holds, once.

    PP 1793 interleaves a pluton `AVERAGE` row after each pluton's samples;
    reading one as a rock would add a synthetic mean to the benchmark and put
    it in the same unit as the rocks it averages.  DS 942 reuses six field
    numbers across different intrusions, which would collide on the
    `source|sample_id` key.  The unit counts pin the grouping the split and
    the bootstrap will cluster on: whole intrusions, not their facies.
    """
    s942, _ = plutonic["dubray_ds942"]
    s1793, _ = plutonic["dubray_pp1793"]
    assert len(s942) == 332
    # one PP 1793 rock carries a mode but no chemistry, which the loader drops
    assert len(s1793) == 247 and s1793["sio2"].notna().sum() == 246
    assert not (s1793["sample_id"] == "AVERAGE").any()
    for samples in (s942, s1793):
        assert samples["sample_id"].is_unique
        assert samples["unit"].notna().all()
    assert {"83-1 (Oracle)", "83-1 (Fort Huachuca)"} <= set(s942["sample_id"])
    assert s942["sample_id"].str.endswith(")").sum() == 12
    assert len(set(canonical_units(s942["unit"]))) == 55
    assert len(set(canonical_units(s1793["unit"]))) == 30


def test_plutonic_modes_close_on_the_whole_rock(plutonic):
    """Whole-rock point counts sum to about 100, with no groundmass.

    That closure is what the whole-rock basis means, and it only holds if the
    accessory, secondary and lumped columns are kept alongside the comparison
    minerals.  The majors are recalculated to 100 volatile-free in both
    sources, the convention the volcanic du Bray tables use.
    """
    for samples, modes in plutonic.values():
        assert set(modes["basis"]) == {PLUTONIC_BASIS}
        assert "groundmass" not in set(modes["magemin_phase"])
        totals = modes.groupby("sample_id")["abundance_pct"].sum()
        assert totals.between(96.0, 102.0).mean() > 0.95
        chem = samples[samples["sio2"].notna()]
        assert np.allclose(chem[MAJORS].sum(axis=1), 100.0, atol=0.01)
        assert (samples["total_crystals_pct"] == 100.0).all()


def test_mode_typos_must_be_found_where_they_are_expected(monkeypatch):
    """A typo correction that does not find its cell fails the harvest.

    The corrections rewrite published text, so each is asserted against the
    cell it names: a revised download that fixed or moved the typo has to be
    looked at, not silently corrected twice or skipped.
    """
    monkeypatch.setitem(harvest.MODE_TYPOS,
                        ("dubray_pp1793", "no-such-rock", "Opq"), ("`0", 0.0))
    with pytest.raises(ValueError, match="not found"):
        parse_pp1793()
    monkeypatch.delitem(harvest.MODE_TYPOS, ("dubray_pp1793", "no-such-rock", "Opq"))
    monkeypatch.setitem(harvest.MODE_TYPOS,
                        ("dubray_pp1793", "28693", "Opq"), ("x0", 0.0))
    with pytest.raises(ValueError, match="expected the typo"):
        parse_pp1793()


def test_plutonic_rocks_stay_out_of_the_volcanic_benchmark():
    """The frozen volcanic tables hold no plutonic rock.

    Every eruption-v2 verdict evaluates all the rocks in its tables, so a
    plutonic rock landing there would move stamped numbers.
    """
    for table in ("samples.csv", "modes.csv"):
        sources = set(pd.read_csv(BENCHMARK_DIR / table)["source"])
        assert not sources & PLUTONIC_SOURCES
    assert harvest.PLUTONIC_DIR != BENCHMARK_DIR

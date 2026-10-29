import numpy as np
import pandas as pd
import pytest
import torch

from cnmineral.eval import real_rock
from cnmineral.eval.real_rock import (
    BENCHMARK_DIR,
    GROUP_INDEX,
    POINTCOUNT_BASIS,
    canonical_units,
    load_benchmark,
    preferred_basis,
    real_rock_metrics,
    score_real_rock,
    split_mask,
)
from cnmineral.models.mode_head import ModeHead
from explorations.real_mineralogy import harvest
from explorations.real_mineralogy import plutonic as plutonic_module
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
from explorations.real_mineralogy.plutonic import (
    PLUTONIC_SPLIT_PATH,
    load_plutonic,
    mafic_total_keys,
    oxides_lumped_keys,
    plutonic_benchmark,
)
from explorations.real_mineralogy.run_plutonic_split import MIN_UNITS_PER_SUBSET

PLUTONIC_SOURCES = {"dubray_ds942", "dubray_pp1793"}
MAJORS = ["sio2", "tio2", "al2o3", "feo_tot", "mno", "mgo", "cao", "na2o",
          "k2o", "p2o5"]

needs_workbooks = pytest.mark.skipif(
    not all((DATA / source).is_dir() for source in
            ("dubray_ds764", "dubray_ds1099", "dubray_ds942", "dubray_pp1793")),
    reason="USGS source workbooks absent; explorations/real_mineralogy/sources.md lists them",
)


@pytest.fixture(scope="module")
def plutonic():
    """Both plutonic workbooks parsed once, as (samples, modes) per source."""
    return {"dubray_ds942": parse_ds942(), "dubray_pp1793": parse_pp1793()}


@pytest.fixture(scope="module")
def composed():
    """Every benchmark's composition tables, built once as (data, markdown, frame)."""
    return {name: tables_script.compose(benchmark)
            for name, benchmark in tables_script.BENCHMARKS.items()}


@pytest.fixture(scope="module")
def counted(tmp_path_factory):
    """The benchmark read on the point counts wherever a rock carries both rulers."""
    out = tmp_path_factory.mktemp("counted")
    samples = pd.read_csv(BENCHMARK_DIR / "samples.csv")
    modes = pd.read_csv(BENCHMARK_DIR / "modes.csv")
    modes["key"] = modes["source"] + "|" + modes["sample_id"].astype(str)
    samples.to_csv(out / "samples.csv", index=False)
    preferred_basis(modes, POINTCOUNT_BASIS).drop(columns="key").to_csv(
        out / "modes.csv", index=False
    )
    return load_benchmark(out)


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


def test_basis_preference_selects_one_ruler_per_rock(counted):
    """A rock measured twice is read once, and the two readings differ.

    The rulers must not sum: both describe the same thin section, so adding
    them would double the crystal cargo.
    """
    visual = load_benchmark()
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
    modes = pd.read_csv(BENCHMARK_DIR / "modes.csv")
    modes["key"] = modes["source"] + "|" + modes["sample_id"].astype(str)
    with pytest.raises(ValueError, match="prefer_basis"):
        preferred_basis(modes, "pointcount")


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


@needs_workbooks
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


@needs_workbooks
@pytest.mark.parametrize("name", sorted(tables_script.BENCHMARKS))
def test_composition_tables_partition_the_scored_set(composed, name):
    """Every appendix table counts the same rocks, exactly once.

    These tables go into the paper, so a row that double-counts a rock measured
    two ways -- or a cut computed on the wrong stage of the funnel -- is a
    published error. Each breakdown has to sum back to the scored total.
    Checked on every benchmark the script describes.
    """
    data, _, frame = composed[name]

    scored = len(frame)
    assert frame["key"].is_unique
    assert scored in data["funnel"].values()
    assert sum(r["rocks"] for r in data["by_survey"]) == scored
    assert sum(r["rocks"] for r in data["by_silica"]) == scored
    assert sum(r["rocks"] for r in data["unit_structure"]) == scored
    assert sum(r["units"] for r in data["unit_structure"]) == data["units_total"]
    assert sum(r["rocks"] for r in data["by_split"]) == scored
    assert sum(data["by_method"].values()) == scored
    assert all(r["trace_only"] >= 0 for r in data["minerals"])
    assert all(r["rocks"] <= scored for r in data["minerals"])


def test_pointcount_truth_is_a_crystal_cargo_like_the_visual_one(counted):
    """Both rulers land on the same seven groups and the same conventions.

    Appendix 10 counts matrix explicitly where the visual estimates do not, so
    this guards that groundmass still drops out rather than entering the
    composition as an eighth component.
    """
    visual = load_benchmark()
    moved = ~np.all(visual.magnitude == counted.magnitude, axis=1)

    # Appendix 10 never looks for pyroxene or olivine, so those groups carry no
    # counted presence on these rocks -- silence, which the loader records as
    # absence and the contrast must not read as a measured zero.
    for group in ("px", "ol"):
        assert counted.presence[moved, GROUP_INDEX[group]].sum() == 0
    assert np.allclose(counted.magnitude[moved].sum(1), 1.0)


@needs_workbooks
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


@needs_workbooks
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


@needs_workbooks
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


@needs_workbooks
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


def test_plutonic_split_covers_every_rock_by_whole_intrusion():
    """The committed plutonic split knows every rock and never splits an intrusion.

    A rock the split does not know drops out of the climatology null's fitting
    set, and an intrusion straddling two subsets puts near-identical rocks on
    both sides of the boundary the split exists to draw.
    """
    benchmark = load_plutonic()
    roster = pd.read_csv(PLUTONIC_SPLIT_PATH)
    keys = roster["source"] + "|" + roster["sample_id"].astype(str)
    assert set(keys) == set(benchmark.bulk_ids)
    assert (roster.groupby("unit")["subset"].nunique() == 1).all()
    units_per_subset = roster.groupby("subset")["unit"].nunique()
    assert set(units_per_subset.index) == {"train", "val", "test"}
    assert (units_per_subset >= MIN_UNITS_PER_SUBSET).all()


def test_unmeasured_categories_are_masked_or_leave_the_evaluated_set():
    """A category nobody measured is not scored as absent.

    The oxides are unmeasured only where the accessories-with-oxides lump
    stands alone; beside an oxide row the lump holds accessories and the rock
    is fully measured.  Such a rock is evaluated with its oxides masked.  A
    mafic total alone leaves every mafic category unmeasured, and the rock
    out of the evaluated set.
    """
    modes = pd.DataFrame({
        "source": ["s"] * 5,
        "sample_id": ["lump_only", "lump_and_oxide", "lump_and_oxide",
                      "mafic_total", "plain"],
        "magemin_phase": ["accessory_opq", "accessory_opq", "opq_unsplit",
                          "mafic_unsplit", "q"],
    })
    assert oxides_lumped_keys(modes) == {"s|lump_only"}
    assert mafic_total_keys(modes) == {"s|mafic_total"}

    benchmark = load_plutonic()
    left_out = set(benchmark.bulk_ids[~benchmark.clean])
    assert len(left_out) == 3
    masked = ~benchmark.measured
    assert masked.sum() == 68 and masked[:, GROUP_INDEX["opq"]].sum() == 68
    assert benchmark.clean[masked.any(1)].all()
    assert not benchmark.presence[masked].any()
    with plutonic_benchmark() as evaluated:
        assert len(evaluated) == len(benchmark) - 3
        assert not set(evaluated.bulk_ids) & left_out


@needs_workbooks
def test_plutonic_composition_tables_describe_the_evaluated_rocks(composed):
    """The plutonic tables break down exactly the rocks the instrument evaluates.

    The tables take their last cut from `mafic_total_keys` directly rather than
    through `load_plutonic`, so a drift between the two would describe one set
    of rocks and evaluate another.  Units come from the split, so a rock the
    split does not know would fall out of every unit and split count silently.
    """
    _, _, frame = composed["plutonic"]
    benchmark = load_plutonic()
    assert set(frame["key"]) == set(benchmark.bulk_ids[benchmark.clean])
    assert frame["unit"].notna().all() and frame["subset"].notna().all()


def test_plutonic_benchmark_restores_the_eruption_split():
    """The eruption split is back in place after the block, even on an error.

    Every eruption-v2 number reads `real_rock.SPLIT_PATH`; if the plutonic
    split leaked past its block, a later eruption evaluation in the same
    process would fit its null on the wrong rocks.
    """
    eruption_split = real_rock.SPLIT_PATH
    with plutonic_benchmark():
        assert real_rock.SPLIT_PATH == PLUTONIC_SPLIT_PATH
    assert real_rock.SPLIT_PATH == eruption_split
    with pytest.raises(RuntimeError):
        with plutonic_benchmark():
            raise RuntimeError("inside the block")
    assert real_rock.SPLIT_PATH == eruption_split


def test_plutonic_benchmark_refuses_a_split_missing_rocks(tmp_path, monkeypatch):
    """A split that does not know a rock fails before anything is evaluated."""
    roster = pd.read_csv(PLUTONIC_SPLIT_PATH)
    stale = tmp_path / "plutonic_split.csv"
    roster.iloc[1:].to_csv(stale, index=False)
    monkeypatch.setattr(plutonic_module, "PLUTONIC_SPLIT_PATH", stale)
    eruption_split = real_rock.SPLIT_PATH
    with pytest.raises(ValueError, match="not in"):
        with plutonic_benchmark():
            pass
    assert real_rock.SPLIT_PATH == eruption_split


def test_zero_shot_scoring_runs_on_the_plutonic_benchmark():
    """The tripwire: the production scoring path reaches a finite number here.

    A random head stands in for the checkpoint.  The failure this guards is
    the null fitting on no rock at all, which `score_real_rock` reaches
    whenever the plutonic rocks are evaluated against a split that does not
    know them.
    """
    torch.manual_seed(0)
    head = ModeHead(d_model=16).eval()
    with plutonic_benchmark() as evaluated:
        benchmark = evaluated.subset(evaluated.measured.all(1))
        fit = split_mask(benchmark, "train") | split_mask(benchmark, "val")
        latents = torch.randn(len(benchmark), 16)
        with torch.no_grad():
            scored = score_real_rock(head, benchmark, latents)
        metrics = real_rock_metrics(scored, benchmark, tag="smoke",
                                    encoder_checkpoint="none")
    assert metrics["climatology"]["n_fit"] == int(fit.sum()) > 0
    assert np.isfinite(metrics["headline"]["joint_nll"])
    assert np.isfinite(metrics["climatology"]["joint_nll"])

import numpy as np
import pandas as pd
import pytest

from cnmineral.eval.real_rock import (
    BENCHMARK_DIR,
    GROUP_INDEX,
    POINTCOUNT_BASIS,
    VISUAL_BASIS,
    load_benchmark,
)
from explorations.real_mineralogy import harvest
from explorations.real_mineralogy import run_composition_tables as tables_script
from explorations.real_mineralogy.harvest import (
    DATA,
    POINTCOUNT_PHASES,
    _pointcount_cell,
    parse_dubray_pointcounts,
)


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

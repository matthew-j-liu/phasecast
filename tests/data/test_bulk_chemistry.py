"""Tests for `cnmineral.data.bulk_chemistry.BulkChemistryDataset`.

Structural tests use a 10-row in-memory CSV fixture so they don't
depend on `data/bulks/gard2019.csv` being present.  One regression-pin
test loads the real file when it's available and skips otherwise; this
guards against silent row-count regressions without coupling the suite
to exact production numbers.
"""

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cnmineral.data.tokens import (
    MASK_TOKEN_ID,
    N_TOKENS,
    N_TOKENS_REDOX,
    REDOX_TOKEN_ID,
    REDOX_TOKEN_INDEX,
    assert_padding_convention,
)
from cnmineral.data.bulk_chemistry import (
    OXIDE_TO_CATION_FACTOR,
    BulkChemistryDataset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CSV = REPO_ROOT / "data" / "bulks" / "gard2019.csv"


# every fixture row carries the same trace block + sample_id so each
# test can override only the columns it cares about.  Defaults are a
# clean igneous bulk that passes the strict require_all_majors filter.
_HEADER = (
    "sample_id,qap_name,rock_group,"
    "sio2,al2o3,cao,mgo,feo,fe2o3,feo_tot,fe2o3_tot,k2o,na2o,tio2,cr2o3,"
    "sr_ppm,zr_ppm,ba_ppm,y_ppm,rb_ppm,ni_ppm,nb_ppm,v_ppm,"
    "la_ppm,ce_ppm,nd_ppm,yb_ppm,th_ppm"
)


def _clean_row(
    *, sample_id: int = 1, qap_name: str = "basalt",
    sio2: float = 50.0, al2o3: float = 15.0, cao: float = 10.0,
    mgo: float = 8.0, feo: float = 7.0, fe2o3: float = 2.0,
    feo_tot: float | str = "", fe2o3_tot: float | str = "",
    k2o: float = 1.0, na2o: float = 3.0, tio2: float = 1.0,
    cr2o3: float | str = "",
    sr_ppm: float = 300.0,
) -> str:
    """Format a single CSV row with sensible defaults.

    Pass empty string for columns that should be NaN.
    """
    return (
        f"{sample_id},{qap_name},igneous,"
        f"{sio2},{al2o3},{cao},{mgo},{feo},{fe2o3},{feo_tot},{fe2o3_tot},"
        f"{k2o},{na2o},{tio2},{cr2o3},"
        f"{sr_ppm},,,,,,,,,,,,"
    )


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join([_HEADER, *rows]) + "\n", encoding="latin-1")
    return path


# ----- structural tests on a tiny in-memory CSV -----


def test_sample_shape_and_dtype(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "tiny.csv", [_clean_row()])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    assert len(ds) == 1
    item = ds[0]
    assert set(item.keys()) == {"src", "frac", "sample_id"}
    assert item["src"].shape == (N_TOKENS,)
    assert item["frac"].shape == (N_TOKENS,)
    assert item["src"].dtype == torch.long
    assert item["frac"].dtype == torch.float32
    assert isinstance(item["sample_id"], int)


def test_obeys_padding_convention(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=i + 1) for i in range(8)
    ])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    loader = DataLoader(ds, batch_size=4)
    for batch in loader:
        assert_padding_convention(batch["src"], batch["frac"])


def test_no_mask_token_in_raw_output(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=i + 1) for i in range(4)
    ])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    src_stack = torch.stack([ds[i]["src"] for i in range(len(ds))])
    assert not (src_stack == MASK_TOKEN_ID).any()


def test_sample_id_stable_across_construction(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=42 + i) for i in range(5)
    ])
    a = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    b = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    for i in range(len(a)):
        assert a[i]["sample_id"] == b[i]["sample_id"]


def test_oxide_to_cation_conversion(tmp_path: Path) -> None:
    """sio2=50 wt% should yield Si frac = 50 * 0.4675 / 100."""
    csv = _write_csv(tmp_path / "tiny.csv", [_clean_row(sio2=50.0)])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    expected_si_frac = 50.0 * OXIDE_TO_CATION_FACTOR["sio2"] / 100.0
    si_token_idx = 0  # MAJORS[0] is Si
    assert ds[0]["src"][si_token_idx] == 14
    assert ds[0]["frac"][si_token_idx].item() == pytest.approx(
        expected_si_frac, rel=1e-5,
    )


def test_iron_aggregation_feo_plus_fe2o3(tmp_path: Path) -> None:
    """feo=10, fe2o3=5 should produce Fe = (10*0.7773 + 5*0.6994) / 100."""
    csv = _write_csv(tmp_path / "tiny.csv", [_clean_row(feo=10.0, fe2o3=5.0)])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    expected_fe_frac = (
        10.0 * OXIDE_TO_CATION_FACTOR["feo"]
        + 5.0 * OXIDE_TO_CATION_FACTOR["fe2o3"]
    ) / 100.0
    fe_token_idx = 4  # MAJORS[4] is Fe
    assert ds[0]["src"][fe_token_idx] == 26
    assert ds[0]["frac"][fe_token_idx].item() == pytest.approx(
        expected_fe_frac, rel=1e-5,
    )


def test_iron_fallback_to_feo_tot(tmp_path: Path) -> None:
    """When feo & fe2o3 both NaN but feo_tot present, fall back to it.

    Requires `require_all_majors=False` since the strict default
    requires both primary iron oxides.
    """
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(feo="", fe2o3="", feo_tot=12.0),
    ])
    ds = BulkChemistryDataset(csv_path=csv, require_all_majors=False, min_class_count=1)
    assert len(ds) == 1
    expected = 12.0 * OXIDE_TO_CATION_FACTOR["feo"] / 100.0
    fe_token_idx = 4
    assert ds[0]["src"][fe_token_idx] == 26
    assert ds[0]["frac"][fe_token_idx].item() == pytest.approx(
        expected, rel=1e-5,
    )


def test_zero_is_nan(tmp_path: Path) -> None:
    """A row with sio2=0 must produce Si in the padded state.

    Uses `require_all_majors=False` because under the strict default
    the zero-as-NaN logic would drop the whole row -- which we cover
    separately in `test_zero_drops_row_under_strict`.
    """
    csv = _write_csv(tmp_path / "tiny.csv", [_clean_row(sio2=0.0)])
    ds = BulkChemistryDataset(csv_path=csv, require_all_majors=False, min_class_count=1)
    assert len(ds) == 1
    si_token_idx = 0
    assert ds[0]["src"][si_token_idx].item() == 0
    assert ds[0]["frac"][si_token_idx].item() == 0.0


def test_zero_drops_row_under_strict(tmp_path: Path) -> None:
    """Strict default + zero-as-NaN + an exact-zero major -> row dropped."""
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=1, sio2=0.0),
        _clean_row(sample_id=2),  # clean: should survive
    ])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    assert len(ds) == 1
    assert ds[0]["sample_id"] == 2


def test_negative_drops_row(tmp_path: Path) -> None:
    """A row with a negative major must not appear in the dataset."""
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=1, sio2=-5.0),
        _clean_row(sample_id=2),
    ])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    assert len(ds) == 1
    assert ds[0]["sample_id"] == 2


def test_negative_trace_nans_out_only_that_slot(tmp_path: Path) -> None:
    """A negative trace value should NaN-out the trace slot but keep the row."""
    csv = _write_csv(tmp_path / "tiny.csv", [_clean_row(sr_ppm=-100.0)])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    assert len(ds) == 1
    sr_token_idx = len(["Si", "Al", "Ca", "Mg", "Fe", "K", "Na", "Ti", "Cr"])
    assert ds[0]["src"][sr_token_idx].item() == 0
    assert ds[0]["frac"][sr_token_idx].item() == 0.0


def test_overrange_magnitudes_nan_the_slot(tmp_path: Path) -> None:
    """Under `drop_overrange_as_nan`, an oxide above 100 wt% or a trace
    above 1e6 ppm pads its slot (transcription garbage would otherwise
    overflow frac past 1); with the knob off the row drops instead via
    the unconditional frac-overflow guard."""
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=1, sio2=405.0, sr_ppm=5e6),
    ])
    ds = BulkChemistryDataset(
        csv_path=csv, require_all_majors=False, min_class_count=1,
        drop_overrange_as_nan=True,
    )
    assert len(ds) == 1
    assert ds[0]["src"][0].item() == 0    # Si padded
    assert ds[0]["src"][9].item() == 0    # Sr padded
    assert float(ds[0]["frac"].max()) <= 1.0

    ds_off = BulkChemistryDataset(
        csv_path=csv, require_all_majors=False, min_class_count=1,
    )
    assert len(ds_off) == 0


def test_frac_overflow_row_dropped(tmp_path: Path) -> None:
    """feo + fe2o3 each under 100 wt% but jointly overflowing the Fe
    slot past frac 1 marks the row corrupt; clean rows survive."""
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=1, feo=95.0, fe2o3=95.0),
        _clean_row(sample_id=2),
    ])
    ds = BulkChemistryDataset(
        csv_path=csv, require_all_majors=False, min_class_count=1,
    )
    assert len(ds) == 1
    assert ds[0]["sample_id"] == 2


def test_qap_name_attribute(tmp_path: Path) -> None:
    """qap_names is parallel-indexed to __getitem__ and preserves the label."""
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=1, qap_name="basalt"),
        _clean_row(sample_id=2, qap_name="dunite"),
    ])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    assert ds.qap_names.tolist() == ["basalt", "dunite"]


def test_rock_group_filter_excludes_sedimentary(tmp_path: Path) -> None:
    rows = [_clean_row(sample_id=1)]
    sed = _clean_row(sample_id=2).replace(",igneous,", ",sedimentary,")
    rows.append(sed)
    csv = _write_csv(tmp_path / "tiny.csv", rows)
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    assert len(ds) == 1
    assert ds[0]["sample_id"] == 1


def test_include_cr2o3_toggle(tmp_path: Path) -> None:
    """Default leaves Cr padded; include_cr2o3=True populates it."""
    csv = _write_csv(tmp_path / "tiny.csv", [_clean_row(cr2o3=0.5)])
    cr_token_idx = 8  # MAJORS[8] is Cr

    ds_default = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    assert ds_default[0]["src"][cr_token_idx].item() == 0
    assert ds_default[0]["frac"][cr_token_idx].item() == 0.0

    ds_with_cr = BulkChemistryDataset(csv_path=csv, include_cr2o3=True, min_class_count=1)
    assert ds_with_cr[0]["src"][cr_token_idx].item() == 24
    expected = 0.5 * OXIDE_TO_CATION_FACTOR["cr2o3"] / 100.0
    assert ds_with_cr[0]["frac"][cr_token_idx].item() == pytest.approx(
        expected, rel=1e-5,
    )


def test_trace_conversion_ppm_to_frac(tmp_path: Path) -> None:
    """sr_ppm=500 should yield Sr frac = 500 / 1e6."""
    csv = _write_csv(tmp_path / "tiny.csv", [_clean_row(sr_ppm=500.0)])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    sr_token_idx = 9  # first trace
    assert ds[0]["src"][sr_token_idx].item() == 38
    assert ds[0]["frac"][sr_token_idx].item() == pytest.approx(
        500.0 / 1e6, rel=1e-6,
    )


def test_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BulkChemistryDataset(csv_path=tmp_path / "nope.csv")


# ----- redox token + corpus-build knobs -----


def test_redox_token_populated_iff_measured_pair(tmp_path: Path) -> None:
    """Only rows with both feo and fe2o3 measured populate the redox slot;
    fallback iron sources carry no speciation so they pad it."""
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=1, feo=7.0, fe2o3=2.0),
        _clean_row(sample_id=2, feo=7.0, fe2o3=""),
        _clean_row(sample_id=3, feo="", fe2o3="", feo_tot=9.0),
    ])
    ds = BulkChemistryDataset(
        csv_path=csv, require_all_majors=False, min_class_count=1,
        include_redox=True,
    )
    assert ds[0]["src"].shape == (N_TOKENS_REDOX,)

    fe2 = 7.0 * OXIDE_TO_CATION_FACTOR["feo"]
    fe3 = 2.0 * OXIDE_TO_CATION_FACTOR["fe2o3"]
    assert ds[0]["src"][REDOX_TOKEN_INDEX].item() == REDOX_TOKEN_ID
    assert ds[0]["frac"][REDOX_TOKEN_INDEX].item() == pytest.approx(
        fe3 / (fe2 + fe3), rel=1e-5,
    )
    for i in (1, 2):
        assert ds[i]["src"][REDOX_TOKEN_INDEX].item() == 0
        assert ds[i]["frac"][REDOX_TOKEN_INDEX].item() == 0.0


def test_redox_dataset_obeys_padding_convention(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=1),
        _clean_row(sample_id=2, fe2o3=""),
    ])
    ds = BulkChemistryDataset(
        csv_path=csv, require_all_majors=False, min_class_count=1,
        include_redox=True,
    )
    loader = DataLoader(ds, batch_size=2)
    for batch in loader:
        assert_padding_convention(batch["src"], batch["frac"])


def test_redox_default_off_keeps_22_tokens(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "tiny.csv", [_clean_row()])
    ds = BulkChemistryDataset(csv_path=csv, min_class_count=1)
    assert ds[0]["src"].shape == (N_TOKENS,)


def test_min_populated_tokens_drops_thin_rows(tmp_path: Path) -> None:
    """A row below the floor disappears; a full row survives."""
    thin = _clean_row(
        sample_id=1, al2o3="", cao="", mgo="", feo="", fe2o3="",
        k2o="", na2o="", tio2="", sr_ppm="",
    )
    csv = _write_csv(tmp_path / "tiny.csv", [thin, _clean_row(sample_id=2)])
    ds_all = BulkChemistryDataset(
        csv_path=csv, require_all_majors=False, min_class_count=1,
    )
    assert len(ds_all) == 2
    ds_floor = BulkChemistryDataset(
        csv_path=csv, require_all_majors=False, min_class_count=1,
        min_populated_tokens=4,
    )
    assert len(ds_floor) == 1
    assert ds_floor[0]["sample_id"] == 2


def test_dedup_chemistry_keeps_first(tmp_path: Path) -> None:
    """Exact tokenized duplicates collapse to the earliest row; attributes
    stay index-parallel with the surviving tensors."""
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=1),
        _clean_row(sample_id=2),          # identical chemistry to 1
        _clean_row(sample_id=3, sio2=60.0),
    ])
    ds = BulkChemistryDataset(
        csv_path=csv, min_class_count=1, dedup_chemistry=True,
    )
    assert len(ds) == 2
    assert ds.sample_ids.tolist() == [1, 3]
    assert len(ds.qap_names) == len(ds) == len(ds.rock_groups)
    assert ds[1]["sample_id"] == 3


def test_dedup_prefer_path_wins_duplicate_group(tmp_path: Path) -> None:
    """A row named in dedup_prefer_path survives its duplicate group even
    when it is not the earliest; unnamed groups keep first as usual."""
    csv = _write_csv(tmp_path / "tiny.csv", [
        _clean_row(sample_id=1),
        _clean_row(sample_id=2),            # identical chemistry to 1
        _clean_row(sample_id=3, sio2=60.0),
        _clean_row(sample_id=4, sio2=60.0),  # identical chemistry to 3
    ])
    prefer = tmp_path / "prefer.csv"
    prefer.write_text("sample_id\n2\n")
    ds = BulkChemistryDataset(
        csv_path=csv, min_class_count=1, dedup_chemistry=True,
        dedup_prefer_path=prefer,
    )
    assert ds.sample_ids.tolist() == [2, 3]


def test_strat_labels_composite_and_rare_relabel(tmp_path: Path) -> None:
    """qap where present, else rock_group; a qap class rarer than
    min_class_count relabels to its rock_group instead of dropping rows."""
    def _null_qap(row: str) -> str:
        return row.replace(",basalt,", ",,")

    rows = [
        _clean_row(sample_id=1, qap_name="basalt"),
        _clean_row(sample_id=2, qap_name="basalt"),
        _clean_row(sample_id=3, qap_name="dunite"),  # rare: 1 member
        _null_qap(_clean_row(sample_id=4)),
        _null_qap(_clean_row(sample_id=5)),
        _null_qap(_clean_row(sample_id=6)).replace(",igneous,", ",sedimentary,"),
        _null_qap(_clean_row(sample_id=7)).replace(",igneous,", ",sedimentary,"),
    ]
    csv = _write_csv(tmp_path / "tiny.csv", rows)
    ds = BulkChemistryDataset(
        csv_path=csv, rock_group=None, require_all_majors=False,
        require_qap_name=False, min_class_count=2,
    )
    assert len(ds) == 7
    # dunite (1 member < 2) relabels to its rock_group, which the two
    # null-qap igneous rows keep above the threshold
    assert ds.strat_labels.tolist() == [
        "basalt", "basalt", "igneous", "igneous", "igneous",
        "sedimentary", "sedimentary",
    ]
    assert ds.rock_groups.tolist() == [
        "igneous", "igneous", "igneous", "igneous", "igneous",
        "sedimentary", "sedimentary",
    ]


def test_strat_labels_still_rare_collapse_to_unknown(tmp_path: Path) -> None:
    """A label still below min_class_count after the rock_group collapse
    lands in the catch-all bucket rather than costing the row."""
    rows = [
        _clean_row(sample_id=1, qap_name="basalt"),
        _clean_row(sample_id=2, qap_name="basalt"),
        _clean_row(sample_id=3, qap_name="dunite"),  # rare, lone igneous fallback
        _clean_row(sample_id=4, qap_name="norite").replace(
            ",igneous,", ",metamorphic,",
        ),  # rare, lone metamorphic fallback
    ]
    csv = _write_csv(tmp_path / "tiny.csv", rows)
    ds = BulkChemistryDataset(
        csv_path=csv, rock_group=None, require_all_majors=False,
        require_qap_name=False, min_class_count=2,
    )
    assert len(ds) == 4
    assert ds.strat_labels.tolist() == [
        "basalt", "basalt", "unknown", "unknown",
    ]


# ----- regression pin against the real corpus (skip if absent) -----


@pytest.mark.skipif(not REAL_CSV.exists(), reason="data/bulks/gard2019.csv absent")
def test_real_corpus_default_filter_size() -> None:
    """Strict default filter against the real CSV lands near ~167k rows.

    Pins the filter outcome so a behavior drift surfaces here rather
    than after a long pretrain run.  Bounds are wide enough to absorb
    minor library-version differences in NaN handling.
    """
    ds = BulkChemistryDataset()
    assert 160_000 <= len(ds) <= 175_000, (
        f"default filter row count drifted: got {len(ds)}; "
        f"expected ~167k"
    )
    # all rows must obey the padding convention
    loader = DataLoader(ds, batch_size=128)
    batch = next(iter(loader))
    assert_padding_convention(batch["src"], batch["frac"])


# ----- filter / threshold tests against the realistic fixture -----


def test_require_qap_name_drops_nulls(corpus_subset_path: Path) -> None:
    """Under require_qap_name=True, no surviving row carries a null label."""
    # Use require_all_majors=False so the null-qap rows aren't filtered
    # out by the major-oxide cut before we get to test the qap filter.
    ds_strict = BulkChemistryDataset(
        csv_path=corpus_subset_path, require_all_majors=False,
        require_qap_name=True, min_class_count=1,
    )
    assert sum(q is None for q in ds_strict.qap_names) == 0

    ds_loose = BulkChemistryDataset(
        csv_path=corpus_subset_path, require_all_majors=False,
        require_qap_name=False, min_class_count=1,
    )
    n_null = sum(q is None for q in ds_loose.qap_names)
    assert n_null > 0, "fixture should carry null-qap rows for this test"
    assert len(ds_strict) == len(ds_loose) - n_null


def test_min_class_count_drops_below_threshold(corpus_subset_path: Path) -> None:
    """A class with fewer than `min_class_count` samples is absent
    from the constructed dataset; a class at or above survives."""
    baseline = BulkChemistryDataset(
        csv_path=corpus_subset_path, min_class_count=1,
    )
    from collections import Counter
    baseline_counts = Counter(baseline.qap_names)
    threshold = 5
    expected_drop = {q for q, c in baseline_counts.items() if c < threshold}
    expected_keep = {q for q, c in baseline_counts.items() if c >= threshold}
    assert expected_drop and expected_keep, (
        "fixture should have classes on both sides of threshold"
    )

    filtered = BulkChemistryDataset(
        csv_path=corpus_subset_path, min_class_count=threshold,
    )
    surviving = set(filtered.qap_names)
    assert surviving.isdisjoint(expected_drop)
    assert surviving == expected_keep


def test_filtered_dataset_passes_padding_convention(
    corpus_subset_path: Path,
) -> None:
    """End-to-end check on the realistic fixture under production-shape filters."""
    ds = BulkChemistryDataset(csv_path=corpus_subset_path, min_class_count=5)
    loader = DataLoader(ds, batch_size=64)
    for batch in loader:
        assert_padding_convention(batch["src"], batch["frac"])


def test_fixture_size_regression_pin(corpus_subset_path: Path) -> None:
    """Pin len(ds) for the strict-default filter on the fixture.

    Detects unintended changes to the fixture, the filter chain, or
    pandas NaN-handling that would otherwise surface only as a downstream
    stratification error.
    """
    ds_strict = BulkChemistryDataset(
        csv_path=corpus_subset_path, min_class_count=1,
    )
    assert len(ds_strict) == 456, (
        f"strict-filter fixture size drifted: got {len(ds_strict)}; "
        f"expected 456.  Regenerate via scripts/build/build_test_fixture.py "
        f"or update this pin if the change is intended."
    )


def test_petrological_axes_returns_finite_values(
    corpus_subset_path: Path,
) -> None:
    """SiO2/Mg#/alkalis reconstruction recovers wt% in a sane range."""
    ds = BulkChemistryDataset(
        csv_path=corpus_subset_path, min_class_count=2,
    )
    if len(ds) == 0:
        pytest.skip("fixture has no rows after filter")
    axes = ds.petrological_axes()
    assert set(axes.keys()) == {"SiO2 (wt%)", "Mg#", "Na2O+K2O (wt%)"}
    sio2 = axes["SiO2 (wt%)"]
    finite = np.isfinite(sio2)
    assert finite.any()
    # SiO2 wt% in nature lives in roughly 35-80 for igneous rocks; the
    # fixture is a 2000-row random sample so the median should land in
    # that band even if individual outlier rows fall outside.
    assert 30 <= float(np.median(sio2[finite])) <= 80
    assert sio2[finite].max() <= 100
    mg_num = axes["Mg#"]
    finite_mg = np.isfinite(mg_num)
    assert ((mg_num[finite_mg] >= 0) & (mg_num[finite_mg] <= 1)).all()


def test_petrological_axes_indices_subset_shape(
    corpus_subset_path: Path,
) -> None:
    """Passing `indices` returns arrays sized to the subset, not the full dataset."""
    ds = BulkChemistryDataset(
        csv_path=corpus_subset_path, min_class_count=2,
    )
    if len(ds) < 3:
        pytest.skip("fixture too small for subset test")
    indices = np.array([0, 1, 2], dtype=np.int64)
    axes = ds.petrological_axes(indices)
    for key, arr in axes.items():
        assert arr.shape == (3,), f"{key} has shape {arr.shape}; expected (3,)"

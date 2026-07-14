"""Tripwire tests for the LEPR extraction.

Locks the pure logic the validation table rides on: the phase-code vocabulary
(coarse map and drop set stay disjoint and complete for the codes we claim to
handle), phase-string parsing, and the duplicate-experiment / conflicting-bulk
dedup. The Excel leg needs the gitignored LEPR download and is exercised by the
CLI run, not here.
"""
import pandas as pd

from explorations.lepr_validation.extract_lepr import (
    COARSE_MAP,
    DROP_CODES,
    dedup_bulk,
    dedup_experiments,
    parse_phases,
)


def test_coarse_map_and_drop_codes_disjoint():
    assert not set(COARSE_MAP) & DROP_CODES


def test_parse_phases_splits_and_strips():
    assert parse_phases("liq, plag ,cpx") == ["liq", "plag", "cpx"]
    assert parse_phases(float("nan")) == ["nan"]  # raw string of NaN; filtered upstream by mapping


def test_dedup_experiments_drops_conflicting_keys():
    exp = pd.DataFrame({
        "Citation": ["a", "a", "b", "b"],
        "Experiment": [1, 1, 2, 2],
        "T (C)": [1000, 1100, 900, 900],
        "P (GPa)": [0.5, 0.5, 0.2, 0.2],
        "Phases": ["liq,plag", "liq,plag", "liq,cpx", "liq,cpx"],
    })
    kept, n_conflict = dedup_experiments(exp)
    # (a, 1) has two distinct T -> conflicting, dropped; (b, 2) is an exact dup -> kept once
    assert n_conflict == 1
    assert list(zip(kept.Citation, kept.Experiment)) == [("b", 2)]


def test_dedup_bulk_flags_disagreeing_duplicates():
    bulk = pd.DataFrame({
        "Citation": ["a", "a", "b", "b"],
        "Experiment": [1, 1, 2, 2],
        "SiO2 value": [55.0, 55.5, 60.0, 60.005],
    })
    deduped, conflicts = dedup_bulk(bulk)
    # (a, 1) disagrees by 0.5 wt% -> conflict; (b, 2) within the 0.01 tolerance -> clean
    assert conflicts == {("a", 1)}
    assert len(deduped) == 2
    assert deduped.set_index(["Citation", "Experiment"]).bulk_conflict.to_dict() == {
        ("a", 1): True, ("b", 2): False}

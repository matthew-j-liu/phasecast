"""Tripwire tests for the bulk -> MAGEMin oxide conversion + Fe provenance tiers.

Locks the contract the corpus generator depends on: total iron is split by the
right tier, H2O is injected (not read from the bulk), MnO/P2O5 are dropped, and
the LOI renormalization is exact.
"""
import pandas as pd

from cnmineral.data.fe_redox import ClassRatioStrategy
from cnmineral.data.magemin_oxides import (
    FE2O3_TO_FEO_MASS,
    bulk_to_magemin,
    loi_normalize,
    split_fe,
)

FE3 = ClassRatioStrategy()


def test_split_fe_measured_tier_passes_values_through():
    row = pd.Series(dict(feo=6.0, fe2o3=1.5))
    s = split_fe(row, "basalt", FE3)
    assert s.provenance == "measured"
    assert (s.feo, s.fe2o3) == (6.0, 1.5)


def test_split_fe_assumed_class_applies_ratio():
    row = pd.Series(dict(feo_tot=10.0))
    r = FE3("andesite")
    s = split_fe(row, "andesite", FE3)
    assert s.provenance == "assumed_class"
    assert abs(s.feo - (1 - r) * 10.0) < 1e-9
    assert abs(s.fe2o3 - r * 10.0 / FE2O3_TO_FEO_MASS) < 1e-9


def test_split_fe_derived_partial_from_measured_feo_plus_total():
    row = pd.Series(dict(feo=5.0, fe2o3_tot=8.0))
    s = split_fe(row, "dacite", FE3)
    assert s.provenance == "derived_partial"
    assert s.feo == 5.0
    assert abs(s.fe2o3 - (8.0 - 5.0 / FE2O3_TO_FEO_MASS)) < 1e-9


def test_split_fe_no_iron_returns_zeros():
    s = split_fe(pd.Series(dict(sio2=60.0)), "rhyolite", FE3)
    assert s.provenance == "no_iron"
    assert (s.feo, s.fe2o3) == (0.0, 0.0)


def test_bulk_to_magemin_injects_h2o_and_drops_mno_p2o5():
    row = pd.Series(dict(sio2=57.0, al2o3=16.0, cao=7.0, mgo=4.0, feo_tot=7.0,
                         k2o=1.0, na2o=4.0, tio2=1.0, mno=0.2, p2o5=0.3))
    out = bulk_to_magemin(row, "andesite", h2o_wt=3.5)
    assert out["h2o_tot"] == 3.5
    assert "mno" not in out and "p2o5" not in out
    assert out["cr2o3"] == 0.0            # missing -> 0
    assert out["fe_provenance"] == "assumed_class"


def test_loi_normalize_is_exact_and_passes_strings():
    out = loi_normalize({"sio2": 50.0, "al2o3": 25.0, "fe_provenance": "measured"})
    numeric = sum(v for v in out.values() if isinstance(v, float))
    assert abs(numeric - 100.0) < 1e-9
    assert out["fe_provenance"] == "measured"


def test_class_ratio_lookup_and_default():
    assert ClassRatioStrategy()("basalt") == 0.15
    assert ClassRatioStrategy()("rhyolite") == 0.45
    assert ClassRatioStrategy()("not a rock") == ClassRatioStrategy().ratios["default"]

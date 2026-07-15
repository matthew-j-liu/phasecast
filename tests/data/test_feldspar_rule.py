import numpy as np

from cnmineral.data.feldspar_rule import anorthite_orthoclase, classify_feldspar

_M_CAO = 56.0774
_M_NA2O = 61.9789
_M_K2O = 94.1960


def _wt(an, ab, orr):
    """Oxide mass fractions whose cation-molar ternary is exactly (an, ab, orr)."""
    return an * _M_CAO, ab * _M_NA2O / 2, orr * _M_K2O / 2


def test_ternary_coordinates_roundtrip():
    an, orr = anorthite_orthoclase(*_wt(0.6, 0.38, 0.02))
    assert np.isclose(an, 0.6) and np.isclose(orr, 0.02)


def test_labradorite_is_plag():
    assert classify_feldspar(*_wt(0.6, 0.38, 0.02))[0] == "plag"


def test_albite_is_plag():
    assert classify_feldspar(*_wt(0.05, 0.93, 0.02))[0] == "plag"


def test_anorthite_is_plag():
    assert classify_feldspar(*_wt(1.0, 0.0, 0.0))[0] == "plag"


def test_sanidine_is_afs():
    assert classify_feldspar(*_wt(0.02, 0.38, 0.60))[0] == "afs"


def test_anorthoclase_is_afs():
    assert classify_feldspar(*_wt(0.05, 0.75, 0.20))[0] == "afs"


def test_miscibility_gap_routes_by_dominant_component():
    assert classify_feldspar(*_wt(0.35, 0.40, 0.25))[0] == "plag"
    assert classify_feldspar(*_wt(0.20, 0.50, 0.30))[0] == "afs"


def test_degenerate_zero_alkalis_is_plag():
    assert classify_feldspar(0.0, 0.0, 0.0)[0] == "plag"


def test_nan_treated_as_zero():
    assert classify_feldspar(1.0 * _M_CAO, np.nan, np.nan)[0] == "plag"


def test_vectorized_input():
    cao, na2o, k2o = zip(*[_wt(0.6, 0.38, 0.02), _wt(0.02, 0.38, 0.60)])
    out = classify_feldspar(np.array(cao), np.array(na2o), np.array(k2o))
    assert out.tolist() == ["plag", "afs"]

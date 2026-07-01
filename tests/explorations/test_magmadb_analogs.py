"""Tripwire test for the `magmadb_analogs` exploration.

Locks the MagmaDB starting-bulk tokenization contract: 8 majors populated
(Cr padded to match production `include_cr2o3=False`), all 13 traces padded,
Fe blended from FeO + Fe2O3 like `_resolve_iron`.  Doubles as a tripwire -- a
`src/` change to `OXIDE_TO_CATION_FACTOR` / the token vocabulary fails here.
The full embed + retrieval pipeline needs a real checkpoint and the external
ArcThermoML CSV, so it is exercised by the CLI run, not here.
"""

import pandas as pd

from cnmineral.data.bulk_chemistry import OXIDE_TO_CATION_FACTOR
from cnmineral.data.tokens import MAJORS, N_TOKENS, TRACES
from explorations.magmadb_analogs.run_analogs import magmadb_tokens

_N_MAJORS = len(MAJORS)
_CR_POS = next(i for i, (sym, _z) in enumerate(MAJORS) if sym == "Cr")
_FE_POS = next(i for i, (sym, _z) in enumerate(MAJORS) if sym == "Fe")
_SI_POS = next(i for i, (sym, _z) in enumerate(MAJORS) if sym == "Si")


def _row() -> pd.DataFrame:
    return pd.DataFrame([{
        "SiO2_start": 52.0, "TiO2_start": 0.8, "Al2O3_start": 17.0,
        "FeO_start": 7.0, "Fe2O3_start": 1.0, "MgO_start": 5.0,
        "CaO_start": 9.0, "Na2O_start": 3.0, "K2O_start": 1.0,
    }])


def test_eight_majors_populated_cr_and_traces_padded():
    src, frac = magmadb_tokens(_row())
    assert src.shape == (1, N_TOKENS)
    assert int((src[0] > 0).sum()) == 8  # 9 majors minus Cr
    assert src[0, _CR_POS].item() == 0
    for trace_offset in range(len(TRACES)):
        assert src[0, _N_MAJORS + trace_offset].item() == 0
    assert frac[0, _CR_POS].item() == 0.0


def test_si_frac_is_cation_wt_pct_over_100():
    _src, frac = magmadb_tokens(_row())
    expected = 52.0 * OXIDE_TO_CATION_FACTOR["sio2"] / 100.0
    assert abs(frac[0, _SI_POS].item() - expected) < 1e-6


def test_fe_blends_feo_and_fe2o3():
    _src, frac = magmadb_tokens(_row())
    expected = (7.0 * OXIDE_TO_CATION_FACTOR["feo"]
                + 1.0 * OXIDE_TO_CATION_FACTOR["fe2o3"]) / 100.0
    assert abs(frac[0, _FE_POS].item() - expected) < 1e-6

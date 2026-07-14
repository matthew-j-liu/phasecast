"""Tests for the volcanic + has-iron subset builder.

Locks the pure logic production corpus tables ride on: the total-Fe-as-FeO
column precedence and the QAPF vocabulary invariants. `build_subset` itself
needs the gard2019 parquet and is exercised by the bulk-table CLI run, not here.
"""
import numpy as np
import pandas as pd

from cnmineral.data.magemin_oxides import FE2O3_TO_FEO_MASS
from cnmineral.data.volcanic_subset import ALKALINE_QAP, VOLCANIC_QAP, total_fe_as_feo


def _fe_frame(**cols):
    base = {"feo": [np.nan], "fe2o3": [np.nan], "feo_tot": [np.nan], "fe2o3_tot": [np.nan]}
    base.update({k: [v] for k, v in cols.items()})
    return pd.DataFrame(base)


def test_measured_feo_and_fe2o3_take_precedence():
    df = _fe_frame(feo=5.0, fe2o3=2.0, feo_tot=99.0)
    assert total_fe_as_feo(df).iloc[0] == 5.0 + 2.0 * FE2O3_TO_FEO_MASS


def test_single_measured_species_counts_alone():
    assert total_fe_as_feo(_fe_frame(feo=5.0)).iloc[0] == 5.0
    assert total_fe_as_feo(_fe_frame(fe2o3=2.0)).iloc[0] == 2.0 * FE2O3_TO_FEO_MASS


def test_feo_tot_fallback_then_fe2o3_tot():
    assert total_fe_as_feo(_fe_frame(feo_tot=7.0, fe2o3_tot=99.0)).iloc[0] == 7.0
    assert total_fe_as_feo(_fe_frame(fe2o3_tot=8.0)).iloc[0] == 8.0 * FE2O3_TO_FEO_MASS


def test_no_iron_is_nan():
    assert np.isnan(total_fe_as_feo(_fe_frame()).iloc[0])


def test_qap_vocabulary_invariants():
    assert len(VOLCANIC_QAP) == 25
    assert len(set(VOLCANIC_QAP)) == 25
    assert ALKALINE_QAP < set(VOLCANIC_QAP)   # proper subset; the rest is subalkaline

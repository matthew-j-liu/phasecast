from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cnmineral.data.magemin_corpus import COMP_OXIDES
from cnmineral.data.magemin_oxides import FE2O3_TO_FEO_MASS
from cnmineral.data.mineralogy_labels import (
    hurdle_targets,
    load_joint_labels,
    load_mode_labels,
)
from cnmineral.data.phase_vocab import PHASE_INDEX, SOLID_PHASES


def _write(corpus_dir, rows):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(corpus_dir / "chunk_0000_rungs.parquet")
    return corpus_dir


_ZERO = {"wt_CaO": 0.0, "wt_Na2O": 0.0, "wt_K2O": 0.0}
_PLAG = {"wt_CaO": 33.6, "wt_Na2O": 11.8, "wt_K2O": 0.9}
_AFS = {"wt_CaO": 1.1, "wt_Na2O": 11.8, "wt_K2O": 28.3}


@pytest.fixture
def corpus_dir(tmp_path):
    return _write(tmp_path / "corpus", [
        dict(bulk=1, draw=0, rung=50, phase="liq", instance=0, mode_wt=0.50, **_ZERO),
        dict(bulk=1, draw=0, rung=50, phase="fsp", instance=0, mode_wt=0.30, **_PLAG),
        dict(bulk=1, draw=0, rung=50, phase="fsp", instance=1, mode_wt=0.10, **_AFS),
        dict(bulk=1, draw=0, rung=50, phase="q", instance=0, mode_wt=0.05, **_ZERO),
        dict(bulk=1, draw=0, rung=50, phase="trd", instance=0, mode_wt=0.05, **_ZERO),
        dict(bulk=1, draw=0, rung=30, phase="liq", instance=0, mode_wt=0.28, **_ZERO),
        dict(bulk=1, draw=0, rung=30, phase="fl", instance=0, mode_wt=0.02, **_ZERO),
        dict(bulk=1, draw=0, rung=30, phase="cpx", instance=0, mode_wt=0.60, **_ZERO),
        dict(bulk=1, draw=0, rung=30, phase="chl", instance=0, mode_wt=0.10, **_ZERO),
        dict(bulk=1, draw=0, rung=40, phase="q", instance=0, mode_wt=1.00, **_ZERO),
        dict(bulk=1, draw=1, rung=50, phase="liq", instance=0, mode_wt=0.60, **_ZERO),
        dict(bulk=1, draw=1, rung=50, phase="amp", instance=0, mode_wt=0.20, **_ZERO),
        dict(bulk=1, draw=1, rung=50, phase="amp", instance=1, mode_wt=0.20, **_ZERO),
    ])


def test_one_row_per_reached_rung(corpus_dir):
    labels = load_mode_labels(corpus_dir)
    keys = set(map(tuple, labels[["bulk", "draw", "rung"]].to_numpy()))
    assert keys == {(1, 0, 50), (1, 0, 30), (1, 1, 50)}
    assert list(labels.columns[3:]) == list(SOLID_PHASES)


def test_crystals_renormalization_and_fsp_routing(corpus_dir):
    labels = load_mode_labels(corpus_dir).set_index(["draw", "rung"])
    r50 = labels.loc[(0, 50)]
    assert np.isclose(r50["plag"], 0.6)
    assert np.isclose(r50["afs"], 0.2)
    assert np.isclose(r50["q"], 0.2)  # q 0.05 + trd 0.05, renormalized


def test_dropped_phase_mass_renormalizes_away(corpus_dir):
    labels = load_mode_labels(corpus_dir).set_index(["draw", "rung"])
    r30 = labels.loc[(0, 30)]
    assert np.isclose(r30["cpx"], 1.0)  # chl dropped, liq/fl excluded
    assert np.isclose(r30[list(SOLID_PHASES)].sum(), 1.0)


def test_instances_sum(corpus_dir):
    labels = load_mode_labels(corpus_dir).set_index(["draw", "rung"])
    assert np.isclose(labels.loc[(1, 50)]["amp"], 1.0)


def test_rung_filter(corpus_dir):
    labels = load_mode_labels(corpus_dir, rungs=(40,))
    assert len(labels) == 1
    assert np.isclose(labels.iloc[0]["q"], 1.0)


def test_unknown_phase_raises(tmp_path):
    corpus = _write(tmp_path / "bad", [
        dict(bulk=1, draw=0, rung=50, phase="wumbo", instance=0, mode_wt=1.0, **_ZERO),
    ])
    with pytest.raises(ValueError, match="wumbo"):
        load_mode_labels(corpus)


def test_missing_corpus_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mode_labels(tmp_path / "nowhere")


def _comp(**named):
    """Full wt_ column dict; `named` sets specific oxides (mass fractions)."""
    out = {f"wt_{o}": 0.0 for o in COMP_OXIDES}
    for k, v in named.items():
        out[f"wt_{k}"] = v
    return out


def _write_joint(tmp_path, rung_rows, draw_rows):
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rung_rows).to_parquet(corpus / "chunk_0000_rungs.parquet")
    pd.DataFrame(draw_rows).to_parquet(corpus / "chunk_0000_draws.parquet")
    return corpus


_DRAW_DEFAULTS = dict(
    bulk=1, draw=0, P_kbar=5.0, H2O=2.0, in_sio2=60.0, in_al2o3=15.0,
    in_cao=6.0, in_mgo=4.0, in_feo=6.0, in_fe2o3=2.0, in_k2o=2.0,
    in_na2o=2.0, in_tio2=0.8, in_cr2o3=0.2,
)


def _joint_corpus(tmp_path):
    plag = _comp(SiO2=0.55, Al2O3=0.28, CaO=0.12, Na2O=0.04, K2O=0.001, FeO=0.009)
    afs = _comp(SiO2=0.66, Al2O3=0.19, K2O=0.10, Na2O=0.03, CaO=0.005, FeO=0.015)
    rung_rows = [
        dict(bulk=1, draw=0, rung=50, T_end=900.0, phase="liq", instance=0,
             mode_wt=0.38, **_comp(SiO2=0.7, Al2O3=0.2, H2O=0.1)),
        dict(bulk=1, draw=0, rung=50, T_end=900.0, phase="fl", instance=1,
             mode_wt=0.02, **_comp(SiO2=0.1, H2O=0.9)),
        dict(bulk=1, draw=0, rung=50, T_end=900.0, phase="fsp", instance=2,
             mode_wt=0.30, **plag),
        dict(bulk=1, draw=0, rung=50, T_end=900.0, phase="fsp", instance=3,
             mode_wt=0.10, **afs),
        # two amp instances with different comps: mass-weighted merge target
        dict(bulk=1, draw=0, rung=50, T_end=900.0, phase="amp", instance=4,
             mode_wt=0.15, **_comp(SiO2=0.4, MgO=0.3, FeO=0.3)),
        dict(bulk=1, draw=0, rung=50, T_end=900.0, phase="amp", instance=5,
             mode_wt=0.05, **_comp(SiO2=0.6, MgO=0.2, FeO=0.2)),
        # trd folds into q's comp by the same mass weighting
        dict(bulk=1, draw=1, rung=50, T_end=850.0, phase="q", instance=0,
             mode_wt=0.30, **_comp(SiO2=1.0)),
        dict(bulk=1, draw=1, rung=50, T_end=850.0, phase="trd", instance=1,
             mode_wt=0.10, **_comp(SiO2=0.9, Al2O3=0.1)),
        dict(bulk=1, draw=1, rung=50, T_end=850.0, phase="cpx", instance=2,
             mode_wt=0.60, **_comp(SiO2=0.5, CaO=0.25, MgO=0.25)),
    ]
    draw_rows = [
        dict(_DRAW_DEFAULTS),
        dict(_DRAW_DEFAULTS, draw=1, P_kbar=2.0, H2O=1.0),
    ]
    return _write_joint(tmp_path, rung_rows, draw_rows)


def test_joint_labels_align_with_mode_labels(tmp_path):
    corpus = _joint_corpus(tmp_path)
    j = load_joint_labels(corpus)
    m = load_mode_labels(corpus).sort_values(["bulk", "draw", "rung"])
    order = np.lexsort(
        (j["index"]["rung"], j["index"]["draw"], j["index"]["bulk"])
    )
    assert np.allclose(
        m.drop(columns=["bulk", "draw", "rung"]).to_numpy(), j["modes"][order]
    )


def test_joint_comp_mass_weighted_instance_merge(tmp_path):
    j = load_joint_labels(_joint_corpus(tmp_path))
    row = int(np.flatnonzero(j["index"]["draw"] == 0)[0])
    amp = j["comp"][row, PHASE_INDEX["amp"]]
    # 0.15 * (0.4, 0.3, 0.3) + 0.05 * (0.6, 0.2, 0.2), mass-normalized
    expect = (0.15 * np.array([0.4, 0.3, 0.3]) + 0.05 * np.array([0.6, 0.2, 0.2])) / 0.2
    got = amp[[COMP_OXIDES.index(o) for o in ("SiO2", "MgO", "FeO")]]
    assert np.allclose(got, expect, atol=1e-3)


def test_joint_feldspar_pair_not_averaged(tmp_path):
    j = load_joint_labels(_joint_corpus(tmp_path))
    row = int(np.flatnonzero(j["index"]["draw"] == 0)[0])
    cao = COMP_OXIDES.index("CaO")
    assert j["comp"][row, PHASE_INDEX["plag"], cao] > 0.1
    assert j["comp"][row, PHASE_INDEX["afs"], cao] < 0.01


def test_joint_trd_comp_merges_into_q(tmp_path):
    j = load_joint_labels(_joint_corpus(tmp_path))
    row = int(np.flatnonzero(j["index"]["draw"] == 1)[0])
    q = j["comp"][row, PHASE_INDEX["q"]]
    # 0.30 * pure SiO2 + 0.10 * (0.9 SiO2, 0.1 Al2O3), mass-normalized
    assert np.isclose(q[COMP_OXIDES.index("SiO2")], 0.975, atol=1e-3)
    assert np.isclose(q[COMP_OXIDES.index("Al2O3")], 0.025, atol=1e-3)


def test_joint_comp_targets_are_eps_clipped_simplexes(tmp_path):
    j = load_joint_labels(_joint_corpus(tmp_path), comp_eps=1e-4)
    assert np.allclose(j["comp"].sum(-1), 1.0)
    assert j["comp"].min() > 0


def test_joint_fixings_and_absent_fluid(tmp_path):
    j = load_joint_labels(_joint_corpus(tmp_path))
    d0 = int(np.flatnonzero(j["index"]["draw"] == 0)[0])
    d1 = int(np.flatnonzero(j["index"]["draw"] == 1)[0])
    assert np.isclose(j["f_liq"][d0], 0.38) and np.isclose(j["f_fl"][d0], 0.02)
    assert np.isclose(j["c_fl"][d0, COMP_OXIDES.index("H2O")], 0.9)
    assert j["f_liq"][d1] == 0 and j["f_fl"][d1] == 0
    assert np.all(j["c_liq"][d1] == 0) and np.all(j["c_fl"][d1] == 0)


def test_joint_cond_targets(tmp_path):
    j = load_joint_labels(_joint_corpus(tmp_path))
    d0 = int(np.flatnonzero(j["index"]["draw"] == 0)[0])
    d1 = int(np.flatnonzero(j["index"]["draw"] == 1)[0])
    assert np.allclose(j["cond"][d0], [900.0, 5.0, 2.0])
    assert np.allclose(j["cond"][d1], [850.0, 2.0, 1.0])


def test_joint_bulk_11_fe_split(tmp_path):
    j = load_joint_labels(_joint_corpus(tmp_path))
    d = _DRAW_DEFAULTS
    feo_total = d["in_feo"] + d["in_fe2o3"] * FE2O3_TO_FEO_MASS
    excess_o = d["in_fe2o3"] * (1 - FE2O3_TO_FEO_MASS)
    total = sum(
        d[k] for k in d if k.startswith("in_") and k != "in_fe2o3"
    ) + d["in_fe2o3"] + d["H2O"]
    row = int(np.flatnonzero(j["index"]["draw"] == 0)[0])
    b = j["bulk_11"][row]
    assert np.isclose(b[COMP_OXIDES.index("FeO")], feo_total / total)
    assert np.isclose(b[COMP_OXIDES.index("O")], excess_o / total)
    assert np.isclose(b.sum(), 1.0)


REAL_VALTEST = Path(__file__).resolve().parents[2] / "data" / "corpus" / "valtest"


@pytest.mark.skipif(
    not REAL_VALTEST.exists(), reason="real valtest corpus not on this machine"
)
def test_joint_labels_close_on_real_corpus():
    """Mass-balance identity through the builder's own outputs.

    Quantile thresholds, not a max: a small tail of corpus rows does not
    self-close (multi-instance melt immiscibility, dropped-phase mass).
    """
    j = load_joint_labels(REAL_VALTEST)
    solids = np.einsum("np,npo->no", j["modes"], j["comp"])
    recon = (
        (1 - j["f_liq"] - j["f_fl"])[:, None] * solids
        + j["f_liq"][:, None] * j["c_liq"]
        + j["f_fl"][:, None] * j["c_fl"]
    )
    resid = np.abs(recon - j["bulk_11"]).max(axis=1)
    assert np.median(resid) < 5e-4
    assert np.quantile(resid, 0.99) < 5e-3


def test_hurdle_targets_cutoff_and_renorm():
    modes = np.array([[0.7, 0.2995, 0.0005]])
    presence, magnitude = hurdle_targets(modes, presence_cutoff=1e-3)
    assert presence.tolist() == [[True, True, False]]
    assert np.allclose(magnitude, [[0.7 / 0.9995, 0.2995 / 0.9995, 0.0]])


def test_hurdle_targets_single_present():
    presence, magnitude = hurdle_targets(np.array([[1.0, 0.0]]), presence_cutoff=1e-3)
    assert presence.tolist() == [[True, False]]
    assert np.allclose(magnitude, [[1.0, 0.0]])

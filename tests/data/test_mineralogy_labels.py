import numpy as np
import pandas as pd
import pytest

from cnmineral.data.mineralogy_labels import hurdle_targets, load_mode_labels
from cnmineral.data.phase_vocab import SOLID_PHASES


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


def test_hurdle_targets_cutoff_and_renorm():
    modes = np.array([[0.7, 0.2995, 0.0005]])
    presence, magnitude = hurdle_targets(modes, presence_cutoff=1e-3)
    assert presence.tolist() == [[True, True, False]]
    assert np.allclose(magnitude, [[0.7 / 0.9995, 0.2995 / 0.9995, 0.0]])


def test_hurdle_targets_single_present():
    presence, magnitude = hurdle_targets(np.array([[1.0, 0.0]]), presence_cutoff=1e-3)
    assert presence.tolist() == [[True, False]]
    assert np.allclose(magnitude, [[1.0, 0.0]])

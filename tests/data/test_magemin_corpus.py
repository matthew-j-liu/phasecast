import json

import numpy as np
import pandas as pd
import pytest

from cnmineral.data.condition_proposal import bulk_proposal, w_max
from cnmineral.data.magemin_corpus import T_LADDER, build_inputs, parse_chunk


def _rec(sid, liq, extra=None, status=0, G=-1000.0, ss=None):
    phases = [dict(name="liq", mode_wt=liq, comp_wt={"SiO2": 60.0})] if liq else []
    phases += extra or [dict(name="fsp", mode_wt=1.0 - liq, comp_wt={"SiO2": 55.0})]
    return dict(sid=sid, P=5.0, T=1000.0, status=status, G=G,
                phases=phases, ss=ss or [])


def _write(tmp_path, recs):
    path = tmp_path / "out.jsonl"
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return path


def _meta(n, bulk=1, draw=0):
    ts = [1100.0 - 20 * i for i in range(n)]
    return pd.DataFrame(dict(sid=range(1, n + 1), bulk=bulk, draw=draw, T_celsius=ts))


def test_ladder_truncates_at_stop_melt(tmp_path):
    liqs = [0.8, 0.5, 0.25, 0.1]
    path = _write(tmp_path, [_rec(i + 1, l) for i, l in enumerate(liqs)])
    ladder, _, stats = parse_chunk(path, _meta(4))
    assert sorted(ladder.T_celsius.unique(), reverse=True) == [1100.0, 1080.0, 1060.0]
    assert stats.T_stop.iloc[0] == 1060.0


def test_rung_pick_is_highest_T_below_target(tmp_path):
    liqs = [0.8, 0.55, 0.45, 0.25]
    path = _write(tmp_path, [_rec(i + 1, l) for i, l in enumerate(liqs)])
    _, rungs, stats = parse_chunk(path, _meta(4))
    ends = rungs.groupby("rung").T_end.first()
    assert ends[60] == 1080.0 and ends[50] == 1060.0
    assert ends[40] == 1040.0 and ends[30] == 1040.0
    assert not any(stats[f"attrition_{m}"].iloc[0] for m in (60, 50, 40, 30))


def test_attrition_when_melt_never_drops(tmp_path):
    liqs = [0.8, 0.7, 0.65, 0.62]
    path = _write(tmp_path, [_rec(i + 1, l) for i, l in enumerate(liqs)])
    ladder, rungs, stats = parse_chunk(path, _meta(4))
    assert rungs.empty
    assert all(stats[f"attrition_{m}"].iloc[0] for m in (60, 50, 40, 30))
    assert len(ladder.T_celsius.unique()) == 4   # kept to the floor
    assert stats.T_stop.iloc[0] == 1040.0


def test_failed_points_skipped_and_counted(tmp_path):
    recs = [_rec(1, 0.8), _rec(2, 0.5, status=1), _rec(3, 0.25)]
    path = _write(tmp_path, recs)
    ladder, _, stats = parse_chunk(path, _meta(3))
    assert stats.n_failed.iloc[0] == 1
    assert 1080.0 not in ladder.T_celsius.values


def test_oversaturated_tag_from_hottest_point(tmp_path):
    fl = [dict(name="fl", mode_wt=0.02, comp_wt={"H2O": 100.0}),
          dict(name="fsp", mode_wt=0.18, comp_wt={"SiO2": 55.0})]
    recs = [_rec(1, 0.8, extra=fl), _rec(2, 0.25)]
    path = _write(tmp_path, recs)
    _, _, stats = parse_chunk(path, _meta(2))
    assert bool(stats.oversaturated.iloc[0])


def test_em_json_matches_duplicate_instances_in_order(tmp_path):
    phases = [dict(name="fsp", mode_wt=0.4, comp_wt={"K2O": 10.0}),
              dict(name="fsp", mode_wt=0.3, comp_wt={"Na2O": 8.0}),
              dict(name="q", mode_wt=0.05, comp_wt={"SiO2": 100.0})]
    ss = [dict(name="fsp", emNames=["ab", "san"], emFrac_wt=[0.2, 0.8]),
          dict(name="fsp", emNames=["ab", "san"], emFrac_wt=[0.9, 0.1])]
    recs = [_rec(1, 0.25, extra=phases, ss=ss)]
    _, rungs, _ = parse_chunk(_write(tmp_path, recs), _meta(1))
    fsp = rungs[(rungs.rung == 30) & (rungs.phase == "fsp")].sort_values("instance")
    assert json.loads(fsp.em_json.iloc[0])["frac"] == [0.2, 0.8]
    assert json.loads(fsp.em_json.iloc[1])["frac"] == [0.9, 0.1]
    q = rungs[(rungs.rung == 30) & (rungs.phase == "q")]
    assert q.em_json.iloc[0] is None


@pytest.fixture
def tiny_bulks():
    bulks = pd.DataFrame(dict(sample_id=[7, 11], qap_name=["andesite", "rhyolite"]))
    majors = pd.DataFrame(
        dict(sample_id=[7, 11], sio2=[58.0, 72.0], al2o3=[16.0, 14.0],
             cao=[7.0, 2.0], mgo=[4.0, 0.5], feo=[6.0, 2.0], fe2o3=[1.5, 0.5],
             k2o=[1.5, 4.0], na2o=[3.5, 3.5], tio2=[1.0, 0.3], cr2o3=[0.01, 0.0],
             sr_ppm=[400.0, 100.0])).set_index("sample_id")
    return bulks, majors


def test_build_inputs_h2o_injection_sums_to_100(tiny_bulks):
    bulks, majors = tiny_bulks
    inputs, meta, draws = build_inputs(bulks, majors, n_draws=4, seed=1)
    assert len(inputs) == 2 * 4 * len(T_LADDER)
    ox = inputs[["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
                 "k2o", "na2o", "tio2", "cr2o3", "h2o_tot"]].sum(axis=1)
    assert np.allclose(ox, 100.0)


def test_build_inputs_q_and_bounds(tiny_bulks):
    bulks, majors = tiny_bulks
    _, _, draws = build_inputs(bulks, majors, n_draws=8, seed=1)
    for bid, qap in [(7, "andesite"), (11, "rhyolite")]:
        d = draws[draws.bulk == bid]
        row = majors.loc[bid]
        bp = bulk_proposal(qap, dict(sr_ppm=float(row.sr_ppm),
                                     tio2=float(row.tio2)))
        assert np.allclose(d.q, bp.q(d.P_kbar.values, d.H2O.values))
    assert (draws.H2O.values <= w_max(draws.P_kbar.values) + 1e-9).all()
    assert draws.sr_ppm.tolist() == [400.0] * 8 + [100.0] * 8


def test_build_inputs_stamps_proposal_routing(tiny_bulks):
    bulks, majors = tiny_bulks
    majors = majors.copy()
    majors.loc[7, ["th_ppm", "nb_ppm", "yb_ppm"]] = [8.0, 4.0, 2.0]
    _, _, draws = build_inputs(bulks, majors, n_draws=4, seed=1)
    arc = draws[draws.bulk == 7]
    fallback = draws[draws.bulk == 11]
    assert (arc.proposal_tier == "trace").all()
    assert (arc.proposal_rung == "th_nb_yb").all()
    assert (fallback.proposal_tier == "fallback").all()
    assert fallback.proposal_rung.isna().all()


def test_build_inputs_draws_stable_across_chunk_boundaries(tiny_bulks):
    bulks, majors = tiny_bulks
    _, _, together = build_inputs(bulks, majors, n_draws=4, seed=9)
    _, _, alone = build_inputs(bulks.iloc[1:], majors, n_draws=4, seed=9)
    got = together[together.bulk == 11].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, alone.reset_index(drop=True))

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "scripts" / "build" / "generate_mineralogy_corpus.py")
spec = importlib.util.spec_from_file_location("generate_mineralogy_corpus", SCRIPT)
gmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gmc)

LADDER = [1100.0, 1080.0]


def _inp(n_draws):
    rows = []
    sid = 0
    for draw in range(n_draws):
        for T in LADDER:
            sid += 1
            rows.append(dict(sample_id=sid, T_celsius=T,
                             P_kbar=1.0 + draw, h2o_tot=2.0))
    return pd.DataFrame(rows)


def _runner(poison, calls=None, salvage=False):
    def run(input_csv, output_jsonl, database):
        if calls is not None:
            calls.append(Path(input_csv).name)
        df = pd.read_csv(input_csv)
        crash = df.sample_id.isin(poison).any()
        if not crash or salvage:
            with open(output_jsonl, "w") as fh:
                for sid in df.sample_id:
                    fh.write(json.dumps(dict(sid=int(sid))) + "\n")
        if crash:
            return SimpleNamespace(returncode=-11, stderr="signal 11")
        return SimpleNamespace(returncode=0, stderr="")
    return run


def _sids(jsonl):
    return [json.loads(line)["sid"] for line in open(jsonl)]


@pytest.fixture(autouse=True)
def _short_ladder(monkeypatch):
    monkeypatch.setattr(gmc, "T_LADDER", LADDER)


def test_clean_group_runs_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gmc, "run_julia", _runner(set(), calls))
    crashed = []
    gmc.run_group(_inp(4), tmp_path / "in_0.csv", tmp_path / "out_0.jsonl",
                  "ig", crashed)
    assert calls == ["in_0.csv"] and crashed == []
    assert _sids(tmp_path / "out_0.jsonl") == list(range(1, 9))


def test_bisection_drops_whole_poison_draws(tmp_path, monkeypatch):
    # sid 3 sits in draw 1 (sids 3-4); the whole draw goes, nothing else
    monkeypatch.setattr(gmc, "run_julia", _runner({3}))
    crashed = []
    gmc.run_group(_inp(6), tmp_path / "in_0.csv", tmp_path / "out_0.jsonl",
                  "ig", crashed)
    assert crashed == [[3, 4]]
    assert _sids(tmp_path / "out_0.jsonl") == [1, 2, 5, 6, 7, 8, 9, 10, 11, 12]
    assert (tmp_path / "in_0.stderr").read_text() == "signal 11"


def test_complete_output_salvaged_from_cleanup_crash(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gmc, "run_julia", _runner({3}, calls, salvage=True))
    crashed = []
    gmc.run_group(_inp(4), tmp_path / "in_0.csv", tmp_path / "out_0.jsonl",
                  "ig", crashed)
    assert calls == ["in_0.csv"] and crashed == []
    assert _sids(tmp_path / "out_0.jsonl") == list(range(1, 9))


def test_crash_cap_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(gmc, "run_julia", _runner(set(range(1, 13))))
    monkeypatch.setattr(gmc, "CRASH_CAP", 2)
    with pytest.raises(RuntimeError, match="systemic"):
        gmc.run_group(_inp(6), tmp_path / "in.csv", tmp_path / "out.jsonl",
                      "ig", [])

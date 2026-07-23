import pytest
import yaml

from cnmineral.eval.trajectory import build_trajectory_figure, load_ledger

GOOD = {
    "instruments": {
        "valtest128-broad": {"draws_per_bulk": 128, "tie_band_nats": 0.03},
        "valtest256-traced": {"draws_per_bulk": 256},
    },
    "versions": [
        {
            "version": "standalone-modehead", "date": "2026-07-18",
            "checkpoint_tag": "mode_head_seed0", "changes": "first head",
            "instrument": "valtest128-broad",
            "heads": {"mode": {"joint_nll": -1.0}},
        },
        {
            "version": "joint-tuned", "date": "2026-07-20",
            "checkpoint_tag": "joint_heads_seed0", "changes": "tuned",
            "instrument": "valtest128-broad",
            "heads": {
                "mode": {"joint_nll": -1.2},
                "comp": {"comp_nll": -53.9},
                "cond": {"cond_nll": 8.3},
            },
        },
        {
            "version": "regen-baseline", "date": "2026-07-28",
            "checkpoint_tag": "joint_heads_regen_seed0", "changes": "new corpus",
            "instrument": "valtest256-traced",
            "heads": {
                "mode": {"joint_nll": -1.1},
                "comp": {"comp_nll": -52.0},
                "cond": {"cond_nll": 8.0},
            },
        },
    ],
}


def _write(tmp_path, payload):
    path = tmp_path / "ledger.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(payload, f)
    return path


def test_load_ledger_validates_schema(tmp_path):
    entries = load_ledger(_write(tmp_path, GOOD))
    assert [e["version"] for e in entries] == [
        "standalone-modehead", "joint-tuned", "regen-baseline"
    ]

    bad = {**GOOD, "versions": [{**GOOD["versions"][0]}]}
    del bad["versions"][0]["heads"]
    with pytest.raises(ValueError, match="missing"):
        load_ledger(_write(tmp_path, bad))

    bad = {**GOOD, "versions": [dict(GOOD["versions"][0], heads={"volume": {}})]}
    with pytest.raises(ValueError, match="unknown heads"):
        load_ledger(_write(tmp_path, bad))

    bad = {**GOOD, "versions": [dict(GOOD["versions"][0], heads={"mode": {"nll": 1.0}})]}
    with pytest.raises(ValueError, match="lacks"):
        load_ledger(_write(tmp_path, bad))

    bad = {**GOOD, "versions": [{**GOOD["versions"][0]}]}
    del bad["versions"][0]["instrument"]
    with pytest.raises(ValueError, match="missing.*instrument"):
        load_ledger(_write(tmp_path, bad))

    bad = {**GOOD, "versions": [dict(GOOD["versions"][0], instrument="valtest9-x")]}
    with pytest.raises(ValueError, match="not defined"):
        load_ledger(_write(tmp_path, bad))


def test_build_trajectory_figure_smoke(tmp_path):
    entries = load_ledger(_write(tmp_path, GOOD))
    out = tmp_path / "trajectory.png"
    build_trajectory_figure(entries, out)
    assert out.exists() and out.stat().st_size > 0

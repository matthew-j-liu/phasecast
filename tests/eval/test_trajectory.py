import pytest
import yaml

from cnmineral.eval.trajectory import (
    build_trajectory_figure,
    load_adapted_levels,
    load_ledger,
)

GOOD = {
    "benchmarks": {
        "eruption-v1": {"n_headline": 635},
        "eruption-v2": {"n_headline": 640, "supersedes": "eruption-v1"},
    },
    "versions": [
        {
            "version": "support-conditioned-magnitude", "date": "2026-07-24",
            "checkpoint_tag": "joint_heads_supportmag_seed0",
            "benchmark": "eruption-v1", "changes": "first real-rock score",
            "n": 635, "joint_nll": 10.404,
        },
        {
            "version": "support-conditioned-magnitude", "date": "2026-07-25",
            "checkpoint_tag": "joint_heads_supportmag_seed0",
            "benchmark": "eruption-v2", "changes": "re-scored after Pinatubo",
            "n": 640, "joint_nll": 10.406,
        },
        {
            "version": "regen-baseline", "date": "2026-07-30",
            "checkpoint_tag": "joint_heads_regen_seed0",
            "benchmark": "eruption-v2", "changes": "traced corpus",
            "n": 640, "joint_nll": 18.396,
        },
    ],
}


def _write(tmp_path, payload):
    path = tmp_path / "real_rock_ledger.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(payload, f)
    return path


def test_load_ledger_validates_schema(tmp_path):
    entries = load_ledger(_write(tmp_path, GOOD))
    assert [e["version"] for e in entries] == [
        "support-conditioned-magnitude", "support-conditioned-magnitude",
        "regen-baseline",
    ]

    bad = {**GOOD, "versions": [{**GOOD["versions"][0]}]}
    del bad["versions"][0]["joint_nll"]
    with pytest.raises(ValueError, match="missing.*joint_nll"):
        load_ledger(_write(tmp_path, bad))

    bad = {**GOOD, "versions": [dict(GOOD["versions"][0], benchmark="eruption-v9")]}
    with pytest.raises(ValueError, match="not defined"):
        load_ledger(_write(tmp_path, bad))


def test_load_adapted_levels_missing_file_is_empty(tmp_path):
    assert load_adapted_levels(tmp_path / "nope.json") == (None, {})


def test_load_adapted_levels_reads_val_slice(tmp_path):
    path = tmp_path / "checkpoint_contrast_matrix.json"
    path.write_text(
        '{"benchmark": "eruption-v2", "levels": '
        '{"joint_heads_supportmag_seed0": {"val": '
        '{"adapted_joint": 1.376}}, "no_val_tag": {"test": '
        '{"adapted_joint": 0.5}}}}'
    )
    assert load_adapted_levels(path) == (
        "eruption-v2", {"joint_heads_supportmag_seed0": 1.376}
    )


def test_build_trajectory_figure_smoke(tmp_path):
    entries = load_ledger(_write(tmp_path, GOOD))
    out = tmp_path / "trajectory.png"
    build_trajectory_figure(
        entries, out, adapted={"joint_heads_regen_seed0": 1.707},
        adapted_benchmark="eruption-v2",
    )
    assert out.exists() and out.stat().st_size > 0

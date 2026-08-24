"""Tests for `cnmineral.training.joint_heads_hpo`.

`tiebreak_best` is unit-tested on fabricated trials (the locked winner
read-out: mode ties pass the sampled-support band-veto then break to comp
then cond, with attr-less trials exempt from the veto; mini-sweep
admissibility under `mode_reference`).  `ledger_tie_band` is unit-tested
against a ledger fixture (the instruments stanza is the single source of
truth; an unfilled stanza refuses to sweep).  The end-to-end test runs a
2-trial sweep on the synthetic joint corpus: enqueued baseline trial,
per-head user attrs incl. the predicted-modes closure and the
sampled-support veto axis, tiebreak winner in the summary, journal +
figures on disk.
"""

import json
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch
import yaml

import cnmineral.eval.joint_heads as joint_heads_eval
import cnmineral.training.joint_heads as joint_heads_train
from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.training.joint_heads_hpo import (
    ledger_tie_band,
    run_hpo,
    tiebreak_best,
)


def _trial(value, *, mode=None, comp=0.0, cond=0.0, sampled=None):
    attrs = {"comp_best": comp, "cond_best": cond}
    if mode is not None:
        attrs["mode_best"] = mode
    if sampled is not None:
        attrs["dirichlet_nll_sampled_support"] = sampled
    return optuna.trial.create_trial(
        params={}, distributions={}, value=value, user_attrs=attrs,
    )


def test_tiebreak_mode_ties_break_to_comp_then_cond():
    leader = _trial(-5.750, comp=-50.0, cond=8.6)
    tied_better_comp = _trial(-5.740, comp=-51.0, cond=8.7)
    outside_band = _trial(-5.700, comp=-60.0, cond=8.0)
    winner = tiebreak_best(
        [leader, tied_better_comp, outside_band], "mode", tie_band=0.03,
    )
    assert winner is tied_better_comp

    cond_breaks = _trial(-5.749, comp=-51.0, cond=8.5)
    winner = tiebreak_best(
        [leader, tied_better_comp, cond_breaks], "mode", tie_band=0.03,
    )
    assert winner is cond_breaks


def test_tiebreak_sampled_support_vetoes_tied_trial():
    leader = _trial(-5.750, comp=-50.0, sampled=-1.50)
    tied_better_comp = _trial(-5.740, comp=-51.0, sampled=-1.40)
    winner = tiebreak_best(
        [leader, tied_better_comp], "mode", tie_band=0.03,
    )
    assert winner is leader

    within_veto_band = _trial(-5.740, comp=-51.0, sampled=-1.48)
    winner = tiebreak_best(
        [leader, within_veto_band], "mode", tie_band=0.03,
    )
    assert winner is within_veto_band


def test_tiebreak_veto_can_reject_the_value_leader():
    leader = _trial(-5.750, comp=-51.0, sampled=-1.40)
    tied_best_sampled = _trial(-5.745, comp=-50.0, sampled=-1.50)
    winner = tiebreak_best(
        [leader, tied_best_sampled], "mode", tie_band=0.03,
    )
    assert winner is tied_best_sampled


def test_tiebreak_trials_without_sampled_attr_are_exempt():
    pre_criterion = _trial(-5.750, comp=-51.0)
    scored = _trial(-5.740, comp=-50.0, sampled=-1.50)
    winner = tiebreak_best([pre_criterion, scored], "mode", tie_band=0.03)
    assert winner is pre_criterion


def test_tiebreak_minisweep_requires_mode_within_band():
    good = _trial(-50.5, mode=-5.74)
    better_but_hurts_mode = _trial(-51.0, mode=-5.60)
    winner = tiebreak_best(
        [good, better_but_hurts_mode], "comp", tie_band=0.03,
        mode_reference=-5.75,
    )
    assert winner is good
    assert (
        tiebreak_best(
            [better_but_hurts_mode], "comp", tie_band=0.03, mode_reference=-5.75,
        )
        is None
    )


def _write_ledger(path, tie_band=0.03):
    """Ledger fixture carrying just the instruments stanza `ledger_tie_band`
    reads."""
    with open(path, "w") as f:
        yaml.safe_dump(
            {"instruments": {"test-instrument": {"tie_band_nats": tie_band}}}, f
        )
    return path


def test_ledger_tie_band_reads_the_instrument_stanza(tmp_path):
    ledger = _write_ledger(tmp_path / "ledger.yaml")
    assert ledger_tie_band("test-instrument", ledger) == 0.03


def test_ledger_tie_band_refuses_unmeasured_instrument(tmp_path):
    ledger = _write_ledger(tmp_path / "ledger.yaml", tie_band=None)
    with pytest.raises(ValueError, match="null"):
        ledger_tie_band("test-instrument", ledger)
    with pytest.raises(KeyError, match="not defined"):
        ledger_tie_band("absent-instrument", ledger)


def _write_hpo_yaml(path, config, storage_dir, output_dir, ledger_path):
    """Minimal hand-written HPO config so the test breaks on `run_hpo`
    mishandling the YAML, not on edits to the production config.  Absolute
    storage / output paths pass straight through the REPO_ROOT join."""
    spec = {
        "study_name": "joint_heads_hpo_test",
        "storage_path": str(storage_dir / "joint_heads_hpo_test.journal"),
        "output_dir": str(output_dir),
        "objective": "mode",
        "instrument": "test-instrument",
        "ledger_path": str(ledger_path),
        "sampler_seed": 7,
        "n_trials": 2,
        "n_startup_trials": 2,
        "base": config,
        "search_space": {
            "lr": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
            "mode_layers": {"type": "int", "low": 1, "high": 2},
        },
        "baseline": {"lr": 1e-3, "mode_layers": 1},
    }
    with open(path, "w") as f:
        yaml.safe_dump(spec, f)
    return path


def test_run_joint_heads_hpo_end_to_end(
    tiny_pretrain_checkpoint, corpus_subset_path, synthetic_joint_corpus,
    tmp_path, monkeypatch,
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, min_class_count=5
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:12]]
    corpus_dir = synthetic_joint_corpus(tmp_path / "corpus", bulk_ids[:8])
    valtest_dir = synthetic_joint_corpus(tmp_path / "valtest", bulk_ids[8:])
    def roster(subset):
        return np.array(bulk_ids[8:], dtype=np.int64)

    monkeypatch.setattr(joint_heads_train, "load_valtest_split", roster)
    monkeypatch.setattr(joint_heads_eval, "load_valtest_split", roster)

    base = {
        "encoder_checkpoint": str(tiny_pretrain_checkpoint),
        "corpus_dir": str(corpus_dir),
        "valtest_corpus_dir": str(valtest_dir),
        "epochs": 2,
        "batch_size": 16,
        "lambda_cond": 0.01,
        "seed": 0,
        "data_kwargs": {"csv_path": str(corpus_subset_path), "min_class_count": 5},
    }
    storage_dir = tmp_path / "optuna"
    storage_dir.mkdir()
    output_dir = tmp_path / "out"
    ledger_path = _write_ledger(tmp_path / "ledger.yaml")
    cfg_path = _write_hpo_yaml(
        tmp_path / "joint_hpo.yaml", base, storage_dir, output_dir, ledger_path,
    )

    if not Path("data/bulks/gard2019.csv").exists():
        pytest.skip("data/bulks/gard2019.csv absent")
    summary = run_hpo(cfg_path, device=torch.device("cpu"))

    assert summary["n_trials_complete"] == 2
    assert summary["objective"] == "mode"
    # resolved from the ledger fixture, not a config copy
    assert summary["tie_band"] == 0.03
    winner = summary["tiebreak_winner"]
    assert winner is not None and "lr" in winner["params"]
    for key in ("mode_best", "mode_best_epoch", "comp_best", "cond_best",
                "closure_l2_mean", "dirichlet_nll_sampled_support",
                "effective_lambda"):
        assert key in winner["user_attrs"]
    assert np.isfinite(winner["user_attrs"]["closure_l2_mean"])
    assert np.isfinite(winner["user_attrs"]["dirichlet_nll_sampled_support"])
    # base config carries no magnitude_weight -> trainer default 1.0 x 21 phases
    assert winner["user_attrs"]["effective_lambda"] == pytest.approx(21.0)
    # trial 0 is the enqueued baseline
    assert summary["config"]["baseline"] == {"lr": 1e-3, "mode_layers": 1}

    run_dir = output_dir / "joint_heads_hpo_test"
    with open(run_dir / "summary.json") as f:
        assert json.load(f)["n_trials_complete"] == 2
    assert (run_dir / "hpo_history.png").stat().st_size > 0
    assert (run_dir / "mode_vs_comp_scatter.png").stat().st_size > 0
    # fANOVA importances are skipped below 4 completed trials
    assert not (run_dir / "hpo_param_importances.png").exists()
    assert list(storage_dir.glob("*.journal"))

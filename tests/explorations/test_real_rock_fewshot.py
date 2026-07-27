"""Tripwire for the few-shot adaptation curve: a src/ change that breaks it fails here."""

import numpy as np
import torch

from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS
from cnmineral.models.real_rock_adapter import TIERS
from cnmineral.training.real_rock_adapter import GroupPredictions

from explorations.real_rock_fewshot.run_curve import (
    BUDGETS,
    WEIGHT_DECAY,
    run_curve,
)

N_GROUPS = len(GROUPS)


def _predictions(n, seed, units_of):
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.5, 8.0, (n, N_GROUPS))
    alpha[:, GROUP_INDEX["ol"]] = 0.0
    numeric = np.zeros((n, N_GROUPS), dtype=bool)
    numeric[:, [GROUP_INDEX["fsp"], GROUP_INDEX["px"], GROUP_INDEX["amp"]]] = True
    magnitude = np.zeros((n, N_GROUPS))
    magnitude[numeric] = rng.dirichlet(np.ones(3), n).reshape(-1)
    return GroupPredictions(
        logits7=torch.as_tensor(rng.normal(0, 1.5, (n, N_GROUPS))),
        alpha7=torch.as_tensor(alpha),
        sio2=torch.as_tensor(rng.uniform(50, 78, n)),
        presence=torch.as_tensor(numeric.astype(np.float64)),
        magnitude=torch.as_tensor(magnitude),
        numeric=torch.as_tensor(numeric.astype(np.float64)),
        unit=np.array([f"unit-{i // units_of}" for i in range(n)]),
    )


def test_weight_decay_is_recorded_for_every_tier():
    """The curve's only use of the val slice is one scalar per tier, fixed up
    front; a tier with no recorded value would silently take a default."""
    assert set(WEIGHT_DECAY) == set(TIERS)
    assert all(v > 0 for v in WEIGHT_DECAY.values())


def test_run_curve_produces_one_row_per_tier_budget_draw():
    train = _predictions(90, seed=0, units_of=3)
    test = _predictions(40, seed=1, units_of=4)
    budgets = (10, 30)
    baseline, rows = run_curve(
        train, test, budgets=budgets, n_draws=2, weight_decay=WEIGHT_DECAY,
    )
    assert len(rows) == len(TIERS) * len(budgets) * 2
    assert {r["tier"] for r in rows} == set(TIERS)
    assert np.isfinite(baseline["joint"]).all()

    for row in rows:
        assert row["n_rocks"] > 0
        assert row["n_units"] > 0
        assert np.isfinite(row["dirichlet"])
        # the paired interval must bracket its own point estimate
        assert row["recovered_lo"] <= row["recovered_dirichlet"] <= row["recovered_hi"]


def test_budgets_are_ascending_so_the_curve_reads_left_to_right():
    assert list(BUDGETS) == sorted(BUDGETS)

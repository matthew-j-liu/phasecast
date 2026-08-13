"""Fitting the measured-label head: the objective it optimizes, the shrinkage
carve-out that keeps its degenerate point sensible, and budget-only inputs."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cnmineral.data.phase_vocab import GROUP_INDEX, GROUPS
from cnmineral.eval.real_rock import raw_chemistry
from cnmineral.models.measured_mode_head import MeasuredModeHead
from cnmineral.training.measured_mode_head import (
    ablate_latent,
    fit_measured_head,
    score_measured_head,
    use_raw_chemistry,
)
from cnmineral.training.real_rock_recalibrator import (
    GroupPredictions,
    recalibrator_nll,
)

N_GROUPS = len(GROUPS)
D_MODEL = 16
N_RAW = 22


def _predictions(n=48, seed=0, units_of=3, latent_shift=0.0):
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
        latent=torch.as_tensor(rng.normal(latent_shift, 1.0, (n, D_MODEL))),
        raw=torch.as_tensor(rng.normal(0, 1.0, (n, N_RAW))),
    )


def _loss(head, data):
    logits, alpha, log_w = head(data.latent, data.presence)
    return recalibrator_nll(
        logits, alpha, data.presence, data.magnitude, data.numeric, log_w=log_w,
    )["joint"].mean().item()


def test_fitting_reduces_the_loss_it_optimizes():
    data = _predictions()
    before = _loss(MeasuredModeHead(D_MODEL).double(), data)
    after = _loss(fit_measured_head(data, weight_decay=1e-3, epochs=200), data)
    assert after < before


def test_biases_are_exempt_from_weight_decay():
    """Shrinking the weights to zero means "ignore the latent", whose sensible
    degenerate point is a constant forecast -- and that lives in the biases.
    Decaying them too would pull it to p = 0.5 and alpha = softplus(0)."""
    data = _predictions()
    head = fit_measured_head(data, weight_decay=1e6, epochs=300)
    weights = [p for n, p in head.named_parameters() if not n.endswith("bias")]
    biases = [p for n, p in head.named_parameters() if n.endswith("bias")]
    assert max(w.abs().max().item() for w in weights) < 1e-3
    assert max(b.abs().max().item() for b in biases) > 1e-2


def test_the_fit_uses_only_the_drawn_rocks_input_statistics():
    """The x-axis is only honest if undrawn rocks contribute nothing at all --
    including label-free input statistics like the standardizer's."""
    data = _predictions(n=48, latent_shift=5.0)
    drawn = data.subset(np.arange(len(data)) < 12)
    head = fit_measured_head(drawn, weight_decay=1e-2, epochs=10)
    assert torch.allclose(head.latent_mean, drawn.latent.mean(0))
    assert not torch.allclose(head.latent_mean, data.latent.mean(0))


def test_scoring_a_held_out_slice_reuses_the_fitted_standardizer():
    data = _predictions()
    head = fit_measured_head(data.subset(np.arange(len(data)) < 24),
                             weight_decay=1e-2, epochs=10)
    held_out = data.subset(np.arange(len(data)) >= 24)
    scored = score_measured_head(head, held_out)
    assert np.isfinite(scored["joint"]).all()
    assert len(scored["joint"]) == len(held_out)


def test_the_input_swaps_replace_what_the_head_reads_and_nothing_else():
    """Each measured arm is the same head on a different input; if a swap moved
    the labels or the rocks too, the arms would not be differenceable."""
    data = _predictions()
    for view, width in ((use_raw_chemistry, N_RAW), (ablate_latent, D_MODEL)):
        swapped = view(data)
        assert swapped.latent.shape == (len(data), width)
        for field in ("presence", "magnitude", "numeric", "logits7", "alpha7"):
            assert torch.equal(getattr(swapped, field), getattr(data, field))
        assert (swapped.unit == data.unit).all()
    assert torch.equal(use_raw_chemistry(data).latent, data.raw)
    assert (ablate_latent(data).latent == 0).all()


def test_raw_chemistry_logs_abundances_and_neutralizes_absent_elements():
    """`log10(0)` has no value to place, and absent means unmeasured rather
    than zero abundance, so the slot has to read as neutral instead."""
    bm = SimpleNamespace(
        bulk_frac=torch.tensor([[0.5, 0.0, 0.01]]),
        bulk_src=torch.tensor([[12, 0, 26]]),
    )
    got = raw_chemistry(bm)
    assert got[0, 0] == pytest.approx(np.log10(0.5))
    assert got[0, 2] == pytest.approx(np.log10(0.01))
    assert got[0, 1] == 0.0


def test_a_head_reading_the_assay_beats_one_reading_nothing():
    """The raw arm must actually be reading its input; if the swap silently
    handed it a constant it would score like the ablated arm."""
    data = _predictions()
    fits = {}
    for name, view in (("raw", use_raw_chemistry), ("ablated", ablate_latent)):
        seen = view(data)
        head = fit_measured_head(seen, weight_decay=1e-3, epochs=300)
        fits[name] = score_measured_head(head, seen)["joint"].mean()
    assert fits["raw"] < fits["ablated"]


def test_a_latentless_slice_is_rejected_rather_than_silently_misfit():
    data = _predictions()
    stripped = GroupPredictions(
        logits7=data.logits7, alpha7=data.alpha7, sio2=data.sio2,
        presence=data.presence, magnitude=data.magnitude, numeric=data.numeric,
        unit=data.unit,
    )
    with pytest.raises(ValueError, match="latent"):
        fit_measured_head(stripped)

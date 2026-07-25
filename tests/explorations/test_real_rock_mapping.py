import numpy as np
import pytest
import torch

from cnmineral.data.phase_vocab import PHASE_INDEX, SOLID_PHASES
from cnmineral.eval.real_rock import (
    GROUPS,
    GROUP_INDEX,
    MINERAL_DENSITY,
    coarsen_presence,
    load_benchmark,
    real_rock_nll,
    truth_forced_support,
)
from cnmineral.models.mode_head import ModeHead
from explorations.real_rock_mapping.run_sensitivity import (
    DENSITY_RANGE,
    alteration_masks,
    coarse_support,
    endpoint_densities,
    imputed_at_bound,
    monte_carlo_densities,
    row_index,
    run_variants,
    score,
    sign_flips,
)


@pytest.fixture(scope="module")
def rig():
    """The shipped benchmark plus a random-weight head over a fake latent.

    The audit is about the truth mapping, not about a trained model, so an
    untrained head exercises every code path a real one does.
    """
    torch.manual_seed(0)
    benchmark = load_benchmark()
    head = ModeHead(d_model=16)
    latents = torch.randn(len(benchmark), 16)
    with torch.no_grad():
        p21 = torch.sigmoid(head.presence_logits(latents)).numpy()
    return benchmark, head, latents, p21, coarsen_presence(p21)


def test_density_ranges_bracket_the_shipped_table():
    """Every shipped density must be a value its own range admits, or the
    Monte Carlo is sampling around a table nobody ships."""
    assert set(DENSITY_RANGE) == set(MINERAL_DENSITY)
    for name, (low, high) in DENSITY_RANGE.items():
        assert low <= MINERAL_DENSITY[name] <= high, name


def test_monte_carlo_and_endpoint_tables_stay_in_range():
    rng = np.random.default_rng(0)
    for table in (monte_carlo_densities(rng), endpoint_densities(True),
                  endpoint_densities(False)):
        for name, value in table.items():
            low, high = DENSITY_RANGE[name]
            assert low <= value <= high, name


def test_endpoint_tables_pull_mafics_and_felsics_apart():
    high = endpoint_densities(mafic_high=True)
    low = endpoint_densities(mafic_high=False)
    assert high["olivine"] > low["olivine"]
    assert high["Fe-Ti oxides"] > low["Fe-Ti oxides"]
    assert high["plagioclase"] < low["plagioclase"]


def test_imputed_composition_matches_a_direct_recomputation():
    """The imputed shares are derived algebra, not a second mapping, so they
    have to agree with rebuilding the composition from the reported volumes."""
    base = load_benchmark()
    imputed = imputed_at_bound(base)
    scored = imputed.has_magnitude
    np.testing.assert_allclose(imputed.magnitude[scored].sum(1), 1.0)
    # a censored group lands exactly on the bound it was known to sit under.
    # Rocks whose every phase was a trace stay unscored, so the headline's row
    # set is the same one the shipped mapping scores.
    censored_on_scored = base.censored & scored[:, None]
    np.testing.assert_allclose(
        imputed.magnitude[censored_on_scored],
        base.censor_bound[censored_on_scored],
    )
    assert not imputed.magnitude[~scored].any()
    # every trace appearance now carries a number, and none is censored
    np.testing.assert_array_equal(
        imputed.numeric, (base.numeric | base.censored) & scored[:, None]
    )
    assert not imputed.censored.any()
    # quantified groups keep their ratios to each other; they only give up the
    # room the bounds occupy
    rock = np.flatnonzero(base.censored.any(1) & (base.numeric.sum(1) > 1))[0]
    keep = base.numeric[rock]
    np.testing.assert_allclose(
        imputed.magnitude[rock, keep] / imputed.magnitude[rock, keep].sum(),
        base.magnitude[rock, keep],
    )


def test_coarse_support_rules_span_and_narrow_the_group():
    presence7 = np.zeros((1, len(GROUPS)))
    presence7[0, GROUP_INDEX["px"]] = 1.0
    p21 = np.zeros((1, len(SOLID_PHASES)))
    p21[0, PHASE_INDEX["cpx"]] = 0.2
    p21[0, PHASE_INDEX["opx"]] = 0.1
    p21[0, PHASE_INDEX["g"]] = 0.9

    every = coarse_support(presence7, p21, "all_members")
    single = coarse_support(presence7, p21, "single_member")
    assert every[0, PHASE_INDEX["cpx"]] == every[0, PHASE_INDEX["opx"]] == 1.0
    assert single[0, PHASE_INDEX["cpx"]] == 1.0
    assert single[0, PHASE_INDEX["opx"]] == 0.0
    # a truth-absent group is off under both rules
    assert every[0, PHASE_INDEX["plag"]] == single[0, PHASE_INDEX["plag"]] == 0.0
    # an unreported phase is left to the model, never forced either way
    assert every[0, PHASE_INDEX["g"]] == single[0, PHASE_INDEX["g"]] == 1.0
    with pytest.raises(ValueError, match="unknown support rule"):
        coarse_support(presence7, p21, "nonsense")


def test_score_reproduces_the_hurdle_nll_it_reports(rig):
    """The scorer must be the shipped loss on the shipped mapping, or every
    variant is measured against something the eval never reported."""
    benchmark, head, latents, p21, p7 = rig
    support = truth_forced_support(benchmark.presence, p21)
    mask = benchmark.clean & benchmark.has_magnitude
    out = score(head, latents, p21, p7, benchmark, support, mask)

    with torch.no_grad():
        alpha = head.magnitude_alpha(
            latents, torch.as_tensor(support, dtype=latents.dtype)
        ).numpy()
    from cnmineral.eval.real_rock import coarsen_alpha
    alpha7, _ = coarsen_alpha(alpha, support)
    expected = real_rock_nll(p7, alpha7, benchmark.presence, benchmark.magnitude,
                             benchmark.numeric)
    assert out["joint_nll"] == pytest.approx(expected["joint"][mask].mean())
    assert out["n"] == int(mask.sum())


def test_alteration_masks_nest_from_strict_to_everything():
    benchmark = load_benchmark()
    masks = alteration_masks(benchmark)
    assert masks["all_rocks_incl_ungraded"].all()
    np.testing.assert_array_equal(masks["grades_1_2"], benchmark.clean)
    # a `+` read as "worse than the digit" can only shrink the clean set
    assert masks["modifier_strict"].sum() < masks["grades_1_2"].sum()
    for tighter, looser in (("grade_1_only", "grades_1_2"),
                            ("grades_1_2", "grades_1_2_3"),
                            ("grades_1_2_3", "all_graded")):
        assert (masks[tighter] <= masks[looser]).all(), tighter


def test_row_index_maps_a_shortened_benchmark_back_onto_the_latents():
    base = load_benchmark()
    idx = row_index(base.bulk_ids, base.bulk_ids[[5, 1, 9]])
    np.testing.assert_array_equal(idx, [5, 1, 9])
    with pytest.raises(KeyError):
        row_index(base.bulk_ids, np.array(["not|a|rock"]))


def test_sign_flips_ignores_offsets_too_small_to_correct():
    def offsets(first, second):
        return [first, second] + [np.nan] * (len(GROUPS) - 2)

    rows = [
        {"variant": "shipped", "n": 500, "off": offsets(0.4, -0.002)},
        {"variant": "big_flip", "n": 500, "off": offsets(-0.4, 0.9)},
        {"variant": "tiny_subset", "n": 6, "off": offsets(-0.4, -0.4)},
    ]
    flips = sign_flips(rows, "off", materiality=0.05)
    # group 1's shipped offset is ~0, so nothing there is a correction to
    # reverse; the tiny subset is excluded by min_n
    assert [f["group"] for f in flips] == [GROUPS[0]]
    assert [v["variant"] for v in flips[0]["flipped_by"]] == ["big_flip"]


def test_run_variants_covers_every_knob_and_reproduces_the_shipped_row(
    rig, monkeypatch
):
    """The tripwire: the whole audit runs, and its `shipped` row is the eval's
    own headline computed through the eval's own functions."""
    benchmark, head, latents, p21, p7 = rig
    monkeypatch.setattr(
        "explorations.real_rock_mapping.run_sensitivity.MC_DRAWS", 2
    )
    rows, mc = run_variants(head, latents, benchmark, p21, p7)

    assert len(mc) == 2
    assert {row["knob"] for row in rows} == {
        "densities", "trace_bound", "alteration", "support", "uncertain_id",
        "reporting_basis",
    }
    shipped = next(row for row in rows if row["variant"] == "shipped")
    mask = benchmark.clean & benchmark.has_magnitude
    expected = score(head, latents, p21, p7, benchmark,
                     truth_forced_support(benchmark.presence, p21), mask)
    assert shipped["joint_nll"] == pytest.approx(expected["joint_nll"])
    # the trace bound cannot move the headline, whatever it is set to
    bounds = [row["joint_nll"] for row in rows
              if row["knob"] == "trace_bound" and "volpct" in row["variant"]]
    assert bounds == pytest.approx([bounds[0]] * len(bounds))

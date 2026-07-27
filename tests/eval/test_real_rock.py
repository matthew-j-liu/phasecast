import numpy as np
import pandas as pd
import pytest
import torch

from cnmineral.data.bulk_chemistry import BulkChemistryDataset, tokenize_bulk_frame
from cnmineral.data.phase_vocab import PHASE_INDEX, SOLID_PHASES
from cnmineral.eval.real_rock import (
    GROUP_INDEX,
    GROUPS,
    MINERAL_DENSITY,
    SPLIT_SUBSETS,
    coarsen_alpha,
    coarsen_presence,
    censored_consistency,
    load_benchmark,
    load_real_rock_split,
    parse_alteration_grade,
    real_rock_nll,
    split_balance,
    split_mask,
    truth_forced_support,
    unit_grouped_split,
)


def _write_benchmark(tmp_path, sample_rows, mode_rows, **mapping):
    """A two-table benchmark on disk with only the columns the loader reads."""
    samples = pd.DataFrame(sample_rows, columns=[
        "source", "sample_id", "alteration_extent", "sio2", "al2o3", "cao",
        "mgo", "feo_tot", "na2o", "k2o", "tio2", "unit",
    ])
    modes = pd.DataFrame(
        mode_rows, columns=["source", "sample_id", "phase", "magemin_phase",
                            "abundance_pct"],
    )
    samples.to_csv(tmp_path / "samples.csv", index=False)
    modes.to_csv(tmp_path / "modes.csv", index=False)
    return load_benchmark(tmp_path, **mapping)


def _sample(sample_id, sio2=60.0, alteration="1", unit=None):
    # one rock per unit unless a test is exercising the grouping, which is the
    # neutral choice: nothing shares a split assignment by accident.
    return ["src", sample_id, alteration, sio2, 16.0, 5.0, 3.0, 5.0, 4.0, 2.0,
            0.8, f"unit-{sample_id}" if unit is None else unit]


def test_coarsen_presence_is_the_noisy_or():
    p = np.zeros((1, len(SOLID_PHASES)))
    p[0, PHASE_INDEX["plag"]] = 0.5
    p[0, PHASE_INDEX["afs"]] = 0.5
    p[0, PHASE_INDEX["amp"]] = 0.3
    out = coarsen_presence(p)
    # feldspar is present if either member is: 1 - 0.5*0.5
    assert out[0, GROUP_INDEX["fsp"]] == pytest.approx(0.75)
    assert out[0, GROUP_INDEX["amp"]] == pytest.approx(0.3)
    assert out[0, GROUP_INDEX["ol"]] == pytest.approx(0.0)


def test_coarsen_alpha_sums_members_on_the_support_only():
    alpha = np.zeros((1, len(SOLID_PHASES)))
    support = np.zeros_like(alpha)
    for phase, value in (("cpx", 3.0), ("opx", 4.0), ("g", 9.0), ("ru", 1.0)):
        alpha[0, PHASE_INDEX[phase]] = value
        support[0, PHASE_INDEX[phase]] = 1.0
    # opx is off the support, so it is not a component of the distribution
    support[0, PHASE_INDEX["opx"]] = 0.0

    out, off_vocab = coarsen_alpha(alpha, support)
    assert out[0, GROUP_INDEX["px"]] == pytest.approx(3.0)
    # g and ru are model-only phases: reported separately, never in a group
    assert off_vocab[0] == pytest.approx(10.0)
    assert out.sum() == pytest.approx(3.0)


def test_coarsen_alpha_matches_dirichlet_aggregation():
    """Merging Dirichlet components sums their concentrations, so the coarsened
    group shares must have the same mean as summing sampled member shares."""
    rng = np.random.default_rng(0)
    alpha = np.zeros((1, len(SOLID_PHASES)))
    members = {"plag": 6.0, "afs": 2.0, "cpx": 4.0, "opx": 1.0, "amp": 3.0}
    for phase, value in members.items():
        alpha[0, PHASE_INDEX[phase]] = value
    support = (alpha > 0).astype(np.float64)

    out, _ = coarsen_alpha(alpha, support)
    on = np.flatnonzero(support[0] > 0)
    draws = rng.dirichlet(alpha[0, on], size=200_000)
    fsp = draws[:, [list(on).index(PHASE_INDEX[p]) for p in ("plag", "afs")]].sum(1)
    np.testing.assert_allclose(
        fsp.mean(), out[0, GROUP_INDEX["fsp"]] / out.sum(), atol=2e-3
    )


def test_parse_alteration_grade_reads_modifiers_and_fresh():
    values = pd.Series(["1", "1+", "2-", "2+", "3", "4", "fresh", None, ""])
    np.testing.assert_array_equal(
        parse_alteration_grade(values), [1, 1, 2, 2, 3, 4, 1, -1, -1]
    )


def test_trace_row_is_present_and_censored_not_zero(tmp_path):
    b = _write_benchmark(
        tmp_path,
        [_sample("a")],
        [["src", "a", "plagioclase", "fsp", 20.0],
         ["src", "a", "hornblende", "amp", None]],
    )
    amp, fsp = GROUP_INDEX["amp"], GROUP_INDEX["fsp"]
    assert b.presence[0, amp] == 1.0
    assert b.censored[0, amp] and not b.numeric[0, amp]
    assert b.magnitude[0, amp] == 0.0
    assert b.censor_bound[0, amp] > 0
    # the quantified phase carries the whole observed sub-composition
    assert b.magnitude[0, fsp] == pytest.approx(1.0)


def test_reported_zero_is_absence_not_presence(tmp_path):
    b = _write_benchmark(
        tmp_path,
        [_sample("a")],
        [["src", "a", "plagioclase", "fsp", 20.0],
         ["src", "a", "quartz", "q", 0.0]],
    )
    q = GROUP_INDEX["q"]
    assert b.presence[0, q] == 0.0
    assert not b.censored[0, q] and not b.numeric[0, q]


def test_groundmass_is_dropped_before_renormalizing(tmp_path):
    b = _write_benchmark(
        tmp_path,
        [_sample("a")],
        [["src", "a", "plagioclase", "fsp", 20.0],
         ["src", "a", "groundmass", "groundmass", 80.0]],
    )
    assert b.presence[0].sum() == 1.0
    assert b.magnitude[0, GROUP_INDEX["fsp"]] == pytest.approx(1.0)


def test_volume_converts_to_mass_before_coarsening(tmp_path):
    """Equal volumes of feldspar and Fe-Ti oxide are not equal masses."""
    b = _write_benchmark(
        tmp_path,
        [_sample("a")],
        [["src", "a", "plagioclase", "fsp", 10.0],
         ["src", "a", "Fe-Ti oxides", "opq_unsplit", 10.0]],
    )
    fsp = 10.0 * MINERAL_DENSITY["plagioclase"]
    opq = 10.0 * MINERAL_DENSITY["Fe-Ti oxides"]
    assert b.magnitude[0, GROUP_INDEX["opq"]] == pytest.approx(opq / (fsp + opq))
    assert b.magnitude[0, GROUP_INDEX["opq"]] > b.magnitude[0, GROUP_INDEX["fsp"]]


def test_group_densities_apply_per_reported_name(tmp_path):
    """Two minerals coarsening into one group each use their own density."""
    b = _write_benchmark(
        tmp_path,
        [_sample("a")],
        [["src", "a", "augite", "cpx", 10.0],
         ["src", "a", "hypersthene", "opx", 10.0],
         ["src", "a", "plagioclase", "fsp", 10.0]],
    )
    px = 10.0 * (MINERAL_DENSITY["augite"] + MINERAL_DENSITY["hypersthene"])
    fsp = 10.0 * MINERAL_DENSITY["plagioclase"]
    assert b.magnitude[0, GROUP_INDEX["px"]] == pytest.approx(px / (px + fsp))


def test_quantified_and_trace_in_one_group_counts_as_quantified(tmp_path):
    """Coarsening can land a number and a trace in the same group; the group
    has a number, so it is observed and its bound is discarded."""
    b = _write_benchmark(
        tmp_path,
        [_sample("a")],
        [["src", "a", "hornblende", "amp", 8.0],
         ["src", "a", "cummingtonite", "amp", None],
         ["src", "a", "plagioclase", "fsp", 20.0]],
    )
    amp = GROUP_INDEX["amp"]
    assert b.numeric[0, amp] and not b.censored[0, amp]
    assert b.censor_bound[0, amp] == 0.0
    assert b.magnitude[0].sum() == pytest.approx(1.0)


def test_rock_without_chemistry_or_modes_is_not_scoreable(tmp_path):
    b = _write_benchmark(
        tmp_path,
        [_sample("has_both"), _sample("no_chem", sio2=None), _sample("no_modes")],
        [["src", "has_both", "plagioclase", "fsp", 20.0],
         ["src", "no_chem", "plagioclase", "fsp", 20.0]],
    )
    assert list(b.bulk_ids) == ["src|has_both"]


def test_unknown_mineral_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no density"):
        _write_benchmark(
            tmp_path,
            [_sample("a")],
            [["src", "a", "zircon", "q", 1.0]],
        )


def test_alteration_grade_masks_rather_than_filters(tmp_path):
    b = _write_benchmark(
        tmp_path,
        [_sample("clean", alteration="2+"), _sample("altered", alteration="3")],
        [["src", "clean", "plagioclase", "fsp", 20.0],
         ["src", "altered", "plagioclase", "fsp", 20.0]],
    )
    assert len(b) == 2
    np.testing.assert_array_equal(b.clean, [True, False])
    assert len(b.subset(b.clean)) == 1


def test_truth_forced_support_respects_the_group_level_only():
    presence7 = np.zeros((1, len(GROUPS)))
    presence7[0, GROUP_INDEX["px"]] = 1.0
    p21 = np.full((1, len(SOLID_PHASES)), 0.9)

    support = truth_forced_support(presence7, p21)
    # truth says pyroxene is there but not which one, so the model resolves it
    assert support[0, PHASE_INDEX["cpx"]] == 1.0
    assert support[0, PHASE_INDEX["opx"]] == 1.0
    # truth says feldspar is absent, so both members are forced off
    assert support[0, PHASE_INDEX["plag"]] == 0.0
    assert support[0, PHASE_INDEX["afs"]] == 0.0
    # a phase no source reports is left to the model, never forced absent
    assert support[0, PHASE_INDEX["g"]] == 1.0


def test_uniform_densities_reproduce_volume_shares(tmp_path):
    """With every density equal the conversion is the identity, so the truth
    shares are the reported volume fractions -- the bound on what converting
    to mass can be worth."""
    rows = [["src", "a", "plagioclase", "fsp", 30.0],
            ["src", "a", "Fe-Ti oxides", "opq_unsplit", 10.0]]
    b = _write_benchmark(tmp_path, [_sample("a")], rows,
                         densities={"plagioclase": 1.0, "Fe-Ti oxides": 1.0})
    assert b.magnitude[0, GROUP_INDEX["fsp"]] == pytest.approx(0.75)
    assert b.magnitude[0, GROUP_INDEX["opq"]] == pytest.approx(0.25)


def test_presence_truth_is_invariant_to_the_density_table(tmp_path):
    """Density enters only the mass conversion, so which phases are there --
    and which carry a number -- cannot move with it."""
    rows = [["src", "a", "plagioclase", "fsp", 30.0],
            ["src", "a", "olivine", "ol", 4.0],
            ["src", "a", "hornblende", "amp", None]]
    base = _write_benchmark(tmp_path, [_sample("a")], rows)
    heavy = _write_benchmark(
        tmp_path, [_sample("a")], rows,
        densities={**MINERAL_DENSITY, "olivine": 9.0, "plagioclase": 1.1},
    )
    np.testing.assert_array_equal(base.presence, heavy.presence)
    np.testing.assert_array_equal(base.numeric, heavy.numeric)
    np.testing.assert_array_equal(base.censored, heavy.censored)
    np.testing.assert_array_equal(base.has_magnitude, heavy.has_magnitude)
    assert not np.allclose(base.magnitude, heavy.magnitude)


def test_headline_arrays_are_invariant_to_the_trace_bound(tmp_path):
    """The bound sizes a censored phase's ceiling and nothing else, so the
    hurdle NLL the headline reports cannot move with it."""
    rows = [["src", "a", "plagioclase", "fsp", 30.0],
            ["src", "a", "quartz", "q", 5.0],
            ["src", "a", "hornblende", "amp", None]]
    tight = _write_benchmark(tmp_path, [_sample("a")], rows,
                             trace_bound_vol_pct=0.1)
    loose = _write_benchmark(tmp_path, [_sample("a")], rows,
                             trace_bound_vol_pct=1.0)
    np.testing.assert_array_equal(tight.magnitude, loose.magnitude)
    np.testing.assert_array_equal(tight.numeric, loose.numeric)
    np.testing.assert_array_equal(tight.presence, loose.presence)
    assert (tight.censor_bound < loose.censor_bound)[0, GROUP_INDEX["amp"]]


def test_member_threshold_controls_which_members_join_the_support():
    presence7 = np.zeros((1, len(GROUPS)))
    presence7[0, GROUP_INDEX["px"]] = 1.0
    p21 = np.zeros((1, len(SOLID_PHASES)))
    p21[0, PHASE_INDEX["cpx"]] = 0.8
    p21[0, PHASE_INDEX["opx"]] = 0.4

    lenient = truth_forced_support(presence7, p21, member_threshold=0.3)
    strict = truth_forced_support(presence7, p21, member_threshold=0.9)
    assert lenient[0, PHASE_INDEX["opx"]] == 1.0
    # nothing clears 0.9, so the rule falls back to the likeliest member
    assert strict[0, PHASE_INDEX["cpx"]] == 1.0
    assert strict[0, PHASE_INDEX["opx"]] == 0.0


def test_truth_forced_support_turns_on_the_likeliest_member_when_none_clears():
    presence7 = np.zeros((1, len(GROUPS)))
    presence7[0, GROUP_INDEX["px"]] = 1.0
    p21 = np.zeros((1, len(SOLID_PHASES)))
    p21[0, PHASE_INDEX["cpx"]] = 0.2
    p21[0, PHASE_INDEX["opx"]] = 0.1

    support = truth_forced_support(presence7, p21)
    assert support[0, PHASE_INDEX["cpx"]] == 1.0
    assert support[0, PHASE_INDEX["opx"]] == 0.0


def test_nll_finite_when_coarsening_leaves_off_support_zeros():
    """Groups with no member on the support coarsen to alpha 0, which would
    make the masked Dirichlet NLL evaluate lgamma(0) and return NaN."""
    n_groups = len(GROUPS)
    alpha7 = np.zeros((1, n_groups))
    numeric = np.zeros((1, n_groups), dtype=bool)
    magnitude = np.zeros((1, n_groups))
    for group, value, share in (("fsp", 6.0, 0.7), ("px", 3.0, 0.3)):
        alpha7[0, GROUP_INDEX[group]] = value
        numeric[0, GROUP_INDEX[group]] = True
        magnitude[0, GROUP_INDEX[group]] = share
    presence = numeric.astype(np.float64)
    p7 = np.full((1, n_groups), 0.5)

    out = real_rock_nll(p7, alpha7, presence, magnitude, numeric)
    assert np.isfinite(out["joint"]).all()
    assert out["bernoulli"][0] == pytest.approx(n_groups * np.log(2.0))


def test_single_group_composition_contributes_no_dirichlet_term():
    n_groups = len(GROUPS)
    alpha7 = np.zeros((1, n_groups))
    alpha7[0, GROUP_INDEX["fsp"]] = 5.0
    numeric = np.zeros((1, n_groups), dtype=bool)
    numeric[0, GROUP_INDEX["fsp"]] = True
    magnitude = numeric.astype(np.float64)

    out = real_rock_nll(np.full((1, n_groups), 0.5), alpha7,
                        numeric.astype(np.float64), magnitude, numeric)
    assert out["dirichlet"][0] == pytest.approx(0.0)


def test_censored_consistency_rewards_predicting_a_small_share():
    n_groups = len(GROUPS)
    alpha7 = np.zeros((2, n_groups))
    censored = np.zeros((2, n_groups), dtype=bool)
    bound = np.zeros((2, n_groups))
    for row, amp_alpha in enumerate((0.2, 40.0)):
        alpha7[row, GROUP_INDEX["fsp"]] = 40.0
        alpha7[row, GROUP_INDEX["amp"]] = amp_alpha
        censored[row, GROUP_INDEX["amp"]] = True
        bound[row, GROUP_INDEX["amp"]] = 0.05
    support = (alpha7 > 0).astype(np.float64)

    out = censored_consistency(alpha7, support, censored, bound)
    amp = GROUP_INDEX["amp"]
    # the model that keeps amphibole tiny agrees with the trace report
    assert out["prob_below_bound"][0, amp] > 0.9
    assert out["prob_below_bound"][1, amp] < 0.1
    assert out["nll"][0, amp] < out["nll"][1, amp]
    assert np.isnan(out["prob_below_bound"][0, GROUP_INDEX["fsp"]])


def test_tokenize_bulk_frame_pads_rather_than_populating_a_zero():
    """A reported zero is below-detection, not a measured zero, so its slot
    pads -- otherwise it would violate the `src == 0 iff frac == 0` convention
    the encoder derives its padding mask from."""
    df = pd.DataFrame({"sio2": [50.0, 50.0], "tio2": [0.0, 0.9], "feo_tot": [8.0, 8.0]})
    src, frac = tokenize_bulk_frame(df)
    ti = 7
    assert src[0, ti] == 0 and frac[0, ti] == 0.0
    assert src[1, ti] == 22 and frac[1, ti] > 0
    assert ((src == 0) == (frac == 0)).all()


def test_tokenize_bulk_frame_agrees_with_the_dataset(corpus_subset_path):
    """The seam is the dataset's own tokenizer, so a frame carrying one
    corpus row must produce that row's tensors."""
    ds = BulkChemistryDataset(csv_path=corpus_subset_path, require_qap_name=False,
                              min_class_count=1)
    row = pd.read_parquet(corpus_subset_path)
    row = row[row[["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
                   "k2o", "na2o", "tio2"]].notna().all(axis=1)]
    row = row[row["sample_id"] == ds.sample_ids[0]]
    src, frac = tokenize_bulk_frame(row)
    torch.testing.assert_close(torch.from_numpy(src[0]), ds[0]["src"])
    torch.testing.assert_close(torch.from_numpy(frac[0]), ds[0]["frac"])


def test_tokenize_bulk_frame_pads_when_iron_columns_are_absent():
    """Real-rock tables report a single iron total, not feo plus fe2o3."""
    df = pd.DataFrame({"sio2": [50.0], "feo_tot": [8.0]})
    src, frac = tokenize_bulk_frame(df)
    assert src[0, 0] == 14  # Si populated
    assert src[0, 4] == 26  # Fe resolved through the feo_tot fallback
    assert frac[0, 4] == pytest.approx(8.0 * 0.7773 / 100.0)
    assert src[0, 1] == 0  # Al absent from the frame, so its slot pads
    assert (src[0, 9:] == 0).all()  # no trace columns at all


def test_load_benchmark_invariants():
    """The shipped benchmark satisfies the contracts the eval relies on."""
    b = load_benchmark()
    assert ((b.magnitude > 0) == b.numeric).all()
    assert not (b.numeric & b.censored).any()
    assert (b.presence >= (b.numeric | b.censored)).all()
    np.testing.assert_allclose(b.magnitude[b.has_magnitude].sum(1), 1.0)
    assert b.bulk_src.shape == b.bulk_frac.shape
    assert len(b.bulk_ids) == len(b.presence) == b.bulk_src.shape[0]
    assert len(b.unit) == len(b.bulk_ids)


def test_no_stratigraphic_unit_straddles_a_split_boundary():
    """One eruption's rocks share nearly identical chemistry, so a unit split
    across train and test leaks the answer -- the same call the corpus makes
    with bulk-level splits."""
    b = load_benchmark()
    seen: dict[str, str] = {}
    for subset in SPLIT_SUBSETS:
        for unit in set(b.unit[split_mask(b, subset)]):
            assert unit not in seen, f"{unit} is in both {seen.get(unit)} and {subset}"
            seen[unit] = subset
    assert len(seen) == len(set(b.unit))


def test_committed_roster_covers_every_benchmark_rock_exactly_once():
    b = load_benchmark()
    masks = [split_mask(b, s) for s in SPLIT_SUBSETS]
    assert sum(int(m.sum()) for m in masks) == len(b)
    assert (sum(m.astype(int) for m in masks) == 1).all()


def test_test_silica_range_sits_inside_train():
    """The adapter carries SiO2 terms on q and amp presence, so a test rock more
    extreme than anything it was fit on measures extrapolation, not adaptation."""
    b = load_benchmark()
    scored = b.clean & b.has_magnitude
    train = b.sio2[split_mask(b, "train") & scored]
    test = b.sio2[split_mask(b, "test") & scored]
    assert train.min() <= test.min() and test.max() <= train.max()


def test_unit_grouped_split_is_deterministic_given_a_seed():
    """The carve-out is the board strategies get compared on; it must not drift."""
    b = load_benchmark()
    first = unit_grouped_split(b, seed=0, n_candidates=20)
    assert (first == unit_grouped_split(b, seed=0, n_candidates=20)).all()
    assert (first != unit_grouped_split(b, seed=7, n_candidates=20)).any()


def test_split_balance_prefers_slices_that_match_the_population():
    """The criterion reads truth marginals only, so selecting on it cannot leak
    a model output -- but it does have to rank a lopsided split as worse."""
    n = 90
    presence = np.zeros((n, len(GROUPS)))
    presence[: n // 2, GROUP_INDEX["amp"]] = 1.0
    sio2 = np.linspace(50, 75, n)
    even = np.array(list(SPLIT_SUBSETS) * (n // 3))
    lopsided = np.array(["train"] * 30 + ["val"] * 30 + ["test"] * 30)
    assert split_balance(even, presence, sio2)["score"] < (
        split_balance(lopsided, presence, sio2)["score"]
    )


def test_unknown_split_subset_is_rejected():
    with pytest.raises(ValueError, match="unknown real-rock subset"):
        load_real_rock_split("holdout")

"""Zero-shot sim-to-real eval: the MAGEMin-trained ModeHead on real volcanic rocks.

Scores the trained heads against `data/real_mineralogy/` (point-counted and
visually-estimated phenocryst modes with matching whole-rock chemistry) with
nothing fit -- a fixed coarsening of the 21-phase vocabulary onto the 7 groups
real modal data actually resolves, then the same hurdle NLL the corpus eval
reports. That number is the sealed transfer gap and the blueprint for the thin
real-world recalibrator that comes next: its residual shape decides which per-group
corrections the recalibrator needs.

The coarsening is exact, not an approximation. Presence composes as a noisy-OR
because the head emits independent Bernoulli marginals; magnitude composes by
Dirichlet aggregation, and dropping the phases real modal data cannot resolve
is the Dirichlet sub-composition property that `masked_dirichlet_nll` already
rests on.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from scipy.special import betainc, betaincinv
from scipy.stats import gaussian_kde, spearmanr
from sklearn.metrics import roc_auc_score

from cnmineral.data.bulk_chemistry import tokenize_bulk_frame
from cnmineral.data.tokens import VOCAB
# GROUPS / GROUP_INDEX / MODEL_TO_GROUP are defined in `phase_vocab` beside the
# 21-phase vocabulary they coarsen, and re-exported here because this module is
# where callers meet the benchmark.
from cnmineral.data.phase_vocab import (  # noqa: F401
    GROUP_INDEX,
    GROUPS,
    MODEL_TO_GROUP,
    PHASE_INDEX,
    SOLID_PHASES,
)
from cnmineral.eval.mode_head import (
    conditional_mean,
    magnitude_parity_stats,
    presence_scores,
    reliability_table,
)
from cnmineral.eval.summary_cards import (
    LEGEND_FS,
    SUPTITLE_FS,
    TICK_FS,
    TITLE_FS,
    XLABEL_FS,
)
from cnmineral.losses.mode_loss import masked_dirichlet_nll, mixture_dirichlet_nll

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPO_ROOT / "data" / "real_mineralogy"
RUNS_ROOT = REPO_ROOT / "runs"

# `modes.csv` magemin_phase -> comparison group. `groundmass` is dropped
# rather than mapped: the label is the crystals-renormalized solid assemblage,
# so the truth vector is renormalized to the phenocryst cargo the same way.
TRUTH_TO_GROUP: dict[str, str] = {
    "fsp": "fsp",
    "px_unsplit": "px", "cpx": "px", "opx": "px",
    "amp": "amp",
    "bi": "bi",
    "ol": "ol",
    "opq_unsplit": "opq",
    "q": "q",
}
DROPPED_TRUTH_PHASES: frozenset[str] = frozenset({"groundmass"})

# Representative densities (g/cm3, webmineral.com measured values) keyed by
# the as-reported mineral name, so a lumped "pyroxene" and a specific
# "hypersthene" convert differently. Point counts and visual estimates are
# volume fractions; MAGEMin modes are mass fractions, so truth converts before
# it is coarsened. Volume is verified rather than assumed: du Bray's workbooks
# never label the abundance block, but DS 764's publication text states the
# units, and MSH's ch30 says "Optical modes, as volume percent". Only ratios
# matter -- the one with real leverage is Fe-Ti
# oxides against the silicates, roughly 1.8x. Solid-solution names interpolate
# end members at an expected composition (arc olivine near Fo80, andesine
# plagioclase), which is an assumption, not a measurement; a label spanning
# distinct minerals takes the plain mean of its members (feldspar over the two
# feldspars, pyroxene over augite/hypersthene/bronzite, Fe-Ti oxides over
# magnetite/ilmenite, silica phases over tridymite/cristobalite -- MSH
# groundmass silica, disjoint from the quartz row).
MINERAL_DENSITY: dict[str, float] = {
    "plagioclase": 2.68,
    "alkali feldspar": 2.57,
    # DS 1099's point counts sometimes tally one total feldspar rather than
    # splitting plagioclase from alkali feldspar; the midpoint covers it, and
    # both names coarsen to `fsp` so the split never reaches the score.
    "feldspar": 2.63,
    "pyroxene": 3.40,
    "augite": 3.40,
    "hypersthene": 3.45,
    "bronzite": 3.35,
    "hornblende": 3.15,
    "cummingtonite": 3.30,
    "biotite": 3.05,
    "olivine": 3.50,
    "Fe-Ti oxides": 4.94,
    "quartz": 2.65,
    "silica phases": 2.32,
}

# du Bray's numeric grid bottoms out at 0.5 vol% (p10 of every abundance it
# does report), so a TR / <1 row means "present, below that grid". Used as the
# upper bound of the left-censored magnitude.
TRACE_BOUND_VOL_PCT: float = 0.5

# `alteration_extent` is graded 1 (least altered) to 4, with +/- modifiers,
# plus a literal "fresh" for the precision-tier eruptions.
CLEAN_GRADES: frozenset[int] = frozenset({1, 2})

PRECISION_SOURCES: frozenset[str] = frozenset(
    {"msh_pallister2008", "pinatubo_pallister1996"}
)

# Most rocks carry one modal measurement, but the 70 du Bray rocks in DS 1099's
# Appendix 10 carry two: the visual phenocryst estimate every du Bray rock has,
# and a microscope point count of the same thin section. Where both exist the
# benchmark reads one of them, and which one is a choice rather than a
# measurement -- so it is a parameter, and its default is the visual estimate
# that every number scored before 2026-08-06 was measured against.
VISUAL_BASIS: str = "phenocryst_vol_visual"
POINTCOUNT_BASIS: str = "optical_vol_pointcount"


@dataclass(frozen=True)
class RealRockBenchmark:
    """One row per scoreable real rock, aligned across every array.

    Duck-types the attributes `eval.latent.bulk_latents` reads (`bulk_ids`,
    `bulk_src`, `bulk_frac`), so the frozen encoder runs over real rocks
    through the same path as the corpus.

    Truth splits three ways per group, because the sources report three
    different things: `presence` is what is there, `numeric` is where an
    abundance was actually quantified (the magnitude vector lives on that
    sub-composition and sums to 1 over it), and `censored` marks
    present-but-unquantified trace phases, whose share is known only to lie
    below `censor_bound`.
    """

    bulk_ids: np.ndarray
    bulk_src: torch.Tensor
    bulk_frac: torch.Tensor
    presence: np.ndarray
    magnitude: np.ndarray
    numeric: np.ndarray
    censored: np.ndarray
    censor_bound: np.ndarray
    has_magnitude: np.ndarray
    source: np.ndarray
    precision_tier: np.ndarray
    grade: np.ndarray
    clean: np.ndarray
    sio2: np.ndarray
    unit: np.ndarray

    def __len__(self) -> int:
        return len(self.bulk_ids)

    def subset(self, mask: np.ndarray) -> "RealRockBenchmark":
        """The rocks selected by a boolean mask, arrays kept aligned."""
        mask = np.asarray(mask, dtype=bool)
        return RealRockBenchmark(
            bulk_ids=self.bulk_ids[mask],
            bulk_src=self.bulk_src[torch.as_tensor(mask)],
            bulk_frac=self.bulk_frac[torch.as_tensor(mask)],
            presence=self.presence[mask],
            magnitude=self.magnitude[mask],
            numeric=self.numeric[mask],
            censored=self.censored[mask],
            censor_bound=self.censor_bound[mask],
            has_magnitude=self.has_magnitude[mask],
            source=self.source[mask],
            precision_tier=self.precision_tier[mask],
            grade=self.grade[mask],
            clean=self.clean[mask],
            sio2=self.sio2[mask],
            unit=self.unit[mask],
        )


def parse_alteration_grade(values: pd.Series) -> np.ndarray:
    """Alteration grade as an int, -1 where ungraded.

    du Bray grades carry modifiers (`1+`, `2-`), which qualify the grade
    rather than change it, so the leading digit is the grade. The
    precision-tier eruptions report a literal `fresh`, which is grade 1 by
    any reading.
    """
    text = values.astype(str).str.strip()
    grade = pd.to_numeric(text.str.extract(r"^(\d)")[0], errors="coerce")
    grade = grade.where(text.str.lower() != "fresh", 1.0)
    return grade.fillna(-1).to_numpy(dtype=np.int64)


def _truth_arrays(
    modes: pd.DataFrame,
    keys: np.ndarray,
    densities: dict[str, float],
    trace_bound_vol_pct: float,
) -> dict[str, np.ndarray]:
    """Build the per-group truth arrays for `keys`, in that order.

    Volume percent converts to mass before coarsening (per-raw-name density),
    the numeric entries renormalize to a composition over the groups that
    carry a number, and each censored group gets the renormalized bound its
    trace rows imply.
    """
    n, p = len(keys), len(GROUPS)
    presence = np.zeros((n, p), dtype=np.float64)
    mass = np.zeros((n, p), dtype=np.float64)
    trace_row = np.zeros((n, p), dtype=bool)
    censor_mass = np.zeros((n, p), dtype=np.float64)

    row_of = {k: i for i, k in enumerate(keys)}
    scored = modes[modes["magemin_phase"].isin(TRUTH_TO_GROUP)]
    for key, phase, magemin_phase, abundance in zip(
        scored["key"], scored["phase"], scored["magemin_phase"],
        scored["abundance_pct"],
    ):
        i = row_of.get(key)
        if i is None:
            continue
        # a reported 0.0 means the phase was looked for and not found, which
        # is the one case where a row in this table is not a presence.
        if pd.notna(abundance) and abundance <= 0:
            continue
        j = GROUP_INDEX[TRUTH_TO_GROUP[magemin_phase]]
        presence[i, j] = 1.0
        density = densities[phase]
        if pd.notna(abundance):
            mass[i, j] += abundance * density
        else:
            trace_row[i, j] = True
            censor_mass[i, j] += trace_bound_vol_pct * density

    # coarsening can put a quantified and a trace mineral in the same group
    # (a numeric hornblende beside a trace cummingtonite). The group then has
    # a number, so it is scored as observed and its bound is discarded --
    # censored means no number for the group at all.
    numeric = mass > 0
    censored = trace_row & ~numeric
    censor_mass[~censored] = 0.0

    total = mass.sum(1, keepdims=True)
    has_magnitude = total[:, 0] > 0
    magnitude = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)

    # the bound is expressed against the mass the rock would have if every
    # trace phase sat exactly at its bound, so it is the conservative reading
    # (a slightly smaller share than renormalizing by the numeric mass alone).
    full = total + censor_mass.sum(1, keepdims=True)
    censor_bound = np.divide(
        censor_mass, full, out=np.zeros_like(censor_mass), where=full > 0
    )
    return {
        "presence": presence,
        "magnitude": magnitude,
        "numeric": numeric,
        "censored": censored,
        "censor_bound": censor_bound,
        "has_magnitude": has_magnitude,
    }


def preferred_basis(modes: pd.DataFrame, prefer_basis: str) -> pd.DataFrame:
    """One modal measurement per rock, `prefer_basis` winning where two exist.

    A rock measured only one way keeps what it has, so this is a no-op for
    every rock but the 70 that carry both a visual estimate and a point count.
    Summing the two would double-count the same thin section.

    A `modes` table with no `basis` column at all is one measurement per rock
    by construction and needs no resolution.
    """
    if "basis" not in modes.columns:
        return modes
    both = modes.groupby("key")["basis"].transform("nunique") > 1
    if both.any() and prefer_basis not in set(modes.loc[both, "basis"]):
        # otherwise a misspelled basis silently drops every doubly-measured
        # rock's modes, and the rock leaves the benchmark rather than raising
        raise ValueError(
            f"prefer_basis {prefer_basis!r} matches none of the measurements on "
            f"the rocks that carry more than one: "
            f"{sorted(set(modes.loc[both, 'basis']))}"
        )
    return modes[~both | (modes["basis"] == prefer_basis)]


def load_benchmark(
    benchmark_dir: Path | str = BENCHMARK_DIR,
    *,
    densities: dict[str, float] = MINERAL_DENSITY,
    trace_bound_vol_pct: float = TRACE_BOUND_VOL_PCT,
    prefer_basis: str = VISUAL_BASIS,
) -> RealRockBenchmark:
    """Every real rock carrying both whole-rock chemistry and modal mineralogy.

    Rocks without chemistry (Pinatubo's `CN6791-h`, which has no Table 7 row
    and takes none rather than borrow a neighbour's) or without mode rows drop
    out here; the freshness cut is a mask on the result, not a filter, so the
    altered rocks stay available as a diagnostic.

    `densities`, `trace_bound_vol_pct` and `prefer_basis` default to the
    shipped mapping and are parameters because none is a measurement -- one is
    a mid-series stand-in for a solid solution, one is read off a reporting
    grid, and one picks between two rulers applied to the same thin section.
    `explorations/real_rock_mapping` perturbs the first two to put an error bar
    on the numbers this benchmark produces;
    `explorations/real_mineralogy/run_pointcount_contrast.py` moves the third.
    """
    benchmark_dir = Path(benchmark_dir)
    samples = pd.read_csv(benchmark_dir / "samples.csv")
    modes = pd.read_csv(benchmark_dir / "modes.csv")
    for frame in (samples, modes):
        frame["key"] = frame["source"] + "|" + frame["sample_id"].astype(str)
    modes = preferred_basis(modes, prefer_basis)

    unknown = set(modes["phase"]) - set(densities) - DROPPED_TRUTH_PHASES
    if unknown:
        raise ValueError(
            f"modes.csv reports minerals with no density: {sorted(unknown)}; "
            "add them to MINERAL_DENSITY"
        )

    scoreable = samples["sio2"].notna() & samples["key"].isin(set(modes["key"]))
    samples = samples[scoreable].reset_index(drop=True)

    src, frac = tokenize_bulk_frame(samples)
    keys = samples["key"].to_numpy()
    truth = _truth_arrays(modes, keys, densities, trace_bound_vol_pct)
    grade = parse_alteration_grade(samples["alteration_extent"])

    return RealRockBenchmark(
        bulk_ids=keys,
        bulk_src=torch.from_numpy(src),
        bulk_frac=torch.from_numpy(frac),
        source=samples["source"].to_numpy(),
        precision_tier=samples["source"].isin(PRECISION_SOURCES).to_numpy(),
        grade=grade,
        clean=np.isin(grade, list(CLEAN_GRADES)),
        sio2=samples["sio2"].to_numpy(dtype=np.float64),
        unit=samples["unit"].to_numpy(),
        **truth,
    )


SPLIT_SUBSETS: tuple[str, ...] = ("train", "val", "test")
SPLIT_FRACTIONS: tuple[float, ...] = (0.60, 0.15, 0.25)
SIO2_QUANTILES: tuple[float, ...] = (0.05, 0.5, 0.95)
SPLIT_PATH = Path(__file__).resolve().parents[1] / "data" / "real_rock_unit_split.csv"


def _unit_sizes(units: np.ndarray) -> list[tuple[str, int]]:
    """Units paired with their rock counts, largest first, name-tiebroken."""
    labels, counts = np.unique(units, return_counts=True)
    return sorted(zip(labels, counts), key=lambda kv: (-kv[1], kv[0]))


def _assign_units(
    order: list[tuple[str, int]], fractions: tuple[float, ...],
    total: int, rng: np.random.Generator,
) -> dict[str, str]:
    """Greedy size-balanced assignment of units to subsets.

    Walks units largest-first and gives each to whichever subset is furthest
    below its rock quota, so no single unit can decide a slice.  Ties break at
    random, which is where candidate diversity comes from -- 73 of the 147
    units are singletons, so most of the assignment is a tie.
    """
    targets = {s: f * total for s, f in zip(SPLIT_SUBSETS, fractions)}
    filled = dict.fromkeys(SPLIT_SUBSETS, 0)
    assignment: dict[str, str] = {}
    for unit, size in order:
        deficits = np.array([targets[s] - filled[s] for s in SPLIT_SUBSETS])
        best = np.flatnonzero(deficits >= deficits.max() - 1e-9)
        subset = SPLIT_SUBSETS[int(rng.choice(best))]
        assignment[unit] = subset
        filled[subset] += size
    return assignment


def split_balance(
    subsets: np.ndarray, presence: np.ndarray, sio2: np.ndarray
) -> dict[str, float]:
    """How far each slice drifts from the whole population it is drawn from.

    Two departures matter for this benchmark and both are checked: a slice
    whose group base rates are off scores a different task, and a slice that
    does not span the silica trend cannot exercise the q / amp corrections
    that are the only silica-dependent part of the recalibrator.  Reads truth
    marginals only -- never a model output -- so selecting on it cannot leak.

    `SIO2_QUANTILES` stays interior on purpose.  The silica tails here are one
    or two rocks -- 6 above 80 wt% with the largest at 90.2 against a 99th
    percentile of 80.0, and 9 below 52.7 -- so scoring the endpoints hands the
    whole split selection to an outlier.  What the endpoints are actually
    needed for is the containment rule in `unit_grouped_split`, which is a
    constraint rather than a term.
    """
    base = presence.mean(0)
    span = max(float(sio2.max() - sio2.min()), 1e-9)
    q_all = np.quantile(sio2, SIO2_QUANTILES)
    rate_dev, sio2_dev = 0.0, 0.0
    for subset in SPLIT_SUBSETS:
        m = subsets == subset
        if not m.any():
            return {"rate_dev": np.inf, "sio2_dev": np.inf, "score": np.inf}
        rate_dev = max(rate_dev, float(np.abs(presence[m].mean(0) - base).max()))
        q = np.quantile(sio2[m], SIO2_QUANTILES)
        sio2_dev = max(sio2_dev, float(np.abs(q - q_all).max() / span))
    return {
        "rate_dev": rate_dev,
        "sio2_dev": sio2_dev,
        "score": rate_dev + sio2_dev,
    }


def unit_grouped_split(
    benchmark: RealRockBenchmark,
    *,
    seed: int = 0,
    fractions: tuple[float, ...] = SPLIT_FRACTIONS,
    n_candidates: int = 400,
) -> np.ndarray:
    """Assign every rock to train / val / test as whole stratigraphic units.

    Rocks from one eruption share nearly identical chemistry, so a rock-level
    split leaks -- the same call the corpus makes with bulk-level splits.  The
    unit distribution is severely skewed (135 units over 640 scoreable rocks,
    median 2, 65 singletons, largest 66), which is why units are assigned by
    rock count rather than by count of units: otherwise which two large units
    land in the test slice decides what the test slice measures.

    Candidates are drawn and scored by `split_balance`, and the best feasible
    one wins.  Feasible means the test slice's silica range sits inside the
    train slice's: the recalibrator carries SiO2 terms on q and amp, and scoring it
    on rocks more extreme than anything it was fit on measures extrapolation
    rather than adaptation.  That constraint is what the tails are for --
    they are too thin to balance on (see `split_balance`).

    Altered rocks are assigned along with the rest so a unit never straddles a
    boundary, but balance is scored on the rocks the headline actually scores.
    """
    rng = np.random.default_rng(seed)
    order = _unit_sizes(benchmark.unit)
    scored = benchmark.clean & benchmark.has_magnitude
    total = int(scored.sum())

    best, best_score, n_feasible = None, np.inf, 0
    for _ in range(n_candidates):
        assignment = _assign_units(order, fractions, total, rng)
        subsets = np.array([assignment[u] for u in benchmark.unit])
        train_sio2 = benchmark.sio2[scored & (subsets == "train")]
        test_sio2 = benchmark.sio2[scored & (subsets == "test")]
        if not train_sio2.size or not test_sio2.size:
            continue
        if test_sio2.min() < train_sio2.min() or test_sio2.max() > train_sio2.max():
            continue
        n_feasible += 1
        balance = split_balance(
            subsets[scored], benchmark.presence[scored], benchmark.sio2[scored]
        )
        if balance["score"] < best_score:
            best, best_score = subsets, balance["score"]
    if best is None:
        raise ValueError(
            f"no candidate of {n_candidates} put the test silica range inside "
            "train's; raise n_candidates or revisit the fractions"
        )
    return best


def load_real_rock_split(subset: str) -> np.ndarray:
    """Bulk ids of the committed real-rock roster subset.

    Generated once by `scripts/build/build_real_rock_split.py` and never
    recomputed at load time: the whole point of a fixed carve-out is that
    models and adaptation strategies are compared on a board that does not
    move between runs.
    """
    if subset not in SPLIT_SUBSETS:
        raise ValueError(
            f"unknown real-rock subset {subset!r}; expected one of {SPLIT_SUBSETS}"
        )
    roster = pd.read_csv(SPLIT_PATH)
    rows = roster[roster["subset"] == subset]
    return (rows["source"] + "|" + rows["sample_id"].astype(str)).to_numpy()


def split_mask(benchmark: RealRockBenchmark, subset: str) -> np.ndarray:
    """Boolean mask selecting `benchmark` rows in the committed roster subset."""
    return np.isin(benchmark.bulk_ids, load_real_rock_split(subset))


LOG_FRAC_FLOOR: float = 1e-12


def raw_chemistry(benchmark: RealRockBenchmark) -> torch.Tensor:
    """The assay itself, in the space the encoder's own inputs live in.

    `log10` abundance where an element is populated and a neutral 0 where it is
    not: absent means unmeasured, not zero abundance, and `log10(0)` has no
    value to place.  Logs also stop the majors from dominating a representation
    the traces should speak to -- the same two conventions the raw-input
    ceiling used, so a head reading this and a head reading the latent are
    being asked the same question about the same chemistry.
    """
    frac = benchmark.bulk_frac.to(torch.float64)
    log_frac = torch.log10(frac.clamp_min(LOG_FRAC_FLOOR))
    return torch.where(benchmark.bulk_src != 0, log_frac, torch.zeros_like(log_frac))


def coarsen_presence(p: np.ndarray) -> np.ndarray:
    """Per-group presence probability from the 21 per-phase marginals.

    A group is present when any member phase is. The head's presence branch
    emits independent Bernoulli marginals given the latent, so the noisy-OR
    `1 - prod(1 - p_i)` is that probability exactly, not an approximation.
    """
    out = np.ones((len(p), len(GROUPS)), dtype=np.float64)
    for phase, group in MODEL_TO_GROUP.items():
        out[:, GROUP_INDEX[group]] *= 1.0 - p[:, PHASE_INDEX[phase]]
    return 1.0 - out


def coarsen_alpha(
    alpha: np.ndarray, support: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group Dirichlet concentrations, plus the off-vocabulary total.

    Merging components of a Dirichlet sums their concentrations, so the
    coarsened magnitude distribution is exact. Only the phases on `support`
    take part -- the distribution lives on the support, and an off-support
    concentration is not a component of it. The phases no source reports are
    returned separately rather than discarded: their share is the
    off-vocabulary mass, which reads as how much of the predicted assemblage
    the benchmark structurally cannot see.
    """
    on = alpha * support if alpha.ndim == 2 else alpha * support[:, None, :]
    out = np.zeros(on.shape[:-1] + (len(GROUPS),), dtype=np.float64)
    for phase, group in MODEL_TO_GROUP.items():
        out[..., GROUP_INDEX[group]] += on[..., PHASE_INDEX[phase]]
    off_vocab = np.array(
        [on[..., PHASE_INDEX[ph]] for ph in SOLID_PHASES if ph not in MODEL_TO_GROUP]
    ).sum(0)
    return out, off_vocab


def coarsen_support(support: np.ndarray) -> np.ndarray:
    """Group-level support: a group is on when any member phase is on.

    The boolean counterpart of `coarsen_alpha` -- it names which groups a
    given 21-phase support actually carries concentration for, which is the
    denominator any share conditioned on that support renormalizes over.
    """
    out = np.zeros((len(support), len(GROUPS)), dtype=bool)
    for phase, group in MODEL_TO_GROUP.items():
        out[:, GROUP_INDEX[group]] |= support[:, PHASE_INDEX[phase]] > 0
    return out


def truth_forced_support(
    presence7: np.ndarray, p21: np.ndarray, *, member_threshold: float = 0.5
) -> np.ndarray:
    """A 21-phase support consistent with the group-level truth.

    The magnitude branch is support-conditioned, but real modal data
    constrains only the group level -- "pyroxene is present" does not say
    whether that is cpx, opx, or both. So truth forces what it knows and the
    model resolves what it does not: within a truth-present group, the member
    phases the model puts above `member_threshold` turn on (its most likely
    member if none clears the bar); truth-absent groups turn fully off. The
    phases no source reports are left to the model, since truth never asserted
    their absence.
    """
    support = (p21 > member_threshold).astype(np.float64)
    for group, gi in GROUP_INDEX.items():
        members = [PHASE_INDEX[ph] for ph, g in MODEL_TO_GROUP.items() if g == group]
        present = presence7[:, gi] > 0
        support[np.ix_(~present, members)] = 0.0
        empty = present & (support[:, members].sum(1) == 0)
        if empty.any():
            best = np.asarray(members)[p21[np.ix_(empty, members)].argmax(1)]
            support[np.flatnonzero(empty), best] = 1.0
    return support


def real_rock_nll(
    p7: np.ndarray,
    alpha7: np.ndarray,
    presence: np.ndarray,
    magnitude: np.ndarray,
    numeric: np.ndarray,
    log_w: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """The hurdle NLL in group space -- the headline, and the recalibrator's loss.

    Assembled exactly as the corpus eval assembles it, on the coarsened
    vocabulary: presence BCE summed over groups, plus the Dirichlet NLL of the
    observed sub-composition. The recalibrator that comes next rescales presence
    logits and magnitude concentrations per group, and both of those are fit
    by minimizing this -- Platt scaling *is* Bernoulli-NLL fitting -- so the
    zero-shot value is that loss at the identity correction and the recalibrator's
    gain reads as nats recovered on one scale.

    Rocks whose observed sub-composition has a single group contribute exactly
    0 to the Dirichlet term (a degenerate point mass, not a free win); the
    caller reports how many.

    `log_w` is None for a single-Dirichlet alpha7 [R, 7]; the mixture head
    passes gate log-weights with alpha7 [R, K, 7].
    """
    p = torch.as_tensor(np.clip(p7, 1e-7, 1 - 1e-7), dtype=torch.float64)
    bern = F.binary_cross_entropy(
        p, torch.as_tensor(presence, dtype=torch.float64), reduction="none",
    ).sum(-1)
    # off-mask concentrations must stay positive going in: the masked NLL
    # forms `mask * lgamma(alpha)`, and lgamma(0) is +inf, so a zero there
    # gives 0 * inf = NaN instead of the intended no-op. Coarsening produces
    # exact zeros wherever a group has no member on the support, and those
    # groups are excluded by the mask anyway, so any positive filler works.
    mask = numeric if log_w is None else numeric[:, None, :]
    alpha = np.where(mask, alpha7, 1.0)
    if log_w is None:
        dirichlet = masked_dirichlet_nll(
            torch.as_tensor(alpha, dtype=torch.float64),
            torch.as_tensor(magnitude, dtype=torch.float64),
            torch.as_tensor(numeric, dtype=torch.float64),
            reduction="none",
        )
    else:
        dirichlet = mixture_dirichlet_nll(
            torch.as_tensor(log_w, dtype=torch.float64),
            torch.as_tensor(alpha, dtype=torch.float64),
            torch.as_tensor(magnitude, dtype=torch.float64),
            torch.as_tensor(numeric, dtype=torch.float64),
            reduction="none",
        )
    return {
        "bernoulli": bern.numpy(),
        "dirichlet": dirichlet.numpy(),
        "joint": (bern + dirichlet).numpy(),
    }


CLIMATOLOGY_GRID: np.ndarray = np.geomspace(0.5, 400, 80)


def climatology(benchmark: RealRockBenchmark) -> dict:
    """The do-nothing forecast every scored figure is read against.

    Per-group base rates and the mean measured composition, quoted for every
    rock regardless of its chemistry -- the reference forecast of weather
    verification.  A cost in nats has no scale until you know what ignoring
    the rock entirely would have cost.

    Fit on the train/val carve-out and scored everywhere, so it never reads
    its own answers back on the rocks it is compared on.  Its one free
    parameter is the Dirichlet concentration, chosen to score as well as it
    possibly can, because a null worth beating should be the strongest
    version of itself.

    Unlike the model it is not zero-shot: it has seen real rocks in
    aggregate.  That asymmetry is the point of the comparison rather than a
    flaw in it -- the gap measures what simulator training buys against a
    little target-domain bookkeeping.
    """
    scored = benchmark.clean & benchmark.has_magnitude
    fit = (split_mask(benchmark, "train") | split_mask(benchmark, "val")) & scored
    rate = (benchmark.presence[fit] > 0).mean(0)
    mean_comp = benchmark.magnitude[fit].sum(0)
    mean_comp /= mean_comp.sum()
    p7 = np.tile(rate, (len(benchmark), 1))
    scores = [
        real_rock_nll(
            p7, np.tile(k * mean_comp, (len(p7), 1)), benchmark.presence,
            benchmark.magnitude, benchmark.numeric,
        )["dirichlet"][fit].mean()
        for k in CLIMATOLOGY_GRID
    ]
    concentration = float(CLIMATOLOGY_GRID[int(np.argmin(scores))])
    return {
        "p7": p7,
        "alpha7": np.tile(concentration * mean_comp, (len(p7), 1)),
        "concentration": concentration,
        "n_fit": int(fit.sum()),
    }


def censored_consistency(
    alpha7: np.ndarray,
    support: np.ndarray,
    censored: np.ndarray,
    censor_bound: np.ndarray,
    log_w: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """How well the model respects the bounds on present-but-unquantified phases.

    A quarter of amphibole's real appearances (a third of quartz's) are
    reported only as a trace, so scoring magnitude on quantified entries alone
    would silently condition the number on the phase being abundant. Those
    entries are left-censored instead: truth says the share sits below a known
    bound, and the Dirichlet marginal on the support is a Beta, so
    `P(share < bound)` is closed-form.

    Kept beside the headline rather than added into it: the exact censored
    likelihood needs an integral over the censored region, and multiplying a
    density on the observed groups by a marginal tail probability is not a
    joint likelihood, because the Dirichlet couples the two. Returns the
    per-instance probabilities and its own NLL, which is what a recalibrator would
    add to the objective if these instances should inform the fit.
    """
    # per-component Beta tails, gate-weighted at the end; the log_w=None path
    # is the K=1 special case with weight exactly 1
    a = alpha7 if log_w is not None else alpha7[:, None, :]
    w = np.exp(log_w) if log_w is not None else np.ones((len(a), 1))
    alpha0 = (a * support[:, None, :]).sum(-1, keepdims=True)
    beta = np.clip(alpha0 - a, 1e-9, None)
    prob_k = np.full(a.shape, np.nan)
    where = censored & (censor_bound > 0)
    where_k = np.broadcast_to(where[:, None, :], a.shape)
    bound_k = np.broadcast_to(
        np.clip(censor_bound, 1e-12, 1 - 1e-12)[:, None, :], a.shape
    )
    prob_k[where_k] = betainc(a[where_k], beta[where_k], bound_k[where_k])
    prob = (w[..., None] * prob_k).sum(1)
    mean_share_k = np.divide(
        a, np.clip(alpha0, 1e-12, None), out=np.zeros_like(a),
        where=alpha0 > 0,
    )
    mean_share = (w[..., None] * mean_share_k).sum(1)
    return {
        "prob_below_bound": prob,
        "nll": -np.log(np.clip(prob, 1e-12, None)),
        "mean_below_bound": (mean_share < censor_bound) & where,
        "scored": where,
    }


def presence_group_stats(p7: np.ndarray, presence: np.ndarray) -> dict[str, np.ndarray]:
    """Per-group AUROC and the threshold maximizing F1.

    AUROC leads because it is threshold-free: the head's probabilities were
    calibrated against how often MAGEMin grows a phase, which is not how often
    a geologist logs it, so any fixed cutoff conflates ranking quality with
    that offset. The best-F1 threshold separates them -- it is a direct read
    on the per-group intercept a Platt correction would learn.
    """
    n_groups = p7.shape[1]
    auroc = np.full(n_groups, np.nan)
    best_threshold = np.full(n_groups, np.nan)
    best_f1 = np.full(n_groups, np.nan)
    for i in range(n_groups):
        y = presence[:, i] > 0
        if y.all() or not y.any():
            continue
        auroc[i] = float(roc_auc_score(y, p7[:, i]))
        grid = np.unique(np.round(p7[:, i], 3))
        f1 = np.array([_f1(p7[:, i] > t, y) for t in grid])
        best = int(np.nanargmax(f1))
        best_threshold[i], best_f1[i] = float(grid[best]), float(f1[best])
    return {"auroc": auroc, "best_threshold": best_threshold, "best_f1": best_f1}


def _f1(pred: np.ndarray, true: np.ndarray) -> float:
    tp = float((pred & true).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (pred.sum() + true.sum())


def magnitude_group_stats(
    pred: np.ndarray, magnitude: np.ndarray, numeric: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-group R2 and Spearman on the groups that carry a reported number.

    Conditional on quantifiably-present by construction: trace-only entries
    have no number to correlate against and are scored by
    `censored_consistency` instead.
    """
    n_groups = magnitude.shape[1]
    r2 = np.full(n_groups, np.nan)
    rho = np.full(n_groups, np.nan)
    n = np.zeros(n_groups, dtype=np.int64)
    for i in range(n_groups):
        m = numeric[:, i]
        n[i] = int(m.sum())
        true, p = magnitude[m, i], pred[m, i]
        if n[i] < 3 or true.std() == 0:
            continue
        r2[i] = float(1.0 - ((true - p) ** 2).sum() / ((true - true.mean()) ** 2).sum())
        if p.std() > 0:
            rho[i] = float(spearmanr(true, p).statistic)
    return {"r2": r2, "spearman": rho, "n": n}


def _group_grid(height: float = 2.7):
    fig, axes = plt.subplots(2, 4, figsize=(4 * 3.1, 2 * height), squeeze=False)
    axes.flat[len(GROUPS)].set_visible(False)
    return fig, axes.flat


def _finish(fig, title: str, out_path: Path) -> None:
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def _wilson_interval(
    freq: np.ndarray, count: np.ndarray, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for a binomial proportion, elementwise.

    Wilson rather than normal-approximation: reliability bins on a skewed
    group hold a handful of rocks, exactly where the normal interval
    collapses to zero width at an observed frequency of 0 or 1.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = 1.0 + z**2 / count
        center = (freq + z**2 / (2 * count)) / denom
        half = z * np.sqrt(
            freq * (1 - freq) / count + z**2 / (4 * count**2)
        ) / denom
    return np.clip(center - half, 0.0, 1.0), np.clip(center + half, 0.0, 1.0)


def unit_bootstrap(
    stat_fn, units: np.ndarray, n_boot: int = 1000, seed: int = 0,
    q: tuple[float, float] = (0.025, 0.975),
) -> tuple[np.ndarray, np.ndarray]:
    """`q` percentile band of `stat_fn` under a unit-clustered bootstrap.

    Rocks within a stratigraphic unit share nearly identical chemistry, so
    resampling rows independently would understate every interval -- units
    are resampled whole, the same clustering the paired NLL comparisons use.
    `stat_fn` maps a row-index array to a stat vector; NaN entries (a group
    with too few quantified rocks in a resample) are ignored per component.
    The indices handed to `stat_fn` are positions into `units`, so a caller
    scoring a subset must pass that subset's units and map back itself --
    silently mixing absolute row indices with subset positions resamples the
    wrong rocks and produces intervals that do not contain the estimate.

    `q` is a pair of fractions, matching the interval constants the card and
    curve figures carry. The default is the 95% band the instrument figures
    label themselves with.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(units == u) for u in np.unique(units)]
    stats = np.stack([
        stat_fn(np.concatenate(
            [groups[j] for j in rng.integers(0, len(groups), len(groups))]
        ))
        for _ in range(n_boot)
    ])
    with np.errstate(invalid="ignore"):
        return (
            np.nanpercentile(stats, 100 * q[0], axis=0),
            np.nanpercentile(stats, 100 * q[1], axis=0),
        )


def plot_presence_calibration(
    p7, presence, stats, out_path: Path, subtitle: str
) -> None:
    """Predicted presence probability against observed frequency, per group.

    The diagonal is what calibration on MAGEMin would buy if the simulator's
    phase frequencies matched a geologist's logs. Systematic departure is the
    offset the recalibrator exists to absorb, so read the vertical gap, not just
    the scatter. Each bin carries its rock count and a Wilson interval, so a
    skewed group's nearly-empty bins read as the noise they are instead of
    as structure.
    """
    fig, axes = _group_grid()
    for i, group in enumerate(GROUPS):
        ax = axes[i]
        rel = reliability_table(p7[:, i], presence[:, i])
        ax.plot([0, 1], [0, 1], lw=0.8, color="0.6", ls="--")
        ok = rel["count"] > 0
        lo, hi = _wilson_interval(rel["freq"], rel["count"])
        ax.errorbar(
            rel["mean_p"][ok], rel["freq"][ok],
            yerr=[np.maximum(rel["freq"] - lo, 0)[ok],
                  np.maximum(hi - rel["freq"], 0)[ok]],
            fmt="o-", ms=4, color="tab:orange", lw=1.0, elinewidth=0.9,
            capsize=2,
        )
        for x, y, c in zip(rel["mean_p"][ok], rel["freq"][ok], rel["count"][ok]):
            ax.annotate(f"{int(c)}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=6, color="0.4")
        ax.axhline(presence[:, i].mean(), lw=0.7, color="tab:blue", alpha=0.6)
        ax.set_title(
            f"{group}   AUROC {stats['auroc'][i]:.2f}   ECE {rel['ece']:.2f}",
            fontsize=9,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("predicted probability", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("observed frequency", fontsize=8)
    _finish(fig, f"presence calibration on real rocks -- {subtitle}\n"
                 "blue line: the group's base rate; bin labels: rock count, "
                 "whiskers: 95% Wilson", out_path)


def plot_magnitude_parity(
    alpha7, magnitude, numeric, stats, out_path, subtitle
) -> None:
    """Predicted share interval against measured share, per group.

    The magnitude branch emits a Dirichlet, so each entry draws as its
    Beta-marginal median with a 10-90% credible bar rather than a point:
    a bar missing the diagonal is a confident miss, and `cov80` is the
    fraction of measured shares the bar contains (0.80 when calibrated).
    Ranking quality is the Spearman; R2 stays in metrics.json, where its
    scale-offset sensitivity can be read next to the MAE instead of
    headlining a plot about distributions.
    """
    mask = numeric.astype(np.float64)
    lo, hi = share_interval(alpha7, mask)
    med, _ = share_interval(alpha7, mask, q=(0.5, 0.5))
    fig, axes = _group_grid()
    for i, group in enumerate(GROUPS):
        ax, m = axes[i], numeric[:, i]
        ax.plot([0, 1], [0, 1], lw=0.8, color="0.6", ls="--")
        x = magnitude[m, i]
        ax.errorbar(
            x, med[m, i],
            yerr=[np.maximum(med[m, i] - lo[m, i], 0),
                  np.maximum(hi[m, i] - med[m, i], 0)],
            fmt=".", ms=3, color="tab:orange", elinewidth=0.6, alpha=0.35,
            capsize=0,
        )
        cov = float(((x >= lo[m, i]) & (x <= hi[m, i])).mean()) if m.any() else np.nan
        ax.set_title(
            f"{group}   rho {stats['spearman'][i]:.2f}   cov80 {cov:.2f}"
            f"   n={stats['n'][i]}", fontsize=9,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("measured share", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("predicted share (median, 10-90%)", fontsize=8)
    _finish(fig, f"magnitude interval parity on real rocks -- {subtitle}\n"
                 "quantified entries only; trace-only appearances scored by their "
                 "censoring bound", out_path)


def plot_gap_summary(p7, pred, benchmark, out_path: Path, subtitle: str) -> None:
    """The two corrections a recalibrator would make, per group, on one axis.

    Whiskers are unit-clustered bootstrap 95% intervals: a bar whose
    interval spans zero is a correction the data does not actually demand.
    """
    def presence_stat(rows: np.ndarray) -> np.ndarray:
        return benchmark.presence[rows].mean(0) - p7[rows].mean(0)

    def scale_stat(rows: np.ndarray) -> np.ndarray:
        out = np.full(len(GROUPS), np.nan)
        for i in range(len(GROUPS)):
            m = rows[benchmark.numeric[rows, i]]
            if len(m) >= 3 and pred[m, i].mean() > 0:
                out[i] = np.log(
                    benchmark.magnitude[m, i].mean() / pred[m, i].mean()
                )
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    y = np.arange(len(GROUPS))
    all_rows = np.arange(len(benchmark))
    for ax, stat_fn, xlabel, title in (
        (ax1, presence_stat, "observed rate - mean predicted probability",
         "presence offset (positive: model under-calls)"),
        (ax2, scale_stat, "log(mean measured share / mean predicted share)",
         "magnitude scale offset (positive: model under-sizes)"),
    ):
        value = stat_fn(all_rows)
        lo, hi = unit_bootstrap(stat_fn, benchmark.unit)
        ax.barh(y, value, color=np.where(value > 0, "tab:blue", "tab:red"))
        ax.errorbar(value, y,
                    xerr=[np.maximum(value - lo, 0), np.maximum(hi - value, 0)],
                    fmt="none", ecolor="0.25", elinewidth=1.0, capsize=2.5)
        ax.axvline(0, lw=0.8, color="0.3")
        ax.set_yticks(y, GROUPS)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
    _finish(fig, f"the corrections a thin recalibrator would learn -- {subtitle}\n"
                 "whiskers: unit-clustered bootstrap 95%", out_path)


# The cohort card renders near 17x12.5 in, several times the single-rock
# card's area, so it carries its own type scale: the `summary_cards`
# constants are illegible at this figure size.  The chemistry panel stacks 21
# elements and cannot take the mineral panels' size, so it keeps its own.
COHORT_ELEMENT_FS = 12
COHORT_GROUP_FS = 15
COHORT_GAP_FS = 16
COHORT_LABEL_FS = 18
COHORT_TITLE_FS = 20
COHORT_LEGEND_FS = 14


def _violin(ax, values, y: float, color: str, *, side: str = "both",
            width: float = 0.38, alpha: float = 0.55,
            min_kde: int = 10) -> None:
    """One horizontal kernel density at row `y`, with a median tick.

    Fewer than `min_kde` rocks fall back to plotting them individually: a
    density over a handful of points is a drawing, not an estimate, and the
    groups where that happens are exactly the ones a reader must not
    over-read.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < min_kde or np.ptp(v) < 1e-9:
        offset = {"up": -0.12, "down": 0.12}.get(side, 0.0)
        if len(v):
            ax.plot(v, np.full(len(v), y + offset), "o", ms=4, color=color)
        return
    grid = np.linspace(v.min(), v.max(), 256)
    dens = gaussian_kde(v)(grid)
    dens = dens / dens.max() * width
    if side in ("both", "up"):
        ax.fill_between(grid, y, y - dens, color=color, alpha=alpha, lw=0)
    if side in ("both", "down"):
        ax.fill_between(grid, y, y + dens, color=color, alpha=alpha, lw=0)
    ax.plot([np.median(v)], [y], "|", ms=10, color=color, mew=1.8)


def _share_ticks(ax, fontsize: int) -> None:
    ax.set_xticks(np.log10([0.001, 0.01, 0.1, 0.5, 1.0]),
                  ["0.1%", "1%", "10%", "50%", "100%"], fontsize=fontsize)


def plot_cohort_card(
    p7: np.ndarray,
    forced_share: np.ndarray,
    model_share: np.ndarray,
    model_support: np.ndarray,
    benchmark: RealRockBenchmark,
    out_path: Path,
) -> None:
    """The single-rock card generalized to the whole cohort.

    Every array is already restricted to the rocks being scored, `benchmark`
    with them.  The four panels follow the pipeline: what went in, which
    minerals the model calls, and how much of each read two ways.

    The two mineralogy panels differ only in which assemblage the magnitude
    branch was conditioned on.  The measured-assemblage panel is teacher
    forced, so its predicted and measured counts are equal by construction
    and a presence failure leaves no trace in it; the predicted-assemblage
    panel is where a mineral the model never calls collapses to a handful of
    rocks.  Both are needed, because the second panel's two violins cover
    different rock sets and its predicted shares renormalize over whatever
    the model kept, so it cannot be read as a like-for-like comparison on its
    own.
    """
    floor_log = np.log10(CARD_SHARE_FLOOR)
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
        2, 2, figsize=(17.0, 12.5), gridspec_kw={"height_ratios": [1.15, 1.0]}
    )

    symbol = {z: sym for sym, z in VOCAB}
    src, frac = benchmark.bulk_src.numpy(), benchmark.bulk_frac.numpy()
    pools = {symbol[int(z)]: np.log10(np.clip(frac[src == z] * 100, 1e-4, None))
             for z in np.unique(src[src > 0])}
    order = sorted(pools, key=lambda s: -np.median(pools[s]))
    for i, sym in enumerate(order):
        _violin(ax0, pools[sym], i, "tab:gray", alpha=0.7)
    ax0.set_yticks(range(len(order)), order, fontsize=COHORT_ELEMENT_FS)
    ax0.invert_yaxis()
    ax0.set_xticks([-3, -2, -1, 0, 1], ["0.001", "0.01", "0.1", "1", "10"],
                   fontsize=COHORT_GROUP_FS)
    ax0.set_xlabel("Cation wt%", fontsize=COHORT_LABEL_FS)
    ax0.set_title("Input: Whole-Rock Chemistry", fontsize=COHORT_TITLE_FS)

    for i in range(len(GROUPS)):
        says = np.mean(p7[:, i] > 0.5) * 100
        truth = np.mean(benchmark.presence[:, i] > 0) * 100
        ax1.plot([truth, says], [i, i], lw=1.4, color="0.7", zorder=0)
        ax1.plot(truth, i, "o", ms=8, color="tab:blue")
        ax1.plot(says, i, "o", ms=8, color="tab:orange", mfc="white", mew=1.8)
        ax1.annotate(f"{says - truth:+.0f}", (112, i), fontsize=COHORT_GAP_FS,
                     color="0.3", va="center", ha="left")
    ax1.set_xlim(-4, 140)
    ax1.set_xticks([0, 50, 100], ["0", "50", "100"], fontsize=COHORT_GROUP_FS)
    ax1.set_xlabel("Fraction of rocks (%)", fontsize=COHORT_LABEL_FS)
    ax1.set_title("Phase Presence", fontsize=COHORT_TITLE_FS)
    ax1.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=9, color="tab:blue",
               label="is there"),
        Line2D([], [], marker="o", ls="none", ms=9, color="tab:orange",
               mfc="white", mew=1.8, label="model calls it"),
    ], fontsize=COHORT_LEGEND_FS, loc="upper left", framealpha=0.9)

    n_total = len(benchmark)
    for ax, share, pmask, title in (
        (ax2, forced_share, benchmark.numeric,
         "Modal Mineralogy | measured assemblage"),
        (ax3, model_share, model_support,
         "Modal Mineralogy | predicted assemblage"),
    ):
        for i in range(len(GROUPS)):
            tm, pm = benchmark.numeric[:, i], pmask[:, i]
            _violin(ax, np.log10(np.maximum(benchmark.magnitude[tm, i],
                                            CARD_SHARE_FLOOR)),
                    i, "tab:blue", side="down")
            _violin(ax, np.log10(np.maximum(share[pm, i], CARD_SHARE_FLOOR)),
                    i, "tab:orange", side="up")
            # each count sits on the side its own violin is drawn on
            ax.annotate(f"{int(pm.sum())}/{n_total}", (0.60, i - 0.20),
                        fontsize=COHORT_GROUP_FS, color="tab:orange",
                        ha="right", va="center")
            ax.annotate(f"{int(tm.sum())}/{n_total}", (0.60, i + 0.20),
                        fontsize=COHORT_GROUP_FS, color="tab:blue",
                        ha="right", va="center")
        ax.set_xlim(floor_log, 0.62)
        _share_ticks(ax, COHORT_GROUP_FS)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.set_xlabel("Fraction of rock's crystalline mass",
                      fontsize=COHORT_LABEL_FS)
        ax.set_title(title, fontsize=COHORT_TITLE_FS)

    for ax in (ax1, ax2, ax3):
        ax.set_yticks(range(len(GROUPS)), GROUPS, fontsize=COHORT_GROUP_FS)
        ax.set_ylim(len(GROUPS) - 0.4, -0.6)
    ax2.legend(handles=[
        Patch(color="tab:orange", alpha=0.55, label="predicted"),
        Patch(color="tab:blue", alpha=0.55, label="measured"),
    ], fontsize=COHORT_LEGEND_FS, loc="upper left", framealpha=0.9)

    # not `_finish`: its suptitle size is tuned for the single-rock card
    fig.suptitle("cohort card (zero-shot)", fontsize=COHORT_TITLE_FS)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_magnitude_ratio(
    share: np.ndarray, magnitude: np.ndarray, numeric: np.ndarray,
    out_path: Path, subtitle: str,
) -> None:
    """Per-rock magnitude error, paired: how many times off, and which way.

    The cohort card's mineralogy panels put two marginal distributions side
    by side, which cannot see a per-rock error -- shuffling the predictions
    among rocks leaves them unchanged.  This is the paired view: one number
    per rock, so the same shuffle destroys it.  Log-spaced because the error
    is multiplicative and the groups span two orders of magnitude in size.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axvline(0, lw=1.0, color="0.35", zorder=0)
    for i, group in enumerate(GROUPS):
        m = numeric[:, i]
        if m.sum() < 3:
            continue
        ratio = np.log2(np.maximum(share[m, i], 1e-9)
                        / np.maximum(magnitude[m, i], 1e-9))
        _violin(ax, ratio, i, "tab:orange", width=0.40)
        ax.annotate(f"median x{2 ** np.median(ratio):.2f}", (1.01, i),
                    fontsize=TICK_FS - 2, color="0.3", va="center",
                    ha="left", xycoords=("axes fraction", "data"), annotation_clip=False)
    ax.set_yticks(range(len(GROUPS)),
                  [f"{g}\nn={int(numeric[:, i].sum())}"
                   for i, g in enumerate(GROUPS)], fontsize=TICK_FS - 1)
    ax.set_ylim(len(GROUPS) - 0.4, -0.6)
    ax.set_xlim(-5, 5)
    ax.set_xticks([-4, -2, -1, 0, 1, 2, 4],
                  ["1/16", "1/4", "1/2", "right", "2x", "4x", "16x"],
                  fontsize=TICK_FS)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.set_xlabel("predicted share divided by measured share",
                  fontsize=XLABEL_FS)
    _finish(fig, f"per-rock magnitude error on real rocks -- {subtitle}\n"
                 "quantified entries only, teacher-forced on the measured "
                 "assemblage", out_path)


def plot_nll_distribution(
    nll: dict, null_nll: dict, out_path: Path, subtitle: str,
) -> None:
    """The headline split into its terms, each against the do-nothing forecast.

    The two halves are different quantities on different scales and neither
    is readable as a single mean, so both are drawn as distributions over
    rocks.  The climatology curve is what the same figure looks like for a
    forecast that ignores the rock; without it a cost in nats has no scale.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, term, name in (
        (axes[0], "bernoulli", "presence"),
        (axes[1], "dirichlet", "composition"),
        (axes[2], "joint", "joint"),
    ):
        for values, color, style, label in (
            (nll[term], "#2a78d6", "-", "model"),
            (null_nll[term], "0.35", ":", "climatology"),
        ):
            v = np.sort(values)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                    lw=2 if style == "-" else 1.6, ls=style, color=color,
                    label=f"{label} (mean {v.mean():.1f})")
        ax.set_xlim(np.percentile(
            np.concatenate([nll[term], null_nll[term]]), [0.5, 98]))
        ax.set_xlabel(f"per-rock {name} NLL (nats)", fontsize=XLABEL_FS)
        ax.set_ylabel("fraction of rocks at or below", fontsize=XLABEL_FS)
        ax.set_title(f"{name}: cost per rock", fontsize=TITLE_FS)
        ax.legend(fontsize=LEGEND_FS - 2, loc="lower right")
    _finish(fig, f"per-rock cost against a forecast that ignores the rock -- "
                 f"{subtitle}\nclimatology: per-group base rates + the mean "
                 "measured composition, fit on the train/val carve-out",
            out_path)


def murphy(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """Reliability and resolution for one group's presence forecasts.

    Brier decomposes as uncertainty - resolution + reliability: how hard the
    question is, credit for separating rocks into groups with different
    outcome rates, and the penalty for probabilities that do not match
    observed frequencies.

    Computed per group and averaged by the caller, never pooled over
    group-rock pairs.  Pooling credits the model with resolution for knowing
    that opaques are common and olivine is rare, which the climatology knows
    too, and inflates the term several-fold.
    """
    base = y.mean()
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0,
                  n_bins - 1)
    rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
            res += m.sum() * (y[m].mean() - base) ** 2
    return rel / len(p), res / len(p)


def plot_presence_brier(
    p7: np.ndarray, null_p7: np.ndarray, presence: np.ndarray,
    unit: np.ndarray, out_path: Path, subtitle: str,
) -> None:
    """Presence scored with calibration folded in, and split into its terms.

    AUROC is invariant to any monotone rescaling of the probabilities, so a
    model that says 95% when it should say 60% keeps a perfect AUROC and a
    terrible Brier.  Since the standing real-rock finding is that ranking
    transfers and scale does not, the ranking-only view is blind to the
    failure being chased; this one is not.

    The decomposition is what makes the figure actionable: resolution the
    model has and the null does not is information about the specific rock,
    and a correction that only rescales probabilities can recover the
    reliability it is losing but cannot manufacture resolution it never had.
    """
    y = (presence > 0).astype(float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2),
                                   gridspec_kw={"width_ratios": [1.6, 1.0]})
    x = np.arange(len(GROUPS) + 1)
    for p, color, label in ((p7, "#2a78d6", "model"),
                            (null_p7, "0.35", "climatology")):
        sq = (p - y) ** 2

        def stat(rows, _s=sq):
            return np.append(_s[rows].mean(0), _s[rows].mean(0).mean())

        value = stat(np.arange(len(y)))
        if label == "climatology":
            for xi, v in zip(x, value):
                ax1.plot([xi - 0.42, xi + 0.42], [v, v], ls=":", lw=1.8,
                         color=color, label=label if xi == 0 else None)
        else:
            lo, hi = unit_bootstrap(stat, unit)
            ax1.bar(x, value, 0.55, color=color, label=label)
            ax1.errorbar(x, value,
                         yerr=[np.maximum(value - lo, 0),
                               np.maximum(hi - value, 0)],
                         fmt="none", ecolor="0.25", elinewidth=1.0, capsize=2)
        terms = np.array([murphy(p[:, i], y[:, i])
                          for i in range(len(GROUPS))]).mean(0)
        j = 0 if label == "model" else 1
        ax2.bar(j - 0.19, terms[0], 0.36, color=color)
        ax2.bar(j + 0.19, terms[1], 0.36, color=color, alpha=0.45, hatch="//")
        ax2.annotate("REL", (j - 0.19, terms[0]), ha="center", va="bottom",
                     fontsize=7, color="0.3")
        ax2.annotate("RES", (j + 0.19, terms[1]), ha="center", va="bottom",
                     fontsize=7, color="0.3")
    ax1.set_xticks(x, list(GROUPS) + ["mean"], fontsize=TICK_FS)
    ax1.set_ylabel("Brier score (lower is better)", fontsize=XLABEL_FS)
    ax1.set_title("presence Brier per group", fontsize=TITLE_FS)
    ax1.legend(fontsize=LEGEND_FS - 2)
    ax2.set_xticks([0, 1], ["model", "climatology"], fontsize=TICK_FS)
    ax2.set_ylabel("per-group mean contribution", fontsize=XLABEL_FS)
    ax2.set_title("Murphy decomposition, per group\nreliability (lower better) "
                  "vs resolution (higher better)", fontsize=TITLE_FS)
    _finish(fig, f"presence Brier and its decomposition -- {subtitle}\n"
                 "whiskers: unit-clustered bootstrap 95%; dotted: climatology; "
                 "means weight each group equally",
            out_path)


ARM_COLORS = ("#2a78d6", "#d9622b", "#7a52a1")


def plot_coverage_sharpness(
    arms: dict[str, tuple[np.ndarray, np.ndarray]],
    magnitude: np.ndarray, numeric: np.ndarray, unit: np.ndarray,
    out_path: Path, subtitle: str, nominal: float = 0.8,
) -> None:
    """Do the stated credible intervals contain the measured shares?

    Coverage alone is gameable -- diluting every forecast until the intervals
    span the simplex reaches nominal vacuously -- so the companion panel shows
    the median interval width that coverage was bought at.  Read together:
    an honest fix moves coverage toward nominal while the width stays useful.

    `arms` maps a label to that forecast's `(lo, hi)` interval arrays; a
    "climatology" arm is drawn dotted like the other promoted panels.  Rocks
    with a single quantified group are excluded: their marginal is a point
    mass that always covers, so they carry no calibration information.
    """
    numeric_b = numeric.astype(bool)
    eligible = numeric_b & (numeric_b.sum(1, keepdims=True) >= 2)
    x = np.arange(len(GROUPS) + 1)
    model_arms = [label for label in arms if label != "climatology"]
    width = 0.8 / max(len(model_arms), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for j, label in enumerate(model_arms):
        lo, hi = arms[label]
        covered = np.where(
            eligible, (magnitude >= lo) & (magnitude <= hi), np.nan
        ).astype(float)

        def stat(rows, _c=covered):
            with np.errstate(invalid="ignore"):
                per_group = np.nanmean(_c[rows], axis=0)
            return np.append(per_group, np.nanmean(per_group))

        value = stat(np.arange(len(magnitude)))
        ci_lo, ci_hi = unit_bootstrap(stat, unit)
        pos = x + (j - (len(model_arms) - 1) / 2) * width
        color = ARM_COLORS[j % len(ARM_COLORS)]
        ax1.bar(pos, value, width * 0.92, color=color, label=label)
        ax1.errorbar(pos, value,
                     yerr=[np.maximum(value - ci_lo, 0),
                           np.maximum(ci_hi - value, 0)],
                     fmt="none", ecolor="0.25", elinewidth=1.0, capsize=2)
        med = [np.nanmedian(np.where(eligible[:, g], hi[:, g] - lo[:, g],
                                     np.nan)) for g in range(len(GROUPS))]
        ax2.bar(pos[:-1], med, width * 0.92, color=color, label=label)

    if "climatology" in arms:
        lo, hi = arms["climatology"]
        cov = np.where(eligible, (magnitude >= lo) & (magnitude <= hi),
                       np.nan).astype(float)
        with np.errstate(invalid="ignore"):
            per_group = np.nanmean(cov, axis=0)
        for xi, v in zip(x, np.append(per_group, np.nanmean(per_group))):
            ax1.plot([xi - 0.45, xi + 0.45], [v, v], ls=":", lw=1.8,
                     color="0.35", label="climatology" if xi == 0 else None)

    ax1.axhline(nominal, lw=1.0, color="0.2", ls="--")
    ax1.annotate(f"nominal {nominal:.0%}", (0.01, nominal), ha="left",
                 va="bottom", xycoords=("axes fraction", "data"), fontsize=8,
                 color="0.2")
    ax1.set_xticks(x, list(GROUPS) + ["mean"], fontsize=TICK_FS)
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("empirical coverage of the stated 80% interval",
                   fontsize=XLABEL_FS)
    ax1.set_title("coverage per group", fontsize=TITLE_FS)
    ax1.legend(fontsize=LEGEND_FS - 2)
    ax2.set_xticks(x[:-1], list(GROUPS), fontsize=TICK_FS)
    ax2.set_ylabel("median interval width (share units)", fontsize=XLABEL_FS)
    ax2.set_title("sharpness: what that coverage costs", fontsize=TITLE_FS)
    _finish(fig, f"interval calibration and its price -- {subtitle}\n"
                 "whiskers: unit-clustered bootstrap 95%; dotted: climatology; "
                 "means weight each group equally; single-group rocks excluded",
            out_path)


def share_interval(
    alpha7: np.ndarray, mask: np.ndarray, q: tuple[float, float] = (0.1, 0.9)
) -> tuple[np.ndarray, np.ndarray]:
    """Central credible interval on each group's predicted share.

    The magnitude branch emits a Dirichlet, not a point estimate, and the
    Dirichlet's marginal for one component is `Beta(alpha_i, alpha_0 - alpha_i)`
    on the conditioned support -- so the interval is closed-form, not sampled.

    Worth plotting rather than reporting the mean alone: much of what the
    sim-to-real correction does is widen this interval (alpha_0 on the scored
    support drops 46 -> 25 under fitting), and a figure showing only the mean
    cannot show that at all.
    """
    on = alpha7 * mask
    alpha0 = on.sum(-1, keepdims=True)
    beta = np.clip(alpha0 - on, 1e-9, None)
    lo, hi = (
        np.where(mask > 0, betaincinv(np.clip(on, 1e-9, None), beta, p), np.nan)
        for p in q
    )
    return lo, hi


def mixture_share_interval(
    alpha7: np.ndarray, log_w: np.ndarray, mask: np.ndarray,
    q: tuple[float, float] = (0.1, 0.9),
) -> tuple[np.ndarray, np.ndarray]:
    """`share_interval` for a K-component Dirichlet mixture.

    Each component's marginal on the conditioned support is a Beta, so the
    mixture CDF is the gate-weighted sum of Beta CDFs -- but its quantiles
    have no closed form, so bisection finds them.  Fifty halvings of [0, 1]
    put the answer well below any share the benchmark resolves.  At K = 1
    this agrees with `share_interval` to that tolerance.
    """
    w = np.exp(log_w)
    on = alpha7 * mask[:, None, :]
    a = np.clip(on, 1e-9, None)
    b = np.clip(on.sum(-1, keepdims=True) - on, 1e-9, None)

    def cdf(t):
        return (w[..., None] * betainc(a, b, t[:, None, :])).sum(1)

    bounds = []
    for p in q:
        lo_t, hi_t = np.zeros(mask.shape), np.ones(mask.shape)
        for _ in range(50):
            mid = 0.5 * (lo_t + hi_t)
            below = cdf(mid) < p
            lo_t = np.where(below, mid, lo_t)
            hi_t = np.where(below, hi_t, mid)
        bounds.append(np.where(mask > 0, 0.5 * (lo_t + hi_t), np.nan))
    return bounds[0], bounds[1]


CARD_SHARE_FLOOR = 5e-4

# The whisker the cards draw, matching the corpus inference card's p25-p75 so
# the two read as one figure family.  `plot_magnitude_parity` deliberately
# keeps `share_interval`'s wider default: its per-panel coverage label asserts
# an 80% band, so narrowing it there would make that figure lie.
CARD_INTERVAL_Q: tuple[float, float] = (0.25, 0.75)


def _card_panels(ax0, ax1, ax2, p7, pred, benchmark, row: int,
                 prefix: str = "", interval=None) -> None:
    """Draw one rock's three card panels onto the given axes.

    Split out of `real_rock_card` so the before/after comparison can stack two
    of them on one figure without duplicating the panel logic.  `prefix` names
    which model a row is showing, in the panel titles rather than only as an
    axis label -- on a stacked figure the reader should not have to infer it.

    Styling follows the corpus inference card (`summary_cards`): open orange
    markers for predictions, filled blue for truth, log share axis, whisker
    intervals -- so the real-rock card and the corpus card read as the same
    figure family.
    """
    if ax0 is not None:
        src, frac = benchmark.bulk_src[row].numpy(), benchmark.bulk_frac[row].numpy()
        populated = np.flatnonzero(src > 0)
        symbol = {z: sym for sym, z in VOCAB}
        ax0.barh(np.arange(len(populated)), frac[populated] * 100,
                 color="tab:gray", alpha=0.8, height=0.6)
        ax0.set_yticks(np.arange(len(populated)),
                       [symbol[int(z)] for z in src[populated]], fontsize=TICK_FS)
        ax0.invert_yaxis()
        ax0.set_xscale("log")
        ax0.set_xlabel("Cation wt%", fontsize=XLABEL_FS)
        ax0.set_title("Input: Whole-Rock Chemistry", fontsize=TITLE_FS)

    y = np.arange(len(GROUPS))
    ax1.plot(p7[row], y - 0.15, "o", ms=5, color="tab:orange", mfc="white")
    ax1.plot(benchmark.presence[row], y + 0.15, "o", ms=5, color="tab:blue")
    ax1.set_yticks(y, GROUPS, fontsize=TICK_FS)
    ax1.invert_yaxis()
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_xticks([0, 0.5, 1])
    ax1.grid(axis="x", lw=0.3, alpha=0.5)
    ax1.set_xlabel("presence probability", fontsize=XLABEL_FS)
    ax1.set_title(f"{prefix}Phase Presence", fontsize=TITLE_FS)

    floor = CARD_SHARE_FLOOR
    for i in range(len(GROUPS)):
        if benchmark.numeric[row, i]:
            ax2.plot(max(benchmark.magnitude[row, i], floor), i + 0.15, "o",
                     ms=5, color="tab:blue")
        elif benchmark.censored[row, i]:
            bound = max(benchmark.censor_bound[row, i], floor)
            ax2.plot(bound, i + 0.15, marker="|", ms=9, mew=1.2,
                     color="tab:blue", alpha=0.7)
            head = max(bound * 10 ** -0.45, floor)
            if head < bound:
                ax2.annotate("", xy=(head, i + 0.15),
                             xytext=(bound, i + 0.15),
                             arrowprops={"arrowstyle": "->", "color": "tab:blue",
                                         "lw": 0.9, "alpha": 0.7})
        value = max(pred[row, i], floor)
        if interval is not None and np.isfinite(interval[0][row, i]):
            lo = max(interval[0][row, i], floor)
            hi = max(interval[1][row, i], floor)
            ax2.errorbar(value, i - 0.15,
                         xerr=[[max(value - lo, 0)], [max(hi - value, 0)]],
                         fmt="o", ms=5, color="tab:orange", mfc="white",
                         lw=1.2, capsize=2.5)
        else:
            ax2.plot(value, i - 0.15, "o", ms=5, color="tab:orange", mfc="white")
    handles = [
        Line2D([], [], marker="o", ls="none", ms=5, color="tab:orange",
               mfc="white", label="predicted"),
        Line2D([], [], marker="o", ls="none", ms=5, color="tab:blue",
               label="measured"),
    ]
    ax2.legend(handles=handles, fontsize=LEGEND_FS, loc="upper left",
               framealpha=0.9)
    ax2.set_yticks(y, GROUPS, fontsize=TICK_FS)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlim(floor, 1.05)
    ax2.grid(axis="x", lw=0.3, alpha=0.5)
    ax2.set_xlabel("Fraction of rock's crystalline mass\n"
                   "whiskers: predicted 25-75%", fontsize=XLABEL_FS)
    ax2.set_title(f"{prefix}Phase Share", fontsize=TITLE_FS)


def real_rock_card(
    p7, pred, benchmark, row: int, out_path: Path, interval=None
) -> None:
    """One real rock end to end: chemistry in, assemblage out, measurement beside it.

    `interval` is the `(lo, hi)` pair from `share_interval`: the predicted
    share draws as a whisker around `pred`, because a point marker alone
    misrepresents a model whose output is a distribution.
    """
    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.0, 0.9, 1.2]}
    )
    _card_panels(ax0, ax1, ax2, p7, pred, benchmark, row, interval=interval)
    fig.suptitle(f"{benchmark.bulk_ids[row]} -- SiO2 {benchmark.sio2[row]:.1f} "
                 "wt%, zero-shot (nothing fit on real rocks)",
                 fontsize=SUPTITLE_FS)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def real_rock_card_comparison(
    before, after, benchmark, row: int, out_path: Path, subtitle: str
) -> None:
    """The same rock zero-shot and adapted, stacked, against one measurement.

    `before` / `after` are `(p7, share, lo, hi)` tuples, the last two the
    Beta credible bounds from `share_interval` (pass `None` to omit). Two rows
    rather than two files
    because the correction is only legible as a difference: a single adapted
    card looks like any other card, and the reader has nothing to subtract.

    Magnitude here is teacher-forced on the measured assemblage, as the
    single-rock card is -- this panel answers "given we know which minerals are
    present, how much of each?". The model standing on its own assemblage is
    `real_rock_scenarios`.
    """
    fig = plt.figure(figsize=(13, 7.0))
    # chemistry spans both rows: the input does not change between them, and
    # two identical panels invite the eye to look for a difference there.
    # `top` reserves room for a multi-line suptitle: the panel titles are two
    # lines themselves, and tight_layout will not arbitrate between them.
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.9, 1.2], wspace=0.3,
                          hspace=0.45, top=0.83, bottom=0.07)
    ax_chem = fig.add_subplot(gs[:, 0])
    rows = (("Zero-shot", before), ("Adapted", after))
    for r, (label, (p7, share, lo, hi)) in enumerate(rows):
        ax1, ax2 = fig.add_subplot(gs[r, 1]), fig.add_subplot(gs[r, 2])
        _card_panels(ax_chem if r == 0 else None, ax1, ax2, p7, share,
                     benchmark, row, prefix=f"{label}: ",
                     interval=None if lo is None else (lo, hi))
        if r == 0:
            ax1.set_xlabel("")
            ax2.set_xlabel("")
    # not `_finish`: its tight_layout would undo the reserved header space
    fig.suptitle(f"{benchmark.bulk_ids[row]} -- SiO2 {benchmark.sio2[row]:.1f} wt%\n"
                 f"{subtitle}", fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def real_rock_scenarios(
    head,
    latent: torch.Tensor,
    p21: np.ndarray,
    *,
    correct_alpha7=None,
    n_samples: int = 512,
    top_k: int = 2,
    seed: int = 0,
) -> list[dict]:
    """The top-K assemblages the model proposes for one rock, on its own support.

    The card teacher-forces the measured assemblage and asks "how much of
    each?". This asks the deployment question instead -- which minerals, and
    how much -- by sampling supports from the presence marginals and letting
    the magnitude branch condition on whatever it drew. Samples are grouped by
    their coarsened 7-group pattern, so a scenario is a distinct mineral list
    with a probability, not a smear over one.

    `correct_alpha7` is the recalibrator's magnitude correction, applied per sample
    because the coupled tiers read the sampled concentrations. Pass `p21`
    already corrected if the recalibrator should also move which supports get drawn.
    """
    rng = np.random.default_rng(seed)
    draws = (rng.random((n_samples, p21.shape[-1])) < p21).astype(np.float64)
    with torch.no_grad():
        alpha21 = head.magnitude_alpha(
            latent.expand(n_samples, -1),
            torch.as_tensor(draws, dtype=latent.dtype, device=latent.device),
        ).cpu().numpy()
    alpha7, _ = coarsen_alpha(alpha21, draws)
    if correct_alpha7 is not None:
        alpha7 = correct_alpha7(alpha7)

    present = np.zeros((n_samples, len(GROUPS)), dtype=bool)
    for phase, group in MODEL_TO_GROUP.items():
        present[:, GROUP_INDEX[group]] |= draws[:, PHASE_INDEX[phase]] > 0
    share = conditional_mean(alpha7, present.astype(np.float64))

    patterns: dict[tuple, list[int]] = {}
    for i, row in enumerate(present):
        patterns.setdefault(tuple(row.tolist()), []).append(i)
    ranked = sorted(patterns.items(), key=lambda kv: -len(kv[1]))[:top_k]
    return [
        {
            "groups": tuple(g for g, on in zip(GROUPS, pattern) if on),
            "present": np.array(pattern),
            "prob": len(idx) / n_samples,
            "share": share[idx].mean(0),
            "share_p10": np.quantile(share[idx], 0.1, axis=0),
            "share_p90": np.quantile(share[idx], 0.9, axis=0),
        }
        for pattern, idx in ranked
    ]


def real_rock_scenario_figure(
    scenarios: list[dict], benchmark, row: int, out_path: Path, subtitle: str,
    tp_samples: list[np.ndarray] | None = None,
) -> None:
    """One rock's top-K proposed assemblages against what was measured.

    `tp_samples`, if given, is one [S, 2] array of (T, P) draws per scenario.
    It is drawn separately because **no source in this benchmark measured
    temperature or pressure**, so that panel has predictions and no truth --
    and CondHead is not corrected by the recalibrator, so it is also the one part of
    the figure still speaking the uncorrected model's language. Labelled as
    such on the axes rather than left for the reader to infer.
    """
    n_panels = 2 if tp_samples else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6.4 * n_panels, 4.3), squeeze=False,
    )
    ax = axes[0][0]
    y = np.arange(len(GROUPS))
    width = 0.8 / max(len(scenarios), 1)
    colors = ["tab:orange", "tab:green", "tab:purple", "tab:red"]
    for i, sc in enumerate(scenarios):
        offset = (i - (len(scenarios) - 1) / 2) * width
        ax.barh(y + offset, sc["share"], width, color=colors[i % len(colors)],
                alpha=0.85, label=f"scenario {i + 1} (p={sc['prob']:.2f})")
    measured = np.where(benchmark.numeric[row], benchmark.magnitude[row], np.nan)
    ax.plot(measured, y, "D", ms=6, color="tab:blue", label="measured",
            ls="none", zorder=5)
    for i in range(len(GROUPS)):
        if benchmark.censored[row, i]:
            ax.annotate("", xy=(benchmark.censor_bound[row, i], i),
                        xytext=(1.0, i),
                        arrowprops={"arrowstyle": "->", "color": "tab:blue",
                                    "lw": 0.9, "alpha": 0.7})
    ax.set_yticks(y, GROUPS, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("share of the crystal cargo", fontsize=9)
    ax.set_title("the model's own assemblages, vs measured", fontsize=10)
    ax.legend(fontsize=7)

    if tp_samples:
        ax2 = axes[0][1]
        for i, tp in enumerate(tp_samples):
            ax2.scatter(tp[:, 0], tp[:, 1], s=5, alpha=0.3,
                        color=colors[i % len(colors)], label=f"scenario {i + 1}")
        ax2.set_xlabel("temperature (C)", fontsize=9)
        ax2.set_ylabel("pressure (kbar)", fontsize=9)
        ax2.set_title("conditions per scenario\n"
                      "(NO measured T/P on this benchmark; and CondHead is "
                      "uncorrected)", fontsize=9)
        ax2.legend(fontsize=7)

    _finish(fig, f"{benchmark.bulk_ids[row]} -- SiO2 {benchmark.sio2[row]:.1f} wt%\n"
                 f"{subtitle}", out_path)


def _headline(nll: dict, mask: np.ndarray, numeric: np.ndarray) -> dict:
    """Mean NLL terms over the rocks selected by `mask`, unweighted per rock.

    Carries `mean_composition_size` because the Dirichlet term is a density
    over however many groups a rock's composition spans, so its scale moves
    with that count. Two subsets are only comparable when their sizes are --
    a subset of simpler rocks scores lower without predicting anything better.
    """
    if not mask.any():
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "joint_nll": float(nll["joint"][mask].mean()),
        "bernoulli_nll": float(nll["bernoulli"][mask].mean()),
        "dirichlet_nll": float(nll["dirichlet"][mask].mean()),
        "mean_composition_size": float(numeric[mask].sum(1).mean()),
    }


def predict(head, latents: torch.Tensor, benchmark: RealRockBenchmark) -> dict:
    """Coarsened zero-shot predictions for every rock in `benchmark`.

    Presence is the model's own call. The magnitude branch is
    support-conditioned, so its support is teacher-forced at the group level
    the truth actually resolves (`truth_forced_support`) and left to the model
    below that. `alpha_model_support` repeats it on the model's own support --
    the same exposure-bias probe the corpus eval reports, which here also says
    how much of the gap is the support being wrong rather than the sizes.
    `model_support7` is the group-level mask that goes with it, which any
    share conditioned on the model's own assemblage renormalizes over.
    """
    with torch.no_grad():
        p21 = torch.sigmoid(head.presence_logits(latents)).cpu().numpy()
    support = truth_forced_support(benchmark.presence, p21)
    model_support = (p21 > 0.5).astype(np.float64)
    support_t = torch.as_tensor(support, dtype=latents.dtype, device=latents.device)
    model_support_t = torch.as_tensor(
        model_support, dtype=latents.dtype, device=latents.device
    )
    out = {"p7": coarsen_presence(p21),
           "model_support7": coarsen_support(model_support),
           "support21": support}
    with torch.no_grad():
        alpha = head.magnitude_alpha(latents, support_t).cpu().numpy()
        alpha_model = head.magnitude_alpha(latents, model_support_t).cpu().numpy()
    out["alpha7"], out["off_vocab_alpha"] = coarsen_alpha(alpha, support)
    out["alpha7_model_support"], _ = coarsen_alpha(alpha_model, model_support)
    return out


@torch.no_grad()
def run_real_rock_eval(
    ckpt,
    benchmark: RealRockBenchmark,
    latents: torch.Tensor,
    *,
    device: torch.device,
    tag: str,
) -> dict:
    """Score a joint checkpoint on the real-rock benchmark, zero-shot.

    Writes figures + `metrics.json` under `runs/real_rock/<tag>/` and returns
    the metrics. The headline is the freshest rocks only; altered rocks and
    the point-counted precision tier are reported beside it so a weak number
    can be attributed rather than just observed.
    """
    output_dir = RUNS_ROOT / "real_rock" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output:  {output_dir}")

    head = ckpt.build_mode_head().to(device)
    pred = predict(head, latents, benchmark)
    p7, alpha7 = pred["p7"], pred["alpha7"]
    numeric_mask = benchmark.numeric.astype(np.float64)
    cond_mean = conditional_mean(alpha7, numeric_mask)

    clean = benchmark.clean & benchmark.has_magnitude
    nll = real_rock_nll(
        p7, alpha7, benchmark.presence, benchmark.magnitude, benchmark.numeric,
    )
    nll_model_support = real_rock_nll(
        p7, pred["alpha7_model_support"], benchmark.presence, benchmark.magnitude,
        benchmark.numeric,
    )
    # the exposure probe asks what the magnitudes would cost if the model had
    # to stand on its own assemblage rather than the measured one. It is only
    # defined where the model's support covers every group that carries a
    # measured share: a group the model calls absent gets no concentration at
    # all, and the density of an observed share under it is undefined, not
    # merely bad. Reported with its coverage so the conditioning is visible.
    alpha_model_on = pred["alpha7_model_support"] > 0
    if alpha_model_on.ndim == 3:
        # coarsened zeros are support-driven, identical across components
        alpha_model_on = alpha_model_on.all(1)
    covered = clean & (
        (benchmark.numeric & alpha_model_on).sum(1) == benchmark.numeric.sum(1)
    )
    censoring = censored_consistency(
        alpha7, (benchmark.numeric | benchmark.censored).astype(np.float64),
        benchmark.censored, benchmark.censor_bound, log_w=log_w,
    )

    subtitle = f"{int(clean.sum())} least-altered rocks"
    p_stats = presence_group_stats(p7[benchmark.clean],
                                   benchmark.presence[benchmark.clean])
    plot_presence_calibration(
        p7[benchmark.clean], benchmark.presence[benchmark.clean], p_stats,
        output_dir / "presence_calibration.png", subtitle,
    )
    clean_bench = benchmark.subset(benchmark.clean)
    mag_stats = magnitude_group_stats(
        cond_mean[benchmark.clean], clean_bench.magnitude, clean_bench.numeric
    )
    parity = magnitude_parity_stats(
        cond_mean[benchmark.clean], clean_bench.magnitude,
        clean_bench.numeric.astype(np.float64),
    )
    plot_magnitude_parity(
        alpha7[benchmark.clean], clean_bench.magnitude, clean_bench.numeric,
        mag_stats, output_dir / "magnitude_parity.png", subtitle,
    )
    plot_gap_summary(
        p7[benchmark.clean], cond_mean[benchmark.clean], clean_bench,
        output_dir / "gap_summary.png", subtitle,
    )
    card_row = int(np.flatnonzero(clean)[np.argsort(nll["joint"][clean])[
        int(clean.sum()) // 2]])
    # the card normalizes over every measured group, trace ones included, so
    # a mineral logged only as a trace still shows what the model says about
    # it next to the bound it has to sit under.
    card_mask = (benchmark.numeric | benchmark.censored).astype(np.float64)
    card_med, _ = share_interval(alpha7, card_mask, q=(0.5, 0.5))
    real_rock_card(
        p7, card_med, benchmark, card_row,
        output_dir / "real_rock_card.png",
        interval=share_interval(alpha7, card_mask, q=CARD_INTERVAL_Q),
    )

    null = climatology(benchmark)
    null_nll = real_rock_nll(
        null["p7"], null["alpha7"], benchmark.presence, benchmark.magnitude,
        benchmark.numeric,
    )
    model_share = conditional_mean(
        pred["alpha7_model_support"], pred["model_support7"].astype(np.float64)
    )
    plot_cohort_card(
        p7[clean], cond_mean[clean], model_share[clean],
        pred["model_support7"][clean], benchmark.subset(clean),
        output_dir / "cohort_card.png",
    )
    plot_magnitude_ratio(
        cond_mean[clean], benchmark.magnitude[clean], benchmark.numeric[clean],
        output_dir / "magnitude_ratio.png", subtitle,
    )
    plot_nll_distribution(
        {k: v[clean] for k, v in nll.items()},
        {k: v[clean] for k, v in null_nll.items()},
        output_dir / "nll_distribution.png", subtitle,
    )
    plot_presence_brier(
        p7[clean], null["p7"][clean], benchmark.presence[clean],
        benchmark.unit[clean], output_dir / "presence_brier.png", subtitle,
    )

    thresholded = presence_scores(p7[benchmark.clean],
                                  benchmark.presence[benchmark.clean])
    scored_mask = benchmark.numeric | benchmark.censored
    alpha0 = (alpha7 * scored_mask).sum(1)
    off_vocab_share = pred["off_vocab_alpha"] / np.clip(
        alpha0 + pred["off_vocab_alpha"], 1e-12, None
    )

    per_group = []
    for i, group in enumerate(GROUPS):
        scored = censoring["scored"][benchmark.clean, i]
        per_group.append({
            "group": group,
            "observed_rate": float(benchmark.presence[benchmark.clean, i].mean()),
            "mean_predicted_p": float(p7[benchmark.clean, i].mean()),
            "auroc": _maybe(p_stats["auroc"][i]),
            "best_threshold": _maybe(p_stats["best_threshold"][i]),
            "best_f1": _maybe(p_stats["best_f1"][i]),
            "recall_at_0.5": _maybe(thresholded["recall"][i]),
            "precision_at_0.5": _maybe(thresholded["precision"][i]),
            "magnitude_r2": _maybe(mag_stats["r2"][i]),
            "magnitude_spearman": _maybe(mag_stats["spearman"][i]),
            "magnitude_mae": _maybe(parity["magnitude_mae"][i]),
            "magnitude_pearson_r": _maybe(parity["magnitude_r"][i]),
            "n_quantified": int(mag_stats["n"][i]),
            "n_trace_only": int(scored.sum()),
            "censored_prob_below_bound": _maybe(
                np.nanmean(censoring["prob_below_bound"][benchmark.clean, i])
                if scored.any() else np.nan
            ),
            "censored_mean_below_bound_rate": _maybe(
                censoring["mean_below_bound"][benchmark.clean, i].sum()
                / scored.sum() if scored.any() else np.nan
            ),
        })

    metrics = {
        "checkpoint_tag": tag,
        "encoder_checkpoint": str(ckpt.encoder_checkpoint),
        "benchmark": {
            "dir": str(BENCHMARK_DIR),
            "groups": list(GROUPS),
            "n_with_chemistry_and_modes": int(len(benchmark)),
            "n_headline": int(clean.sum()),
            "n_altered_or_ungraded": int((~benchmark.clean).sum()),
            "n_precision_tier": int(benchmark.precision_tier.sum()),
            "n_no_quantified_phase": int((~benchmark.has_magnitude).sum()),
            "n_single_group_composition": int(
                (benchmark.numeric.sum(1) == 1).sum()
            ),
            "trace_bound_vol_pct": TRACE_BOUND_VOL_PCT,
            "clean_grades": sorted(CLEAN_GRADES),
        },
        "headline": _headline(nll, clean, benchmark.numeric),
        # the null is recorded, not just drawn: a headline in nats is only
        # readable beside what ignoring the rock would have cost
        "climatology": {
            "concentration": null["concentration"],
            "n_fit": null["n_fit"],
            **_headline(null_nll, clean, benchmark.numeric),
        },
        "altered": _headline(nll, ~benchmark.clean & benchmark.has_magnitude,
                             benchmark.numeric),
        "all_rocks": _headline(nll, benchmark.has_magnitude, benchmark.numeric),
        "precision_tier": _headline(
            nll, benchmark.precision_tier & benchmark.has_magnitude,
            benchmark.numeric,
        ),
        "exposure_bias": {
            "n": int(covered.sum()),
            "coverage": float(covered.sum() / max(int(clean.sum()), 1)),
            "dirichlet_nll_truth_support": float(nll["dirichlet"][covered].mean())
            if covered.any() else None,
            "dirichlet_nll_model_support": float(
                nll_model_support["dirichlet"][covered].mean()
            ) if covered.any() else None,
            "delta": float(
                nll_model_support["dirichlet"][covered].mean()
                - nll["dirichlet"][covered].mean()
            ) if covered.any() else None,
        },
        "censored": {
            "n_instances": int(censoring["scored"][benchmark.clean].sum()),
            "mean_prob_below_bound": float(
                np.nanmean(censoring["prob_below_bound"][benchmark.clean])
            ),
            "mean_nll": float(
                censoring["nll"][benchmark.clean][
                    censoring["scored"][benchmark.clean]
                ].mean()
            ),
        },
        "off_vocabulary_mass": {
            "mean": float(off_vocab_share[benchmark.clean].mean()),
            "p90": float(np.quantile(off_vocab_share[benchmark.clean], 0.9)),
        },
        "per_group": per_group,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"headline joint NLL: {metrics['headline']['joint_nll']:.4f} nats "
          f"over {metrics['headline']['n']} rocks")
    return metrics


def _maybe(value) -> float | None:
    """JSON-safe scalar: NaN becomes null rather than an invalid literal."""
    value = float(value)
    return None if np.isnan(value) else value

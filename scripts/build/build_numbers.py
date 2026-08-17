"""Builds paper/numbers.tex from the tracked ledgers and result JSONs, so no
number is ever hand-typed in paper prose. Rerun after any source moves:
`conda run -n alignn_fix python scripts/build/build_numbers.py`. A missing source
degrades to a visible [TBD] macro and a stderr warning, never a broken build.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import json
import re
import sys
from pathlib import Path

import yaml

from cnmineral.data.phase_vocab import SOLID_PHASES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "numbers.tex"

FEWSHOT = ROOT / "explorations" / "real_rock_fewshot" / "results"
SCALING = ROOT / "explorations" / "corpus_scaling" / "results"
SEDIMENT = ROOT / "explorations" / "sediment_transfer" / "results"
PROBE = ROOT / "explorations" / "input_probe_ceiling" / "results"
REALROCK = ROOT / "runs" / "real_rock"

# `Prod` is the frozen production stack and nothing else. It used to name
# joint_heads_supportmag_seed0, which also happens to be the first arm of the
# sharper-sampling ladder, so one suffix meant two different checkpoints and
# retargeting it silently dropped the ladder. The two roles are now separate
# dicts: a macro suffixed Prod is the production model, and a macro suffixed
# Broad / Sharp / SharpWeight is a rung of the ladder.
PROD = "joint_heads_allrock_tuned_seed0"
ARMS = {
    "Broad": "joint_heads_supportmag_seed0",
    "Sharp": "joint_heads_regen_seed0",
    "SharpWeight": "joint_heads_regen_tuned_seed0",
}
IGNEOUS = "joint_heads_igneous_tuned_seed0"
ENCODER_RUN = "pretrain_allrock_redox_seed0_epoch_best"
LEDGER_TAGS = {"Prod": PROD, **ARMS}

macros = []
missing = []
emitted = set()


def emit(name, value, comment=None):
    # a repeated \newcommand is a LaTeX hard error at the owner's end, and the
    # second definition is always the accident; fail here instead
    if name in emitted:
        raise ValueError(f"macro {name} emitted twice")
    emitted.add(name)
    line = f"\\newcommand{{\\{name}}}{{{value}}}"
    if comment:
        line += f"  % {comment}"
    macros.append(line)


def tbd(name, why):
    missing.append(f"{name}: {why}")
    emit(name, "\\paperTBD", f"MISSING: {why}")


def fmt(x, nd=2, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s


def sci(x, nd=2):
    from math import floor, log10

    e = floor(log10(abs(x)))
    m = x / 10**e
    return f"${m:.{nd}f}\\times10^{{{e}}}$"


def pow10(x):
    from math import log10

    e = round(log10(x))
    if abs(x - 10**e) <= 1e-9 * x:
        return f"$10^{{{e}}}$"
    return sci(x, 1)


def section(title):
    macros.append("")
    macros.append(f"% {title}")


# real-rock zero-shot entries (results/real_rock_ledger.yaml)
section("real-rock ledger: zero-shot, benchmark eruption-v2 unless noted")
try:
    rr = yaml.safe_load((ROOT / "results" / "real_rock_ledger.yaml").read_text())
    entries = {}
    for v in rr["versions"]:
        entries[(v["checkpoint_tag"], v["benchmark"])] = v
    for short, tag in LEDGER_TAGS.items():
        e = entries.get((tag, "eruption-v2"))
        if e is None:
            tbd(f"rrZsJoint{short}", f"no eruption-v2 entry for {tag}")
            continue
        emit(f"rrZsJoint{short}", fmt(e["joint_nll"], 1), tag)
        emit(f"rrZsBern{short}", fmt(e["bernoulli_nll"], 1))
        emit(f"rrZsDir{short}", fmt(e["dirichlet_nll"], 1))
    # the ladder rungs, as the deltas section 3 actually states: each step's
    # cost and how much of it is the abundance factor
    for lo, hi, label in (("Broad", "Sharp", "BroadSharp"),
                          ("Sharp", "SharpWeight", "SharpWeight")):
        before = entries.get((ARMS[lo], "eruption-v2"))
        after = entries.get((ARMS[hi], "eruption-v2"))
        if before is None or after is None:
            continue
        dj = after["joint_nll"] - before["joint_nll"]
        dd = after["dirichlet_nll"] - before["dirichlet_nll"]
        emit(f"rrZsStep{label}", fmt(dj, 2, sign=True), f"{hi} minus {lo}, headline 640")
        emit(f"rrZsStep{label}DirPct", f"{100 * dd / dj:.0f}",
             "share of the step carried by the abundance factor")
    v1 = entries.get((ARMS["Broad"], "eruption-v1"))
    if v1:
        emit("rrZsJointBroadVOne", fmt(v1["joint_nll"], 1),
             "first-ever real-rock score, eruption-v1, n=635")
    b = rr["benchmarks"]["eruption-v2"]
    emit("rrBenchN", str(b["n_headline"]), "least-altered headline cohort")
    emit("rrBenchTotal", str(b["n_with_chemistry_and_modes"]))
    emit("rrBenchGroups", str(len(b["groups"])))
except Exception as exc:
    tbd("rrZsJointProd", f"real_rock_ledger.yaml unreadable: {exc}")

# corpus-val ledger (results/ledger.yaml); numbers valid only within their
# stamped instrument -- prose must label the instrument wherever these appear
section("corpus ledger: teacher-forced val NLLs, instrument valtest128-broad")
try:
    lg = yaml.safe_load((ROOT / "results" / "ledger.yaml").read_text())
    by_tag = {v["checkpoint_tag"]: v for v in lg["versions"]}
    for short, tag in LEDGER_TAGS.items():
        v = by_tag.get(tag)
        if v is None:
            tbd(f"cvModeJoint{short}", f"no ledger entry for {tag}")
            continue
        # the instrument is part of the number: the sharper-sampling arms are
        # scored against a validation set regenerated by their own sampler, so
        # a cvModeJoint pair is comparable ONLY within one instrument
        emit(f"cvModeJoint{short}", fmt(v["heads"]["mode"]["joint_nll"], 2),
             f"{tag} @ {v['instrument']}")
except Exception as exc:
    tbd("cvModeJointProd", f"ledger.yaml unreadable: {exc}")

# three-way recalibrated contrast; the test slice's one sealed read
section("three-way checkpoint contrast, matrix tier fit on the 368-rock "
        "train roster (explorations/real_rock_fewshot/results/"
        "checkpoint_contrast_matrix_three_way.json); test = the one sealed read")
try:
    tw = json.loads((FEWSHOT / "checkpoint_contrast_matrix_three_way.json").read_text())
    for short, tag in ARMS.items():
        lv = tw["levels"][tag]
        emit(f"rrTestAd{short}", fmt(lv["test"]["adapted_joint"], 2), tag)
        emit(f"rrValAd{short}", fmt(lv["val"]["adapted_joint"], 2))
        emit(f"rrValZs{short}", fmt(lv["val"]["zero_shot_joint"], 1),
             "val-slice zero-shot, same rocks as the adapted number")
    pairs = {
        ("Broad", "Sharp"): "BroadSharp",
        ("Broad", "SharpWeight"): "BroadSharpWeight",
        ("Sharp", "SharpWeight"): "SharpSharpWeight",
    }
    for split in ("val", "test"):
        con = tw["paired_contrasts_adapted_joint"][split]
        for (a, b), label in pairs.items():
            key = f"{ARMS[a]} - {ARMS[b]}"
            if key not in con:
                continue
            c = con[key]
            pre = "rrValCon" if split == "val" else "rrTestCon"
            emit(f"{pre}{label}Mean", fmt(c["mean"], 2, sign=True),
                 f"{a} minus {b}, unit-clustered")
            emit(f"{pre}{label}Lo", fmt(c["lo"], 2, sign=True))
            emit(f"{pre}{label}Hi", fmt(c["hi"], 2, sign=True))
except Exception as exc:
    tbd("rrTestAdBroad", f"three-way contrast unreadable: {exc}")

# encoder-breadth contrast on volcanic rocks, the companion to the sedimentary
# one (S5). Both arms carry their own tuned dials, so this is an arm contrast
# rather than an encoder-only ablation; prose must say so.
section("encoder-breadth recalibrated val contrast, matrix tier "
        "(checkpoint_contrast_matrix.json); reported as production minus the "
        "igneous-only-encoder arm, so negative = the all-rock encoder is better")
try:
    two = json.loads((FEWSHOT / "checkpoint_contrast_matrix.json").read_text())
    con = two["paired_contrasts_adapted_joint"]["val"]
    key_fwd, key_rev = f"{PROD} - {IGNEOUS}", f"{IGNEOUS} - {PROD}"
    if key_fwd in con:
        c, flip = con[key_fwd], False
    elif key_rev in con:
        c, flip = con[key_rev], True
    else:
        raise KeyError("no production/igneous-encoder pair in contrasts")
    mean = -c["mean"] if flip else c["mean"]
    lo = -c["hi"] if flip else c["lo"]
    hi = -c["lo"] if flip else c["hi"]
    emit("rrValConProdIgneousMean", fmt(mean, 2, sign=True))
    emit("rrValConProdIgneousLo", fmt(lo, 2, sign=True))
    emit("rrValConProdIgneousHi", fmt(hi, 2, sign=True))
    emit("rrValAdIgneous", fmt(two["levels"][IGNEOUS]["val"]["adapted_joint"], 2))
except Exception as exc:
    tbd("rrValConProdIgneousMean", f"two-way contrast unreadable: {exc}")

# few-shot curve at the 25-rock budget: median joint NLL over draws, per
# recalibrator tier (vector / matrix), val slice
section("few-shot curve and benchmark splits, frozen production stack, val "
        "slice (explorations/real_rock_fewshot/results/<PROD>/curve_val.json)")
try:
    import statistics

    cv = json.loads((FEWSHOT / PROD / "curve_val.json").read_text())
    sp = cv["splits"]
    for split, short in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        emit(f"rr{short}NRocks", str(sp[split]["rocks"]))
        emit(f"rr{short}NUnits", str(sp[split]["units"]))
    emit("sysBenchTrainRocks", str(sp["train"]["rocks"]),
         "recalibration fit roster, split by stratigraphic unit")
    emit("fsBudgetDraws", str(len({p["draw"] for p in cv["curve"]})),
         "independent unit-draws behind each budget point")
except Exception as exc:
    tbd("rrValNRocks", f"curve_val.json unreadable: {exc}")

section("recalibration tier ladder, frozen production stack, val slice "
        "(<PROD>/tailed_acceptance.json for the full roster, curve_val.json "
        "for the 25-rock budget); every number here is val, never test")
try:
    ta = json.loads((FEWSHOT / PROD / "tailed_acceptance.json").read_text())
    cv = json.loads((FEWSHOT / PROD / "curve_val.json").read_text())
    lv = ta["val_levels"]
    zs = lv["zero_shot"]["joint"]
    emit("rrValZsProd", fmt(zs, 1), "val-slice zero-shot, frozen stack")
    emit("rrValZsBernProd", fmt(lv["zero_shot"]["bernoulli"], 2))
    emit("rrValZsDirProd", fmt(lv["zero_shot"]["dirichlet"], 2))
    clim = ta["internal_climatology_val_joint"]
    emit("rrValClimProd", fmt(clim, 2), "climatology fit and scored inside val")
    for tier in ("vector", "matrix", "tailed"):
        full = lv[tier]["joint"]
        emit(f"rrVal{tier.capitalize()}Prod", fmt(full, 2))
        pts = [p["joint"] for p in cv["curve"]
               if p["arm"] == tier and p["budget"] == 25]
        if not pts:
            tbd(f"fsTwentyFive{tier.capitalize()}Prod",
                f"no {tier} budget-25 points in curve_val.json")
            continue
        at25 = statistics.median(pts)
        emit(f"fsTwentyFive{tier.capitalize()}Prod", fmt(at25, 2),
             f"median over {len(pts)} draws")
        emit(f"fsTwentyFive{tier.capitalize()}GainProd", fmt(zs - at25, 1),
             "nats the 25-rock fit recovers off zero-shot")
        emit(f"fsRecovered{tier.capitalize()}Pct", f"{100 * (zs - at25) / (zs - full):.0f}",
             "share of that tier's own recoverable nats bought by 25 rocks")
    emit("rrValAheadProd", fmt(clim - lv["tailed"]["joint"], 2),
         "canonical tailed tier, nats ahead of the val climatology")
    emit("rrValAdProd", fmt(lv["tailed"]["joint"], 2),
         "ALIAS of rrValTailedProd; tailed is the canonical layer")
    con = ta["paired_contrasts"]
    emit("rrTailedGainMean", fmt(con["matrix_minus_tailed_joint"]["mean"], 2, sign=True),
         "matrix minus tailed, unit-clustered; positive = tailed is better")
    emit("rrTailedGainLo", fmt(con["matrix_minus_tailed_joint"]["lo"], 2, sign=True))
    emit("rrTailedGainHi", fmt(con["matrix_minus_tailed_joint"]["hi"], 2, sign=True))
    emit("rrTailedBernGain", fmt(con["matrix_minus_tailed_bernoulli"]["mean"], 3, sign=True),
         "the correction moves abundance only; presence contrast is exactly zero")
    emit("rrTailedEps", fmt(ta["tailed_params"]["eps"], 3))
    emit("rrTailedK", fmt(ta["tailed_params"]["k"], 3))
except Exception as exc:
    tbd("rrValZsProd", f"tailed_acceptance.json unreadable: {exc}")

# what the frozen encoder is worth away from the mineralogy task: recovering
# element concentrations withheld from it, and how well rock type reads off the
# latent. The imputation readout lives under gitignored runs/; every probe
# accuracy comes paired with the same probe on the raw assay, since the
# compilation computes qap_name from the majors the encoder reads.
section("encoder capability away from the mineralogy task (runs/imputation/"
        f"{ENCODER_RUN}/readout/readout_logspace_metrics.json; "
        "explorations/input_probe_ceiling/results/probe_ceiling.json)")
try:
    import statistics

    im = json.loads((ROOT / "runs" / "imputation" / ENCODER_RUN / "readout"
                     / "readout_logspace_metrics.json").read_text())
    per = im["per_element"].values()
    majors = [v["log_r2"] for v in per if "wt%" in v["unit"]]
    traces = [v["log_r2"] for v in per if "ppm" in v["unit"]]
    emit("encImputeTestN", f"{im['n_test']:,}".replace(",", "{,}"),
         "held-out whole-rock analyses, pretraining test split")
    emit("encImputeMajorN", str(len(majors)),
         "majors the production pretraining set populates; Cr is never reported")
    emit("encImputeMajorRSq", fmt(statistics.median(majors), 2),
         "median log-space R^2, one masked element at a time")
    emit("encImputeTraceN", str(len(traces)))
    emit("encImputeTraceRSq", fmt(statistics.median(traces), 2))
    # per-element cells for the supplementary imputation table; the redox row
    # is neither major nor trace and stays out of the table and the medians
    for el, v in im["per_element"].items():
        if "wt%" in v["unit"] or "ppm" in v["unit"]:
            emit(f"encImputeRSq{el}", fmt(v["log_r2"], 2))
except Exception as exc:
    tbd("encImputeTraceRSq", f"imputation readout unreadable: {exc}")
# the scored slice is the production encoder's test split restricted to
# pretrain_inherited_test_roster.csv (32,950 rows), which is also where both
# probe arms are fit -- latent and assay see one set of rows, not two
why = ""
try:
    pc = json.loads((PROBE / "probe_ceiling.json").read_text())
    qap, sg = pc["headline"]["qap_name"], pc["headline"]["supergroup"]
    raw = qap["best_raw_mlp_representation"]
    probe = {
        "encQapProbeAcc": fmt(qap["latent_mlp"], 3),
        "encQapRawMlp": fmt(qap["best_raw_mlp"], 3),
        "encQapProbeAccTuned": fmt(
            pc["tuned"]["latent"]["final_mlp"]["test_accuracy"], 3),
        "encQapRawMlpTuned": fmt(
            pc["tuned"][raw]["final_mlp"]["test_accuracy"], 3),
        "encQapLatentLin": fmt(qap["latent_logreg"], 3),
        "encQapRawLin": fmt(qap["best_raw_logreg"], 3),
        "encSgLatentLin": fmt(sg["latent_logreg"], 3),
        "encSgRawLin": fmt(sg["best_raw_logreg"], 3),
        "encQapProbeChance": fmt(pc["label_stats"]["qap_name"]["chance"], 3),
        "encQapProbeClasses": str(pc["n_classes"]),
        "encQapProbeMajority": fmt(
            pc["label_stats"]["qap_name"]["majority_share"], 2),
    }
except Exception as exc:
    probe = {}
    why = f"probe_ceiling.json unreadable: {exc}"

# emitted from a fixed list rather than from `probe` itself: on a partial read
# the missing macros must still reach the .tex as [TBD], since an undefined
# \newcommand is a hard LaTeX error at the owner's end
for name, comment in (
    ("encQapProbeAcc", "MLP probe on the frozen latent, redox padded"),
    ("encQapRawMlp", "same probe, same rows, reading the assay instead"),
    ("encQapProbeAccTuned", None),
    ("encQapRawMlpTuned",
     "both arms tuned under one search space; selects on test, so the gap "
     "is the statistic, not the level"),
    ("encQapLatentLin", None),
    ("encQapRawLin",
     "a linear readout is invariant to linear reparameterization, so this "
     "bounds any linear representation of the assay, PCA included"),
    ("encSgLatentLin", None),
    ("encSgRawLin", None),
    ("encQapProbeChance", "frequency-matched guess, sum p^2; NOT 1/n-classes"),
    ("encQapProbeClasses", "QAPF names in the probe's label space; all 70 in both slices"),
    ("encQapProbeMajority", "always answering basalt, the rock-blind floor"),
):
    if name in probe:
        emit(name, probe[name], comment)
    else:
        tbd(name, why)
# the log-vs-linear objective ablation cited in the pretraining appendix; no
# metrics file survives on disk, so the values are carried from ADR 007's
# recorded table (gate median trace log-R2, igneous-era encoder, 0.839 vs
# 0.877; reproducible from tag v3.1-logmse-objective)
emit("encImputeTraceRSqLinearObj", "0.84",
     "docs/decisions/007_ssl_pretrain_objective.md: linear-space objective")
emit("encImputeTraceRSqLogObj", "0.88",
     "docs/decisions/007_ssl_pretrain_objective.md: standardized log objective")

# the three heads re-supervised on measured proportions (#53), which differ only
# in what they are given about a rock: the latent representation, only the
# assay, or neither.  Levels at the knee and the full roster, plus the paired
# per-draw contrasts that are the honest statistic -- the 10-90 bands across
# draws overlap almost completely, since the arms are paired within a draw.
section("re-supervised heads against the recalibration layer, frozen production "
        "stack, val slice (<PROD>/curve_val.json)")
try:
    import statistics

    cv = json.loads((FEWSHOT / PROD / "curve_val.json").read_text())
    ARM_TAG = {"tailed": "Recal", "measured": "Latent",
               "measured_raw": "Assay", "measured_ablated": "Assemblage"}
    KNEE, FULL = 25, 400

    def rows(arm, budget):
        return [r for r in cv["curve"]
                if r["arm"] == arm and r["budget"] == budget]

    for budget, when in ((KNEE, "Knee"), (FULL, "Full")):
        emit(f"fsRocks{when}",
             str(round(statistics.mean([r["n_rocks"] for r in rows("measured", budget)]))),
             "realized labelled rocks, whole eruptions")
        for arm, tag in ARM_TAG.items():
            for term, mac in (("joint", "Joint"), ("bernoulli", "Presence"),
                              ("dirichlet", "Amount"), ("share_mae", "Mae")):
                digits = 4 if term == "share_mae" else 2
                emit(f"fs{tag}{mac}{when}",
                     fmt(statistics.median([r[term] for r in rows(arm, budget)]), digits))
    emit("fsMaeZeroShot", fmt(cv["zero_shot"]["share_mae"], 4),
         "nothing fit; several arms sit above this at the smallest budgets")
    emit("fsMaeZeroShotPct", fmt(cv["zero_shot"]["share_mae"] * 100, 1),
         "fsMaeZeroShot in percentage points")
    for budget, when in ((KNEE, "Knee"), (FULL, "Full")):
        emit(f"fsRecalMae{when}Pct",
             fmt(statistics.median(
                 [r["share_mae"] for r in rows("tailed", budget)]) * 100, 1),
             f"fsRecalMae{when} in percentage points")

    # paired within the draw: median of the per-draw differences, never the
    # difference of medians, which at the knee reads +2.28 instead of +1.38
    # because the two arms' worst draws are different draws
    def paired(a, b, budget, term="joint"):
        x = {r["draw"]: r[term] for r in rows(a, budget)}
        y = {r["draw"]: r[term] for r in rows(b, budget)}
        d = [x[k] - y[k] for k in x]
        return statistics.median(d), sum(1 for v in d if v > 0), len(d)

    for a, tag in (("measured_raw", "OverAssay"),
                   ("measured_ablated", "OverAssemblage"),
                   ("tailed", "OverRecal")):
        for budget, when in ((KNEE, "Knee"), (FULL, "Full")):
            med, favour, total = paired(a, "measured", budget)
            emit(f"fsLatent{tag}{when}", fmt(med, 2, sign=True))
            emit(f"fsLatent{tag}{when}Draws", f"{favour}/{total}")
    # the encoder margin quoted as a range across every budget swept
    spans = [paired("measured_raw", "measured", b)[0]
             for b in sorted({r["budget"] for r in cv["curve"]})]
    emit("fsLatentOverAssayLo", fmt(min(spans), 2, sign=True),
         "smallest paired margin of the latent over the assay, any budget")
    emit("fsLatentOverAssayHi", fmt(max(spans), 2, sign=True))
    # at the smallest budget the assay-given head is behind the one given
    # neither: 22 raw numbers overfit where the latent does not
    med, favour, total = paired("measured_raw", "measured_ablated", KNEE)
    emit("fsAssemblageOverAssayKnee", fmt(med, 2, sign=True),
         "positive = the head given neither beats the one given only the assay")
    emit("fsAssemblageOverAssayKneeDraws", f"{favour}/{total}")

    # the recalibration layer's share error only beats doing nothing with
    # labels behind it; at the knee it is worse than no fit at all
    zs_mae = cv["zero_shot"]["share_mae"]
    cross = [b for b in sorted({r["budget"] for r in cv["curve"]})
             if statistics.median(
                 [r["share_mae"] for r in rows("tailed", b)]) < zs_mae]
    if cross:
        emit("fsRecalMaeBeatsZeroShotAt",
             str(round(statistics.mean(
                 [r["n_rocks"] for r in rows("measured", cross[0])]))),
             "realized labelled rocks at which the recalibration layer's "
             "median share error first clears the zero-shot line")
        emit("fsRecalMaeBeatsZeroShotPct",
             fmt(statistics.median(
                 [r["share_mae"] for r in rows("tailed", cross[0])]) * 100, 1),
             "median tailed share error at that crossing budget, percentage points")
    else:
        tbd("fsRecalMaeBeatsZeroShotAt",
            "no budget's median share error clears zero-shot")

    def sweeps_share_mae(budget):
        for arm in ("measured", "measured_raw", "measured_ablated"):
            _, favour, total = paired(arm, "tailed", budget, "share_mae")
            if favour != total:
                return False
        return True

    unanimous = [b for b in sorted({r["budget"] for r in cv["curve"]})
                 if sweeps_share_mae(b)]
    if unanimous:
        emit("fsRecalMaeUnanimousAt",
             str(round(statistics.mean(
                 [r["n_rocks"] for r in rows("measured", unanimous[0])]))),
             "labelled rocks from which the simulation-trained head's share "
             "error beats all three re-supervised heads on every draw")
    else:
        tbd("fsRecalMaeUnanimousAt", "no budget wins share error on every draw")
except Exception as exc:
    tbd("fsLatentJointFull", f"curve_val.json unreadable for the #53 arms: {exc}")

# the inference card's fit is one unit-draw at a stamped seed, scored on the
# sealed test slice -- a single figure-scoped read, not the few-shot median,
# and the only test-slice read of the production stack
section("inference card: tailed tier on one 25-rock unit draw, sealed test "
        "slice (paper/figures/data/inference_card_*.csv, written by "
        "figures/src/extract_inference_card.py)")
try:
    figdata = ROOT / "paper" / "figures" / "data"
    with open(figdata / "inference_card_meta.csv") as f:
        meta = next(csv.DictReader(f))
    emit("icFitRocks", meta["n_fit_rocks"],
         "drawn as whole stratigraphic units from the train roster")
    emit("icFitUnits", meta["n_fit_units"])
    emit("icDrawSeed", meta["draw_seed"])
    emit("icNSamples", f"{int(meta['n_samples']):,}".replace(",", "{,}"),
         "candidate assemblages the card's abundance panels summarize")
    emit("icRockSilica", meta["sio2"], f"SiO2 wt% of rendered rock {meta['bulk_id']}")
    zs_j = float(meta["test_joint_zero_shot"])
    ad_j = float(meta["test_joint_recalibrated"])
    zs_d = float(meta["test_dirichlet_zero_shot"])
    ad_d = float(meta["test_dirichlet_recalibrated"])
    emit("icTestZsJoint", fmt(zs_j, 2),
         "test-slice zero-shot, the same rocks the card's fit is scored on")
    emit("icTestAdJoint", fmt(ad_j, 2))
    emit("icTestZsBern", fmt(float(meta["test_bernoulli_zero_shot"]), 2))
    emit("icTestAdBern", fmt(float(meta["test_bernoulli_recalibrated"]), 2))
    emit("icTestZsDir", fmt(zs_d, 2))
    emit("icTestAdDir", fmt(ad_d, 2))
    emit("icDirPct", f"{100 * (zs_d - ad_d) / (zs_j - ad_j):.0f}",
         "share of the recovery carried by the abundance factor")
    emit("icOpqRatePct", f"{100 * float(meta['test_opq_rate']):.0f}",
         "test rocks carrying iron-titanium oxides; the card's false positive "
         "lands on this base rate")
    with open(figdata / "inference_card_groups.csv") as f:
        card = {(r["arm"], r["group"]): r for r in csv.DictReader(f)}
    for name, arm, group, col, nd in (
        ("icBiPZs", "zero_shot", "bi", "p_pred", 2),
        ("icBiPAd", "recalibrated", "bi", "p_pred", 2),
        ("icOpqPZs", "zero_shot", "opq", "p_pred", 2),
        ("icOpqPAd", "recalibrated", "opq", "p_pred", 2),
        ("icBiShareZsPct", "zero_shot", "bi", "share_pred", 1),
        ("icBiShareAdPct", "recalibrated", "bi", "share_pred", 1),
        ("icBiShareMeasPct", "recalibrated", "bi", "measured_share", 1),
        ("icQShareZsPct", "zero_shot", "q", "share_pred", 1),
        ("icQShareAdPct", "recalibrated", "q", "share_pred", 1),
        ("icQShareMeasPct", "recalibrated", "q", "measured_share", 1),
        ("icOpqShareAdPct", "recalibrated", "opq", "share_pred", 1),
    ):
        v = float(card[(arm, group)][col])
        emit(name, fmt(100 * v if name.endswith("Pct") else v, nd))
except Exception as exc:
    tbd("icTestZsJoint", f"inference card CSVs unreadable: {exc}")

section("appendix composition/conditions cards, constants parsed from the "
        "card script; point at an extractor meta CSV when the card graduates "
        "to figures/src")
try:
    src = (ROOT / "tmp" / "comp_cond_card" / "comp_cond_card.py").read_text()
    n_cond = int(re.search(r"^N_SAMPLES = (\d+)", src, re.M).group(1))
    n_comp = int(re.search(r"^COMP_DRAWS = (\d+)", src, re.M).group(1))
    emit("acCondSamples", str(n_cond),
         "assemblages sampled from the presence branch for the condition panels")
    emit("acCompDraws", f"{n_comp:,}".replace(",", "{,}"),
         "draws from the predicted Dirichlet per composition panel")
except Exception as exc:
    tbd("acCondSamples", f"comp_cond_card.py unreadable: {exc}")
    tbd("acCompDraws", f"comp_cond_card.py unreadable: {exc}")

section("phase-vocabulary presence rates over the training labels, percent "
        "(runs/corpus_stats.json from scripts/build/build_corpus_stats.py)")
try:
    rates = json.loads(
        (ROOT / "runs" / "corpus_stats.json").read_text())["presence_rate_pct"]
    for token in SOLID_PHASES:
        emit(f"pv{token.capitalize()}", fmt(rates[token], 1))
except Exception as exc:
    for token in SOLID_PHASES:
        tbd(f"pv{token.capitalize()}", f"corpus_stats.json unreadable: {exc}")

section("zero-shot decomposition and per-group discrimination, headline "
        "cohort, frozen production stack (runs/real_rock/<PROD>/metrics.json)")
try:
    mt = json.loads((REALROCK / PROD / "metrics.json").read_text())
    hd, cl = mt["headline"], mt["climatology"]
    emit("rrClimJointProd", fmt(cl["joint_nll"], 2), "rock-independent reference, headline cohort")
    emit("rrClimBernProd", fmt(cl["bernoulli_nll"], 2))
    emit("rrClimDirProd", fmt(cl["dirichlet_nll"], 2))
    emit("rrClimNFit", str(cl["n_fit"]), "non-test rocks the cohort-wide reference is fit on")
    gaps = {k: hd[f"{k}_nll"] - cl[f"{k}_nll"] for k in ("joint", "bernoulli", "dirichlet")}
    emit("rrGapJointProd", fmt(gaps["joint"], 2), "model minus climatology, headline cohort")
    emit("rrGapBernProd", fmt(gaps["bernoulli"], 2))
    emit("rrGapDirProd", fmt(gaps["dirichlet"], 2))
    emit("rrGapDirPct", f"{100 * gaps['dirichlet'] / gaps['joint']:.0f}",
         "share of the zero-shot gap carried by the abundance factor")
    for g in mt["per_group"]:
        emit(f"rrAuroc{g['group'].capitalize()}", fmt(g["auroc"], 3))
        emit(f"rrSpearman{g['group'].capitalize()}", fmt(g["magnitude_spearman"], 3))
        emit(f"rrNQuant{g['group'].capitalize()}", str(g["n_quantified"]),
             "rocks with a numeric abundance for the category")
    emit("rrSpearmanPosGroups",
         str(sum(g["magnitude_spearman"] > 0 for g in mt["per_group"])),
         f"of {len(mt['per_group'])} groups with positive rank correlation")
    emit("rrMagRSqNegGroups",
         str(sum(g["magnitude_r2"] < 0 for g in mt["per_group"])),
         "groups whose magnitude R^2 is negative while their ranking holds")
    # the abundance miss in the units a geologist reads, teacher-forced
    # conditional-mean share against measured share -- the quantity Figure 1c
    # draws, so the range and the panel cannot disagree
    mag_mae = {g["group"]: g["magnitude_mae"] for g in mt["per_group"]}
    lo_group = min(mag_mae, key=mag_mae.get)
    hi_group = max(mag_mae, key=mag_mae.get)
    emit("rrMagMaeLo", fmt(mag_mae[lo_group], 3),
         f"best group ({lo_group}), mean absolute share error, zero-shot")
    emit("rrMagMaeHi", fmt(mag_mae[hi_group], 3), f"worst group ({hi_group})")
    emit("rrOffVocabPctProd", fmt(mt["off_vocabulary_mass"]["mean"] * 100, 2),
         "mean predicted crystal mass on the 11 unscored phases, percent")
    emit("rrAlteredJointProd", fmt(mt["altered"]["joint_nll"], 2),
         "alteration grades 3-4, scored separately")
    emit("rrAlteredN", str(mt["altered"]["n"]))
    emit("rrExposureBias", fmt(mt["exposure_bias"]["delta"], 2),
         "sampled-support minus true-support abundance NLL; a diagnostic, not a score")
    # the abundance-branch mixture arm. Its zero-shot read survives here; the
    # recalibrated contrast does NOT have a generated source any more (the
    # fixed-output-name contrast JSON was overwritten), so prose may cite the
    # zero-shot only.
    mx = json.loads((REALROCK / "joint_heads_allrock_mixture_seed0" / "metrics.json").read_text())
    emit("rrZsJointMixture", fmt(mx["headline"]["joint_nll"], 1))
    emit("rrZsDirMixture", fmt(mx["headline"]["dirichlet_nll"], 1))
except Exception as exc:
    tbd("rrGapJointProd", f"real_rock metrics.json unreadable: {exc}")

section("corpus-size ladder, tailed tier "
        "(explorations/corpus_scaling/results/scaling_realrock_tailed.json); "
        "simulator-side against measured-rock, same recalibration throughout")
try:
    sc = json.loads((SCALING / "scaling_realrock_tailed.json").read_text())
    pts = sorted((p for p in sc["points"] if p["seed"] == 0), key=lambda p: p["n_bulks"])
    lo, hi = pts[0], pts[-1]
    emit("csBulksLo", f"{lo['n_bulks']:,}".replace(",", "{,}"))
    emit("csBulksHi", f"{hi['n_bulks']:,}".replace(",", "{,}"))
    emit("csFold", str(round(hi["n_bulks"] / lo["n_bulks"])), "size ratio across the ladder")
    emit("csCorpusValLo", fmt(lo["corpus_val"]["joint_nll"], 2))
    emit("csCorpusValHi", fmt(hi["corpus_val"]["joint_nll"], 2))
    emit("csCorpusValGain", fmt(lo["corpus_val"]["joint_nll"] - hi["corpus_val"]["joint_nll"], 2),
         "simulator-side improvement across the ladder, monotone")
    dz = hi["zero_shot_640"]["joint_nll"] - lo["zero_shot_640"]["joint_nll"]
    dzd = hi["zero_shot_640"]["dirichlet_nll"] - lo["zero_shot_640"]["dirichlet_nll"]
    emit("csZsDelta", fmt(dz, 2, sign=True), "zero-shot change across the ladder, headline 640")
    emit("csZsDeltaDirPct", f"{100 * dzd / dz:.0f}")
    for key, short in (("full_ladder", "csSlope"), ("top_of_ladder", "csTopSlope")):
        s = sc["slopes_adapted_val_joint"][key]
        emit(f"{short}Mean", fmt(s["mean"], 3, sign=True), f"{key}, nats per doubling")
        emit(f"{short}Lo", fmt(s["lo"], 3, sign=True))
        emit(f"{short}Hi", fmt(s["hi"], 3, sign=True))
    sr = sc["seed_repeat"]["contrast_seed0_minus_seed1"]
    emit("csSeedMean", fmt(sr["mean"], 3, sign=True),
         f"seed repeat at {sc['seed_repeat']['n_bulks']} bulks, prices training noise")
    emit("csSeedLo", fmt(sr["lo"], 3, sign=True))
    emit("csSeedHi", fmt(sr["hi"], 3, sign=True))
    mae = [p["adapted_val"]["share_mae_mean"] for p in pts]
    emit("csShareMaeLo", fmt(min(mae), 3))
    emit("csShareMaeHi", fmt(max(mae), 3))
    bern = [p["adapted_val"]["bernoulli"] for p in pts]
    emit("csAdBernLo", fmt(min(bern), 2))
    emit("csAdBernHi", fmt(max(bern), 2))
except Exception as exc:
    tbd("csCorpusValGain", f"scaling_realrock_tailed.json unreadable: {exc}")

section("sedimentary XRD benchmark: a rock class and an instrument the heads "
        "never saw (explorations/sediment_transfer/results/)")
try:
    mf = json.loads((SEDIMENT / "manifest.json").read_text())
    rc = json.loads((SEDIMENT / "recalibrate.json").read_text())
    ec = json.loads((SEDIMENT / "encoder_contrast.json").read_text())
    emit("sedN", str(mf["n_rocks"]))
    emit("sedZsJoint", fmt(mf["headline_nll"]["joint"]["model"], 2))
    emit("sedClimJoint", fmt(mf["headline_nll"]["joint"]["climatology"], 2))
    emit("sedGap", fmt(mf["headline_nll"]["joint"]["model"]
                       - mf["headline_nll"]["joint"]["climatology"], 2),
         "cohort-wide zero-shot failure, all rocks")
    ov = mf["truth"]["off_vocab_share_p5_50_95"]
    emit("sedOffVocabPct", f"{100 * ov[1]:.0f}",
         "median truth mass outside the 21-phase vocabulary")
    emit("sedOffVocabLoPct", f"{100 * ov[0]:.0f}", "5th percentile; no rock escapes it")
    emit("sedOffVocabHiPct", f"{100 * ov[2]:.0f}", "95th percentile")
    emit("sedPopulatedTokens", str(mf["inputs"]["populated_tokens"]),
         "input rows this cohort can fill, of sysInputTokens")
    emit("sedBootstrapRatio", fmt(mf["bootstrap"]["eruption_v2_width_ratio"], 2),
         "per-rock intervals are this many times narrower than the "
         "unit-clustered ones on eruption-v2; no unit labels exist here")
    for group in ("px", "amp", "opq"):
        emit(f"sedBaseRate{group.capitalize()}Pct",
             f"{100 * mf['truth']['presence_base_rate'][group]:.0f}",
             "share of rocks containing it; fsp and q are in every rock")
    for split, short in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        emit(f"sed{short}N", str(rc["n"][split]))
    emit("sedTestGap", fmt(rc["zero_shot"]["joint"] - rc["climatology"]["joint"], 2),
         "the same failure on the held-out slice alone; quote this beside the recovery")
    # stored as [mean, lo, hi] of a negative delta; prose states the recovery
    # as a magnitude, so the bounds swap
    dv = rc["tiers"]["tailed"]["joint_delta_vs_zero_shot"]
    emit("sedRecovMean", fmt(-dv[0], 2))
    emit("sedRecovLo", fmt(-dv[2], 2))
    emit("sedRecovHi", fmt(-dv[1], 2))
    emit("sedAhead", fmt(-rc["best_vs_climatology"], 2),
         "nats ahead of the sedimentary climatology after fitting")
    # the whole margin above is abundance: the recovered presence term does not
    # beat a forecast that never looks at the rock, which is the appendix's
    # scoping claim and must stay quotable as a pair
    emit("sedClimTestJoint", fmt(rc["climatology"]["joint"], 2),
         "test-slice climatology; sedClimJoint is the cohort-wide one, do not blur")
    # presence at 3 decimals: the pair differs by under 0.01, and rounding to 2
    # renders a tie as a visible gap, which is the opposite of the claim
    for term, short, places in (("bernoulli", "Bern", 3), ("dirichlet", "Dir", 2)):
        emit(f"sedTailed{short}", fmt(rc["tiers"]["tailed"][term], places))
        emit(f"sedClim{short}", fmt(rc["climatology"][term], places))
    for term, short in (("bernoulli", "Bern"), ("dirichlet", "Dir")):
        d = ec["paired_delta_allrock_minus_igneous"][term]
        emit(f"sedEnc{short}Mean", fmt(d[0], 2, sign=True),
             "all-rock minus igneous-only encoder; negative = all-rock better")
        emit(f"sedEnc{short}Lo", fmt(d[1], 2, sign=True))
        emit(f"sedEnc{short}Hi", fmt(d[2], 2, sign=True))
    fs = json.loads((SEDIMENT / "fewshot_budget.json").read_text())
    emit("sedFewshotBudget", str(fs["budget"]))
    emit("sedFewshotDraws", str(fs["n_draws"]))
    for tier in ("matrix", "tailed"):
        emit(f"sedFewshot{tier.capitalize()}", fmt(fs["tiers"][tier]["median_joint"], 2),
             f"median over the draws; compare against sedClimTestJoint")
except Exception as exc:
    tbd("sedGap", f"sediment_transfer results unreadable: {exc}")

# the sedimentary cohort's 43% only means something beside the volcanic
# benchmark's own truth-side figure, which is a different quantity from the
# predicted-side rrOffVocabPctProd and had never been measured
section("volcanic truth-side off-vocabulary mass, du Bray scored set "
        "(explorations/real_mineralogy/results/benchmark_composition.json)")
try:
    bc = json.loads((ROOT / "explorations" / "real_mineralogy" / "results"
                     / "benchmark_composition.json").read_text())["off_vocabulary"]
    emit("rrTruthInVocabN", str(bc["as_scored"]["rocks_fully_in_vocabulary"]),
         "rocks whose whole measured assemblage the seven groups represent")
    emit("rrTruthScoredN", str(bc["as_scored"]["rocks"]))
    emit("rrGroundmassN", str(bc["point_count_basis"]["rocks_affected"]),
         "point-counted rocks itemizing groundmass, the sole out-of-vocabulary category")
    emit("rrGroundmassPct",
         f"{100 * bc['point_count_basis']['share_among_affected_p50']:.0f}",
         "median share of those rocks, volume basis; excluded like the labels' melt")
except Exception as exc:
    tbd("rrTruthInVocabN", f"benchmark_composition.json unreadable: {exc}")

section("controls: each defends a claim and asserts nothing on its own "
        "(explorations/real_mineralogy/results/<PROD>/, explorations/redox_padding/results/)")
try:
    CTL = ROOT / "explorations" / "real_mineralogy" / "results" / PROD

    def interval(prefix, d, nd=2):
        emit(f"{prefix}Mean", fmt(d["mean"], nd, sign=True))
        emit(f"{prefix}Lo", fmt(d["lo"], nd, sign=True))
        emit(f"{prefix}Hi", fmt(d["hi"], nd, sign=True))

    pc = json.loads((CTL / "pointcount_contrast.json").read_text())
    emit("ctlRulerN", str(pc["n_paired"]), "rocks carrying both a visual estimate and a point count")
    emit("ctlRulerEruptions", str(pc["n_eruptions"]),
         "clusters behind the interval; too few to call the joint contrast either way")
    for term, short in (("joint", "Joint"), ("dirichlet", "Dir"), ("bernoulli", "Bern")):
        interval(f"ctlRuler{short}", pc["contrasts"][f"visual_minus_pointcount_{term}"])
    opq = next(g for g in pc["truth_disagreement"] if g["group"] == "opq")
    emit("ctlRulerOpqVisual", fmt(opq["median_share_visual"], 3))
    emit("ctlRulerOpqPointcount", fmt(opq["median_share_pointcount"], 3))

    ul = json.loads((CTL / "unit_leakage.json").read_text())
    emit("ctlLeakFlagged", str(ul["n_flagged"]), "validation rocks whose unit name near-merges a training unit")
    emit("ctlLeakFreeN", str(ul["val"]["leak_free"]["n"]))
    # NOTE the two rows disagree in direction. Recalibrated, removing the
    # flagged rocks IMPROVES the score (0.885 < rrValTailedProd), so those
    # rocks were harder, not easier, and cannot have inflated the reported
    # numbers. Zero-shot, removal moves the other way. State the recalibrated
    # pair, since that is the scale every reported result uses.
    emit("ctlLeakFreeTailed", fmt(ul["val"]["leak_free"]["tailed_joint"], 3))
    emit("ctlLeakFreeZs", fmt(ul["val"]["leak_free"]["zero_shot_joint"], 2),
         "zero-shot on the survivors; moves opposite to the recalibrated pair")

    sv = json.loads((CTL / "scale_convention.json").read_text())
    emit("ctlScaleN", str(sv["n_rescaled_scored_nontest"]))
    interval("ctlScale", sv["contrasts"]["published_minus_undone_joint"], 3)
    emit("ctlScaleDoseSpearman", fmt(sv["dose_response"]["spearman"], 3),
         "correction size against per-rock benefit; no dose response")

    RDX = ROOT / "explorations" / "redox_padding" / "results"
    ms = json.loads((RDX / "missingness.json").read_text())
    gm = json.loads((RDX / "geometry.json").read_text())
    emit("ctlRedoxAurocInputs", fmt(ms["arms"]["inputs"]["auroc"], 3),
         "blankness of the redox slot predicted from the other inputs")
    emit("ctlRedoxAurocLatent", fmt(ms["arms"]["latent"]["auroc"], 3))
    emit("ctlRedoxPopulatedPct", f"{100 * ms['populated_fraction']:.1f}",
         "share of pretraining rows reporting the ratio")
    emit("ctlRedoxOffCurve", fmt(gm["padded_to_curve_over_span"]["p50"], 3),
         "padded point's distance from the true-ratio curve, in units of the sweep span")
except Exception as exc:
    tbd("ctlRulerJointMean", f"control results unreadable: {exc}")

# system facts not read from a metrics file; provenance in the comment.
# UNVERIFIED entries must be confirmed on disk before submission.
section("system facts (hand-carried, provenance per line)")
emit("sysEffLambdaFactor", "21", "docs/heads.md: BCE phase-mean vs Dirichlet sum")
emit("sysLambdaOneNominal", "0.048", "docs/heads.md: magnitude_weight at effective lambda 1")
emit("sysEffLambdaBaseline", "5.3", "docs/2026_08_03_recap.md: production dials")
emit("sysEffLambdaTuned", "163", "docs/2026_08_03_recap.md: reversed trial-22 dials")
emit("sysPhaseVocab", "21", "cnmineral.data.phase_vocab")
emit("sysRealGroups", "7", "coarsened real-rock groups")
emit("sysTraceTokens", "13")
emit("sysInputTokens", "22", "9 majors + 13 traces; populated slots")
emit("sysEncoderTokenSlots", "23", "22 + the Fe-redox slot, padded in production")
emit("sysMatTwoVecDim", "200", "mat2vec.csv width")
emit("sysPretrainBulksIgneous", "167k", "docs/encoder.md, igneous 22-token generation")
emit("sysFewshotBudgetKnee", "25", "few-shot curve grid point where most of the gain lands (#52)")
emit("sysRecalibEpochs", "400", "full-batch Adam steps, training/real_rock_recalibrator.fit_recalibrator")

# counted 2026-08-08 by instantiating the production BulkChemistryDataset with
# pretrain_allrock_redox's data_kwargs; settles the encoder.md 788k vs
# acceptance-memory 760k discrepancy in favour of 788k
emit("sysPretrainBulksAllrock", "787{,}721", "counted from data/bulks/gard2019.csv under the production filter+dedup")

# corpus facts counted from data/corpus/*_rungs.parquet at rungs 50/30
emit("sysCorpusTrainBulks", "10{,}000", "stratified draw; 9,946 realized")
emit("sysCorpusTrainBulksRealized", "9{,}946")
emit("sysCorpusTrainDraws", "48", "old broad-draw corpus; verified on disk")
emit("sysCorpusTrainRows", "949{,}228", "(bulk, draw, rung) triples at rungs 50/30")
emit("sysCorpusValtestDraws", "128")
emit("sysCorpusValtestBulks", "493", "realized of the 500 committed when this corpus was generated")
emit("sysCorpusValtestRows", "125{,}105")
emit("sysCorpusValBulks", "218", "realized val slice, instrument valtest128-broad")
emit("sysCorpusValRows", "55{,}576")
emit("sysCorpusTestBulks", "274", "the remaining realized valtest compositions; one carries no subset label")
emit("sysCorpusTestRows", "69{,}521")

# head architecture and loss weights from the stamped checkpoint; optimizer,
# batch, and stopping knobs from the config that produced it (not stamped)
section("production joint-head configuration (architecture/loss stamped in "
        "checkpoints/<PROD>/heads_best.pt; optimization from "
        "configs/heads/joint_heads.yaml)")
try:
    import numpy as np
    import torch

    from cnmineral.data.magemin_corpus import COMP_OXIDES

    hc = yaml.safe_load((ROOT / "configs" / "heads" / "joint_heads.yaml").read_text())
    ck = torch.load(ROOT / "checkpoints" / PROD / "heads_best.pt",
                    map_location="cpu", weights_only=False)
    head_cfg, loss_cfg, mon = ck["head_config"], ck["loss_config"], ck["monitor"]
    emit("sysCompOxides", str(len(COMP_OXIDES)),
         "MAGEMin output oxides, cnmineral.data.magemin_corpus.COMP_OXIDES")
    emit("hdCompEps", pow10(ck["label_config"]["comp_eps"]),
         "hard-zero oxide clip on composition targets, stamped label_config")
    emit("hdModeHidden", str(head_cfg["mode_hidden"]))
    emit("hdModeLayers", str(head_cfg["mode_layers"]))
    emit("hdCompHidden", str(head_cfg["comp_hidden"] or head_cfg["d_model"] // 2),
         "null in the config resolves to d_model // 2")
    emit("hdCompLayers", str(head_cfg["comp_layers"]))
    emit("hdCondHidden", str(head_cfg["cond_hidden"]))
    emit("hdCondLayers", str(head_cfg["cond_layers"]))
    emit("hdDropout", fmt(head_cfg["dropout"], 3))
    emit("hdCondComponents", str(head_cfg["cond_n_components"]),
         "mixture components per condition axis")
    emit("hdMagWeight", fmt(loss_cfg["magnitude_weight"], 4))
    emit("hdEffLambda", fmt(loss_cfg["magnitude_weight"] * 21, 2),
         "magnitude weight x the 21-phase averaging mismatch (docs/heads.md)")
    emit("hdLambdaComp", fmt(loss_cfg["lambda_comp"], 1))
    emit("hdLambdaCond", fmt(loss_cfg["lambda_cond"], 2),
         "also absorbs the raw-unit scale of the conditions NLL")
    emit("hdLambdaBalance", sci(loss_cfg["lambda_balance"]))
    emit("hdLr", sci(hc["lr"]))
    emit("hdWeightDecay", sci(hc["weight_decay"]))
    emit("hdBatchSize", str(hc["batch_size"]))
    emit("hdEpochsCeiling", str(hc["epochs"]), "ceiling, not a target")
    emit("hdPatience", str(hc["early_stop_patience"]),
         "per head, on its own raw val term")
    emit("hdSeed", str(hc["seed"]))
    mode_sel = [b + n for b, n in
                zip(mon["val_presence_bce"], mon["val_mode_nll"])]
    emit("hdEpochsRun", str(len(mon["train"])),
         "early stop: every head past its own best by the patience")
    emit("hdModeBestEpoch", str(1 + int(np.argmin(mode_sel))))
    emit("hdCompBestEpoch", str(1 + int(np.argmin(mon["val_comp_nll"]))))
    emit("hdCondBestEpoch", str(1 + int(np.argmin(mon["val_cond_nll"]))))
except Exception as exc:
    tbd("hdModeHidden", f"head config unreadable: {exc}")

section("joint single-stage HPO (configs/heads/joint_heads_hpo.yaml + "
        "runs/heads_hpo/joint_heads_allrock_hpo/summary.json)")
try:
    hp = yaml.safe_load(
        (ROOT / "configs" / "heads" / "joint_heads_hpo.yaml").read_text())
    sm = json.loads((ROOT / "runs" / "heads_hpo" / "joint_heads_allrock_hpo"
                     / "summary.json").read_text())
    emit("hpoTrials", str(sm["n_trials_complete"]), "completed TPE trials")
    emit("hpoStartupTrials", str(hp["n_startup_trials"]),
         "random startup trials seeding the sampler")
    emit("hpoTrialEpochs", str(hp["base"]["epochs"]),
         "per-trial epoch ceiling; production retrains at the 120 ceiling")
    emit("hpoTieBand", fmt(sm["tie_band"], 2),
         "nats; the instrument's tie band from results/ledger.yaml")
    emit("hpoWinnerTrial", str(sm["tiebreak_winner"]["number"]))
    ss = hp["search_space"]

    def logrange(spec):
        return f"{pow10(spec['low'])} to {pow10(spec['high'])}"

    emit("hpoRangeLr", logrange(ss["lr"]), "log-uniform")
    emit("hpoRangeWeightDecay", logrange(ss["weight_decay"]), "log-uniform")
    emit("hpoRangeMagWeight",
         f"{fmt(ss['magnitude_weight']['low'], 4)} to "
         f"{fmt(ss['magnitude_weight']['high'], 3)}",
         "log-uniform; effective lambda 0.1 to 1")
    emit("hpoRangeLambdaBalance", logrange(ss["lambda_balance"]), "log-uniform")
    emit("hpoRangeDropout",
         f"{ss['dropout']['low']:g} to {ss['dropout']['high']:g}", "uniform")
    emit("hpoRangeModeHidden",
         ", ".join(str(c) for c in ss["mode_hidden"]["choices"]), "categorical")
    emit("hpoRangeModeLayers",
         f"{ss['mode_layers']['low']} to {ss['mode_layers']['high']}", "integer")
except Exception as exc:
    tbd("hpoTrials", f"HPO config or summary unreadable: {exc}")

section("recalibration fitting protocol (fit defaults from "
        "training/real_rock_recalibrator.fit_recalibrator; selected shrinkage "
        "from <PROD>/tailed_acceptance.json)")
try:
    import inspect

    from cnmineral.training.real_rock_recalibrator import (
        fit_recalibrator,
        select_weight_decay,
    )

    ta = json.loads((FEWSHOT / PROD / "tailed_acceptance.json").read_text())
    fit_defaults = {
        k: v.default for k, v in
        inspect.signature(fit_recalibrator).parameters.items()
    }
    emit("rcFitLr", f"{fit_defaults['lr']:g}",
         "full-batch Adam, fit_recalibrator default")
    emit("rcTailedWeightDecay", fmt(ta["weight_decay"]["tailed"], 1),
         "chosen once on the fixed val slice, shared across budgets and draws")
    grid = sorted(
        inspect.signature(select_weight_decay).parameters["grid"].default)
    emit("rcWdGridLo", pow10(grid[0]))
    emit("rcWdGridHi", f"{grid[-1]:g}")
except Exception as exc:
    tbd("rcTailedWeightDecay", f"recalibrator fit sources unreadable: {exc}")

# architecture + param counts read from the frozen production checkpoints
section("architecture and parameter counts (read from checkpoints)")
try:
    import torch
    ck = torch.load(
        ROOT / "checkpoints" / "joint_heads_allrock_tuned_seed0" / "heads_best.pt",
        map_location="cpu", weights_only=False,
    )
    enc = torch.load(
        ROOT / "checkpoints" / "pretrain_allrock_redox_seed0" / "epoch_best.pt",
        map_location="cpu", weights_only=False,
    )

    def count(state):
        return sum(v.numel() for v in state.values() if hasattr(v, "shape"))

    def grouped(n):
        return f"{n:,}".replace(",", "{,}")

    per_head = {k: count(ck[k]) for k in ("mode", "comp", "cond")}
    emit("sysModeHeadParams", grouped(per_head["mode"]))
    emit("sysCompHeadParams", grouped(per_head["comp"]))
    emit("sysCondHeadParams", grouped(per_head["cond"]))
    emit("sysHeadParams", grouped(sum(per_head.values())))
    # the two FractionalEncoder sinusoidal tables are fixed buffers, not
    # learned; the paper quotes trainable only
    buffers = 2 * enc["config"]["frac_resolution"] * (enc["config"]["d_model"] // 2)
    emit("sysEncoderParams", grouped(count(enc["encoder"]) - buffers),
         "trainable; excludes the fixed sinusoidal lookup tables")
    ec = enc["config"]
    emit("sysEncoderDModel", str(ec["d_model"]))
    emit("sysEncoderLayers", str(ec["n_layers"]))
    emit("sysEncoderHeads", str(ec["n_heads"]))
    emit("sysMaskProb", f"{ec['mask_prob']:.2f}")
    emit("sysMaskPct", f"{100 * ec['mask_prob']:.0f}", "same value, for prose")
except Exception as exc:
    for name in ("sysHeadParams", "sysEncoderParams", "sysModeHeadParams",
                 "sysEncoderDModel", "sysEncoderLayers", "sysEncoderHeads",
                 "sysMaskProb"):
        tbd(name, f"checkpoint unreadable: {exc}")

# recalibrator tier sizes, counted from the module rather than recalled
section("recalibrator tier parameter counts")
try:
    from cnmineral.models.real_rock_recalibrator import RealRockRecalibrator

    tiers = {t: sum(p.numel() for p in RealRockRecalibrator(t).parameters())
             for t in ("vector", "matrix", "tailed")}
    emit("sysRecalibVectorParams", str(tiers["vector"]))
    emit("sysRecalibMatrixParams", str(tiers["matrix"]))
    emit("sysRecalibParams", str(tiers["tailed"]), "canonical tailed tier")

    # the capacity mismatch the #53 contrast runs against: a re-supervised head
    # is ~280x the recalibration layer, whose correction acts on outputs only
    from cnmineral.models.measured_mode_head import MeasuredModeHead

    n_latent = sum(p.numel() for p in MeasuredModeHead(256).parameters())
    emit("sysResupervisedParams", f"{n_latent:,}".replace(",", "{,}"),
         "head given the latent representation, d_model 256")
    n_assay = sum(p.numel() for p in MeasuredModeHead(22).parameters())
    emit("sysResupervisedAssayParams", f"{n_assay:,}".replace(",", "{,}"),
         "head given only the assay, 22 element tokens")
    emit("sysResupervisedOverRecalib", str(round(n_latent / tiers["tailed"])),
         "parameter ratio, re-supervised head over the recalibration layer")
except Exception as exc:
    tbd("sysRecalibParams", f"recalibrator import failed: {exc}")

# the re-supervised heads' own fit, which Appendix C describes: architecture
# and optimizer from the fitting function's fixed defaults (nothing here is
# tuned per arm), the shrinkage constant and budget grid from the curve that
# produced the fs* macros
section("re-supervised head configuration and sweep protocol "
        "(training.measured_mode_head.fit_measured_head defaults; "
        "<PROD>/curve_val.json for the grid and shrinkage)")
try:
    import inspect

    from cnmineral.training.measured_mode_head import fit_measured_head
    from cnmineral.training.real_rock_recalibrator import draw_units

    defaults = {k: v.default
                for k, v in inspect.signature(fit_measured_head).parameters.items()}
    emit("rsHidden", str(defaults["hidden"]), "trunk width, both branches")
    emit("rsComponents", str(defaults["n_components"]),
         "Dirichlet mixture components, matched to the tailed tier")
    emit("rsLr", fmt(defaults["lr"], 2), "full-batch Adam, fixed a priori")
    emit("rsEpochs", str(defaults["epochs"]), "steps to a train-loss plateau")
    emit("rsOvershootPct",
         str(round(inspect.signature(draw_units).parameters["overshoot"].default * 100)),
         "a unit overshooting the target budget by more than this is skipped")

    cv = json.loads((FEWSHOT / PROD / "curve_val.json").read_text())
    wd = {cv["weight_decay"][a]
          for a in ("measured", "measured_raw", "measured_ablated")}
    if len(wd) != 1:
        raise ValueError(f"re-supervised arms disagree on weight decay: {wd}")
    emit("rsWeightDecay", fmt(wd.pop(), 1),
         "one shrinkage constant, shared by all three re-supervised arms")
    grid = sorted({r["budget"] for r in cv["curve"]})
    emit("rsBudgetGrid", ", ".join(str(b) for b in grid[:-1]) + f", and {grid[-1]}",
         "target budgets; the largest exceeds the roster, so it realizes as all "
         "of it")
except Exception as exc:
    tbd("rsHidden", f"measured-head defaults unreadable: {exc}")

# the sign of the zero-shot abundance miss, rock by rock. Read from the
# figure's own extract so the table and panel (c) cannot disagree: both
# violins of a category cover the same rocks in the same order, so pairing by
# position is exact.
section("zero-shot abundance bias per mineral category "
        "(paper/figures/data/cohort_card_abundance.csv, written by "
        "figures/src/extract_cohort_card.py)")
try:
    import statistics

    pairs = {}
    order = []
    with open(ROOT / "paper" / "figures" / "data" / "cohort_card_abundance.csv") as fh:
        for row in csv.DictReader(fh):
            if row["group"] not in order:
                order.append(row["group"])
            pairs.setdefault((row["group"], row["kind"]), []).append(float(row["share"]))
    for g in order:
        meas, pred = pairs[(g, "measured")], pairs[(g, "predicted")]
        if len(meas) != len(pred):
            raise ValueError(f"{g}: {len(meas)} measured against {len(pred)} predicted")
        over = sum(p > m for p, m in zip(pred, meas))
        emit(f"rrOverPct{g.capitalize()}", str(round(100 * over / len(meas))),
             "percent of quantified rocks whose predicted share exceeds the "
             "measured one" if g == order[0] else None)
        emit(f"rrMagFold{g.capitalize()}",
             fmt(statistics.median(p / m for p, m in zip(pred, meas) if m > 0), 2),
             "median per-rock predicted/measured share" if g == order[0] else None)
except Exception as exc:
    tbd("rrOverPctFsp", f"cohort card abundance extract unreadable: {exc}")

header = [
    "% numbers.tex -- GENERATED by scripts/build/build_numbers.py; do not edit.",
    "% Every number the paper cites lives here as a macro with provenance.",
    "\\newcommand{\\paperTBD}{\\textbf{[TBD]}}",
]
OUT.write_text("\n".join(header + macros) + "\n")
print(f"wrote {OUT} ({sum(1 for m in macros if m.startswith(chr(92)))} macros)")
if missing:
    print("missing sources:", file=sys.stderr)
    for m in missing:
        print(f"  {m}", file=sys.stderr)

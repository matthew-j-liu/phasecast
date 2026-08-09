"""Builds paper/numbers.tex from the tracked ledgers and result JSONs, so no
number is ever hand-typed in paper prose. Rerun after any source moves:
`conda run -n alignn_fix python scripts/build/build_numbers.py`. A missing source
degrades to a visible [TBD] macro and a stderr warning, never a broken build.
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "numbers.tex"

FEWSHOT = ROOT / "explorations" / "real_rock_fewshot" / "results"
TAGS = {
    "Prod": "joint_heads_supportmag_seed0",
    "Regen": "joint_heads_regen_seed0",
    "RegenTuned": "joint_heads_regen_tuned_seed0",
    "Allrock": "joint_heads_allrock_seed0",
}

macros = []
missing = []


def emit(name, value, comment=None):
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
    for short, tag in TAGS.items():
        e = entries.get((tag, "eruption-v2"))
        if e is None:
            tbd(f"rrZsJoint{short}", f"no eruption-v2 entry for {tag}")
            continue
        emit(f"rrZsJoint{short}", fmt(e["joint_nll"], 1), tag)
        emit(f"rrZsBern{short}", fmt(e["bernoulli_nll"], 1))
        emit(f"rrZsDir{short}", fmt(e["dirichlet_nll"], 1))
    v1 = entries.get((TAGS["Prod"], "eruption-v1"))
    if v1:
        emit("rrZsJointProdVOne", fmt(v1["joint_nll"], 1),
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
    for short in ("Prod", "Allrock"):
        v = by_tag.get(TAGS[short])
        if v is None:
            tbd(f"cvModeJoint{short}", f"no ledger entry for {TAGS[short]}")
            continue
        emit(f"cvModeJoint{short}", fmt(v["heads"]["mode"]["joint_nll"], 2))
except Exception as exc:
    tbd("cvModeJointProd", f"ledger.yaml unreadable: {exc}")

# three-way recalibrated contrast; the test slice's one sealed read
section("three-way checkpoint contrast, matrix tier fit on the 368-rock "
        "train roster (explorations/real_rock_fewshot/results/"
        "checkpoint_contrast_matrix_three_way.json); test = the one sealed read")
try:
    tw = json.loads((FEWSHOT / "checkpoint_contrast_matrix_three_way.json").read_text())
    for short in ("Prod", "Regen", "RegenTuned"):
        lv = tw["levels"][TAGS[short]]
        emit(f"rrTestAd{short}", fmt(lv["test"]["adapted_joint"], 2))
        emit(f"rrValAd{short}", fmt(lv["val"]["adapted_joint"], 2))
        emit(f"rrValZs{short}", fmt(lv["val"]["zero_shot_joint"], 1),
             "val-slice zero-shot, same rocks as the adapted number")
    emit("rrValNRocks", str(tw["levels"][TAGS["Prod"]]["val"]["n_rocks"]))
    emit("rrTestNRocks", str(tw["levels"][TAGS["Prod"]]["test"]["n_rocks"]))
    pairs = {
        ("Prod", "Regen"): "ProdRegen",
        ("Prod", "RegenTuned"): "ProdTuned",
        ("Regen", "RegenTuned"): "RegenTuned",
    }
    for split in ("val", "test"):
        con = tw["paired_contrasts_adapted_joint"][split]
        for (a, b), label in pairs.items():
            key = f"{TAGS[a]} - {TAGS[b]}"
            if key not in con:
                continue
            c = con[key]
            pre = "rrValCon" if split == "val" else "rrTestCon"
            emit(f"{pre}{label}Mean", fmt(c["mean"], 2, sign=True),
                 f"{a} minus {b}, unit-clustered")
            emit(f"{pre}{label}Lo", fmt(c["lo"], 2, sign=True))
            emit(f"{pre}{label}Hi", fmt(c["hi"], 2, sign=True))
except Exception as exc:
    tbd("rrTestAdProd", f"three-way contrast unreadable: {exc}")

# encoder-swap contrast (val only; test untouched)
section("allrock-vs-production recalibrated val contrast "
        "(checkpoint_contrast_matrix.json); reported as allrock minus "
        "production, so positive = the all-rock encoder is worse")
try:
    two = json.loads((FEWSHOT / "checkpoint_contrast_matrix.json").read_text())
    con = two["paired_contrasts_adapted_joint"]["val"]
    key_fwd = f"{TAGS['Allrock']} - {TAGS['Prod']}"
    key_rev = f"{TAGS['Prod']} - {TAGS['Allrock']}"
    if key_fwd in con:
        c, flip = con[key_fwd], False
    elif key_rev in con:
        c, flip = con[key_rev], True
    else:
        raise KeyError("no supportmag/allrock pair in contrasts")
    mean = -c["mean"] if flip else c["mean"]
    lo = -c["hi"] if flip else c["lo"]
    hi = -c["lo"] if flip else c["hi"]
    emit("rrValConAllrockProdMean", fmt(mean, 2, sign=True))
    emit("rrValConAllrockProdLo", fmt(lo, 2, sign=True))
    emit("rrValConAllrockProdHi", fmt(hi, 2, sign=True))
except Exception as exc:
    tbd("rrValConAllrockProdMean", f"two-way contrast unreadable: {exc}")

# few-shot curve at the 25-rock budget: median joint NLL over draws, per
# recalibrator tier (vector / matrix), val slice
section("few-shot curves, val slice, budget 25 "
        "(explorations/real_rock_fewshot/results/<tag>/curve_val.json)")
try:
    import statistics
    for short in ("Prod", "Regen", "RegenTuned"):
        path = FEWSHOT / TAGS[short] / "curve_val.json"
        cv = json.loads(path.read_text())
        for tier in ("vector", "matrix"):
            pts = [p["joint"] for p in cv["curve"]
                   if p["tier"] == tier and p["budget"] == 25]
            if not pts:
                tbd(f"fsTwentyFive{tier.capitalize()}{short}",
                    f"no {tier} budget-25 points in {path.name}")
                continue
            emit(f"fsTwentyFive{tier.capitalize()}{short}",
                 fmt(statistics.median(pts), 2),
                 f"median over {len(pts)} draws")
except Exception as exc:
    tbd("fsTwentyFiveVectorProd", f"curve_val.json unreadable: {exc}")

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
emit("sysBenchTrainRocks", "368", "eruption-v2 split by stratigraphic unit")
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
emit("sysCorpusValBulks", "218", "realized val slice, instrument valtest128-broad")

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
except Exception as exc:
    tbd("sysRecalibParams", f"recalibrator import failed: {exc}")

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

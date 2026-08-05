"""Builds paper/numbers.tex from the tracked ledgers and result JSONs, so no
number is ever hand-typed in paper prose. Rerun after any source moves:
`conda run -n alignn_fix python paper/src/build_numbers.py`. A missing source
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
emit("sysHeadParams", "895{,}169", "slides/overview.py, from heads_best.pt")
emit("sysEncoderParams", "3.23M", "slides/overview.py, frozen encoder")
emit("sysRecalibParams", "121", "UNVERIFIED-EXACT: recap says ~121; count from fit code before submission")
emit("sysPhaseVocab", "21", "cnmineral.data.phase_vocab")
emit("sysRealGroups", "7", "coarsened real-rock groups")
emit("sysTraceTokens", "13")
emit("sysInputTokens", "22", "9 majors + 13 traces; all-rock encoder adds a 23rd redox slot")
emit("sysPretrainBulksIgneous", "167k", "docs/encoder.md, igneous 22-token generation")
emit("sysPretrainBulksAllrock", "788k", "UNVERIFIED: encoder.md says ~788k rows; acceptance memory says ~760k after dedup -- reconcile")
emit("sysCorpusTrainBulks", "10{,}000", "old broad-draw corpus, docs/encoder.md")
emit("sysCorpusTrainDraws", "48")
emit("sysCorpusValtestDraws", "128")
emit("sysBenchTrainRocks", "368", "eruption-v2 split by stratigraphic unit")
emit("sysFewshotBudgetKnee", "25", "few-shot curve grid point where most of the gain lands (#52)")

header = [
    "% numbers.tex -- GENERATED by paper/src/build_numbers.py; do not edit.",
    "% Every number the paper cites lives here as a macro with provenance.",
    "\\newcommand{\\paperTBD}{\\textbf{[TBD]}}",
]
OUT.write_text("\n".join(header + macros) + "\n")
print(f"wrote {OUT} ({sum(1 for m in macros if m.startswith(chr(92)))} macros)")
if missing:
    print("missing sources:", file=sys.stderr)
    for m in missing:
        print(f"  {m}", file=sys.stderr)

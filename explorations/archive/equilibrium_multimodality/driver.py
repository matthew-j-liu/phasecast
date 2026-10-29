"""Sample (P, H2O) per test bulk, cool on the ladder, run MAGEMin equilibrium.

Per bulk: Fe-split the majors (class-conditional Middlemost, provenance tagged),
draw N (P, H2O) by Latin-Hypercube from the LOCKED solubility-bounded proposal
`q = Uniform(P[0.5,12] kbar x H2O[0, min(8, S(P))] wt%)` (regen spec, now ADR 004,
"Bounded sampler"; `cnmineral.data.condition_proposal`), and expand each draw
over a cooling T-ladder. H2O is injected as an exact wt fraction (anhydrous
majors renormalized to 100 - H2O) so the sampled H2O axis means the same thing
across bulks. MAGEMin runs per bulk (one Julia call each) for resilience +
progress; raw JSONL lands under the chosen work dir (gitignored).

The pre-redesign flat-box run (q uniform on the full P x H2O rectangle) is kept
at work/ root; bounded runs default to work/bounded/ so water_ab.py can compare
the two on the same bulks.

    python explorations/equilibrium_multimodality/driver.py            # full: 36 bulks x N=128
    python explorations/equilibrium_multimodality/driver.py --smoke    # 2 bulks x N=8
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from cnmineral.data.condition_proposal import draw_conditions
from cnmineral.data.fe_redox import ClassRatioStrategy
from cnmineral.data.magemin_oxides import bulk_to_magemin

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
RUNNER = REPO / "magemin" / "run_magemin.jl"
PARQUET = REPO / "data" / "bulks" / "gard2019.parquet"

# Cooling ladder: liquidus-ish down to a sub-solidus floor. Floor is 600 (below
# the prototype's 640) to probe the wet-bulk near-solidus attrition tail; kept
# for the water A/B so flat and bounded runs share the same ladder.
T_LADDER = np.arange(1200.0, 599.0, -20.0)

OXIDE_KEYS = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3", "k2o", "na2o", "tio2", "cr2o3"]
INPUT_COLS = ["sample_id", "T_celsius", "P_kbar", *OXIDE_KEYS, "h2o_tot"]


def load_majors():
    df = pd.read_parquet(PARQUET).set_index("sample_id")
    return df


def build_inputs(sel, majors, n, seed, work):
    """Write one MAGEMin input CSV per bulk; return the per-point meta table."""
    (work / "inputs").mkdir(parents=True, exist_ok=True)
    fe3 = ClassRatioStrategy()
    meta = []
    sid = 0
    for _, b in sel.iterrows():
        row = majors.loc[b.sample_id]
        # archived record: bulk_to_magemin lost fe3_strategy= on 2026-10-29
        # and always applies ClassRatioStrategy, the value passed here
        base = bulk_to_magemin(row, b.qap_name, h2o_wt=0.0, fe3_strategy=fe3)
        anh = {k: base[k] for k in OXIDE_KEYS}
        anh_sum = sum(anh.values())
        rng = np.random.default_rng(seed + int(b.sample_id))
        P, W, Q = draw_conditions(n, rng)
        rows = []
        for draw, (p, w, q) in enumerate(zip(P, W, Q)):
            scale = (100.0 - w) / anh_sum
            ox = {k: v * scale for k, v in anh.items()}
            for T in T_LADDER:
                sid += 1
                rows.append(dict(sample_id=sid, T_celsius=float(T), P_kbar=float(p),
                                 h2o_tot=float(w), **ox))
                meta.append(dict(sid=sid, bulk=int(b.sample_id), qap_name=b.qap_name,
                                 series=b.series, sio2_bulk=b.sio2, draw=draw,
                                 P_kbar=float(p), H2O=float(w), q=float(q),
                                 T_celsius=float(T), fe_provenance=base["fe_provenance"]))
        pd.DataFrame(rows)[INPUT_COLS].to_csv(work / "inputs" / f"bulk_{int(b.sample_id)}.csv", index=False)
    meta = pd.DataFrame(meta)
    meta.to_csv(work / "meta.csv", index=False)
    return meta


def run_all(sel, work):
    (work / "raw").mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(sel.itertuples(), 1):
        bid = int(b.sample_id)
        inp = work / "inputs" / f"bulk_{bid}.csv"
        out = work / "raw" / f"bulk_{bid}.jsonl"
        print(f"[{i}/{len(sel)}] bulk {bid} ({b.qap_name}) ...", flush=True)
        subprocess.run(
            ["julia", f"--project={REPO/'magemin'}", "-t", "auto", str(RUNNER),
             "--input", str(inp), "--output", str(out), "--quiet"],
            cwd=REPO, check=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 bulks x N=8")
    ap.add_argument("--n", type=int, default=128, help="draws per bulk")
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--inputs-only", action="store_true", help="build inputs, skip MAGEMin")
    ap.add_argument("--work", default="work/bounded",
                    help="work dir (relative to this exploration); flat-box baseline is work/")
    args = ap.parse_args()

    work = HERE / args.work
    sel = pd.read_csv(RESULTS / "selected_bulks.csv")
    n = args.n
    if args.smoke:
        # one mafic-intermediate + one alkaline, small N
        sel = pd.concat([sel[sel.qap_name == "andesite"].head(1),
                         sel[sel.series == "alkaline"].head(1)]).reset_index(drop=True)
        n = 8

    print(f"{len(sel)} bulks x N={n} draws x {len(T_LADDER)} T-steps "
          f"= {len(sel)*n*len(T_LADDER):,} points -> {work}")
    build_inputs(sel, load_majors(), n, args.seed, work)
    if not args.inputs_only:
        run_all(sel, work)
    print("done")


if __name__ == "__main__":
    main()

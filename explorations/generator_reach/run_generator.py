"""Run the MAGEMin generator on the benchmark rocks' own chemistry: 128
(P, H2O) draws per rock from the broad box, the whole cooling ladder kept.

  python explorations/generator_reach/run_generator.py --smoke
  python explorations/generator_reach/run_generator.py --max-chunks 1
  python explorations/generator_reach/run_generator.py
  python explorations/generator_reach/run_generator.py --benchmark plutonic

Run from the repo root with Julia and MAGEMin installed (magemin/Project.toml).
The composition table (`results/compositions.csv`) is built on the first run
from the benchmark and checked against it on every later run. The rocks are
cut into chunks of `--chunk-size`, each expanded over `--draws` Latin-hypercube
draws from `draw_conditions` (P 0.5 to 12 kbar, H2O 0 to min(8, S(P)) wt%)
and the 31-point ladder (1200 to 600 C), run through a fresh Julia process per
`--bulks-per-call` rocks with the production script's timeout and crash
bisection, parsed into two tables (every converged ladder point with its melt
fraction, modes only; one row per draw with its rung read-outs, its last
converged step and failure counts) and flushed atomically under `--out` with
a manifest line. Rerun the same command to resume: manifested chunks are
skipped and at most the one in-flight chunk is redone. `--smoke` runs the first rock at two draws
into `work/smoke` (`work/smoke_plutonic`) and prints the assemblage at each
rung and the last step.

`--benchmark eruption` (#78) runs the evaluated eruption rocks read at 50 and
30% melt into `work/broad128`. `--benchmark plutonic` (#83 Stage A) runs every
plutonic rock with chemistry, the evaluated ones and the rest, read near the
solidus at 10 and 5% melt into `work/plutonic128`; its iron split keys on each
rock's reported class rather than the TAS field.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.condition_proposal import draw_conditions
from cnmineral.data.magemin_corpus import INPUT_COLS, OXIDE_KEYS, T_LADDER
from cnmineral.data.magemin_oxides import bulk_to_magemin
from cnmineral.data.phase_vocab import NON_SOLID_PHASES
from cnmineral.eval.real_rock import BENCHMARK_DIR, load_benchmark
from explorations.generator_reach.compositions import (
    composition_table, load_compositions, plutonic_composition_table)
from explorations.real_mineralogy.harvest import PLUTONIC_DIR
from explorations.real_mineralogy.plutonic import PLUTONIC_SPLIT_PATH, load_plutonic

RESULTS = Path(__file__).resolve().parent / "results"
WORK = Path(__file__).resolve().parent / "work"

# the production orchestrator is a script, not a package; its Julia call,
# timeout and crash bisection are reused as they are
_spec = importlib.util.spec_from_file_location(
    "generate_mineralogy_corpus", REPO / "scripts" / "build" / "generate_mineralogy_corpus.py")
gmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gmc)

RUNGS = {"eruption": (0.50, 0.30), "plutonic": (0.10, 0.05)}
DEFAULT_SEED = 20260714
LADDER_COLS = ["bulk", "draw", "T_celsius", "liq_frac", "phase", "instance", "mode_wt"]


def build_inputs(comps, n_draws, seed):
    """Expand composition rows into `run_magemin.jl` input rows, per-point meta
    and one row per draw; the per-rock rng is seeded `seed + bulk`."""
    inputs, meta, draws = [], [], []
    sid = 0
    for _, b in comps.iterrows():
        bulk = int(b.bulk)
        base = bulk_to_magemin(b, b.rock_class, h2o_wt=0.0)
        anh = {k: base[k] for k in OXIDE_KEYS}
        anh_sum = sum(anh.values())
        rng = np.random.default_rng(seed + bulk)
        P, W, Q = draw_conditions(n_draws, rng)
        for draw, (p, w, q) in enumerate(zip(P, W, Q)):
            scale = (100.0 - w) / anh_sum
            ox = {k: v * scale for k, v in anh.items()}
            for T in T_LADDER:
                sid += 1
                inputs.append(dict(sample_id=sid, T_celsius=float(T),
                                   P_kbar=float(p), h2o_tot=float(w), **ox))
                meta.append(dict(sid=sid, bulk=bulk, draw=draw, T_celsius=float(T)))
            draws.append(dict(bulk=bulk, draw=draw, P_kbar=float(p), H2O=float(w),
                              q=float(q), **{f"in_{k}": ox[k] for k in OXIDE_KEYS}))
    return pd.DataFrame(inputs)[INPUT_COLS], pd.DataFrame(meta), pd.DataFrame(draws)


def parse_chunk(jsonl_path, meta, rungs):
    """Parse one chunk's JSONL into (ladder_df, stats_df): every converged
    point with its melt fraction, and per draw the rung read-outs (highest-T
    converged point below the rung's melt fraction, NaN when never reached)
    and the last converged point with its melt fraction."""
    recs = {}
    with open(jsonl_path) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["sid"]] = r

    ladder, stats = [], []
    for (bulk, draw), g in meta.groupby(["bulk", "draw"], sort=False):
        g = g.sort_values("T_celsius", ascending=False)
        pts = []
        n_failed = 0
        for sid, T in zip(g.sid, g.T_celsius):
            r = recs.get(sid)
            if r is None or r["status"] != 0 or not np.isfinite(r["G"]) or r["G"] >= 0:
                n_failed += 1
                continue
            liq = sum(p["mode_wt"] for p in r["phases"] if p["name"] == "liq")
            pts.append((T, liq, r))

        st = dict(bulk=bulk, draw=draw, n_failed=n_failed, n_points=len(pts),
                  oversaturated=bool(pts) and any(p["name"] == "fl" for p in pts[0][2]["phases"]))
        for T, liq, r in pts:
            for inst, p in enumerate(r["phases"]):
                ladder.append(dict(bulk=bulk, draw=draw, T_celsius=T, liq_frac=liq,
                                   phase=p["name"], instance=inst, mode_wt=float(p["mode_wt"])))
        st["T_last"] = pts[-1][0] if pts else np.nan
        st["liq_last"] = pts[-1][1] if pts else np.nan
        for m in rungs:
            hit = next((T for T, liq, _ in pts if liq < m), np.nan)
            st[f"T_end_{int(m * 100)}"] = hit
            st[f"attrition_{int(m * 100)}"] = bool(np.isnan(hit))
        stats.append(st)

    return pd.DataFrame(ladder, columns=LADDER_COLS), pd.DataFrame(stats)


def plutonic_compositions(results=RESULTS):
    """`compositions` for the plutonic benchmark: every rock with chemistry."""
    benchmark = load_plutonic()
    path = results / "plutonic_compositions.csv"
    if path.exists():
        comps = load_compositions(path)
        if list(comps.key) != list(benchmark.bulk_ids):
            raise SystemExit(f"{path} does not list the plutonic benchmark's rocks in "
                             "order; delete it to rebuild")
        return comps
    comps = plutonic_composition_table(
        benchmark, pd.read_csv(PLUTONIC_DIR / "samples.csv", dtype={"sample_id": str}),
        pd.read_csv(PLUTONIC_SPLIT_PATH, dtype={"sample_id": str}))
    results.mkdir(parents=True, exist_ok=True)
    comps.to_csv(path, index=False)
    print(f"wrote {path}: {len(comps)} rocks, {int(comps.evaluated.sum())} evaluated")
    print("iron classes: " + ", ".join(f"{k} {v}" for k, v in comps.rock_class.value_counts().items()))
    return comps


def compositions(results=RESULTS, benchmark_dir=BENCHMARK_DIR):
    """The composition table on disk, built from the benchmark on first use
    and checked against it afterwards (`bulk` is positional)."""
    benchmark = load_benchmark(benchmark_dir)
    keys = benchmark.bulk_ids[benchmark.clean & benchmark.has_magnitude]
    path = results / "compositions.csv"
    if path.exists():
        comps = load_compositions(path)
        if list(comps.key) != list(keys):
            raise SystemExit(f"{path} does not list the benchmark's evaluated rocks in "
                             "order; delete it to rebuild")
        return comps
    comps = composition_table(benchmark, pd.read_csv(benchmark_dir / "samples.csv"))
    results.mkdir(parents=True, exist_ok=True)
    comps.to_csv(path, index=False)
    print(f"wrote {path}: {len(comps)} rocks")
    print("tas fields: " + ", ".join(f"{k} {v}" for k, v in comps.tas_field.value_counts().items()))
    print(f"outside every field, nearest taken: {int((~comps.tas_inside).sum())}")
    print("fe3+/sum fe: " + ", ".join(f"{k:.2f} x{v}" for k, v in
                                      comps.fe3_ratio.value_counts().sort_index().items()))
    return comps


def run_chunk(idx, comps, args, out):
    t0 = time.time()
    work = out / "work" / f"chunk_{idx:04d}"
    work.mkdir(parents=True, exist_ok=True)

    parts = {t: [] for t in ("draws", "ladder", "stats")}
    groups = [comps.iloc[i:i + args.bulks_per_call]
              for i in range(0, len(comps), args.bulks_per_call)]
    n_points = 0
    crash_recs = []
    for gi, grp in enumerate(groups):
        inp, meta, draws = build_inputs(grp, args.draws, args.seed)
        csv, jsonl = work / f"in_{gi}.csv", work / f"out_{gi}.jsonl"
        crashed = []
        gmc.run_group(inp, csv, jsonl, args.database, crashed)
        if crashed:
            m = meta.set_index("sid").loc[[s for d in crashed for s in d]]
            crash_recs += [dict(bulk=int(b), draw=int(d), n_points=len(g))
                           for (b, d), g in m.groupby(["bulk", "draw"])]
        ladder, stats = parse_chunk(jsonl, meta, RUNGS[args.benchmark])
        parts["draws"].append(draws)
        parts["ladder"].append(ladder)
        parts["stats"].append(stats)
        n_points += len(inp)

    draws = pd.concat(parts["draws"], ignore_index=True).merge(
        pd.concat(parts["stats"], ignore_index=True), on=["bulk", "draw"])
    tables = dict(draws=draws, ladder=pd.concat(parts["ladder"], ignore_index=True))
    for name, df in tables.items():
        tmp = out / f"chunk_{idx:04d}_{name}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.rename(out / f"chunk_{idx:04d}_{name}.parquet")

    if not args.keep_raw:
        shutil.rmtree(work)

    rec = dict(chunk=idx, n_bulks=len(comps),
               bulk_range=[int(comps.bulk.iloc[0]), int(comps.bulk.iloc[-1])],
               n_draws_per_bulk=args.draws,
               n_points_run=n_points, n_ladder_rows=len(tables["ladder"]),
               mean_failed_per_draw=round(float(draws.n_failed.mean()), 3),
               solver_crashes=crash_recs,
               wall_s=round(time.time() - t0, 1))
    with open(out / "manifest.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def print_smoke(out, rungs):
    draws = pd.read_parquet(out / "chunk_0000_draws.parquet")
    ladder = pd.read_parquet(out / "chunk_0000_ladder.parquet")
    for d in draws.itertuples():
        print(f"draw {d.draw}: P {d.P_kbar:.2f} kbar, H2O {d.H2O:.2f} wt%, "
              f"{d.n_points} points converged, {d.n_failed} failed")
        reads = [(f"rung {int(m * 100)}", getattr(d, f"T_end_{int(m * 100)}")) for m in rungs]
        for name, T in reads + [("last step", d.T_last)]:
            if np.isnan(T):
                print(f"  {name}: never reached")
                continue
            pt = ladder[(ladder.draw == d.draw) & (ladder.T_celsius == T)]
            solids = pt[~pt.phase.isin(NON_SOLID_PHASES)].groupby("phase").mode_wt.sum()
            assemblage = " ".join(f"{p} {m:.3f}" for p, m in
                                  solids.sort_values(ascending=False).items())
            print(f"  {name} at {T:.0f} C, melt {pt.liq_frac.iloc[0]:.2f}: {assemblage}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=tuple(RUNGS), default="eruption")
    ap.add_argument("--out", type=Path,
                    help="default work/broad128 or work/plutonic128 (work/smoke or "
                         "work/smoke_plutonic with --smoke)")
    ap.add_argument("--draws", type=int, default=128, help="(P, H2O) draws per rock")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--chunk-size", type=int, default=100, help="rocks per chunk")
    ap.add_argument("--bulks-per-call", type=int, default=25,
                    help="rocks per Julia invocation (bounds Julia-side memory)")
    ap.add_argument("--database", default="ig")
    ap.add_argument("--max-chunks", type=int, help="stop after this many new chunks")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep per-chunk input CSVs and raw JSONL")
    ap.add_argument("--smoke", action="store_true", help="first rock, two draws")
    args = ap.parse_args()

    plutonic = args.benchmark == "plutonic"
    comps = plutonic_compositions() if plutonic else compositions()
    if args.smoke:
        comps, args.draws, args.chunk_size = comps.iloc[:1], 2, 1
    default = {(False, False): "broad128", (False, True): "smoke",
               (True, False): "plutonic128", (True, True): "smoke_plutonic"}
    out = args.out or WORK / default[plutonic, args.smoke]
    out.mkdir(parents=True, exist_ok=True)

    chunks = [comps.iloc[i:i + args.chunk_size] for i in range(0, len(comps), args.chunk_size)]
    done = gmc.read_manifest(out)

    # a changed rock table or draw count would silently misalign chunk indices
    for idx, rec in done.items():
        expect = (None if idx >= len(chunks) else
                  [int(chunks[idx].bulk.iloc[0]), int(chunks[idx].bulk.iloc[-1])])
        if rec["bulk_range"] != expect or rec["n_draws_per_bulk"] != args.draws:
            raise SystemExit(
                f"chunk {idx} in the manifest was generated from a different rock "
                f"table or --draws; use a fresh --out for a changed run")

    # discard any unmanifested chunk files (the at-most-one in-flight chunk)
    for f in list(out.glob("chunk_*.parquet*")):
        if int(f.name.split("_")[1]) not in done:
            f.unlink()

    pending = [i for i in range(len(chunks)) if i not in done]
    print(f"{len(comps)} rocks x {args.draws} draws x {len(T_LADDER)} T-steps; "
          f"{len(chunks)} chunks, {len(done)} done, {len(pending)} pending")

    for k, idx in enumerate(pending):
        if args.max_chunks is not None and k >= args.max_chunks:
            print(f"stopping after {args.max_chunks} chunks (--max-chunks)")
            break
        print(f"[chunk {idx + 1}/{len(chunks)}] {len(chunks[idx])} rocks ...", flush=True)
        rec = run_chunk(idx, chunks[idx], args, out)
        print(f"  {rec['n_points_run']:,} points in {rec['wall_s']}s "
              f"(failed/draw {rec['mean_failed_per_draw']})", flush=True)

    if len(gmc.read_manifest(out)) == len(chunks):
        print(f"all {len(chunks)} chunks complete -> {out}")
    if args.smoke:
        print_smoke(out, RUNGS[args.benchmark])


if __name__ == "__main__":
    main()

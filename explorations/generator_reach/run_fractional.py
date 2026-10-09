"""Run fractional crystallization on the benchmark rocks' own chemistry: the
equilibrium run's 128 (P, H2O) draws per rock and its cooling ladder, every
phase but the melt removed after each step, the whole path kept.

  python explorations/generator_reach/run_fractional.py --smoke
  python explorations/generator_reach/run_fractional.py --smoke --benchmark plutonic
  python explorations/generator_reach/run_fractional.py --benchmark plutonic
  python explorations/generator_reach/run_fractional.py

Run from the repo root with Julia and MAGEMin installed (magemin/Project.toml).
Paired draw for draw with `run_generator.py`: the same composition table, the
same `draw_conditions` draws from `seed + bulk`, the same 31-point ladder, run
through `run_fractional.jl` in a fresh Julia process per `--bulks-per-call`
rocks with the production script's timeout and crash bisection, here at path
boundaries. A path starts from the rock's 11-oxide bulk at the top of the
ladder; after each converged step the melt's own composition becomes the next
step's bulk and the system mass is scaled by the melt's weight fraction, so
everything else, exsolved fluid included, is removed; the path ends when no
melt is stable. Rerun the same command to resume: manifested chunks are
skipped and at most the one in-flight chunk is redone.

Per chunk, three tables in `run_generator.LADDER_COLS` with `mode_wt` on the
starting mass and `liq_frac` the melt's fraction of the starting mass:
`chunk_XXXX_step.parquet`, each step's phases as they form (the cargo);
`chunk_XXXX_pile.parquet`, the solids and fluid accumulated down the path with
the residual melt as its `liq` rows (the plutonic label); and
`chunk_XXXX_melt.parquet`, one row per converged step with the melt's
Fe3+/sum Fe and H2O, the step's status and G. `chunk_XXXX_draws.parquet` has
one row per draw with its rung read-outs at 50, 30, 10 and 5% melt, where the
melt ran out, and failure counts. `run_read.build_points(work, table)` reads
the step or pile table. Outputs land under `work/plutonic128_fc` and
`work/broad128_fc` (`work/smoke_fc`, `work/smoke_plutonic_fc` with `--smoke`,
which runs the first rock at the equilibrium smoke's two draws and prints
every step).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_WARNINGS", "off")

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.data.magemin_corpus import T_LADDER
from cnmineral.data.phase_vocab import NON_SOLID_PHASES
from explorations.generator_reach import run_generator as rg
from explorations.generator_reach.run_generator import DEFAULT_SEED, LADDER_COLS, WORK

RUNNER = Path(__file__).resolve().with_name("run_fractional.jl")
RUNGS = (0.50, 0.30, 0.10, 0.05)
TABLES = ("draws", "step", "pile", "melt")
MELT_COLS = ["bulk", "draw", "T_celsius", "liq_frac", "mass", "status", "G", "fe3_melt", "h2o_melt", "end"]
MW_FEO, MW_O = 71.844, 15.999
CRASH_CAP = 48


def build_inputs(comps, n_draws, seed):
    """`run_generator.build_inputs` with a `path_id` per (rock, draw) so the
    runner can group a cooling path."""
    inp, meta, draws = rg.build_inputs(comps, n_draws, seed)
    inp = inp.copy()
    inp["path_id"] = (meta["bulk"] * n_draws + meta["draw"]).to_numpy()
    return inp, meta, draws


def run_julia(input_csv, output_jsonl, database, melt_floor, timeout):
    cmd = ["julia", f"--project={REPO / 'magemin'}", "-t", "auto", str(RUNNER),
           "--input", str(input_csv), "--output", str(output_jsonl),
           "--database", database, "--melt-floor", str(melt_floor), "--quiet"]
    try:
        return subprocess.run(cmd, cwd=REPO, stderr=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(
            cmd, returncode=-1,
            stderr=f"killed: no output after {timeout:.0f}s (solver hang)\n{stderr}")


def _complete_output(jsonl, path_ids):
    """A crash after the last record was written still leaves every path
    ended; a path's rows below its end are never written."""
    if not jsonl.exists():
        return False
    ended = set()
    try:
        with open(jsonl) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("end"):
                    ended.add(r["path_id"])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return ended >= set(path_ids)


def run_group(inp, csv, jsonl, database, melt_floor, crashed):
    """Run one call group, bisecting at path boundaries around solver crashes."""
    inp.to_csv(csv, index=False)
    # the equilibrium runner's budget (30 ms per point, 300 s floor for Julia
    # startup); a fractional step costs about the same and ends early
    proc = run_julia(csv, jsonl, database, melt_floor, timeout=300 + 0.03 * len(inp))
    if proc.returncode == 0 or _complete_output(jsonl, inp["path_id"].unique()):
        return
    csv.with_suffix(".stderr").write_text(proc.stderr or "")
    paths = inp["path_id"].unique()
    if len(paths) <= 1:
        crashed.append([int(s) for s in inp.sample_id])
        r = inp.iloc[0]
        print(f"  solver crash isolated, path dropped ({len(inp)} points): "
              f"P={r.P_kbar:.2f} kbar H2O={r.h2o_tot:.2f} wt%", flush=True)
        if len(crashed) > CRASH_CAP:
            raise RuntimeError(
                f"{len(crashed)} dropped paths in one call group -- systemic, "
                "not pathological inputs; investigate before resuming")
        jsonl.write_text("")
        return
    print(f"  solver died on {len(paths)} paths (exit {proc.returncode}); "
          f"bisecting (stderr -> {csv.with_suffix('.stderr').name})", flush=True)
    first = inp["path_id"].isin(paths[:len(paths) // 2])
    for side, part in enumerate((inp[first], inp[~first])):
        run_group(part, csv.with_name(f"{csv.stem}.{side}.csv"),
                  jsonl.with_name(f"{jsonl.stem}.{side}.jsonl"), database, melt_floor, crashed)
    with open(jsonl, "w") as out:
        for side in (0, 1):
            out.write(jsonl.with_name(f"{jsonl.stem}.{side}.jsonl").read_text())


def _converged(r):
    return r["status"] == 0 and r["G"] is not None and np.isfinite(r["G"]) and r["G"] < 0


def _melt_chemistry(r):
    """Fe3+/sum Fe and H2O wt% of the melt from its oxide fractions, which
    carry all iron as FeO plus the excess oxygen `O`."""
    liq = next((p for p in r["phases"] if p["name"] == "liq"), None)
    if liq is None:
        return np.nan, np.nan
    c = liq["comp_wt"]
    fe = c.get("FeO", 0.0) / MW_FEO
    fe3 = 2.0 * c.get("O", 0.0) / MW_O
    return (fe3 / fe if fe > 0 else np.nan), 100.0 * c.get("H2O", 0.0)


def parse_chunk(jsonl_path, meta, rungs):
    """Parse one chunk's JSONL into (step_df, pile_df, melt_df, stats_df).
    Every mass is on the starting mass: a phase's step mass is the record's
    `mass` times its weight fraction, the pile the running sum of the step
    masses of everything but the melt, and the melt fraction the record's
    `mass` times the melt's weight fraction. A failed step counts and leaves
    no rows; the path's rows below its end were never run."""
    recs = {}
    with open(jsonl_path) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["sid"]] = r

    step, pile, melt, stats = [], [], [], []
    for (bulk, draw), g in meta.groupby(["bulk", "draw"], sort=False):
        g = g.sort_values("T_celsius", ascending=False)
        pts, acc = [], {}
        n_failed = n_run = 0
        end, T_end = None, np.nan
        for sid, T in zip(g.sid, g.T_celsius):
            r = recs.get(sid)
            if r is None:
                break
            n_run += 1
            if r.get("end"):
                end, T_end = r["end"], T
            if not _converged(r):
                n_failed += 1
                continue
            mass = r["mass"]
            liq = mass * sum(p["mode_wt"] for p in r["phases"] if p["name"] == "liq")
            key = dict(bulk=bulk, draw=draw, T_celsius=T, liq_frac=liq)
            for inst, p in enumerate(r["phases"]):
                m = mass * float(p["mode_wt"])
                step.append(dict(**key, phase=p["name"], instance=inst, mode_wt=m))
                if p["name"] != "liq":
                    acc[p["name"]] = acc.get(p["name"], 0.0) + m
            for inst, (name, m) in enumerate(acc.items()):
                pile.append(dict(**key, phase=name, instance=inst, mode_wt=m))
            pile.append(dict(**key, phase="liq", instance=len(acc), mode_wt=liq))
            fe3, h2o = _melt_chemistry(r)
            melt.append(dict(**key, mass=mass, status=r["status"], G=r["G"],
                             fe3_melt=fe3, h2o_melt=h2o, end=r.get("end")))
            pts.append((T, liq))

        st = dict(bulk=bulk, draw=draw, n_failed=n_failed, n_steps_run=n_run, n_points=len(pts),
                  oversaturated=bool(pts) and any(
                      p["name"] == "fl" for p in recs[g.sid.iloc[0]]["phases"]),
                  end=end, T_liquid_out=T_end if end in ("melt_out", "melt_floor") else np.nan)
        st["T_last"] = pts[-1][0] if pts else np.nan
        st["liq_last"] = pts[-1][1] if pts else np.nan
        for m in rungs:
            hit = next((T for T, liq in pts if liq < m), np.nan)
            st[f"T_end_{int(m * 100)}"] = hit
            st[f"attrition_{int(m * 100)}"] = bool(np.isnan(hit))
        stats.append(st)

    return (pd.DataFrame(step, columns=LADDER_COLS), pd.DataFrame(pile, columns=LADDER_COLS),
            pd.DataFrame(melt, columns=MELT_COLS), pd.DataFrame(stats))


def run_chunk(idx, comps, args, out):
    t0 = time.time()
    work = out / "work" / f"chunk_{idx:04d}"
    work.mkdir(parents=True, exist_ok=True)

    parts = {t: [] for t in (*TABLES, "stats")}
    groups = [comps.iloc[i:i + args.bulks_per_call]
              for i in range(0, len(comps), args.bulks_per_call)]
    n_points = 0
    crash_recs = []
    for gi, grp in enumerate(groups):
        inp, meta, draws = build_inputs(grp, args.draws, args.seed)
        csv, jsonl = work / f"in_{gi}.csv", work / f"out_{gi}.jsonl"
        crashed = []
        run_group(inp, csv, jsonl, args.database, args.melt_floor, crashed)
        if crashed:
            m = meta.set_index("sid").loc[[s for d in crashed for s in d]]
            crash_recs += [dict(bulk=int(b), draw=int(d), n_points=len(g))
                           for (b, d), g in m.groupby(["bulk", "draw"])]
        step, pile, melt, stats = parse_chunk(jsonl, meta, RUNGS)
        parts["draws"].append(draws)
        parts["step"].append(step)
        parts["pile"].append(pile)
        parts["melt"].append(melt)
        parts["stats"].append(stats)
        n_points += len(inp)

    tables = {t: pd.concat(parts[t], ignore_index=True) for t in TABLES}
    tables["draws"] = tables["draws"].merge(pd.concat(parts["stats"], ignore_index=True),
                                            on=["bulk", "draw"])
    for name, df in tables.items():
        tmp = out / f"chunk_{idx:04d}_{name}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.rename(out / f"chunk_{idx:04d}_{name}.parquet")

    if not args.keep_raw:
        shutil.rmtree(work)

    draws = tables["draws"]
    rec = dict(chunk=idx, n_bulks=len(comps),
               bulk_range=[int(comps.bulk.iloc[0]), int(comps.bulk.iloc[-1])],
               n_draws_per_bulk=args.draws, recipe="fractional", melt_floor=args.melt_floor,
               n_points_run=int(draws.n_steps_run.sum()), n_points_laddered=n_points,
               n_ladder_rows=len(tables["step"]), n_pile_rows=len(tables["pile"]),
               mean_failed_per_draw=round(float(draws.n_failed.mean()), 3),
               draws_by_end={k: int(v) for k, v in draws["end"].value_counts(dropna=False).items()},
               solver_crashes=crash_recs,
               wall_s=round(time.time() - t0, 1))
    with open(out / "manifest.jsonl", "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def _assemblage(rows):
    solids = rows[~rows.phase.isin(NON_SOLID_PHASES)].groupby("phase").mode_wt.sum()
    return " ".join(f"{p} {m:.3f}" for p, m in solids.sort_values(ascending=False).items()
                    if m > 5e-4)


def print_smoke(out, rungs):
    draws = pd.read_parquet(out / "chunk_0000_draws.parquet")
    step = pd.read_parquet(out / "chunk_0000_step.parquet")
    pile = pd.read_parquet(out / "chunk_0000_pile.parquet")
    melt = pd.read_parquet(out / "chunk_0000_melt.parquet")
    for d in draws.itertuples():
        print(f"draw {d.draw}: P {d.P_kbar:.2f} kbar, H2O {d.H2O:.2f} wt%, {d.n_steps_run} steps run, "
              f"{d.n_points} converged, {d.n_failed} failed, end {d.end} at {d.T_last:.0f} C")
        print("  T      melt   fluid  Fe3/Fe  H2O_melt  G_sys    step crystals | pile")
        for m in melt[melt.draw == d.draw].sort_values("T_celsius", ascending=False).itertuples():
            s = step[(step.draw == d.draw) & (step.T_celsius == m.T_celsius)]
            p = pile[(pile.draw == d.draw) & (pile.T_celsius == m.T_celsius)]
            fl = s[s.phase == "fl"].mode_wt.sum()
            print(f"  {m.T_celsius:5.0f}  {m.liq_frac:6.3f} {fl:6.3f}  {m.fe3_melt:5.2f}   "
                  f"{m.h2o_melt:5.2f}  {m.G:8.1f}  {_assemblage(s)} | {_assemblage(p)}")
        reads = [(f"rung {int(m * 100)}", getattr(d, f"T_end_{int(m * 100)}")) for m in rungs]
        for name, T in reads + [("last step", d.T_last)]:
            if np.isnan(T):
                print(f"  {name}: never reached")
                continue
            p = pile[(pile.draw == d.draw) & (pile.T_celsius == T)]
            print(f"  {name} at {T:.0f} C, melt {p.liq_frac.iloc[0]:.3f}: pile {_assemblage(p)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=("eruption", "plutonic"), default="eruption")
    ap.add_argument("--out", type=Path,
                    help="default work/broad128_fc or work/plutonic128_fc (work/smoke_fc or "
                         "work/smoke_plutonic_fc with --smoke)")
    ap.add_argument("--draws", type=int, default=128, help="(P, H2O) draws per rock")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--chunk-size", type=int, default=100, help="rocks per chunk")
    ap.add_argument("--bulks-per-call", type=int, default=25,
                    help="rocks per Julia invocation (bounds Julia-side memory)")
    ap.add_argument("--database", default="ig")
    ap.add_argument("--melt-floor", type=float, default=1e-4,
                    help="a path ends when the melt falls below this fraction of the starting mass")
    ap.add_argument("--max-chunks", type=int, help="stop after this many new chunks")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep per-chunk input CSVs and raw JSONL")
    ap.add_argument("--smoke", action="store_true", help="first rock, two draws")
    args = ap.parse_args()

    plutonic = args.benchmark == "plutonic"
    comps = rg.plutonic_compositions() if plutonic else rg.compositions()
    if args.smoke:
        comps, args.draws, args.chunk_size = comps.iloc[:1], 2, 1
    default = {(False, False): "broad128_fc", (False, True): "smoke_fc",
               (True, False): "plutonic128_fc", (True, True): "smoke_plutonic_fc"}
    out = args.out or WORK / default[plutonic, args.smoke]
    out.mkdir(parents=True, exist_ok=True)

    chunks = [comps.iloc[i:i + args.chunk_size] for i in range(0, len(comps), args.chunk_size)]
    done = rg.gmc.read_manifest(out)

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
    print(f"{len(comps)} rocks x {args.draws} draws x {len(T_LADDER)} T-steps, fractional; "
          f"{len(chunks)} chunks, {len(done)} done, {len(pending)} pending")

    for k, idx in enumerate(pending):
        if args.max_chunks is not None and k >= args.max_chunks:
            print(f"stopping after {args.max_chunks} chunks (--max-chunks)")
            break
        print(f"[chunk {idx + 1}/{len(chunks)}] {len(chunks[idx])} rocks ...", flush=True)
        rec = run_chunk(idx, chunks[idx], args, out)
        print(f"  {rec['n_points_run']:,} of {rec['n_points_laddered']:,} points run in {rec['wall_s']}s "
              f"(failed/draw {rec['mean_failed_per_draw']}, ends {rec['draws_by_end']})", flush=True)

    if len(rg.gmc.read_manifest(out)) == len(chunks):
        print(f"all {len(chunks)} chunks complete -> {out}")
    if args.smoke:
        print_smoke(out, RUNGS)


if __name__ == "__main__":
    main()

"""Chunked, resumable MAGEMin corpus generation (corpus_regen_spec "Orchestration").

Wraps `magemin/run_magemin.jl` unchanged. The bulk table is cut into chunks;
each chunk is expanded (bounded-proposal draws x cooling ladder), run through
a fresh Julia process per call group (the juliacall/MAGEMin leak mitigation),
parsed into the three corpus tables, flushed atomically (tmp + rename), and
recorded in `manifest.jsonl`. Resume = rerun the same command: manifested
chunks are skipped, unmanifested files are discarded, so at most the one
in-flight chunk is redone.

The full fixed ladder (1200 -> 600 C) runs in one batch and the stored ladder
is truncated at the 30%-melt stop while parsing: on the measured crossing
distribution the ~29% of points below the stop cost about the same as the ~31
extra Julia startups per chunk that adaptive per-step cooling would need, and
one batch keeps the chunk a single clean unit of work.

Tiers run as separate invocations (own --out, --draws):

    conda run -n alignn_fix python scripts/build/generate_mineralogy_corpus.py \\
        --bulks data/corpus/train_bulks.csv --out data/corpus/train --draws 48

Pause anytime (ctrl-C); rerun the same command to continue.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

from cnmineral.data.magemin_corpus import T_LADDER, build_inputs, parse_chunk

RUNNER = REPO / "magemin" / "run_magemin.jl"
MAJORS = REPO / "data" / "bulks" / "gard2019.parquet"
TABLES = ("draws", "ladder", "rungs")


def read_manifest(out):
    path = out / "manifest.jsonl"
    if not path.exists():
        return {}
    return {rec["chunk"]: rec for rec in map(json.loads, open(path))}


def run_julia(input_csv, output_jsonl, database):
    subprocess.run(
        ["julia", f"--project={REPO / 'magemin'}", "-t", "auto", str(RUNNER),
         "--input", str(input_csv), "--output", str(output_jsonl),
         "--database", database, "--quiet"],
        cwd=REPO, check=True)


def run_chunk(idx, bulks, majors, args, out):
    t0 = time.time()
    work = out / "work" / f"chunk_{idx:04d}"
    work.mkdir(parents=True, exist_ok=True)

    parts = {t: [] for t in ("draws", "ladder", "rungs", "stats")}
    groups = [bulks.iloc[i:i + args.bulks_per_call]
              for i in range(0, len(bulks), args.bulks_per_call)]
    n_points = 0
    for gi, grp in enumerate(groups):
        inp, meta, draws = build_inputs(grp, majors, args.draws, args.seed)
        csv, jsonl = work / f"in_{gi}.csv", work / f"out_{gi}.jsonl"
        inp.to_csv(csv, index=False)
        run_julia(csv, jsonl, args.database)
        ladder, rungs, stats = parse_chunk(jsonl, meta)
        parts["draws"].append(draws)
        parts["ladder"].append(ladder)
        parts["rungs"].append(rungs)
        parts["stats"].append(stats)
        n_points += len(inp)

    draws = pd.concat(parts["draws"], ignore_index=True).merge(
        pd.concat(parts["stats"], ignore_index=True), on=["bulk", "draw"])
    tables = dict(draws=draws,
                  ladder=pd.concat(parts["ladder"], ignore_index=True),
                  rungs=pd.concat(parts["rungs"], ignore_index=True))
    for name, df in tables.items():
        tmp = out / f"chunk_{idx:04d}_{name}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.rename(out / f"chunk_{idx:04d}_{name}.parquet")

    if not args.keep_raw:
        shutil.rmtree(work)

    rec = dict(chunk=idx, n_bulks=len(bulks),
               bulk_range=[int(bulks.sample_id.iloc[0]), int(bulks.sample_id.iloc[-1])],
               n_draws_per_bulk=args.draws,
               n_points_run=n_points, n_ladder_rows=len(tables["ladder"]),
               n_rung_rows=len(tables["rungs"]),
               mean_failed_per_draw=round(float(draws.n_failed.mean()), 3),
               wall_s=round(time.time() - t0, 1))
    with open(out / "manifest.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulks", required=True,
                    help="bulk table CSV: sample_id + passthrough columns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, required=True, help="(P, H2O) draws per bulk")
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--chunk-size", type=int, default=100, help="bulks per chunk")
    ap.add_argument("--bulks-per-call", type=int, default=25,
                    help="bulks per Julia invocation (bounds Julia-side memory)")
    ap.add_argument("--database", default="ig")
    ap.add_argument("--max-chunks", type=int, help="stop after this many new chunks")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep per-chunk input CSVs and raw JSONL")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bulks = pd.read_csv(args.bulks)
    majors = pd.read_parquet(MAJORS).set_index("sample_id")

    chunks = [bulks.iloc[i:i + args.chunk_size]
              for i in range(0, len(bulks), args.chunk_size)]
    done = read_manifest(out)

    # a changed bulk table or draw count would silently misalign chunk indices
    for idx, rec in done.items():
        expect = (None if idx >= len(chunks) else
                  [int(chunks[idx].sample_id.iloc[0]), int(chunks[idx].sample_id.iloc[-1])])
        if rec["bulk_range"] != expect or rec["n_draws_per_bulk"] != args.draws:
            raise SystemExit(
                f"chunk {idx} in the manifest was generated from a different bulk "
                f"table or --draws; use a fresh --out for a changed run")

    # discard any unmanifested chunk files (the at-most-one in-flight chunk)
    for f in list(out.glob("chunk_*.parquet*")):
        if int(f.name.split("_")[1]) not in done:
            f.unlink()

    pending = [i for i in range(len(chunks)) if i not in done]
    print(f"{len(bulks)} bulks x {args.draws} draws x {len(T_LADDER)} T-steps; "
          f"{len(chunks)} chunks, {len(done)} done, {len(pending)} pending")

    for k, idx in enumerate(pending):
        if args.max_chunks is not None and k >= args.max_chunks:
            print(f"stopping after {args.max_chunks} chunks (--max-chunks)")
            break
        print(f"[chunk {idx + 1}/{len(chunks)}] {len(chunks[idx])} bulks ...", flush=True)
        rec = run_chunk(idx, chunks[idx], majors, args, out)
        print(f"  {rec['n_points_run']:,} points in {rec['wall_s']}s "
              f"(failed/draw {rec['mean_failed_per_draw']})", flush=True)

    if len(read_manifest(out)) == len(chunks):
        print(f"all {len(chunks)} chunks complete -> {out}")


if __name__ == "__main__":
    main()

"""Production FC corpus generation (ADR 005, plan item 3).

Reads data/bulks_v1_with_T.parquet (from 01_find_T_FC_end_per_bulk.py). For
each kept bulk: generates n_P x n_H2O paths (default 4 x 2 = 8), runs ptt
FC per path from T_start (class-derived) down to that bulk's T_FC_end_bulk,
extracts the in-situ modal vector at the final cooling step (cumsum solids
+ residual liq + fl), normalizes to 1.0.

Output: data/train_fc_v1.parquet — one row per (bulk, path). Schema is the
union of: input oxide columns (h2o_tot replaced by path's H2O), P_kbar,
T_used, path_idx, T_FC_end_attrition (from pre-pass), provenance flags,
plus modal phase columns summing to 1.0.

Multiprocessing: --workers N spawns a pool with per-worker ptt init.

Run:
    python scripts/production/02_generate_corpus.py --workers 8 --limit 50   # smoke
    python scripts/production/02_generate_corpus.py --workers 8              # full
    python scripts/production/02_generate_corpus.py --clean                  # drop cache
"""
import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "eq_vs_frac"))
sys.path.insert(0, str(REPO / "scripts" / "production"))
from cnmineral.data.tp_priors import PUTIRKA_2008  # noqa: E402
from cnmineral.data.h2o import WALLACE_PLANK_2013  # noqa: E402
from _phase_map import to_short  # noqa: E402

# importlib trick: pull CLASS_FALLBACK from the sibling pre-pass script.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_prepass", REPO / "scripts" / "production" / "01_find_T_FC_end_per_bulk.py")
_prepass = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_prepass)
CLASS_FALLBACK = _prepass.CLASS_FALLBACK
bulk_to_ptt_dict = _prepass.bulk_to_ptt_dict
resolve_class = _prepass.resolve_class

_PTT = None  # per-worker module reference


def class_axes(qap_name: str) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return ((P_lo, P_hi) kbar, (H2O_lo, H2O_hi) wt%, T_start C) for a class.

    Class is first resolved via CLASS_FALLBACK if needed. Defaults applied
    when a class lacks an H2O entry (rare; fallback to broad 0.5-3.0 range).
    """
    cls = resolve_class(qap_name)
    if cls is None:
        raise ValueError(f"unsupported class: {qap_name}")
    P = PUTIRKA_2008[cls]["P_kbar"]
    T_hi = PUTIRKA_2008[cls]["T_C"][1]
    H = WALLACE_PLANK_2013.get(cls, (0.5, 3.0))
    return P, H, T_hi + 50.0


def build_paths(qap_name: str, n_P: int, n_H2O: int) -> list[tuple[float, float, float]]:
    """Return list of (P_bar, H2O_wt, T_start_C) for n_P x n_H2O cross-product.

    P samples: linspace across class P range, inclusive.
    H2O samples: linspace across class H2O range, exclusive ends to avoid edge
      values (interior anchors at q33/q66 for n_H2O=2 ish).
    """
    (P_lo, P_hi), (H_lo, H_hi), T_start_C = class_axes(qap_name)
    P_grid = np.linspace(P_lo, P_hi, n_P) if n_P > 1 else np.array([0.5 * (P_lo + P_hi)])
    if n_H2O > 1:
        # interior quantiles, avoiding the literal extremes that may be implausible
        edges = np.linspace(0, 1, n_H2O + 2)[1:-1]
        H_grid = H_lo + edges * (H_hi - H_lo)
    else:
        H_grid = np.array([0.5 * (H_lo + H_hi)])
    paths = [(float(P) * 1000.0, float(H), float(T_start_C))
             for P in P_grid for H in H_grid]
    return paths


def final_modal_from_res(res: dict) -> dict[str, float]:
    """Sum cumsum solids + residual liq + fluid at final cooling step, return
    a dict short_phase -> mass (g). Caller normalizes.
    """
    out: dict[str, float] = {}
    mg = res.get("mass_g")
    if mg is None or len(mg) == 0:
        return out
    last = mg.iloc[-1]
    for col in mg.columns:
        if col.endswith("_cumsum"):
            short = to_short(col[: -len("_cumsum")])
            if short is None:
                continue
            v = last[col]
            v = float(v) if not pd.isna(v) else 0.0
            if v > 0:
                out[short] = out.get(short, 0.0) + v
    for liq_col, short in (("liquid1", "liq"), ("fluid1", "fl")):
        if liq_col in mg.columns:
            v = last[liq_col]
            v = float(v) if not pd.isna(v) else 0.0
            if v > 0:
                out[short] = out.get(short, 0.0) + v
    return out


def _worker_init():
    global _PTT
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import petthermotools as _ptt_mod
    _ptt_mod.activate_petthermotools_env()
    _PTT = _ptt_mod


def _worker_process_chunk(payload: dict) -> dict:
    chunk_idx = payload["chunk_idx"]
    cache_dir = Path(payload["cache_dir"])
    chunk_path = cache_dir / f"chunk_{chunk_idx:05d}.parquet"
    if chunk_path.exists():
        return {"chunk_idx": chunk_idx, "status": "cached", "n": 0}
    chunk = payload["chunk_df"]
    out = process_chunk(chunk_idx, chunk, _PTT,
                        n_P=payload["n_P"], n_H2O=payload["n_H2O"],
                        dt_C=payload["dt_C"],
                        timeout_per_path=payload["timeout_per_path"])
    out.to_parquet(chunk_path, index=False)
    return {"chunk_idx": chunk_idx, "status": "ok",
            "n": len(out), "n_bulks": len(chunk)}


def process_chunk(chunk_idx: int, chunk: pd.DataFrame, ptt,
                  n_P: int, n_H2O: int, dt_C: float,
                  timeout_per_path: int) -> pd.DataFrame:
    # Julia GC handle. Each ptt FC call leaks ~15 MB into the Julia heap (cooling
    # trajectory DataFrames, JIT artifacts, intermediate arrays Julia GC doesn't
    # reclaim on its own under our call pattern). Forcing a full GC between paths
    # keeps per-worker RSS bounded; without this the 2026-05-17 run grew workers
    # at ~0.7 GB/min and hit Jetsam.
    from juliacall import Main as _jl
    rows = []
    for _, bulk in chunk.iterrows():
        T_end = bulk.get("T_FC_end_bulk")
        if pd.isna(T_end):
            continue  # dropped at pre-pass (unsupported class)
        try:
            paths = build_paths(bulk["qap_name"], n_P, n_H2O)
        except ValueError:
            continue
        for path_idx, (P_bar, H2O_wt, T_start_C) in enumerate(paths):
            bulk_for_ptt = bulk.copy()
            bulk_for_ptt["h2o_tot"] = H2O_wt
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ptt.isobaric_crystallisation(
                        Model="Green2025",
                        bulk=bulk_to_ptt_dict(bulk_for_ptt),
                        Frac_solid=True,
                        T_start_C=T_start_C,
                        T_end_C=float(T_end),
                        dt_C=dt_C,
                        P_bar=P_bar,
                        Suppress=["rutile", "tridymite"],
                        multi_processing=False,
                        timeout=timeout_per_path,
                    )
                modal_g = final_modal_from_res(res)
                if not modal_g:
                    rows.append({
                        "sample_id": bulk["sample_id"], "path_idx": path_idx,
                        "qap_name": bulk["qap_name"], "P_kbar": P_bar / 1000.0,
                        "h2o_path_wt": H2O_wt, "T_FC_end_used": int(T_end),
                        "T_FC_end_attrition": bool(bulk.get("T_FC_end_attrition", False)),
                        "fe_provenance": bulk.get("fe_provenance", ""),
                        "h2o_provenance": bulk.get("h2o_provenance", ""),
                        "fc_status": "empty_modal",
                    })
                    continue
                total = sum(modal_g.values())
                modal = {k: v / total for k, v in modal_g.items()}
                rows.append({
                    "sample_id": bulk["sample_id"], "path_idx": path_idx,
                    "qap_name": bulk["qap_name"], "P_kbar": P_bar / 1000.0,
                    "h2o_path_wt": H2O_wt, "T_FC_end_used": int(T_end),
                    "T_FC_end_attrition": bool(bulk.get("T_FC_end_attrition", False)),
                    "fe_provenance": bulk.get("fe_provenance", ""),
                    "h2o_provenance": bulk.get("h2o_provenance", ""),
                    "fc_status": "ok",
                    "sio2": float(bulk["sio2"]), "al2o3": float(bulk["al2o3"]),
                    "cao": float(bulk["cao"]), "mgo": float(bulk["mgo"]),
                    "feo": float(bulk["feo"]), "fe2o3": float(bulk["fe2o3"]),
                    "k2o": float(bulk["k2o"]), "na2o": float(bulk["na2o"]),
                    "tio2": float(bulk["tio2"]), "cr2o3": float(bulk["cr2o3"]),
                    "h2o_tot": H2O_wt,
                    **modal,
                })
            except Exception as e:
                rows.append({
                    "sample_id": bulk["sample_id"], "path_idx": path_idx,
                    "qap_name": bulk["qap_name"], "P_kbar": P_bar / 1000.0,
                    "h2o_path_wt": H2O_wt, "T_FC_end_used": int(T_end),
                    "T_FC_end_attrition": bool(bulk.get("T_FC_end_attrition", False)),
                    "fc_status": f"fc_error:{type(e).__name__}",
                })
            finally:
                # Drop the Python reference to ptt's return dict (carries refs to
                # Julia heap data via PythonCall) and trigger Julia full GC so the
                # leaked-per-call memory actually frees.
                try:
                    del res
                except NameError:
                    pass
                _jl.GC.gc(True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path,
                    default=REPO / "data" / "bulks_v1_with_T.parquet")
    ap.add_argument("--output", type=Path,
                    default=REPO / "data" / "train_fc_v1.parquet")
    ap.add_argument("--cache-dir", type=Path,
                    default=REPO / "data" / "fc_corpus_cache")
    ap.add_argument("--chunk-size", type=int, default=100,
                    help="bulks per chunk (each bulk = n_P*n_H2O paths)")
    ap.add_argument("--n-P", type=int, default=4)
    ap.add_argument("--n-H2O", type=int, default=2)
    ap.add_argument("--dt-C", type=float, default=25.0)
    ap.add_argument("--timeout-per-path", type=int, default=120)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.clean and args.cache_dir.exists():
        print(f"removing {args.cache_dir}")
        shutil.rmtree(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    bulks = pd.read_parquet(args.inputs)
    # Drop bulks that pre-pass couldn't handle (unsupported_class -> NaN T)
    bulks = bulks[bulks["T_FC_end_bulk"].notna()].reset_index(drop=True)
    if args.limit:
        bulks = bulks.head(args.limit).reset_index(drop=True)
    print(f"loaded {len(bulks)} usable bulks; n_P={args.n_P} n_H2O={args.n_H2O} "
          f"-> {args.n_P * args.n_H2O} paths/bulk; workers={args.workers}")

    n_chunks = (len(bulks) + args.chunk_size - 1) // args.chunk_size
    print(f"{n_chunks} chunks; expected total rows ~{len(bulks) * args.n_P * args.n_H2O}")

    common = dict(n_P=args.n_P, n_H2O=args.n_H2O,
                  dt_C=args.dt_C, timeout_per_path=args.timeout_per_path)

    t0 = time.time()
    n_processed_paths = 0

    if args.workers <= 1:
        import petthermotools as ptt
        ptt.activate_petthermotools_env()
        pbar = tqdm(range(n_chunks), desc="chunks")
        for i in pbar:
            chunk_path = args.cache_dir / f"chunk_{i:05d}.parquet"
            if chunk_path.exists():
                pbar.write(f"chunk {i}: cached, skipping")
                continue
            chunk = bulks.iloc[i * args.chunk_size : (i + 1) * args.chunk_size]
            out = process_chunk(i, chunk, ptt, **common)
            out.to_parquet(chunk_path, index=False)
            n_processed_paths += len(out)
            elapsed = time.time() - t0
            rate = n_processed_paths / max(elapsed, 1)
            pbar.write(f"chunk {i}: {len(out)} path-rows, rate={rate:.2f} paths/s")
    else:
        payloads = []
        for i in range(n_chunks):
            chunk_path = args.cache_dir / f"chunk_{i:05d}.parquet"
            if chunk_path.exists():
                continue
            chunk = bulks.iloc[i * args.chunk_size : (i + 1) * args.chunk_size].copy()
            payloads.append({"chunk_idx": i, "chunk_df": chunk,
                             "cache_dir": str(args.cache_dir),
                             **common})
        n_skipped = n_chunks - len(payloads)
        if n_skipped:
            print(f"skipping {n_skipped} cached chunks; dispatching {len(payloads)}")
        ctx = mp.get_context("spawn")
        # maxtasksperchild=1: each worker handles one chunk then exits. juliacall +
        # MAGEMin leaks memory across calls (RSS grew ~7 GB -> 180 GB per worker over
        # 4 hr in the 2026-05-17 run -> Jetsam cascade -> kernel panic). Recycling
        # adds ~30 s spawn overhead per chunk per worker = ~10% wall-time tax for
        # bounded memory.
        with ctx.Pool(processes=args.workers, initializer=_worker_init,
                      maxtasksperchild=1) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker_process_chunk, payloads),
                total=len(payloads), desc="chunks",
            ):
                if result["status"] != "cached":
                    n_processed_paths += result["n"]
                    elapsed = time.time() - t0
                    rate = n_processed_paths / max(elapsed, 1)
                    tqdm.write(f"chunk {result['chunk_idx']:5d}: "
                               f"{result['n']} path-rows, rate={rate:.2f} paths/s")

    # Merge
    parts = sorted(args.cache_dir.glob("chunk_*.parquet"))
    print(f"\nmerging {len(parts)} chunk parquets")
    merged = pd.concat([pd.read_parquet(p) for p in parts],
                       ignore_index=True, sort=False)
    # Modal columns may have NaN where a phase wasn't present in that path.
    # Distinguish modal columns by treating non-meta columns as modal.
    meta_cols = {"sample_id", "path_idx", "qap_name", "P_kbar", "h2o_path_wt",
                 "T_FC_end_used", "T_FC_end_attrition",
                 "fe_provenance", "h2o_provenance", "fc_status",
                 "sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
                 "k2o", "na2o", "tio2", "cr2o3", "h2o_tot"}
    modal_cols = [c for c in merged.columns if c not in meta_cols]
    merged[modal_cols] = merged[modal_cols].fillna(0.0)
    merged.to_parquet(args.output, index=False)
    print(f"wrote {args.output}: {merged.shape}")

    ok = merged[merged["fc_status"] == "ok"]
    print("\nsummary:")
    print(f"  total path-rows:    {len(merged)}")
    print(f"  ok:                 {len(ok)}")
    print(f"  fc_errors:          {merged['fc_status'].str.startswith('fc_error').sum()}")
    print(f"  empty_modal:        {(merged['fc_status']=='empty_modal').sum()}")
    print(f"  modal phase count:  {len(modal_cols)}")
    print(f"  modal phases:       {modal_cols}")
    if len(ok):
        liq_mean = float(ok["liq"].mean()) if "liq" in ok.columns else 0.0
        print(f"  mean liq mass frac: {liq_mean:.4f}  (v0 baseline was 0.70)")


if __name__ == "__main__":
    main()

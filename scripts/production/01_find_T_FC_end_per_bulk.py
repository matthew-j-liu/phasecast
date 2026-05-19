"""Per-bulk T_FC_end pre-pass (ADR 005, plan item 2).

For each bulk in data/magemin_inputs.csv: run one ptt.isobaric_crystallisation
sweep at the bulk's actual H2O and the class P midpoint, cooling from above
the class liquidus to a SiO2-tier floor. From the per-step liq trajectory,
pick T_FC_end_bulk = highest T (snapped to 25 C) at which liq < 5 wt%.

If liq never falls below 5 wt% within the calibration floor, the bulk is
flagged (T_FC_end_attrition=True) and T_FC_end_bulk is set to the floor.
The production orchestrator can keep or drop flagged bulks.

Output: data/bulks_v1_with_T.parquet — same schema as magemin_inputs.csv plus
T_FC_end_bulk, liq_at_T_FC_end_wt, T_FC_end_attrition. Per-chunk caches in
data/T_FC_end_cache/ allow resume after Ctrl-C.

Parallelization: --workers N uses mp.Pool with spawn context. Each worker
imports ptt once via the initializer and processes its chunks serially with
ptt.multi_processing=False, avoiding (python workers × julia workers)
process explosion.

Run:
    python scripts/production/01_find_T_FC_end_per_bulk.py --limit 100              # smoke serial
    python scripts/production/01_find_T_FC_end_per_bulk.py --workers 4 --limit 200  # smoke parallel
    python scripts/production/01_find_T_FC_end_per_bulk.py --workers 4              # full parallel
    python scripts/production/01_find_T_FC_end_per_bulk.py --clean                  # drop cache
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
from cnmineral.data.tp_priors import PUTIRKA_2008  # noqa: E402
from _phase_map import to_short  # noqa: E402  (unused here but pin imports)

# Worker-local ptt module handle, set by _worker_init().
_PTT = None

MW_FEO = 71.844
MW_FE2O3 = 159.69

# Classes in magemin_inputs.csv that are not direct keys of PUTIRKA_2008.
# Most are compound names (e.g. "phonolitic tephrite") with an obvious parent.
# Truly unmodellable compositions for MAGEMin ig (carbonate-rich, silexite)
# map to None and are dropped.
CLASS_FALLBACK: dict[str, str | None] = {
    "clinopyroxene norite":               "norite",
    "orthopyroxene gabbro":               "gabbro",
    "phonolitic tephrite":                "tephrite",
    "phonolitic basanite":                "basanite",
    "tephritic phonolite":                "phonolite",
    "quartz latite":                      "latite",
    "foid-bearing monzodiorite":          "monzodiorite",
    "quartz-rich rhyolite":               "rhyolite",
    "quartz-rich granitoid":              "granodiorite",
    "foidolite":                          "foidite",
    "foid-bearing monzonite":             "monzonite",
    "tephritic foidite":                  "foidite",
    "olivine websterite":                 "lherzolite",
    "alkali feldspar quartz trachyte":    "trachyte",
    "alkali feldspar rhyolite":           "rhyolite",
    "foid-bearing syenite":               "syenite",
    "foid-bearing trachyte":              "trachyte",
    "alkali feldspar quartz syenite":     "quartz syenite",
    "phonolitic foidite":                 "foidite",
    "foid-bearing alkali feldspar trachyte": "trachyte",
    "quartz trachyte":                    "trachyte",
    "foid diorite":                       "diorite",
    "olivine orthopyroxenite":            "harzburgite",
    "wherlite":                           "lherzolite",
    "foid-bearing alkali feldspar syenite": "alkali feldspar syenite",
    "plagioclase-bearing websterite":     "norite",
    "silicocarbonatite":                  None,
    "carbonatite":                        None,
    "silexite":                           None,
}


def resolve_class(qap_name: str) -> str | None:
    """Return tp_priors-resolvable class name, or None if bulk should be dropped."""
    if qap_name in PUTIRKA_2008:
        return qap_name
    return CLASS_FALLBACK.get(qap_name, None)


def sio2_tier_floor(sio2_wt: float) -> float:
    """SiO2-based T_floor for the FC sweep. Calibrated against v2's per-class
    floors and the MAGEMin ig database edge:
      ultramafic (<45):   800 C  (ig database limit; ultramafic don't go below)
      mafic (45-52):      575 C  (matches v2 basalt/gabbro floor 600 with -25C margin)
      intermediate (52-63): 525 C  (v2 diorite/andesite 625; -100C margin for trapped melt)
      felsic (63+):       500 C  (matches v2 felsic floors 500-525)
    """
    if sio2_wt < 45:
        return 800.0
    if sio2_wt < 52:
        return 575.0
    if sio2_wt < 63:
        return 525.0
    return 500.0


def class_to_sweep_TP(qap_name: str, sio2_wt: float) -> tuple[float, float, float]:
    """Return (T_start_C, T_floor_C, P_bar) for the FC sweep.

    T_start: 50 C above the class's Putirka upper-T bound (well above
    liquidus). T_floor: SiO2-tier (per-bulk), not class-derived, since
    Putirka T_lo is *crystallization* T_lo, ~200-400 C above MAGEMin's
    subsolidus calibrated range. P: class midpoint in bar.
    """
    entry = PUTIRKA_2008[qap_name]
    T_lo, T_hi = entry["T_C"]
    P_lo, P_hi = entry["P_kbar"]
    T_start = T_hi + 50.0
    T_floor = sio2_tier_floor(sio2_wt)
    P_bar = 0.5 * (P_lo + P_hi) * 1000.0
    return T_start, T_floor, P_bar


def bulk_to_ptt_dict(row: pd.Series) -> dict:
    feo, fe2o3 = float(row["feo"]), float(row["fe2o3"])
    feot = feo + fe2o3 * (2 * MW_FEO / MW_FE2O3)
    mol_fe2 = feo / MW_FEO
    mol_fe3 = 2 * fe2o3 / MW_FE2O3
    fe3fet = mol_fe3 / (mol_fe2 + mol_fe3) if (mol_fe2 + mol_fe3) > 0 else 0.0
    return {
        "SiO2": float(row["sio2"]), "Al2O3": float(row["al2o3"]),
        "CaO": float(row["cao"]), "MgO": float(row["mgo"]),
        "FeOt": feot, "K2O": float(row["k2o"]), "Na2O": float(row["na2o"]),
        "TiO2": float(row["tio2"]), "Cr2O3": float(row["cr2o3"]),
        "H2O": float(row["h2o_tot"]), "Fe3Fet": fe3fet,
    }


def pick_T_from_trajectory(mass_g: pd.DataFrame, liq_threshold_wt: float,
                           T_quantize_C: int) -> tuple[int, float, bool]:
    """From a ptt FC mass_g trajectory, return (T_FC_end, liq_at_T, attrition).

    Highest T at which liquid1 < threshold (mass conserved at 100 g so column
    is already in wt%). Snap down to nearest T_quantize_C boundary. If never
    crosses below threshold, attrition=True and T_FC_end is the lowest T in
    the sweep.
    """
    if mass_g is None or len(mass_g) == 0 or "T_C" not in mass_g.columns:
        return 0, np.nan, True
    liq = mass_g["liquid1"].fillna(0.0).values
    Ts = mass_g["T_C"].values
    below = liq < liq_threshold_wt
    if not below.any():
        T_end = int(Ts[-1])
        return (T_end // T_quantize_C) * T_quantize_C, float(liq[-1]), True
    # rows are cooling-ordered (high T -> low T); first True is the
    # highest-T crossing.
    first = int(np.argmax(below))
    T_at = float(Ts[first])
    T_snap = (int(T_at) // T_quantize_C) * T_quantize_C
    return T_snap, float(liq[first]), False


def _worker_init():
    """mp.Pool initializer: each worker imports ptt once and activates the Julia env."""
    global _PTT
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import petthermotools as _ptt_mod
    _ptt_mod.activate_petthermotools_env()
    _PTT = _ptt_mod


def _worker_process_chunk(payload: dict) -> dict:
    """mp.Pool task. payload: chunk_idx, chunk_df_pickle bytes (via pandas), params, cache_dir.
    Skips if chunk parquet already exists. Writes parquet + returns lightweight stats.
    """
    chunk_idx = payload["chunk_idx"]
    cache_dir = Path(payload["cache_dir"])
    chunk_path = cache_dir / f"chunk_{chunk_idx:05d}.parquet"
    if chunk_path.exists():
        return {"chunk_idx": chunk_idx, "status": "cached", "n": 0, "flagged": 0}
    chunk = payload["chunk_df"]
    out = process_chunk(chunk_idx, chunk, _PTT,
                        dt_C=payload["dt_C"],
                        liq_threshold_wt=payload["liq_threshold_wt"],
                        T_quantize_C=payload["T_quantize_C"],
                        timeout_per_bulk=payload["timeout_per_bulk"])
    out.to_parquet(chunk_path, index=False)
    return {"chunk_idx": chunk_idx, "status": "ok",
            "n": len(out), "flagged": int(out["T_FC_end_attrition"].sum())}


def process_chunk(chunk_idx: int, chunk: pd.DataFrame, ptt, dt_C: float,
                  liq_threshold_wt: float, T_quantize_C: int,
                  timeout_per_bulk: int) -> pd.DataFrame:
    rows = []
    for _, row in chunk.iterrows():
        cls = resolve_class(row["qap_name"])
        if cls is None:
            rows.append({**row.to_dict(),
                         "T_FC_end_bulk": np.nan,
                         "liq_at_T_FC_end_wt": np.nan,
                         "T_FC_end_attrition": True,
                         "drop_reason": "unsupported_class"})
            continue
        T_start, T_floor, P_bar = class_to_sweep_TP(cls, float(row["sio2"]))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ptt.isobaric_crystallisation(
                    Model="Green2025",
                    bulk=bulk_to_ptt_dict(row),
                    Frac_solid=True,
                    T_start_C=T_start, T_end_C=T_floor, dt_C=dt_C,
                    P_bar=P_bar,
                    Suppress=["rutile", "tridymite"],
                    multi_processing=False,
                    timeout=timeout_per_bulk,
                )
            mg = res.get("mass_g")
            cond = res.get("Conditions")
            if mg is not None and cond is not None and "T_C" in cond.columns:
                mg = mg.reset_index(drop=True).copy()
                mg["T_C"] = cond["T_C"].values
            T_FC_end, liq_at, attrition = pick_T_from_trajectory(
                mg, liq_threshold_wt, T_quantize_C)
            rows.append({**row.to_dict(),
                         "T_FC_end_bulk": int(T_FC_end),
                         "liq_at_T_FC_end_wt": liq_at,
                         "T_FC_end_attrition": bool(attrition),
                         "drop_reason": ""})
        except Exception as e:
            rows.append({**row.to_dict(),
                         "T_FC_end_bulk": np.nan,
                         "liq_at_T_FC_end_wt": np.nan,
                         "T_FC_end_attrition": True,
                         "drop_reason": f"fc_error: {type(e).__name__}"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path,
                    default=REPO / "data" / "magemin_inputs.csv")
    ap.add_argument("--output", type=Path,
                    default=REPO / "data" / "bulks_v1_with_T.parquet")
    ap.add_argument("--cache-dir", type=Path,
                    default=REPO / "data" / "T_FC_end_cache")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--liq-threshold-wt", type=float, default=5.0)
    ap.add_argument("--T-quantize-C", type=int, default=25)
    ap.add_argument("--dt-C", type=float, default=25.0)
    ap.add_argument("--timeout-per-bulk", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None, help="cap bulks for smoke runs")
    ap.add_argument("--clean", action="store_true", help="wipe cache and start over")
    ap.add_argument("--workers", type=int, default=1,
                    help="python-mp workers (1=serial in main; N>1=spawn pool)")
    args = ap.parse_args()

    if args.clean and args.cache_dir.exists():
        print(f"removing {args.cache_dir}")
        shutil.rmtree(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    bulks = pd.read_csv(args.inputs)
    if args.limit:
        bulks = bulks.head(args.limit).reset_index(drop=True)
    print(f"loaded {len(bulks)} bulks; chunk_size={args.chunk_size}; workers={args.workers}")

    n_chunks = (len(bulks) + args.chunk_size - 1) // args.chunk_size
    print(f"{n_chunks} chunks total")

    common_params = dict(
        dt_C=args.dt_C,
        liq_threshold_wt=args.liq_threshold_wt,
        T_quantize_C=args.T_quantize_C,
        timeout_per_bulk=args.timeout_per_bulk,
    )

    t0 = time.time()
    n_processed = 0

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
            out = process_chunk(i, chunk, ptt, **common_params)
            out.to_parquet(chunk_path, index=False)
            n_processed += len(out)
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed, 1)
            pbar.write(f"chunk {i}: {len(out)} bulks, "
                       f"{out['T_FC_end_attrition'].sum()} flagged, "
                       f"rate={rate:.2f} bulks/s")
    else:
        payloads = []
        for i in range(n_chunks):
            chunk_path = args.cache_dir / f"chunk_{i:05d}.parquet"
            if chunk_path.exists():
                continue
            chunk = bulks.iloc[i * args.chunk_size : (i + 1) * args.chunk_size].copy()
            payloads.append({"chunk_idx": i, "chunk_df": chunk,
                             "cache_dir": str(args.cache_dir),
                             **common_params})
        n_skipped = n_chunks - len(payloads)
        if n_skipped:
            print(f"skipping {n_skipped} cached chunks; dispatching {len(payloads)} to pool")
        ctx = mp.get_context("spawn")
        # maxtasksperchild=1 prevents juliacall+MAGEMin's per-worker memory leak
        # (RSS grew 7 GB -> 180 GB over 4 hr in production runs). See 2026-05-17
        # crash post-mortem.
        with ctx.Pool(processes=args.workers, initializer=_worker_init,
                      maxtasksperchild=1) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker_process_chunk, payloads),
                total=len(payloads), desc="chunks",
            ):
                if result["status"] != "cached":
                    n_processed += result["n"]
                    elapsed = time.time() - t0
                    rate = n_processed / max(elapsed, 1)
                    tqdm.write(f"chunk {result['chunk_idx']:5d}: "
                               f"{result['n']} bulks, {result['flagged']} flagged, "
                               f"rate={rate:.2f} bulks/s")

    # Merge
    parts = sorted(args.cache_dir.glob("chunk_*.parquet"))
    print(f"\nmerging {len(parts)} chunk parquets")
    merged = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    merged.to_parquet(args.output, index=False)
    print(f"wrote {args.output}: {merged.shape}")

    print("\nsummary:")
    print(f"  total bulks:      {len(merged)}")
    print(f"  dropped (class):  {(merged['drop_reason']=='unsupported_class').sum()}")
    print(f"  fc_errors:        {merged['drop_reason'].str.startswith('fc_error').sum()}")
    print(f"  attrition (>thr): {merged['T_FC_end_attrition'].sum()}")
    print(f"  clean (T usable): {((~merged['T_FC_end_attrition']) & (merged['drop_reason']=='')).sum()}")
    print("\nT_FC_end_bulk distribution (clean only):")
    clean = merged[~merged["T_FC_end_attrition"] & (merged["drop_reason"] == "")]
    print(clean.groupby("qap_name")["T_FC_end_bulk"].agg(["count", "median", "min", "max"]).to_string())


if __name__ == "__main__":
    main()

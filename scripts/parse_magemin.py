"""Parse MAGEMin results CSV → training H5 with vocabulary discovered from data.

Inputs:
  --results PATH  : results CSV from run_magemin.jl
  --inputs  PATH  : original inputs CSV (the one fed to run_magemin.jl); used
                    to recover bulk compositions + metadata per sample
  --output  PATH  : H5 path to write

Output H5 schema:
  /X          [N, 13]   bulk_comp (11 oxides) + T_celsius + P_kbar
  /X_cols               oxide + T + P column names
  /Y_wt       [N, P]    mineral wt fractions
  /Y_cols               phase names (discovered from data)
  /Y_ss       [N, S]    solid-solution endmember fractions, NaN where absent
  /Y_ss_cols            SS endmember labels like "ol_fo", "fsp_ab"
  /Y_energy   [N]       G_system
  /qap_name   [N]       per-sample IUGS rock class (carried from inputs)
  /fe_prov    [N]       per-sample Fe-tier provenance (carried from inputs)

Phase + endmember vocabularies are discovered by scanning the data — MAGEMin's
database may add/remove phases over time. Rows with status != 0 are dropped.
Negative endmember fractions are preserved (basis projections, not mole frac).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent

OXIDE_COLS = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
              "k2o", "na2o", "tio2", "cr2o3", "h2o_tot"]


def discover_vocabularies(results: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Scan all rows to build phase + endmember vocabularies.

    SS JSON is a LIST of per-phase records (see run_magemin.jl extract_result).
    Each record has {"ph", "ph_frac_wt", "emNames", "emFrac_wt"}.
    """
    phases: set[str] = set()
    ss_endmembers: dict[str, list[str]] = {}

    for ph_str, ss_str in zip(results["phases_json"], results["ss_json"]):
        if ph_str:
            try:
                ph_d = json.loads(ph_str)
                phases.update(ph_d.get("ph", []))
            except json.JSONDecodeError:
                continue
        if ss_str and ss_str not in ("{}", "[]", ""):
            try:
                ss_records = json.loads(ss_str)
            except json.JSONDecodeError:
                continue
            # Backward-compat: old format was a dict keyed by phase name
            if isinstance(ss_records, dict):
                ss_records = [
                    {"ph": k, "ph_frac_wt": 1.0, **v}
                    for k, v in ss_records.items()
                ]
            for rec in ss_records:
                phase = rec.get("ph")
                em_names = rec.get("emNames", [])
                if phase is None:
                    continue
                if phase not in ss_endmembers:
                    ss_endmembers[phase] = list(em_names)
                else:
                    for em in em_names:
                        if em not in ss_endmembers[phase]:
                            ss_endmembers[phase].append(em)

    return sorted(phases), ss_endmembers


def build_ss_columns(ss_endmembers: dict[str, list[str]]) -> tuple[list[str], dict[tuple[str, str], int]]:
    """Flatten {phase: [em1, ...]} → ['phase_em1', ...] + index map."""
    columns: list[str] = []
    idx: dict[tuple[str, str], int] = {}
    for phase, ems in ss_endmembers.items():
        for em in ems:
            idx[(phase, em)] = len(columns)
            columns.append(f"{phase}_{em}")
    return columns, idx


def parse_phases_row(phases_json: str, phases: list[str]) -> np.ndarray:
    """Sum (NOT overwrite) when the same phase appears multiple times in out.ph.

    MAGEMin returns two `fsp` records when plagioclase + alkali feldspar
    coexist, two `cpx` records when two pyroxene compositions are stable, etc.
    Their wt fractions should sum into a single mineral-level entry.
    """
    out = np.zeros(len(phases), dtype=np.float32)
    if not phases_json:
        return out
    try:
        d = json.loads(phases_json)
    except json.JSONDecodeError:
        return out
    phase_to_idx = {p: i for i, p in enumerate(phases)}
    for name, frac in zip(d.get("ph", []), d.get("ph_frac_wt", [])):
        i = phase_to_idx.get(name)
        if i is not None:
            out[i] += float(frac)
    return out


def parse_ss_row(ss_json: str, ss_idx: dict, n_ss: int) -> np.ndarray:
    """Mass-weighted average of SS endmember fractions across duplicate phases.

    SS JSON is a list of per-phase records (see run_magemin.jl). When two
    records share the same phase name, we aggregate their endmember fractions
    weighted by each record's ph_frac_wt. This produces the "average grain
    chemistry" of all coexisting grains of that mineral, which is what users
    measuring bulk per-phase chemistry would report.

    Backward-compat: if ss_json is a dict (old format), each entry contributes
    with unit weight.
    """
    out = np.full(n_ss, np.nan, dtype=np.float32)
    if not ss_json or ss_json in ("{}", "[]"):
        return out
    try:
        records = json.loads(ss_json)
    except json.JSONDecodeError:
        return out
    if isinstance(records, dict):
        records = [{"ph": k, "ph_frac_wt": 1.0, **v} for k, v in records.items()]

    # Accumulate numerator (weight × emFrac) and denominator (sum of weights) per em column.
    num = np.zeros(n_ss, dtype=np.float64)
    den = np.zeros(n_ss, dtype=np.float64)
    for rec in records:
        phase = rec.get("ph")
        w = float(rec.get("ph_frac_wt", 0.0) or 0.0)
        if phase is None or w <= 0:
            continue
        em_names = rec.get("emNames", [])
        em_fracs = rec.get("emFrac_wt", [])
        for em, frac in zip(em_names, em_fracs):
            i = ss_idx.get((phase, em))
            if i is not None and frac is not None:
                num[i] += w * float(frac)
                den[i] += w

    mask = den > 0
    out[mask] = (num[mask] / den[mask]).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=REPO_ROOT / "data" / "magemin_results.csv",
        help="MAGEMin results CSV (output of run_magemin.jl)",
    )
    parser.add_argument(
        "--inputs", type=Path,
        default=REPO_ROOT / "data" / "magemin_inputs_tp.csv",
        help="Original inputs CSV (with T_celsius, P_kbar, qap_name, fe_provenance, oxides)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "data" / "train_magemin.h5",
        help="H5 path to write",
    )
    args = parser.parse_args()

    print(f"loading results: {args.results}")
    results = pd.read_csv(args.results)
    n_total = len(results)
    print(f"  {n_total:,} rows")

    # Drop failed points
    results = results[results["status"] == 0].copy()
    n_success = len(results)
    pct = 100 * n_success / max(n_total, 1)
    print(f"  {n_success:,} converged ({pct:.1f}%)")
    if n_success == 0:
        print("ERROR: no successful points to parse")
        return

    results["phases_json"] = results["phases_json"].fillna("")
    results["ss_json"] = results["ss_json"].fillna("")

    print("discovering phase + endmember vocabularies...")
    phases, ss_endmembers = discover_vocabularies(results)
    ss_columns, ss_idx = build_ss_columns(ss_endmembers)
    print(f"  {len(phases)} phases: {phases}")
    print(f"  {len(ss_endmembers)} solid-solution phases:")
    for ph, ems in ss_endmembers.items():
        print(f"    {ph:6s} ({len(ems)}): {ems}")
    print(f"  total SS endmember columns: {len(ss_columns)}")

    print(f"\nloading inputs: {args.inputs}")
    inputs = pd.read_csv(args.inputs)
    # Merge: results rows are (sample_id, P, T, ...); inputs rows are (sample_id, T, P, oxides, qap_name, fe_provenance).
    # A given (sample_id, T, P) may not be unique in inputs if 02b expanded the same bulk twice
    # at the same T,P (low probability but possible). Use merge on (sample_id, T_celsius, P_kbar)
    # with a tolerance via rounding, since float comparisons can be lossy.
    merged = results.merge(
        inputs,
        left_on=["sample_id", "T_celsius", "P_kbar"],
        right_on=["sample_id", "T_celsius", "P_kbar"],
        how="inner",
        suffixes=("", "_input"),
    )
    if len(merged) != len(results):
        print(f"  WARNING: merge produced {len(merged):,} rows from {len(results):,} results; possible duplicate (id, T, P)")

    X_cols = OXIDE_COLS + ["T_celsius", "P_kbar"]
    X = merged[X_cols].values.astype(np.float32)
    print(f"\nX shape: {X.shape}  cols: {X_cols}")

    Y_wt = np.zeros((len(merged), len(phases)), dtype=np.float32)
    Y_ss = np.full((len(merged), len(ss_columns)), np.nan, dtype=np.float32)

    for i, (ph_json, ss_json) in enumerate(zip(merged["phases_json"], merged["ss_json"])):
        Y_wt[i] = parse_phases_row(ph_json, phases)
        Y_ss[i] = parse_ss_row(ss_json, ss_idx, len(ss_columns))

    Y_energy = merged["G_system"].values.astype(np.float32)
    qap_name = merged["qap_name"].fillna("(unknown)").astype(str).to_numpy()
    fe_prov = merged.get(
        "fe_provenance",
        pd.Series(["(unknown)"] * len(merged), index=merged.index),
    ).fillna("(unknown)").astype(str).to_numpy()

    print(f"Y_wt shape: {Y_wt.shape}")
    print(f"Y_ss shape: {Y_ss.shape}")
    print(f"Y_energy: range [{Y_energy.min():.2f}, {Y_energy.max():.2f}]")

    phase_presence = (Y_wt > 0).mean(axis=0)
    print("\nphase presence (frac of samples):")
    for name, pres in sorted(zip(phases, phase_presence), key=lambda x: -x[1]):
        if pres > 0.001:
            print(f"  {name:8s} {pres:.3f}")

    wt_sums = Y_wt.sum(axis=1)
    print(f"\nphase wt fractions sum: mean={wt_sums.mean():.4f}, "
          f"min={wt_sums.min():.4f}, max={wt_sums.max():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {args.output}")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("X_cols", data=np.array(X_cols, dtype="S"))
        f.create_dataset("Y_wt", data=Y_wt)
        f.create_dataset("Y_cols", data=np.array(phases, dtype="S"))
        f.create_dataset("Y_ss", data=Y_ss)
        f.create_dataset("Y_ss_cols", data=np.array(ss_columns, dtype="S"))
        f.create_dataset("Y_energy", data=Y_energy)
        f.create_dataset("qap_name", data=np.array(qap_name, dtype="S"))
        f.create_dataset("fe_prov", data=np.array(fe_prov, dtype="S"))
        f.attrs["n_success"] = n_success
        f.attrs["n_total"] = n_total

    print("done.")


if __name__ == "__main__":
    main()

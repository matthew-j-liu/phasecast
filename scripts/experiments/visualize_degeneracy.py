"""Visualize the per-bulk (P, H2O) degeneracy structure.

One real Arc basalt from complete.csv (sample_id=356546). Run PetThermoTools
FC across 100 (P, H2O) pairs sampled from the Arc tectonic-class prior. For
each path, find per-path T_FC_end where liq < 5 wt%. Extract per-path modal
+ per-phase composition at T_FC_end.

Then plot:
  - distribution of modal abundance per phase (the within-bulk spread that
    drives ModeHead per-path R^2 ceiling)
  - for each major phase, distribution of per-oxide composition (the spread
    CompHead would have to predict against)

Outputs to data/diagnostics/degeneracy_viz/.
"""
import os, sys, warnings, json, time, re
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "diagnostics" / "degeneracy_viz"
OUT.mkdir(parents=True, exist_ok=True)

# Share the phase mapping with production: ptt name (with trailing digits) -> short
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "eq_vs_frac"))
from _phase_map import to_short  # noqa: E402

_TRAIL_DIGITS = re.compile(r"\d+$")

# All-side phase abbreviations PTT uses in res["All"] column suffixes.
# These are distinct from the long-name keys in mass_g (which map via to_short).
_ALL_SHORT_TO_SHORT = {
    "Cpx": "cpx", "Cpx2": "cpx", "cpx": "cpx",
    "Opx": "opx",
    "Ol": "ol",
    "Plag": "fsp", "Kspar": "fsp", "Afs": "fsp",
    "Sp": "spl",
    "Rhm": "ilm",
    "Amp": "amp", "amp": "amp",
    "Bi": "bi", "bi": "bi",
    "Mu": "mu", "mu": "mu",
    "Gar": "g", "g": "g",
    "Crd": "cd", "cd": "cd",
    "Ep": "ep", "ep": "ep",
    "Ne": "ne", "ne": "ne",
    "Q": "q", "q": "q",
    "Liq": "liq",
    "Fl": "fl",
    "Ttn": "sph", "sph": "sph", "Sph": "sph",
    "Ru": "ru", "ru": "ru",
}

# The chosen bulk (sample_id=356546) — Arc basalt, Th/Yb=1.0, Nb/Yb=3.0
# feo_tot=11.83 split via typical arc Fe3+/SumFe ratio = 0.20
FE_TOT = 11.83
FE3_RATIO = 0.20
MW_FEO = 71.844
MW_FE2O3 = 159.69

# Fe partitioning: feo_tot in wt% → split into FeO and Fe2O3
# fe3_ratio is moles_Fe3 / moles_Fe_total; mass_FeO = (1 - r) * feo_tot;
# mass_Fe2O3 = r * feo_tot / (2*MW_FeO/MW_Fe2O3)
FEO = (1.0 - FE3_RATIO) * FE_TOT
FE2O3 = FE3_RATIO * FE_TOT * MW_FE2O3 / (2 * MW_FEO)

BULK = {
    "SiO2": 47.07, "TiO2": 1.42, "Al2O3": 14.36,
    "FeO": FEO, "Fe2O3": FE2O3,
    "MnO": 0.18, "MgO": 5.21, "CaO": 11.75,
    "Na2O": 1.77, "K2O": 0.06, "Cr2O3": 0.0,
}

# Arc tectonic-class prior (from per-setting literature: Plank 2013, Wallace 2005)
P_RANGE_KBAR = (1.0, 5.0)       # arc crustal magma chamber depth range
H2O_RANGE_WT = (2.0, 6.0)       # arc magma H2O range
T_START_C = 1280                 # above arc basalt liquidus
T_FLOOR_C = 600                  # mafic SiO2 tier floor
LIQ_THRESHOLD = 0.05             # 5 wt% liq threshold for T_FC_end
DT_STEP_C = 10                   # FC cooling step
N_PATHS = 100                    # number of (P, H2O) pairs

# Sampling: 10x10 grid (deterministic, covers the rectangle uniformly)
N_P = 10
N_H = 10
P_grid = np.linspace(P_RANGE_KBAR[0], P_RANGE_KBAR[1], N_P)
H_grid = np.linspace(H2O_RANGE_WT[0], H2O_RANGE_WT[1], N_H)
PATHS = [(p, h) for p in P_grid for h in H_grid]
assert len(PATHS) == N_PATHS, f"expected {N_PATHS} paths, got {len(PATHS)}"


def _ptt_short(ptt_phase: str) -> str | None:
    """Strip trailing digits, then map. Try mass_g long-name table first
    (clinopyroxene1 -> cpx), then fall back to the All-side short table
    (Cpx -> cpx)."""
    base = _TRAIL_DIGITS.sub("", ptt_phase)
    s = to_short(base) or to_short(ptt_phase)
    if s is not None:
        return s
    # Try All-side short codes
    return _ALL_SHORT_TO_SHORT.get(base) or _ALL_SHORT_TO_SHORT.get(ptt_phase)


def run_one_fc(p_kbar: float, h2o_wt: float, ptt) -> dict | None:
    """Run one FC sweep at (P, H2O), find T_FC_end where liquid1 < 5 g
    (system mass conserved at 100 g; matches production T_FC_end logic).
    Modal = cumulate (cumsum solids + residual liq + fluid) at T_FC_end row.
    Composition = snapshot per-phase composition from res['All'] same row.
    """
    bulk = {**BULK, "H2O": float(h2o_wt)}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ptt.isobaric_crystallisation(
                Model="Green2025", bulk=bulk, Frac_solid=True,
                T_start_C=T_START_C, T_end_C=T_FLOOR_C, dt_C=DT_STEP_C,
                P_bar=float(p_kbar) * 1000.0,
                Suppress=["rutile", "tridymite"],
                multi_processing=False, timeout=180,
            )
    except Exception as e:
        return {"status": f"fc_error:{type(e).__name__}", "P_kbar": p_kbar, "H2O_wt": h2o_wt}

    mg = res.get("mass_g")
    All = res.get("All")
    if mg is None or len(mg) == 0 or "liquid1" not in mg.columns:
        return {"status": "empty_mass_g", "P_kbar": p_kbar, "H2O_wt": h2o_wt}
    if All is None or "T_C" not in All.columns or len(All) != len(mg):
        return {"status": "all_missing_or_misaligned", "P_kbar": p_kbar, "H2O_wt": h2o_wt}

    # T_FC_end: highest T where liquid1 < 5 g (production convention, mass conserved 100 g).
    # mass_g and All are row-aligned by FC step; T_C lives in All.
    Ts = All["T_C"].to_numpy()
    liq_g = mg["liquid1"].fillna(0.0).to_numpy()
    below = liq_g < 5.0
    if not below.any():
        idx = len(mg) - 1
        attrition = True
    else:
        idx = int(np.argmax(below))
        attrition = False
    T_FC_end = float(Ts[idx])
    liq_at_T = float(liq_g[idx])

    out = {"status": "ok", "P_kbar": p_kbar, "H2O_wt": h2o_wt,
           "T_FC_end": T_FC_end, "attrition": attrition,
           "liq_at_T_g": liq_at_T}

    # Modal (cumulate): sum *_cumsum + residual liquid1 + fluid1, in grams,
    # normalized to fractions. Map ptt phase names to MAGEMin short codes.
    last = mg.iloc[idx]
    modal_g: dict[str, float] = {}
    for col in mg.columns:
        if col.endswith("_cumsum"):
            short = _ptt_short(col[: -len("_cumsum")])
            if short is None:
                continue
            v = last[col]
            v = float(v) if not pd.isna(v) else 0.0
            if v > 0:
                modal_g[short] = modal_g.get(short, 0.0) + v
    for liq_col, short in (("liquid1", "liq"), ("fluid1", "fl")):
        if liq_col in mg.columns:
            v = last[liq_col]
            v = float(v) if not pd.isna(v) else 0.0
            if v > 0:
                modal_g[short] = modal_g.get(short, 0.0) + v
    total = sum(modal_g.values())
    modal_frac = {k: v / total for k, v in modal_g.items()} if total > 0 else {}
    out["modal_frac"] = modal_frac
    out["modal_total_g"] = total

    # Per-phase cumulate composition over [0, idx]. In FC, each step removes
    # solids — the cumulate is the mass-weighted average of per-step
    # compositions across all steps where the phase crystallized.
    # mass_g columns: "<ptt_name>" = incremental mass (g) of phase at that step;
    # All columns: "<oxide>_<PtTPhaseInAll>" = composition at that step.
    # Use the mass_g (not _cumsum) column for per-step mass; map ptt name to
    # the corresponding "mass%_*" phase code in All to read composition.
    phase_codes_all = set()
    for col in All.columns:
        if col.startswith("mass%_"):
            phase_codes_all.add(col[len("mass%_"):])
    phases_seen = sorted(phase_codes_all)

    # Map ptt_name (in mass_g) -> PtTPhaseInAll (in All). Both follow same
    # base+digit pattern; the All-side name capitalizes ("Cpx" vs "clinopyroxene1").
    # The mass% column with the highest cumulative mass at idx whose short
    # matches the same short as the mass_g column is the right pair.
    def _short_of(name: str) -> str | None:
        return _ptt_short(name) or _ptt_short(name.lower())

    # Group All-side phase codes by short
    all_by_short: dict[str, list[str]] = {}
    for ph in phase_codes_all:
        s = _short_of(ph)
        if s is None:
            continue
        all_by_short.setdefault(s, []).append(ph)

    # For each mass_g phase column (non-cumsum, non-liquid/fluid), accumulate
    # mass-weighted composition over [0, idx].
    composition: dict[str, dict[str, float]] = {}
    OXIDES = ["SiO2","TiO2","Al2O3","Cr2O3","FeOt","MnO","MgO","CaO","Na2O","K2O","H2O"]
    for mg_col in mg.columns:
        if mg_col.endswith("_cumsum") or mg_col in ("liquid1", "fluid1"):
            continue
        short = _short_of(mg_col)
        if short is None or short in ("liq", "fl"):
            continue
        all_phases = all_by_short.get(short, [])
        if not all_phases:
            continue
        # Per-step incremental mass for this ptt phase
        delta_m = mg[mg_col].fillna(0.0).to_numpy()[: idx + 1]
        if delta_m.sum() <= 0:
            continue
        # Pick the matching All-side phase by name proximity (Cpx vs cpx3 etc.)
        # Fallback: any phase with matching short whose mass% is populated.
        sum_oxide = {ox: 0.0 for ox in OXIDES}
        total_m = 0.0
        for i in range(idx + 1):
            dm = delta_m[i]
            if dm <= 0:
                continue
            # Pick the All-phase with the largest mass% at this step that maps to `short`
            row_all_i = All.iloc[i]
            best_ph = None
            best_m = 0.0
            for ph in all_phases:
                v = row_all_i.get(f"mass%_{ph}", 0.0)
                if pd.notna(v) and v > best_m:
                    best_m = float(v)
                    best_ph = ph
            if best_ph is None:
                continue
            for ox in OXIDES:
                v = row_all_i.get(f"{ox}_{best_ph}", np.nan)
                if pd.notna(v):
                    sum_oxide[ox] += dm * float(v)
            total_m += dm
        if total_m <= 0:
            continue
        avg = {ox: sum_oxide[ox] / total_m for ox in OXIDES if sum_oxide[ox] != 0.0}
        # Multiple ptt names can collapse to same short (Cpx + Cpx2 + cpx3);
        # mass-weight across them too if we hit it twice.
        if short in composition:
            prior = composition[short]
            prior_m = prior.pop("_total_g", total_m)
            new_total = prior_m + total_m
            for ox in OXIDES:
                a = prior.get(ox, 0.0) * prior_m
                b = avg.get(ox, 0.0) * total_m
                if a + b != 0.0:
                    prior[ox] = (a + b) / new_total
            prior["_total_g"] = new_total
        else:
            avg["_total_g"] = total_m
            composition[short] = avg
    for short, comp in composition.items():
        comp.pop("_total_g", None)
    out["phases_seen_ptt"] = phases_seen
    out["composition_wt_pct"] = composition
    return out


def main():
    print(f"running {N_PATHS} FC paths for sample_id=356546 (Arc basalt)")
    print(f"  P range: {P_RANGE_KBAR}  H2O range: {H2O_RANGE_WT}")
    print(f"  T_start={T_START_C} T_floor={T_FLOOR_C} dt={DT_STEP_C}")
    print(f"  per-path T_FC_end where liq < {LIQ_THRESHOLD*100}%\n")

    import petthermotools as ptt
    ptt.activate_petthermotools_env()

    results = []
    t0 = time.time()
    for i, (p_kbar, h2o_wt) in enumerate(PATHS):
        res = run_one_fc(p_kbar, h2o_wt, ptt)
        results.append(res)
        # Clean up Julia heap to avoid memory drift
        from juliacall import Main as _jl
        _jl.GC.gc(True)
        if (i+1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i+1) / max(elapsed, 1)
            eta = (N_PATHS - i - 1) / max(rate, 0.01)
            ok = sum(1 for r in results if r.get("status") == "ok")
            print(f"  [{i+1:3d}/{N_PATHS}] ok={ok}  rate={rate:.2f} paths/s  eta={eta:.0f}s")

    # Save raw results
    with open(OUT / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nraw results -> {OUT / 'raw_results.json'}")

    # Build summary tables
    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\nconverged: {len(ok)} / {N_PATHS}")
    if not ok:
        print("no successful FC runs; aborting plotting")
        return

    # Modal table: rows=paths, cols=phases (mass fraction of cumulate)
    all_phases = sorted({ph for r in ok for ph in r["modal_frac"]})
    modal_rows = []
    for r in ok:
        row = {"P_kbar": r["P_kbar"], "H2O_wt": r["H2O_wt"], "T_FC_end": r["T_FC_end"]}
        for ph in all_phases:
            row[ph] = r["modal_frac"].get(ph, 0.0)
        modal_rows.append(row)
    modal_df = pd.DataFrame(modal_rows)
    modal_df.to_parquet(OUT / "modal_per_path.parquet")
    print(f"modal table: {modal_df.shape} -> {OUT / 'modal_per_path.parquet'}")

    # Composition table: long-format rows=(path, phase, oxide), value=wt%
    comp_rows = []
    for r in ok:
        for ph, comp in r["composition_wt_pct"].items():
            for ox, val in comp.items():
                comp_rows.append({"P_kbar": r["P_kbar"], "H2O_wt": r["H2O_wt"],
                                  "T_FC_end": r["T_FC_end"],
                                  "phase": ph, "oxide": ox, "wt_pct": val})
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_parquet(OUT / "composition_per_path.parquet")
    print(f"composition table: {comp_df.shape} -> {OUT / 'composition_per_path.parquet'}")

    # Per-path T_FC_end summary (the "T depends on (P, H2O)" demonstration)
    t_summary = modal_df[["P_kbar", "H2O_wt", "T_FC_end"]].describe()
    print(f"\nT_FC_end across the {len(ok)} paths:")
    print(t_summary[["T_FC_end"]].to_string())

    print(f"\nDone. Plotting script next will read these parquets.")


if __name__ == "__main__":
    main()

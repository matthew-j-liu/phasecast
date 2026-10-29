"""Harvest real volcanic bulk-chemistry + modal-mineralogy pairs into a unified
benchmark under data/real_mineralogy/ (gitignored, regenerable).

Sources (open-access; see README for the screen they passed):
  - du Bray et al. USGS DS 764 (Bodie Hills) and DS 1099 (Tonopah/Divide/
    Goldfield): whole-rock XRF + visual phenocryst abundance estimates, joined
    on Field_ID. Coarse modes (visual, not point-counted) -- presence tier.
  - DS 1099 Appendix 10: microscope point counts of the Fraction and Heller
    Tuff samples, machine-readable, covering 70 rocks the visual estimates
    already describe. Emitted alongside those estimates rather than replacing
    them, so the same rock carries both rulers and `load_benchmark` picks.
  - Pallister et al. 2008 (MSH 2004-06, PP1750 ch. 30) and Pallister et al.
    1996 (Pinatubo): downloaded here; table extraction is manual/vision work,
    parsed separately (precision tier).
  - du Bray et al. USGS DS 942 (1.4 Ga A-type granitoids) and PP 1793
    (Boulder batholith): whole-rock point counts of plutonic rocks, a second
    benchmark written to data/real_mineralogy/plutonic/ rather than joined to
    the volcanic one.

Output: samples.csv (one row per sample: chemistry + crystallinity + QC) and
modes.csv (one row per sample x phase: reported abundance + basis), with phase
names mapped to the MAGEMin vocabulary alongside the as-reported names.

    conda run -n mineral python explorations/real_mineralogy/harvest.py
"""
import re

import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "real_mineralogy"

# du Bray phenocryst-abundance columns -> (reported name, MAGEMin phase).
# px is unsplit cpx/opx in the source (the Cpx_Opx column carries presence and
# a dominance ordering -- `C`/`O`, dominant uppercase -- not a ratio, on 357 of
# the 717 benchmark rocks); opq is unsplit magnetite/ilmenite.
DUBRAY_PHASES = {
    "AbdQtz": ("quartz", "q"),
    "AbdAlkFld": ("alkali feldspar", "fsp"),
    "AbdPl": ("plagioclase", "fsp"),
    "AbdHbl": ("hornblende", "amp"),
    "AbdBt": ("biotite", "bi"),
    "AbdPx": ("pyroxene", "px_unsplit"),
    "AbdOl": ("olivine", "ol"),
    "AbdOpq": ("Fe-Ti oxides", "opq_unsplit"),
}

OXIDES = {
    "SiO2_pct": "sio2", "TiO2_pct": "tio2", "Al2O3_pct": "al2o3",
    "FeO*_pct": "feo_tot", "MnO_pct": "mno", "MgO_pct": "mgo",
    "CaO_pct": "cao", "Na2O_pct": "na2o", "K2O_pct": "k2o",
    "P2O5_pct": "p2o5", "LOI_pct": "loi",
}
# Output names are the tokenizer's `<sym>_ppm` trace columns (cnmineral.data.tokens
# TRACES: Sr Zr Ba Y Rb Ni Nb V La Ce Nd Yb Th) so samples.csv feeds the encoder
# path directly; source column -> tokenizer column.
TRACES = {"Sr_ppm": "sr_ppm", "Zr_ppm": "zr_ppm", "Ba_ppm": "ba_ppm", "Y_ppm": "y_ppm",
          "Rb_ppm": "rb_ppm", "Ni_ppm": "ni_ppm", "Nb_ppm": "nb_ppm", "V_ppm": "v_ppm",
          "La_ppm": "la_ppm", "Ce_ppm": "ce_ppm", "Nd_ppm": "nd_ppm", "Yb_ppm": "yb_ppm",
          "Th_ppm": "th_ppm"}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def parse_dubray(source, pet_path, geo_path, tot_col):
    pet = pd.read_excel(DATA / pet_path, header=1)
    geo = pd.read_excel(DATA / geo_path, header=1)
    # DS1099 uses section-label rows (e.g. 'TONOPAH:') with no data
    geo = geo[geo.get("SiO2_pct").notna() | geo.iloc[:, 2].notna()]
    pet = pet.dropna(subset=["Field_ID"]).set_index("Field_ID")
    geo = geo.dropna(subset=["Field_ID"]).set_index("Field_ID")
    joint = pet.index.intersection(geo.index)

    unit_col = "Strat_Name" if "Strat_Name" in pet.columns else "Unit"
    samples, modes = [], []
    for fid in joint:
        p, g = pet.loc[fid], geo.loc[fid]
        if isinstance(p, pd.DataFrame):
            p = p.iloc[0]
        if isinstance(g, pd.DataFrame):
            g = g.iloc[0]
        row = dict(source=source, sample_id=str(fid), unit=p.get(unit_col),
                   total_crystals_pct=_num(pd.Series([p.get(tot_col)])).iloc[0],
                   alteration_extent=str(p.get("AltExtnt")),
                   texture=p.get("Texture"), mode_basis="phenocryst_vol_visual")
        for src, dst in {**OXIDES, **TRACES}.items():
            if src in g.index:
                row[dst] = _num(pd.Series([g[src]])).iloc[0]
        samples.append(row)
        for col, (name, mgm) in DUBRAY_PHASES.items():
            raw = p.get(col)
            if pd.isna(raw):
                continue
            val = _num(pd.Series([raw])).iloc[0]
            qual = None if not pd.isna(val) else str(raw).strip()
            modes.append(dict(source=source, sample_id=str(fid), phase=name,
                              magemin_phase=mgm, abundance_pct=val,
                              qualifier=qual, basis="phenocryst_vol_visual"))
    return pd.DataFrame(samples), pd.DataFrame(modes)


# DS 1099 Appendix 10, "Results of microscope-based point counts of Fraction
# Tuff and Heller Tuff samples".  Machine-readable, unlike the other two
# point-counted sources, and it covers rocks the visual estimates already
# describe -- so these rows join the ones parse_dubray emits instead of
# replacing them, and `preferred_basis` decides which ruler a scoring run
# reads (`load_benchmark` keeps the visual estimates).
#
# Seven categories are counted and no others.  Pyroxene and olivine are never
# looked for here, so their absence from a row is silence rather than a reported
# absence -- the eval must not read these rocks as px-free.  A blank cell in a
# counted category IS an absence (the phase was in the tally scheme and no point
# landed on it), which matches the visual rows' convention of omitting absent
# phases rather than zero-rowing them.
POINTCOUNT_PHASES = {
    "Quartz": ("quartz", "q"),
    "Plagioclase": ("plagioclase", "fsp"),
    "Alkali feldspar": ("alkali feldspar", "fsp"),
    "Biotite": ("biotite", "bi"),
    "Hornblende": ("hornblende", "amp"),
    "Opaques": ("Fe-Ti oxides", "opq_unsplit"),
    "Matrix": ("groundmass", "groundmass"),
}
POINTCOUNT_PCT_TOL = 1e-9


def _pointcount_cell(raw):
    """One Appendix 10 cell as (counted points, reported-name override, qualifier).

    Four forms occur in the workbook.  A bare integer is the count.  `TR`, with
    an optional `?` or `(?)`, is present-but-uncounted and takes the same
    left-censored path du Bray's `TR` rows take.  `1(?)` is a count whose
    identification the author flagged as uncertain.  And `87 (total feldspar)`
    in the plagioclase column means plagioclase and alkali feldspar could not be
    separated, so the count is their sum -- which costs nothing downstream,
    since both coarsen to `fsp`.
    """
    text = str(raw).strip()
    if text.upper().startswith("TR"):
        return None, None, text
    match = re.match(r"^(\d+)\s*(.*)$", text)
    if match is None:
        raise ValueError(f"unparsed Appendix 10 cell {raw!r}")
    count, rest = int(match.group(1)), match.group(2)
    name = "feldspar" if "feldspar" in rest.lower() else None
    return count, name, "?" if "?" in rest else None


def parse_dubray_pointcounts(known_ids):
    """Appendix 10 mode rows, plus each counted rock's section total.

    Percentages are derived from the raw-count block rather than read from the
    workbook's own percent block, then checked against it.  That makes the
    conversion reproducible instead of trusted, and a revised download that
    changed either block could not land silently -- the same guarantee
    `_check_pinatubo_chem` gives the transcribed chunk.

    `Total` runs a little above the sum of the counted categories on most rows
    (an unlisted residual), and the workbook's own percentages divide by it, so
    this does too.  It cancels regardless: the truth vector renormalizes over
    the crystal cargo once groundmass drops.
    """
    a10 = pd.read_excel(DATA / "dubray_ds1099" / "Appendix10.xlsx", header=2)
    a10 = a10.dropna(subset=["Field_ID"])
    modes, n_points, skipped = [], {}, 0
    for _, row in a10.iterrows():
        fid = str(row["Field_ID"])
        if fid not in known_ids:
            skipped += 1
            continue
        total = float(row["Total"])
        n_points[fid] = total
        for col, (name, mgm) in POINTCOUNT_PHASES.items():
            raw = row.get(col)
            if pd.isna(raw):
                continue
            count, override, qualifier = _pointcount_cell(raw)
            pct = None if count is None else count / total * 100.0
            published = _num(pd.Series([row.get(f"{col}.1")])).iloc[0]
            if pct is not None and pd.notna(published):
                if abs(pct - published) > POINTCOUNT_PCT_TOL:
                    raise ValueError(
                        f"appendix 10 {fid} {col}: {count}/{total} = {pct} does "
                        f"not reproduce the published {published} -- check the "
                        "Appendix10.xlsx download"
                    )
            modes.append(dict(source="dubray_ds1099", sample_id=fid,
                              phase=override or name, magemin_phase=mgm,
                              abundance_pct=pct, qualifier=qualifier,
                              basis="optical_vol_pointcount"))
    return pd.DataFrame(modes), n_points, skipped


# MSH 2004-06 dome, Pallister 2008 Table 2 optical modes (vol%, thin-section
# basis: 1,000-1,300 pt counts; Matrix = groundmass, so phenocryst/groundmass
# split is explicit). Transcribed from the chapter PDF (GIF/PDF table, not
# machine-readable). Values are TS optical mode; silica phases n.d. -> 0.
# (name, magemin, {sample: vol%}); chem_key joins to OFR 2008-1130 chemistry.
MSH_CHEM_KEY = {"SH304-2A": "SH304-2", "SH304-2G1": "SH304-2G1", "SH305-1A": "SH305-1",
                "SH315-4": "SH315-4", "SH319-1": "SH319-1", "SH321-1A": "SH321-1",
                "SH325-1A": "SH325-1"}
MSH_MODES = {  # phenocryst vol%, thin-section optical mode
    "plagioclase": ("fsp", dict(zip(MSH_CHEM_KEY, [30, 31, 22, 39, 36.3, 37.1, 33.8]))),
    "hypersthene": ("opx", dict(zip(MSH_CHEM_KEY, [4, 4, 3, 6, 5.2, 6.3, 2.5]))),
    "hornblende": ("amp", dict(zip(MSH_CHEM_KEY, [8, 2, 2, 4, 5.5, 3.5, 2.6]))),
    "Fe-Ti oxides": ("opq_unsplit", dict(zip(MSH_CHEM_KEY, [2, 2, 1, 2, 1.4, 1.1, 1.4]))),
    "silica phases": ("q", dict(zip(MSH_CHEM_KEY, [0, 0, 0, 0.2, 1.1, 1.0, 1.1]))),
    "groundmass": ("groundmass", dict(zip(MSH_CHEM_KEY, [55, 59, 72, 49, 48.8, 49.2, 56.8]))),
}
MSH_OXIDE_ROW = {8: "sio2", 9: "tio2", 10: "al2o3", 11: "feo_tot", 12: "mno",
                 13: "mgo", 14: "cao", 15: "na2o", 16: "k2o", 17: "p2o5"}


def parse_msh():
    raw = pd.read_excel(DATA / "msh_pallister2008" / "of2008-1130_v1.1.xls",
                        sheet_name="3. Sample Chemistry", header=None)
    names = [str(x).strip() for x in raw.iloc[2].tolist()]
    col = {n: j for j, n in enumerate(names)}
    samples, modes = [], []
    for sid, chem_key in MSH_CHEM_KEY.items():
        j = col.get(chem_key)
        row = dict(source="msh_pallister2008", sample_id=sid, unit="MSH 2004-06 dome",
                   total_crystals_pct=100 - MSH_MODES["groundmass"][1][sid],
                   alteration_extent="fresh", texture="porphyritic",
                   mode_basis="optical_vol_pointcount")
        if j is not None:
            for r, dst in MSH_OXIDE_ROW.items():
                row[dst] = _num(pd.Series([raw.iloc[r, j]])).iloc[0]
        samples.append(row)
        for name, (mgm, vals) in MSH_MODES.items():
            modes.append(dict(source="msh_pallister2008", sample_id=sid, phase=name,
                              magemin_phase=mgm, abundance_pct=vals[sid],
                              qualifier=None, basis="optical_vol_pointcount"))
    return pd.DataFrame(samples), pd.DataFrame(modes)


# Pinatubo 1991, Pallister et al. 1996 Table 1: modal point counts (400x mag,
# vol%, phenocryst basis; Groundmass explicit). Transcribed from table1.gif.
# tr/<1 -> present-but-trace (abundance NaN, qualifier kept); '-' -> absent (omitted).
# cummingtonite is a Fe-Mg amphibole -> amp; bronzite -> opx; augite -> cpx.
PIN_SAMPLES = ["EW910615-1", "EW910615-2", "PH13d", "7-1-91-1A", "CN6791-h", "CN6791-i"]
PIN_ROCK = ["dacite", "dacite", "dacite", "andesite", "andesite", "basalt"]
PIN_PTS = [3206, 1802, 1618, 1606, 2061, 2280]
PIN_MODES = {  # phase: (magemin, [six values]); "tr"/"<1"/"-" as strings
    "plagioclase": ("fsp", [31, 20, 11, 17, 31, 22]),
    "hornblende": ("amp", [12, 6, 3, 12, 12, 31]),
    "cummingtonite": ("amp", [1, "<1", "tr", "<1", "-", "-"]),
    "biotite": ("bi", ["tr", "tr", "-", "<1", 1, "-"]),
    "augite": ("cpx", ["-", "-", "-", 4, 5, 12]),
    "olivine": ("ol", ["tr", "tr", "-", 2, 2, 3]),
    "bronzite": ("opx", ["tr", "-", "-", "-", "-", "-"]),
    "Fe-Ti oxides": ("opq_unsplit", [1, "<1", "<1", "<1", 2, 2]),
    "quartz": ("q", [1, "<1", 1, 1, "<1", 1]),
    "groundmass": ("groundmass", [53, 74, 85, 62, 48, 29]),
}


# Pinatubo chemistry, Pallister et al. 1996 Table 7, transcribed from
# table7p1.gif (majors + Ni) and table7p2.gif (traces).  Keyed by Table 1's
# sample name; the value is that sample's Table 7 lab number, which is what
# identifies the row -- Table 7 groups by magma type and spells some names
# differently (PH13d appears as PH13D).
#
# CN6791-h is absent: Table 1's andesite lava has no Table 7 row.  The nearest
# is CN6791d, a different split of the same locality and date, and equating
# them would invent a measurement rather than record one -- so this rock keeps
# no chemistry and stays out of the benchmark.
#
# Replicate analyses (the `x`-suffixed lab numbers) are left aside; each
# sample takes its primary row.  Y, Nb and V are not in Table 7 at all, so
# they pad -- 10 of the tokenizer's 13 traces are populated here.
#
# `fe2o3_tot` is Table 7's FeTO3 column verbatim (footnote 1: total iron
# expressed as Fe2O3).  The tokenizer already accepts it as an iron source and
# converts it itself, so the source's own column is what gets recorded --
# no hand conversion to the `feo_tot` the other sources report.  Pallister
# also titrated Fe2O3 and FeO separately; those are a redox split, and the
# encoder takes total Fe cation wt%, so the total is the faithful record.
PIN_CHEM_COLS = ["sio2", "al2o3", "fe2o3_tot", "mgo", "cao", "na2o", "k2o",
                 "tio2", "p2o5", "mno", "loi", "rb_ppm", "sr_ppm", "zr_ppm",
                 "ba_ppm", "la_ppm", "ce_ppm", "nd_ppm", "yb_ppm", "th_ppm",
                 "ni_ppm"]
PIN_CHEM = {
    "EW910615-1": ("D-379505", [64.8, 16.6, 4.30, 2.28, 5.28, 4.47, 1.52, .52,
                                .19, .09, .97, 38.0, 569, 108, 455, 15.9, 34.4,
                                14.3, 1.20, 4.69, 17.0]),
    "EW910615-2": ("D-379506", [64.3, 16.6, 4.37, 2.53, 5.47, 4.42, 1.47, .50,
                                .18, .09, 1.01, 39.9, 567, 102, 433, 15.1, 31.2,
                                15.1, 1.32, 4.39, 17.0]),
    "PH13d": ("D-500917", [64.1, 17.1, 4.28, 2.38, 5.28, 4.62, 1.47, .48,
                           .19, .10, 1.32, 42.0, 590, 104, 454, 18.8, 40.2,
                           16.7, 1.49, 4.97, 14.5]),
    "7-1-91-1A": ("D-500918", [59.2, 16.1, 6.13, 4.86, 7.17, 3.91, 1.56, .64,
                               .29, .12, 1.04, 39.0, 515, 101, 380, 17.5, 35.8,
                               18.0, 1.44, 4.84, 43.6]),
    "CN6791-i": ("D-504517", [50.5, 14.7, 9.32, 9.21, 10.55, 3.02, 1.34, .90,
                              .37, .16, .17, 44.0, 592, 120, 313, 22.6, 49.4,
                              26.3, 1.76, 5.96, 105.0]),
}

# Table 7's majors are normalized volatile-free, so each row's ten oxides sum
# to 100 up to the printed rounding.  That makes the transcription checkable
# rather than trusted, and a retranscription cannot regress silently.
PIN_MAJORS = PIN_CHEM_COLS[:10]
PIN_SUM_TOL = 0.15


def _check_pinatubo_chem():
    for sid, (lab, values) in PIN_CHEM.items():
        row = dict(zip(PIN_CHEM_COLS, values))
        total = sum(row[c] for c in PIN_MAJORS)
        if abs(total - 100.0) > PIN_SUM_TOL:
            raise ValueError(
                f"pinatubo {sid} ({lab}) majors sum to {total:.2f}, not 100 "
                f"+/- {PIN_SUM_TOL} -- check the table7p1.gif transcription"
            )


def parse_pinatubo():
    _check_pinatubo_chem()
    samples, modes = [], []
    for i, sid in enumerate(PIN_SAMPLES):
        chem = PIN_CHEM.get(sid)
        samples.append(dict(source="pinatubo_pallister1996", sample_id=sid,
                            unit=f"Pinatubo 1991 {PIN_ROCK[i]}", rock_class=PIN_ROCK[i],
                            total_crystals_pct=100 - PIN_MODES["groundmass"][1][i],
                            alteration_extent="fresh", texture="porphyritic",
                            mode_basis="optical_vol_pointcount",
                            n_points=PIN_PTS[i],
                            chem_status="table7" if chem else "no_table7_row",
                            **(dict(zip(PIN_CHEM_COLS, chem[1])) if chem else {})))
        for name, (mgm, vals) in PIN_MODES.items():
            v = vals[i]
            if v == "-":
                continue
            num = float(v) if not isinstance(v, str) else np.nan
            modes.append(dict(source="pinatubo_pallister1996", sample_id=sid, phase=name,
                              magemin_phase=mgm, abundance_pct=num,
                              qualifier=v if isinstance(v, str) else None,
                              basis="optical_vol_pointcount"))
    return pd.DataFrame(samples), pd.DataFrame(modes)


# The plutonic benchmark: du Bray DS 942 (1.4 Ga A-type granitoids) and PP 1793
# (Boulder batholith).  Both are point counts of the whole rock in volume
# percent -- DS 942's Data Fields table says so, and PP 1793's appendix defers
# to DS 454's -- so there is no groundmass and a rock's modes sum to about 100.
# They are written to their own directory: every eruption-v2 verdict evaluates
# all the rocks in `DATA`'s tables, so these must never join them.
PLUTONIC_DIR = DATA / "plutonic"
PLUTONIC_BASIS = "wholerock_vol_pointcount"

# mode column -> (reported name, MAGEMin phase).  Accessories, the
# accessories-with-oxides lump, the mafic-total-only column and the secondary
# (alteration) minerals are kept as rows outside the seven comparison groups,
# so each rock's modes still close and a later freshness or cumulate rule can
# read them.  The two lumps also record where a comparison group was not
# measured: on a rock carrying `accessory_opq` and no Fe-Ti oxide row the
# oxides were never counted apart from the accessories, and on one carrying
# `mafic_unsplit` no mafic mineral was counted individually, so an absent row
# there is silence rather than absence.  Where a rock carries both
# `accessory_opq` and an oxide row, the modes still sum to 100, so the lump
# holds the accessories alone and nothing is counted twice.
DS942_PHASES = {
    "Abd_Pl": ("plagioclase", "fsp"),
    "Abd_AlkFld": ("alkali feldspar", "fsp"),
    "Abd_Qtz": ("quartz", "q"),
    "Abd_Hbl": ("hornblende", "amp"),
    "Abd_Opx": ("orthopyroxene", "opx"),
    "Abd_Cpx": ("clinopyroxene", "cpx"),
    "Abd_Bt": ("biotite", "bi"),
    "Abd_Mu": ("muscovite", "mu"),
    "Abd_Opq": ("Fe-Ti oxides", "opq_unsplit"),
    "Abd_Tot_mafics": ("mafic minerals", "mafic_unsplit"),
    "Abd_Acc": ("accessory minerals", "accessory"),
    "Abd_Acces (incl opq)": ("accessory minerals and Fe-Ti oxides", "accessory_opq"),
    "Abd_Alt": ("secondary minerals", "secondary"),
}
# PP 1793 never counts muscovite, so its absence there is silence too.  `Pyx`
# is lumped pyroxene; a rock carrying both `Pyx` and a split column counts
# them as separate minerals (they sum with the rest to 100), so both are kept.
PP1793_PHASES = {
    "Plag": ("plagioclase", "fsp"),
    "Kspar": ("alkali feldspar", "fsp"),
    "Qtz": ("quartz", "q"),
    "Hb": ("hornblende", "amp"),
    "Pyx": ("pyroxene", "px_unsplit"),
    "Opx": ("orthopyroxene", "opx"),
    "Cpx": ("clinopyroxene", "cpx"),
    "Ol": ("olivine", "ol"),
    "Bi": ("biotite", "bi"),
    "Opq": ("Fe-Ti oxides", "opq_unsplit"),
    "Acc": ("accessory minerals", "accessory"),
    "Acces(incl opq)": ("accessory minerals and Fe-Ti oxides", "accessory_opq"),
    "Alt": ("secondary minerals", "secondary"),
}

# Majors are recalculated to 100 volatile-free with all Fe as FeO in both
# sources, the convention du Bray's volcanic tables use, so they pass through
# unrenormalized.  The pre-recalculation analytical total and the volatile
# total are carried so that convention stays checkable and invertible per rock.
DS942_CHEM = {
    "SiO2_pct": "sio2", "TiO2_pct": "tio2", "Al2O3_pct": "al2o3",
    "FeO_pct": "feo_tot", "MnO_pct": "mno", "MgO_pct": "mgo",
    "CaO_pct": "cao", "Na2O_pct": "na2o", "K2O_pct": "k2o",
    "P2O5_pct": "p2o5", "LOI_pct": "loi",
    "Total_I_pct": "analytical_total_pct", "Volatile_pct": "volatile_pct",
    **TRACES,
}
PP1793_CHEM = {
    "SiO2": "sio2", "TiO2": "tio2", "Al2O3": "al2o3", "FeO*": "feo_tot",
    "MnO": "mno", "MgO": "mgo", "CaO": "cao", "Na2O": "na2o", "K2O": "k2o",
    "P2O5": "p2o5", "LOI": "loi",
    "total": "analytical_total_pct", "vol_sum": "volatile_pct",
    **{src.removesuffix("_ppm"): dst for src, dst in TRACES.items()},
}

# Cells whose published text is a typo for a number: the letter l for the
# digit 1, and a stray backtick before a zero.  Keyed by the cell, and every
# entry must be found, so a revised download that fixes or moves one fails
# here instead of passing through unexamined.
MODE_TYPOS = {
    ("dubray_ds942", "N13", "Abd_Hbl"): ("l.0", 1.0),
    ("dubray_ds942", "N13", "Abd_Opq"): ("l.0", 1.0),
    ("dubray_pp1793", "28693", "Opq"): ("`0", 0.0),
}


def _mode_cell(raw):
    """One plutonic mode cell as (abundance, qualifier), or None when blank.

    `TR` / `tr` is a trace -- below 0.5 vol% in DS 942, generally below 0.1 in
    PP 1793 (DS 454) -- and `<x` is present below x.  Both are left-censored
    presences: no number, qualifier kept.  A reported zero is a row with
    abundance 0, which the loader reads as looked-for-and-not-found.  A cell
    holding only whitespace is blank.
    """
    if pd.isna(raw):
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.lower() == "tr" or text.startswith("<"):
        return np.nan, text
    try:
        return float(text), None
    except ValueError:
        raise ValueError(f"unparsed plutonic mode cell {raw!r}") from None


def _plutonic_rocks(source, frame, ids, units, phases, chem, rock_class, form):
    """samples / modes rows for one plutonic workbook, one rock per `frame` row.

    Only rocks with at least one mode cell are emitted.  Every rock is
    crystalline throughout, so `total_crystals_pct` is 100, and neither source
    grades alteration per rock -- each already removed the rocks its chemical
    screens flag -- so `alteration_extent` is left blank.
    """
    samples, modes, typos_found = [], [], set()
    for (_, row), sid, unit in zip(frame.iterrows(), ids, units):
        rock_modes = []
        for col, (name, mgm) in phases.items():
            raw = row[col]
            typo = MODE_TYPOS.get((source, sid, col))
            if typo is not None:
                if str(raw).strip() != typo[0]:
                    raise ValueError(
                        f"{source} {sid} {col}: expected the typo {typo[0]!r}, "
                        f"found {raw!r} -- check the download"
                    )
                raw = typo[1]
                typos_found.add((source, sid, col))
            cell = _mode_cell(raw)
            if cell is None:
                continue
            rock_modes.append(dict(source=source, sample_id=sid, phase=name,
                                   magemin_phase=mgm, abundance_pct=cell[0],
                                   qualifier=cell[1], basis=PLUTONIC_BASIS))
        if not rock_modes:
            continue
        modes.extend(rock_modes)
        rock = dict(source=source, sample_id=sid, unit=unit,
                    rock_class=row[rock_class], intrusive_form=row[form],
                    total_crystals_pct=100.0, alteration_extent=np.nan,
                    mode_basis=PLUTONIC_BASIS)
        for src, dst in chem.items():
            rock[dst] = _num(pd.Series([row[src]])).iloc[0]
        samples.append(rock)
    missing = {k for k in MODE_TYPOS if k[0] == source} - typos_found
    if missing:
        raise ValueError(f"typo cells not found in the download: {sorted(missing)}")
    return pd.DataFrame(samples), pd.DataFrame(modes)


def parse_ds942():
    """DS 942's `Data` tab, the rocks that passed the source's screens.

    The `Alt` tab (chemically altered or off the 97-103 analytical-total
    window) and `PoorLocSamps` (no usable location) stay out, as they stay out
    of the source's own primary dataset.

    The unit is `Shorthand`, not `Intrs_Name`: `Intrs_Name` names the facies of
    one intrusion separately (the San Isabel Granite's coarse and medium
    facies, the Virginia Dale core and cap rocks), and the split and the
    bootstrap would then treat parts of one body as independent.

    Six `Field_Number`s each label two different rocks in different
    intrusions, so those take their unit as a suffix to stay unique.
    """
    frame = pd.read_excel(DATA / "dubray_ds942" / "1.4GaIntrsChemModData.xlsx",
                          sheet_name="Data")
    frame.columns = [c.strip() for c in frame.columns]
    frame = frame[frame[list(DS942_PHASES)].notna().any(axis=1)]
    ids = frame["Field_Number"].astype(str).str.strip()
    shared = ids.duplicated(keep=False)
    ids[shared] = ids[shared] + " (" + frame.loc[shared, "Shorthand"] + ")"
    samples, modes = _plutonic_rocks(
        "dubray_ds942", frame, ids, frame["Shorthand"], DS942_PHASES,
        DS942_CHEM, "Lithology", "Ign_Form",
    )
    if samples["sample_id"].duplicated().any():
        raise ValueError("DS 942 sample ids are not unique after the unit suffix")
    return samples, modes


def parse_pp1793():
    """PP 1793's Appendix 1, less the pluton averages interleaved with it.

    The appendix places an `AVERAGE` row after each pluton's samples; those
    are summaries, not rocks.  The unit is `intrs_name`, which names whole
    intrusions (its `alt name` neighbor lumps fifteen separate satellite
    stocks under `Satellitic`).  The rock name is `modal_lithology`; the
    `lithology` column is the source's unverified name and is blank wherever a
    mode exists.
    """
    frame = pd.read_excel(DATA / "dubray_pp1793" / "Appendix_1.xls", header=2)
    frame.columns = [c.strip() for c in frame.columns]
    ids = frame["sample _number"].astype(str).str.strip()
    keep = frame["sample _number"].notna() & (ids != "AVERAGE")
    frame, ids = frame[keep], ids[keep]
    samples, modes = _plutonic_rocks(
        "dubray_pp1793", frame, ids, frame["intrs_name"], PP1793_PHASES,
        PP1793_CHEM, "modal_lithology", "ign_form",
    )
    if samples["sample_id"].duplicated().any():
        raise ValueError("PP 1793 sample ids are not unique")
    return samples, modes


def main():
    s764, m764 = parse_dubray("dubray_ds764", "dubray_ds764/Appendix2.xls",
                              "dubray_ds764/Appendix3.xls", "TotXls")
    s1099, m1099 = parse_dubray("dubray_ds1099", "dubray_ds1099/Appendix02.xlsx",
                                "dubray_ds1099/Appendix03.xlsx", "TotXtls")
    mpc, pc_points, pc_skipped = parse_dubray_pointcounts(set(s1099.sample_id))
    s1099["n_points"] = s1099.sample_id.map(pc_points)
    smsh, mmsh = parse_msh()
    spin, mpin = parse_pinatubo()
    samples = pd.concat([s764, s1099, smsh, spin], ignore_index=True)
    modes = pd.concat([m764, m1099, mpc, mmsh, mpin], ignore_index=True)

    samples.to_csv(DATA / "samples.csv", index=False)
    modes.to_csv(DATA / "modes.csv", index=False)

    sio2 = samples.sio2
    fresh = samples.alteration_extent.isin(["1", "2", "1+"])
    print(f"samples: {len(samples)} ({dict(samples.source.value_counts())})")
    print(f"modes rows: {len(modes)}, phases: {sorted(modes.phase.unique())}")
    print(f"SiO2 {sio2.min():.0f}-{sio2.max():.0f}, n in 52-63: {((sio2>=52)&(sio2<=63)).sum()}")
    print(f"least-altered (AltExtnt 1-2): {fresh.sum()}/{len(samples)}")
    print(f"crystallinity known: {samples.total_crystals_pct.notna().sum()}")
    print(f"point-counted (appendix 10): {len(pc_points)} joined, "
          f"{pc_skipped} with no petrography row")

    s942, m942 = parse_ds942()
    s1793, m1793 = parse_pp1793()
    plutonic = pd.concat([s942, s1793], ignore_index=True)
    PLUTONIC_DIR.mkdir(exist_ok=True)
    plutonic.to_csv(PLUTONIC_DIR / "samples.csv", index=False)
    pd.concat([m942, m1793], ignore_index=True).to_csv(
        PLUTONIC_DIR / "modes.csv", index=False)
    print(f"plutonic samples: {len(plutonic)} ({dict(plutonic.source.value_counts())}), "
          f"SiO2 {plutonic.sio2.min():.0f}-{plutonic.sio2.max():.0f}")


if __name__ == "__main__":
    main()

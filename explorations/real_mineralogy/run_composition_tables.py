"""What the real-rock benchmark is made of, as tables the paper can cite.

The benchmark section needs a reader to see the funnel from what USGS published
down to what gets scored, and to see that "629 rocks" is a much smaller number
than it looks once rocks are grouped by the eruption they came from. Both are
easy to state wrongly from memory, so they are generated from the data instead:
markdown for the appendix, JSON for the numbers-as-macros path, regenerated
whenever the harvest changes.

Scoped to du Bray, which is the headline population (owner decision 2026-08-06).
The MSH and Pinatubo rocks are counted in one row and otherwise left out --
they are 11 hand-transcribed rocks from other arcs and belong in a separate
out-of-population line, not averaged into a table about this one.

Each benchmark is one entry in `BENCHMARKS`. The breakdowns by survey, unit,
split, silica and mineral are shared; each entry brings its own funnel,
measurement-method count and prose, since what a source published and how it
measured differ from source to source. The plutonic benchmark (`plutonic.py`)
is the second entry.

    conda run -n mineral python explorations/real_mineralogy/run_composition_tables.py
"""

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cnmineral.eval.real_rock import (
    BENCHMARK_DIR,
    CLEAN_GRADES,
    POINTCOUNT_BASIS,
    SPLIT_PATH,
    TRUTH_TO_GROUP,
    VISUAL_BASIS,
    parse_alteration_grade,
    preferred_basis,
)
from explorations.real_mineralogy.harvest import (
    DS942_PHASES,
    PLUTONIC_BASIS,
    PLUTONIC_DIR,
    PP1793_PHASES,
)
from explorations.real_mineralogy.plutonic import PLUTONIC_SPLIT_PATH, mafic_total_keys

RESULTS = Path(__file__).resolve().parent / "results"
DUBRAY = ("dubray_ds764", "dubray_ds1099")
SURVEY_LABEL = {
    "dubray_ds764": "DS 764, Bodie Hills CA/NV",
    "dubray_ds1099": "DS 1099, Tonopah / Divide / Goldfield NV",
}
# The two workbooks each publish a petrography appendix and a geochemistry
# appendix; a rock enters only where both describe it.
APPENDICES = {
    "dubray_ds764": ("dubray_ds764/Appendix2.xls", "dubray_ds764/Appendix3.xls"),
    "dubray_ds1099": ("dubray_ds1099/Appendix02.xlsx", "dubray_ds1099/Appendix03.xlsx"),
}
PLUTONIC_SURVEY_LABEL = {
    "dubray_ds942": "DS 942, 1.4 Ga A-type granitoids",
    "dubray_pp1793": "PP 1793, Boulder batholith MT",
}
SILICA_EDGES = (0, 52, 57, 63, 70, 100)
SILICA_LABELS = ("basalt (<52)", "basaltic andesite (52-57)", "andesite (57-63)",
                 "dacite (63-70)", "rhyolite (>70)")
# plutonic rock names follow the mode, not silica
PLUTONIC_SILICA_LABELS = ("<52", "52-57", "57-63", "63-70", ">70")
UNIT_SIZE_BINS = ((1, 1, "1 rock only"), (2, 5, "2-5"), (6, 20, "6-20"),
                  (21, 10**6, "21+"))
KEY_ORDER = ("off_vocabulary", "funnel", "by_survey", "by_method", "unit_structure",
             "units_total", "by_split", "by_silica", "minerals", "published_by_survey")


@dataclass(frozen=True, slots=True)
class Benchmark:
    """One benchmark's tables: where its rocks live and what is its own.

    `describe` returns the benchmark's own keys of `KEY_ORDER` (funnel,
    measurement method, off-vocabulary mass, published counts) and the
    evaluated rocks with `key`, `unit`, `subset` and `point_counted` set.
    """

    data_dir: Path
    stem: str
    survey_label: tuple[tuple[str, str], ...]
    silica_labels: tuple[str, ...]
    describe: Callable[[pd.DataFrame, pd.DataFrame], tuple[dict, pd.DataFrame]]
    render: Callable[[dict, pd.DataFrame, pd.DataFrame], str]


def _keys(frame: pd.DataFrame) -> pd.Series:
    return frame["source"] + "|" + frame["sample_id"].astype(str)


def _scoped(modes: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    scoped = modes.assign(key=_keys(modes))
    return scoped[scoped["key"].isin(set(frame["key"]))]


def published_counts() -> dict:
    """How many rocks each survey described, analyzed, and did both to.

    Read from the workbooks rather than from the harvested tables, because the
    interesting number is what was published and dropped -- which the harvest
    output no longer contains.
    """
    out = {}
    for source, (pet_path, geo_path) in APPENDICES.items():
        pet = pd.read_excel(BENCHMARK_DIR / pet_path, header=1)
        geo = pd.read_excel(BENCHMARK_DIR / geo_path, header=1)
        described = set(pet.dropna(subset=["Field_ID"])["Field_ID"].astype(str))
        analyzed = set(geo.dropna(subset=["Field_ID"])["Field_ID"].astype(str))
        out[source] = {
            "described": len(described),
            "analyzed": len(analyzed),
            "both": len(described & analyzed),
            "described_not_analyzed": len(described - analyzed),
            "analyzed_not_described": len(analyzed - described),
        }
    return out


def funnel(samples: pd.DataFrame, modes: pd.DataFrame, published: dict) -> dict:
    """Every rock lost between publication and the scored set, with its reason."""
    dubray = samples[samples["source"].isin(DUBRAY)].copy()
    dubray["key"] = _keys(dubray)
    mode_keys = set(_keys(modes))

    usable = dubray[dubray["sio2"].notna() & dubray["key"].isin(mode_keys)]
    grade = parse_alteration_grade(usable["alteration_extent"])
    fresh = usable[np.isin(grade, list(CLEAN_GRADES))]

    quantified = modes[modes["abundance_pct"].notna() & (modes["abundance_pct"] > 0)]
    scored = fresh[fresh["key"].isin(set(_keys(quantified)))]

    return {
        "described": sum(v["described"] for v in published.values()),
        "with_chemistry": sum(v["both"] for v in published.values()),
        "usable": len(usable),
        "least_altered": len(fresh),
        "scored": len(scored),
        "analyzed_never_described": sum(
            v["analyzed_not_described"] for v in published.values()
        ),
        "scored_keys": sorted(scored["key"]),
    }


def scored_frame(samples: pd.DataFrame, scored_keys: list[str]) -> pd.DataFrame:
    frame = samples.copy()
    frame["key"] = _keys(frame)
    frame = frame[frame["key"].isin(set(scored_keys))].copy()
    split = pd.read_csv(SPLIT_PATH)
    split_keys = _keys(split)
    # the split's units are canonical, the ones its dealer and the bootstrap group by
    frame["unit"] = frame["key"].map(dict(zip(split_keys, split["unit"])))
    frame["subset"] = frame["key"].map(dict(zip(split_keys, split["subset"])))
    return frame


def describe_eruption(samples: pd.DataFrame, modes: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    published = published_counts()
    counts = funnel(samples, modes, published)
    frame = scored_frame(samples, counts.pop("scored_keys"))
    counted = set(_keys(modes[modes["basis"] == POINTCOUNT_BASIS]))
    frame["point_counted"] = frame["key"].isin(counted)
    return {
        "off_vocabulary": off_vocabulary(_scoped(modes, frame)),
        "funnel": counts,
        "by_method": {
            "visual_estimate_only": int((~frame["point_counted"]).sum()),
            "point_count_available": int(frame["point_counted"].sum()),
        },
        "published_by_survey": published,
    }, frame


def plutonic_published_counts() -> dict:
    """Rows in each source's primary table, and how many carry a point count.

    Read from the workbooks, as `published_counts` is, with the harvest's own
    table cuts: DS 942's `Data` tab alone, PP 1793 less its pluton averages.
    """
    ds942 = pd.read_excel(BENCHMARK_DIR / "dubray_ds942" / "1.4GaIntrsChemModData.xlsx",
                          sheet_name="Data")
    ds942.columns = [c.strip() for c in ds942.columns]
    pp1793 = pd.read_excel(BENCHMARK_DIR / "dubray_pp1793" / "Appendix_1.xls", header=2)
    pp1793.columns = [c.strip() for c in pp1793.columns]
    ids = pp1793["sample _number"].astype(str).str.strip()
    pp1793 = pp1793[pp1793["sample _number"].notna() & (ids != "AVERAGE")]
    return {
        source: {
            "in_table": len(frame),
            "point_counted": int(frame[list(phases)].notna().any(axis=1).sum()),
        }
        for source, frame, phases in (("dubray_ds942", ds942, DS942_PHASES),
                                      ("dubray_pp1793", pp1793, PP1793_PHASES))
    }


def describe_plutonic(samples: pd.DataFrame, modes: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    published = plutonic_published_counts()
    chemistry = samples[samples["sio2"].notna()]
    frame = chemistry.assign(key=_keys(chemistry))
    frame = frame[~frame["key"].isin(mafic_total_keys(modes))].copy()

    split = pd.read_csv(PLUTONIC_SPLIT_PATH)
    split_keys = _keys(split)
    # the split's units are canonical, the ones its dealer and the bootstrap group by
    frame["unit"] = frame["key"].map(dict(zip(split_keys, split["unit"])))
    frame["subset"] = frame["key"].map(dict(zip(split_keys, split["subset"])))
    counted = set(_keys(modes[modes["basis"] == PLUTONIC_BASIS]))
    frame["point_counted"] = frame["key"].isin(counted)

    in_tables = sum(v["in_table"] for v in published.values())
    point_counted = sum(v["point_counted"] for v in published.values())
    return {
        "off_vocabulary": {
            "basis": "volume percent as published",
            "as_evaluated": _off_vocab_shares(_scoped(modes, frame)),
        },
        "funnel": {
            "in_tables": in_tables,
            "point_counted": point_counted,
            "with_chemistry": len(chemistry),
            "evaluated": len(frame),
            "never_point_counted": in_tables - point_counted,
        },
        "by_method": {"point_count_whole_rock": int(frame["point_counted"].sum())},
        "published_by_survey": published,
    }, frame


def breakdowns(frame: pd.DataFrame, modes: pd.DataFrame, benchmark: Benchmark) -> dict:
    sizes = frame["unit"].value_counts()
    unit_structure = []
    for lo, hi, label in UNIT_SIZE_BINS:
        picked = sizes[(sizes >= lo) & (sizes <= hi)]
        unit_structure.append({"size": label, "units": int(len(picked)),
                               "rocks": int(picked.sum())})

    split_rows = []
    for subset, group in frame.groupby("subset"):
        split_rows.append({
            "subset": subset,
            "rocks": int(len(group)),
            "units": int(group["unit"].nunique()),
            "point_counted": int(group["point_counted"].sum()),
        })

    silica = pd.cut(frame["sio2"], bins=list(SILICA_EDGES),
                    labels=list(benchmark.silica_labels)).value_counts()

    # one ruler per rock, the same one the scored numbers were measured against;
    # counting both would report a doubly-measured rock twice
    scored_modes = preferred_basis(_scoped(modes, frame), VISUAL_BASIS)
    minerals = []
    for phase, group in scored_modes[scored_modes["phase"] != "groundmass"].groupby("phase"):
        rocks = int(group["key"].nunique())
        numeric = int(group["abundance_pct"].notna().sum())
        minerals.append({"mineral": phase, "rocks": rocks, "with_a_number": numeric,
                         "trace_only": rocks - numeric})

    survey_label = dict(benchmark.survey_label)
    return {
        "by_survey": [{"survey": survey_label[s], "rocks": int(n)}
                      for s, n in frame["source"].value_counts().items()],
        "unit_structure": unit_structure,
        "units_total": int(frame["unit"].nunique()),
        "by_split": sorted(split_rows, key=lambda r: r["subset"]),
        "by_silica": [{"band": label, "rocks": int(silica.get(label, 0))}
                      for label in benchmark.silica_labels],
        "minerals": sorted(minerals, key=lambda r: -r["rocks"]),
    }


def _off_vocab_shares(frame: pd.DataFrame) -> dict:
    """Per-rock `1 - scoreable / reported` over one measurement basis."""
    quantified = frame[frame["abundance_pct"].notna() & (frame["abundance_pct"] > 0)]
    total = quantified.groupby("key")["abundance_pct"].sum()
    kept = (quantified[quantified["magemin_phase"].isin(TRUTH_TO_GROUP)]
            .groupby("key")["abundance_pct"].sum().reindex(total.index).fillna(0.0))
    share = (1.0 - kept / total).to_numpy()
    affected = share > 1e-9
    categories = (quantified[~quantified["magemin_phase"].isin(TRUTH_TO_GROUP)]
                  ["magemin_phase"].value_counts())
    return {
        "rocks": int(len(share)),
        "rocks_fully_in_vocabulary": int((~affected).sum()),
        "rocks_affected": int(affected.sum()),
        "share_among_affected_p50": (float(np.median(share[affected]))
                                     if affected.any() else None),
        "categories": {k: int(v) for k, v in categories.items()},
    }


def off_vocabulary(scoped_modes: pd.DataFrame) -> dict:
    """Measured mass the seven scored groups cannot represent, per rock.

    Same definition as the sedimentary cohort's (`load_andrews.py`),
    `1 - scoreable / reported`, so the two are comparable. `_truth_arrays`
    drops every row whose `magemin_phase` falls outside `TRUTH_TO_GROUP` and
    renormalizes what remains without recording what it dropped; this is that
    record.

    Reported per measurement basis because the answer differs between them:
    groundmass is itemized only by point counts, so the visual estimates the
    scored numbers are measured against carry no out-of-vocabulary mass at
    all. Shares are on the published volume-percent basis, against weight
    percent for XRD Rietveld, so the rock counts rather than the shares are
    what compares across cohorts.
    """
    return {
        "basis": "volume percent as published",
        "as_scored": _off_vocab_shares(preferred_basis(scoped_modes, VISUAL_BASIS)),
        "point_count_basis": _off_vocab_shares(
            scoped_modes[scoped_modes["basis"] == POINTCOUNT_BASIS]
        ),
    }


def _markdown_table(rows: list[dict], columns: list[str], total: str | None) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = "".join(
        "| " + " | ".join(str(row[c]) for c in columns) + " |\n" for row in rows
    )
    return head + body + (total or "")


def render_eruption(data: dict, samples: pd.DataFrame, modes: pd.DataFrame) -> str:
    # counted the same way every other number here is: rocks that carry both
    # chemistry and mode rows, not raw table rows
    other = samples[~samples["source"].isin(DUBRAY)]
    n_other = int((other["sio2"].notna()
                   & _keys(other).isin(set(_keys(modes)))).sum())
    f = data["funnel"]
    ov = data["off_vocabulary"]
    scored, pc = ov["as_scored"], ov["point_count_basis"]
    lines = [
        "# Benchmark composition (du Bray)",
        "",
        "Generated by `explorations/real_mineralogy/run_composition_tables.py`.",
        "Scoped to the du Bray surveys, which are the headline population; the",
        f"{n_other} hand-transcribed MSH and Pinatubo rocks are reported",
        "separately as an out-of-population check and are not in any table below.",
        "",
        "## Funnel: from what USGS published to what gets scored",
        "",
        "| Step | Rocks |",
        "|---|---|",
        f"| Petrographic descriptions published | {f['described']:,} |",
        f"| ...that also have whole-rock chemistry | {f['with_chemistry']} |",
        f"| ...that parse cleanly (chemistry + at least one mineral row) | {f['usable']} |",
        f"| ...least altered (alteration grade 1 or 2) | {f['least_altered']} |",
        f"| ...with at least one mineral carrying a number | **{f['scored']}** |",
        "",
        f"A further {f['analyzed_never_described']} rocks were chemically analyzed but",
        "never described petrographically, so they can never enter: no modal",
        "mineralogy to score against.",
        "",
        "## The scored set, by survey",
        "",
        _markdown_table(data["by_survey"], ["survey", "rocks"], None),
        "## Modal measurement method",
        "",
        "| Method | Rocks |",
        "|---|---|",
        f"| Visual phenocryst estimate only | {data['by_method']['visual_estimate_only']} |",
        f"| Microscope point count also available | {data['by_method']['point_count_available']} |",
        "",
        "## Unit structure",
        "",
        "A geological unit is the named rock body a sample came from. Same unit",
        "means same eruption means near-identical chemistry, so splits and",
        "bootstrap intervals cluster by unit rather than by rock.",
        "",
        _markdown_table(data["unit_structure"], ["size", "units", "rocks"], None),
        "## Split, in units as well as rocks",
        "",
        _markdown_table(data["by_split"], ["subset", "rocks", "units", "point_counted"], None),
        "## Composition of the scored set",
        "",
        _markdown_table(data["by_silica"], ["band", "rocks"], None),
        "All volcanic; no plutonic rocks, effectively no basalt.",
        "",
        "## Reported minerals, over the scored set",
        "",
        "A rock measured twice is counted once, under the visual estimate that",
        "the scored numbers were measured against.",
        "",
        _markdown_table(data["minerals"], ["mineral", "rocks", "with_a_number", "trace_only"], None),
        "## Measured mass the seven groups cannot represent",
        "",
        f"As scored, all {scored['rocks']} rocks report their whole measured",
        "assemblage inside the seven groups, so nothing is discarded before",
        "renormalizing. The one out-of-vocabulary category anywhere in the",
        f"benchmark is {'/'.join(pc['categories']) or 'none'}, itemized only by",
        f"point counts: {pc['rocks_affected']} of the {pc['rocks']} point-counted",
        f"rocks report it, at a median {pc['share_among_affected_p50']:.0%} of the",
        "rock. Groundmass is quenched melt, which the simulated labels also",
        "exclude before renormalizing, so both sides are crystals only either",
        f"way. Shares are on the {ov['basis']}.",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_plutonic(data: dict, samples: pd.DataFrame, modes: pd.DataFrame) -> str:
    f = data["funnel"]
    ov = data["off_vocabulary"]["as_evaluated"]
    categories = ", ".join(f"`{k}` {v}" for k, v in ov["categories"].items())
    lines = [
        "# Benchmark composition (plutonic)",
        "",
        "Generated by `explorations/real_mineralogy/run_composition_tables.py`.",
        "Whole-rock point counts from du Bray DS 942 and PP 1793, over the rocks",
        "`plutonic.py` evaluates.",
        "",
        "## Funnel: from the sources' tables to the evaluated set",
        "",
        "| Step | Rocks |",
        "|---|---|",
        f"| Rows in the sources' primary tables | {f['in_tables']:,} |",
        f"| ...with a point count | {f['point_counted']} |",
        f"| ...that also have whole-rock chemistry | {f['with_chemistry']} |",
        f"| ...with the mafic categories counted apart | **{f['evaluated']}** |",
        "",
        f"The other {f['never_point_counted']:,} rows carry no point count, so they",
        "can never enter. A rock reporting only a mafic total stays in the split",
        "but outside the evaluated set. Where a source counts the Fe-Ti oxides",
        "with the accessories, the rock is evaluated with the oxides marked",
        "unmeasured (#83, 2026-10-05).",
        "",
        "## The evaluated set, by source",
        "",
        _markdown_table(data["by_survey"], ["survey", "rocks"], None),
        "## Modal measurement method",
        "",
        "| Method | Rocks |",
        "|---|---|",
        f"| Whole-rock point count | {data['by_method']['point_count_whole_rock']} |",
        "",
        "## Unit structure",
        "",
        "A unit is the intrusion a rock came from (DS 942 `Shorthand`, PP 1793",
        "`intrs_name`). Rocks of one intrusion are related, so the split and",
        "bootstrap intervals cluster by intrusion rather than by rock.",
        "",
        _markdown_table(data["unit_structure"], ["size", "units", "rocks"], None),
        "## Split, in intrusions as well as rocks",
        "",
        _markdown_table(data["by_split"], ["subset", "rocks", "units"], None),
        "## Composition of the evaluated set",
        "",
        "SiO2, wt%.",
        "",
        _markdown_table(data["by_silica"], ["band", "rocks"], None),
        "## Reported minerals, over the evaluated set",
        "",
        _markdown_table(data["minerals"], ["mineral", "rocks", "with_a_number", "trace_only"], None),
        "## Measured mass the seven groups cannot represent",
        "",
        f"{ov['rocks_affected']} of the {ov['rocks']} evaluated rocks report mass outside the",
        f"seven groups, at a median {ov['share_among_affected_p50']:.1%} of the rock. Rocks",
        f"reporting each category: {categories}. The evaluation drops that",
        "mass and renormalizes what remains, so a measured share is a share of the",
        "seven groups. Shares are volume percent as published.",
        "",
    ]
    return "\n".join(lines) + "\n"


BENCHMARKS = {
    "eruption_v2": Benchmark(
        data_dir=BENCHMARK_DIR,
        stem="benchmark_composition",
        survey_label=tuple(SURVEY_LABEL.items()),
        silica_labels=SILICA_LABELS,
        describe=describe_eruption,
        render=render_eruption,
    ),
    "plutonic": Benchmark(
        data_dir=PLUTONIC_DIR,
        stem="plutonic_composition",
        survey_label=tuple(PLUTONIC_SURVEY_LABEL.items()),
        silica_labels=PLUTONIC_SILICA_LABELS,
        describe=describe_plutonic,
        render=render_plutonic,
    ),
}


def compose(benchmark: Benchmark) -> tuple[dict, str, pd.DataFrame]:
    samples = pd.read_csv(benchmark.data_dir / "samples.csv")
    modes = pd.read_csv(benchmark.data_dir / "modes.csv")
    own, frame = benchmark.describe(samples, modes)
    merged = {**own, **breakdowns(frame, modes, benchmark)}
    data = {k: merged[k] for k in KEY_ORDER}
    return data, benchmark.render(data, samples, modes), frame


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, benchmark in BENCHMARKS.items():
        data, markdown, frame = compose(benchmark)
        (RESULTS / f"{benchmark.stem}.json").write_text(json.dumps(data, indent=2))
        (RESULTS / f"{benchmark.stem}.md").write_text(markdown)
        print(f"{name}: {len(frame)} rocks across {data['units_total']} units; "
              f"wrote {RESULTS / benchmark.stem}.{{json,md}}")


if __name__ == "__main__":
    main()

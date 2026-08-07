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

    conda run -n alignn_fix python explorations/real_mineralogy/run_composition_tables.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cnmineral.eval.real_rock import (
    BENCHMARK_DIR,
    CLEAN_GRADES,
    POINTCOUNT_BASIS,
    SPLIT_PATH,
    VISUAL_BASIS,
    parse_alteration_grade,
    preferred_basis,
)

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
SILICA_EDGES = (0, 52, 57, 63, 70, 100)
SILICA_LABELS = ("basalt (<52)", "basaltic andesite (52-57)", "andesite (57-63)",
                 "dacite (63-70)", "rhyolite (>70)")
UNIT_SIZE_BINS = ((1, 1, "1 rock only"), (2, 5, "2-5"), (6, 20, "6-20"),
                  (21, 10**6, "21+"))


def _keys(frame: pd.DataFrame) -> pd.Series:
    return frame["source"] + "|" + frame["sample_id"].astype(str)


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
    frame["subset"] = frame["key"].map(dict(zip(_keys(split), split["subset"])))
    return frame


def tables(frame: pd.DataFrame, modes: pd.DataFrame, funnel_counts: dict) -> dict:
    counted = set(_keys(modes[modes["basis"] == POINTCOUNT_BASIS]))
    frame = frame.assign(point_counted=frame["key"].isin(counted))
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
                    labels=list(SILICA_LABELS)).value_counts()

    # one ruler per rock, the same one the scored numbers were measured against;
    # counting both would report a doubly-measured rock twice
    scored_modes = modes.assign(key=_keys(modes))
    scored_modes = preferred_basis(scored_modes, VISUAL_BASIS)
    scored_modes = scored_modes[scored_modes["key"].isin(set(frame["key"]))]
    minerals = []
    for phase, group in scored_modes[scored_modes["phase"] != "groundmass"].groupby("phase"):
        rocks = int(group["key"].nunique())
        numeric = int(group["abundance_pct"].notna().sum())
        minerals.append({"mineral": phase, "rocks": rocks, "with_a_number": numeric,
                         "trace_only": rocks - numeric})

    return {
        "funnel": {k: v for k, v in funnel_counts.items() if k != "scored_keys"},
        "by_survey": [{"survey": SURVEY_LABEL[s], "rocks": int(n)}
                      for s, n in frame["source"].value_counts().items()],
        "by_method": {
            "visual_estimate_only": int((~frame["point_counted"]).sum()),
            "point_count_available": int(frame["point_counted"].sum()),
        },
        "unit_structure": unit_structure,
        "units_total": int(frame["unit"].nunique()),
        "by_split": sorted(split_rows, key=lambda r: r["subset"]),
        "by_silica": [{"band": label, "rocks": int(silica.get(label, 0))}
                      for label in SILICA_LABELS],
        "minerals": sorted(minerals, key=lambda r: -r["rocks"]),
    }


def _markdown_table(rows: list[dict], columns: list[str], total: str | None) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = "".join(
        "| " + " | ".join(str(row[c]) for c in columns) + " |\n" for row in rows
    )
    return head + body + (total or "")


def render(data: dict, n_other: int) -> str:
    f = data["funnel"]
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
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    samples = pd.read_csv(BENCHMARK_DIR / "samples.csv")
    modes = pd.read_csv(BENCHMARK_DIR / "modes.csv")
    published = published_counts()
    counts = funnel(samples, modes, published)
    frame = scored_frame(samples, counts["scored_keys"])
    data = tables(frame, modes, counts)
    data["published_by_survey"] = published
    # counted the same way every other number here is: rocks that carry both
    # chemistry and mode rows, not raw table rows
    other = samples[~samples["source"].isin(DUBRAY)]
    n_other = int((other["sio2"].notna()
                   & _keys(other).isin(set(_keys(modes)))).sum())

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "benchmark_composition.json").write_text(json.dumps(data, indent=2))
    (RESULTS / "benchmark_composition.md").write_text(render(data, n_other))
    print(f"scored {data['funnel']['scored']} du Bray rocks across "
          f"{data['units_total']} units; wrote {RESULTS}")


if __name__ == "__main__":
    main()

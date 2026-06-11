"""Cross-encoder probe-HPO calibration aggregation.

Reads N probe-HPO summary JSONs (produced by
`cnmineral.eval.probe.hpo`, one per encoder), tabulates each HP's
distribution across encoders, and emits:

  1. `calibration_report.md` -- a per-HP convergence table with
     mean / median / range / coefficient of variation + a converged/
     scattered flag.
  2. `calibrated_probe.yaml` -- the consensus / median value per HP,
     ready to feed into a future probe run as the calibrated default.

Convergence rules:

  - Continuous HPs (`dropout`, `label_smoothing`): coefficient of
    variation (std / mean) across encoders; CV < 0.5 = converged.
  - Log-sampled HPs (`lr`, `weight_decay`): log10 range across
    encoders; range < 0.5 decade = converged.
  - Categorical HPs (`hidden_dim`, `n_hidden_layers`,
    `l2_normalize_inputs`, `use_cosine_schedule`): require >= N-1 of
    N encoders to agree on the same value.
"""

import json
import math
import statistics
from collections import Counter
from pathlib import Path

import yaml

CONTINUOUS_HPS: set[str] = {"dropout", "label_smoothing"}
LOG_CONTINUOUS_HPS: set[str] = {"lr", "weight_decay"}
CATEGORICAL_HPS: set[str] = {
    "hidden_dim",
    "n_hidden_layers",
    "l2_normalize_inputs",
    "use_cosine_schedule",
}

CV_THRESHOLD: float = 0.5
LOG_RANGE_THRESHOLD: float = 0.5  # decades; max-min in log10 space


def _cv(values: list[float]) -> float | None:
    """Coefficient of variation = std / |mean|.  None when mean ~ 0."""
    if not values:
        return None
    m = statistics.fmean(values)
    if abs(m) < 1e-12:
        return None
    s = statistics.pstdev(values)
    return s / abs(m)


def _log_range(values: list[float]) -> float | None:
    """log10(max) - log10(min) for positive values.  Independent of
    the absolute magnitude; only reflects spread."""
    if not values or any(v <= 0 for v in values):
        return None
    return math.log10(max(values)) - math.log10(min(values))


def _categorical_consensus(values: list) -> tuple[object, int, int]:
    """Most-common value, its count, and total count."""
    counts = Counter(values)
    (most_common, n_most), = counts.most_common(1)
    return most_common, n_most, len(values)


def load_summaries(paths: list[Path]) -> list[dict]:
    """Read N summary JSONs, drop ones with no completed trials."""
    out = []
    for p in paths:
        with open(p) as f:
            summary = json.load(f)
        if not summary.get("best_params"):
            print(f"  skipping {p.name}: no best_params (0 completed trials?)")
            continue
        out.append(summary)
    return out


def build_report(summaries: list[dict]) -> dict:
    """Aggregate best_params across encoders into a per-HP report."""
    if not summaries:
        raise SystemExit("no usable summaries; nothing to calibrate")
    encoders = [s["encoder_tag"] for s in summaries]
    per_hp: dict[str, list] = {}
    for s in summaries:
        for k, v in s["best_params"].items():
            per_hp.setdefault(k, []).append(v)

    report = {
        "n_encoders": len(summaries),
        "encoders": encoders,
        "hyperparameters": {},
        "consensus": {},
    }

    for hp, values in per_hp.items():
        entry: dict = {"values": values}
        if hp in CONTINUOUS_HPS:
            cv = _cv([float(v) for v in values])
            entry["type"] = "continuous"
            entry["mean"] = statistics.fmean(values)
            entry["median"] = statistics.median(values)
            entry["range"] = [min(values), max(values)]
            entry["cv"] = cv
            entry["converged"] = cv is not None and cv < CV_THRESHOLD
            report["consensus"][hp] = entry["median"]
        elif hp in LOG_CONTINUOUS_HPS:
            lr = _log_range([float(v) for v in values])
            entry["type"] = "log_continuous"
            entry["geometric_mean"] = (
                math.exp(statistics.fmean(math.log(v) for v in values))
                if all(v > 0 for v in values)
                else None
            )
            entry["median"] = statistics.median(values)
            entry["range"] = [min(values), max(values)]
            entry["log_range_decades"] = lr
            entry["converged"] = lr is not None and lr < LOG_RANGE_THRESHOLD
            report["consensus"][hp] = (
                entry["geometric_mean"]
                if entry["geometric_mean"] is not None
                else entry["median"]
            )
        elif hp in CATEGORICAL_HPS:
            most, n_most, n_total = _categorical_consensus(values)
            entry["type"] = "categorical"
            entry["mode"] = most
            entry["mode_count"] = n_most
            entry["total"] = n_total
            entry["converged"] = n_most >= max(1, n_total - 1)
            report["consensus"][hp] = most
        else:
            entry["type"] = "unknown"
            entry["converged"] = None
            report["consensus"][hp] = values[0]

        report["hyperparameters"][hp] = entry

    return report


def write_report_markdown(report: dict, path: Path) -> None:
    """Per-HP table in the order the HPs appear in the report."""
    lines: list[str] = []
    lines.append("# Probe-HPO cross-encoder calibration\n")
    lines.append(f"N encoders: {report['n_encoders']}")
    lines.append("Encoders compared:")
    for tag in report["encoders"]:
        lines.append(f"  - {tag}")
    lines.append("")
    lines.append("| HP | type | values | summary | converged? |")
    lines.append("|---|---|---|---|---|")
    for hp, entry in report["hyperparameters"].items():
        values_str = ", ".join(repr(v) for v in entry["values"])
        if entry["type"] == "continuous":
            summary = (
                f"mean={entry['mean']:.4g}, median={entry['median']:.4g}, "
                f"CV={entry['cv']:.3f}"
                if entry["cv"] is not None
                else f"median={entry['median']:.4g}, CV=n/a"
            )
        elif entry["type"] == "log_continuous":
            summary = (
                f"geom_mean={entry['geometric_mean']:.4g}, "
                f"median={entry['median']:.4g}, "
                f"log_range={entry['log_range_decades']:.2f} decades"
                if entry["log_range_decades"] is not None
                else f"median={entry['median']:.4g}, log_range=n/a"
            )
        elif entry["type"] == "categorical":
            summary = (
                f"mode={entry['mode']!r} ({entry['mode_count']}/{entry['total']})"
            )
        else:
            summary = "(unknown HP type; no judgement)"
        flag = (
            "yes" if entry["converged"] is True
            else ("no" if entry["converged"] is False else "n/a")
        )
        lines.append(f"| `{hp}` | {entry['type']} | {values_str} | {summary} | {flag} |")

    lines.append("")
    lines.append("## Consensus / calibrated_probe.yaml values\n")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(report["consensus"], sort_keys=True).rstrip())
    lines.append("```")
    lines.append("")
    lines.append("**Reading the table:**")
    lines.append(
        f"- continuous HPs: converged when CV < {CV_THRESHOLD}"
    )
    lines.append(
        f"- log-continuous HPs: converged when log10-range < "
        f"{LOG_RANGE_THRESHOLD} decade"
    )
    lines.append(
        "- categorical HPs: converged when at least N-1 of N encoders agree"
    )
    lines.append(
        "- scattered HPs are signals to either narrow the probe-HPO search"
        " space or accept that probe accuracy comparisons across encoders"
        " carry HP noise"
    )

    path.write_text("\n".join(lines))


def write_calibrated_yaml(report: dict, path: Path) -> None:
    """Emit the consensus HPs as a YAML the next probe run can consume."""
    payload = {
        "_provenance": {
            "n_encoders": report["n_encoders"],
            "encoders": report["encoders"],
            "convergence_thresholds": {
                "continuous_cv": CV_THRESHOLD,
                "log_continuous_range_decades": LOG_RANGE_THRESHOLD,
                "categorical_agreement": "N-1 of N",
            },
        },
        "calibrated_probe": report["consensus"],
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=True)


def run_calibration(summary_paths: list[Path], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(summary_paths)
    report = build_report(summaries)
    md_path = output_dir / "calibration_report.md"
    yaml_path = output_dir / "calibrated_probe.yaml"
    write_report_markdown(report, md_path)
    write_calibrated_yaml(report, yaml_path)
    print(f"\nreport -> {md_path}")
    print(f"calibrated -> {yaml_path}")
    return report

"""Build the static technique index.

Reads docs/methodology/techniques.yaml, scans src/cnmineral/, scripts/,
configs/ for each technique's substring patterns, and writes the matching
file paths to docs/methodology/technique_index.json.  Re-runnable; the
output is regenerated from scratch each invocation so stale entries
cannot survive.

    conda run -n mineral python scripts/build/build_technique_index.py

Output JSON shape (grouped by technique type):

    {
      "<type>": {
        "<technique_name>": [
          "configs/probe/probe_hpo.yaml",
          "scripts/training/pretrain.py",
          "src/cnmineral/eval/probes.py"
        ],
        ...
      },
      ...
    }

Line numbers and match snippets are deliberately omitted -- they churn
on every edit, which the project rule "reference code by symbol, not
line number" already addresses.  The file list is enough to grep from.

Precision-first: patterns in techniques.yaml should be distinctive
symbol names, not generic words.
"""

import json
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TECHNIQUES_YAML = REPO_ROOT / "docs" / "methodology" / "techniques.yaml"
OUTPUT_JSON = REPO_ROOT / "docs" / "methodology" / "technique_index.json"

SCAN_ROOTS = ("src/cnmineral", "scripts", "configs")
SCAN_SUFFIXES = (".py", ".yaml", ".yml")
SKIP_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints"}
SKIP_FILES = {Path(__file__).resolve()}  # don't scan this script


def iter_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        if p.suffix not in SCAN_SUFFIXES:
            continue
        if p.resolve() in SKIP_FILES:
            continue
        out.append(p)
    return sorted(out)


def file_matches_any(path: Path, patterns: list[str]) -> bool:
    """True if `path` contains any of `patterns` as a substring."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return any(pat in text for pat in patterns)


def build_index(techniques: list[dict], files: list[Path]) -> dict:
    grouped: dict[str, dict[str, list[str]]] = {}
    for tech in techniques:
        name = tech["name"]
        ttype = tech["type"]
        patterns = tech["patterns"]
        hits = sorted(
            f.relative_to(REPO_ROOT).as_posix()
            for f in files
            if file_matches_any(f, patterns)
        )
        grouped.setdefault(ttype, {})[name] = hits
    return {
        ttype: dict(sorted(by_name.items()))
        for ttype, by_name in sorted(grouped.items())
    }


def print_summary(index: dict) -> None:
    n_types = len(index)
    counts: list[tuple[str, str, int]] = [
        (ttype, name, len(files))
        for ttype, by_name in index.items()
        for name, files in by_name.items()
    ]
    n_techniques = len(counts)
    zero_hit = [(ttype, name) for ttype, name, n in counts if n == 0]

    print(f"types: {n_types}  techniques: {n_techniques}")
    print(f"total file hits: {sum(n for _, _, n in counts)}")
    print("per-type counts:")
    for ttype, by_name in index.items():
        n = sum(len(files) for files in by_name.values())
        print(f"  {n:4d}  {ttype}  ({len(by_name)} techniques)")
    print("top hits (techniques):")
    for ttype, name, n in sorted(counts, key=lambda x: -x[2])[:5]:
        print(f"  {n:4d}  {name}  ({ttype})")
    if zero_hit:
        print(f"zero hits ({len(zero_hit)}):")
        for ttype, name in zero_hit:
            print(f"  {name}  ({ttype})")
    else:
        print("zero hits: none")


def main() -> None:
    if not TECHNIQUES_YAML.exists():
        sys.exit(f"techniques.yaml not found at {TECHNIQUES_YAML}")

    with open(TECHNIQUES_YAML) as f:
        doc = yaml.safe_load(f)
    techniques = doc["techniques"]

    files: list[Path] = []
    for root in SCAN_ROOTS:
        files.extend(iter_files(REPO_ROOT / root))

    index = build_index(techniques, files)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    print_summary(index)
    print(f"\nwrote {OUTPUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

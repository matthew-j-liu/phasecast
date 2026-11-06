"""Stage the public data release: the simulated label corpus as one zip, the
pretraining set as one parquet, and the record README.

    conda run -n mineral python scripts/build/build_zenodo_release.py --out ../phasecast_zenodo

The pretraining set is rebuilt from the production encoder config and its
split assignment is checked against the test indices stamped in the
checkpoint, so the file is the set the encoder saw, not a re-draw.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.model_selection import train_test_split

from cnmineral.data.bulk_chemistry import BulkChemistryDataset, _load_raw_dataframe
from cnmineral.training.splits import _strat_labels_for_indices, reserve_test

REPO_ROOT = Path(__file__).resolve().parents[2]
ENCODER_CONFIG = REPO_ROOT / "configs/pretrain/pretrain_allrock_redox.yaml"
ENCODER_CHECKPOINT = REPO_ROOT / "checkpoints/pretrain_allrock_redox_seed0/epoch_best.pt"
CORPUS_DIR = REPO_ROOT / "data/corpus"
CORPUS_SPLITS = ("train", "valtest")
CORPUS_ROSTERS = ("train_bulks.csv", "valtest_bulks.csv")
RECORD_README = REPO_ROOT / "docs/data_release.md"


def build_pretraining_set(out: Path) -> dict:
    config = yaml.safe_load(ENCODER_CONFIG.read_text())
    dataset = BulkChemistryDataset(**config["data_kwargs"])
    splits = reserve_test(dataset, config)
    train_idx, val_idx = train_test_split(
        splits["trainval"], test_size=config["val_split"], random_state=config["seed"],
        stratify=_strat_labels_for_indices(dataset, splits["trainval"]),
    )
    stamped = torch.load(ENCODER_CHECKPOINT, map_location="cpu", weights_only=False)
    if sorted(stamped["test_indices"]) != sorted(splits["test"]):
        raise RuntimeError("rebuilt test split differs from the checkpoint's test_indices")

    split = np.empty(len(dataset), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[splits["test"]] = "test"

    raw = _load_raw_dataframe(Path(config["data_kwargs"]["csv_path"]))
    raw = raw.drop_duplicates("sample_id").set_index("sample_id")
    table = raw.loc[dataset.sample_ids].reset_index()
    table.insert(1, "split", split)
    table.to_parquet(out / "pretraining_set.parquet", index=False)
    counts = table["split"].value_counts().to_dict()
    print(f"pretraining set: {len(table)} rows, {counts}")
    return counts


def build_corpus_zip(out: Path) -> dict:
    zip_path = out / "simulated_corpus.zip"
    rows = {}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in CORPUS_ROSTERS:
            zf.write(CORPUS_DIR / name, f"data/corpus/{name}")
        for split in CORPUS_SPLITS:
            for path in sorted((CORPUS_DIR / split).rglob("*")):
                if path.is_dir() or "work" in path.relative_to(CORPUS_DIR).parts:
                    continue
                zf.write(path, str(Path("data/corpus") / path.relative_to(CORPUS_DIR)))
                if path.suffix == ".parquet":
                    kind = path.stem.split("_")[-1]
                    rows[(split, kind)] = rows.get((split, kind), 0) + pq.read_metadata(path).num_rows
    for (split, kind), n in sorted(rows.items()):
        print(f"{split} {kind}: {n} rows")
    print(f"zip: {zip_path.stat().st_size / 1e9:.2f} GB")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-pretraining-set", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    os.chdir(REPO_ROOT)
    if not args.skip_pretraining_set:
        build_pretraining_set(out)
    build_corpus_zip(out)
    shutil.copy(RECORD_README, out / "README.md")
    print(f"staged in {out}")


if __name__ == "__main__":
    main()

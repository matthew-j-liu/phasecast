"""
Train/val/test split logic.

`reserve_test` is the outer 80/20 that peels off the test indices
before any training touches the data, stratified by the dataset's
stratification labels (`strat_labels` where the dataset carries them --
qap_name with a rock_group fallback -- else `qap_names`).  It rejects a
corpus with null labels: such corpora cannot be stratified and belong
on the inference path (`extract_latents`), not here.

An optional `test_roster_path` config key pins named `sample_id` rows
into the test split before the stratified draw -- the split-inheritance
mechanism that keeps a re-pretrained encoder scoreable on the previous
encoder's test rows.

The inner train/val 80/20 lives in `ssl_pretrain.py::run_pretraining` because
it depends on per-trial config and runs many times during HPO.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def reserve_test(dataset: Dataset, config: dict) -> dict[str, list[int]]:
    """Outer 80/20 split: peel off the test set, keep the trainval pool.

    The test indices are reserved before any training touches the data.
    `run_pretraining` receives `splits["trainval"]` and structurally
    cannot reach the held-out test rows.  The test evaluation happens
    once at the very end via `evaluate_on_test`.

    Splits are stratified by `dataset.strat_labels` when present, else
    `dataset.qap_names`, so per-class proportions are preserved on both
    sides.

    Parameters
    ----------
    dataset
        Already-constructed dataset; ownership lives at the entry
        point (`run_single` / `run_hpo`) so the corpus is built once
        per invocation, not once per call into this helper.
    config
        Required: `seed`.  Optional: `test_split` (default 0.20);
        `test_roster_path`, a one-column `sample_id` CSV whose rows --
        where they survive the corpus filters -- are pinned into the
        test split, with the stratified draw topping test up to the
        `test_split` fraction from the remaining rows.

    Returns
    -------
    dict[str, list[int]]
        `{"trainval": [int, ...], "test": [int, ...]}` indexing into
        `dataset`.

    Raises
    ------
    ValueError
        If any row's stratification label is None (a no-label corpus).
        Such corpora cannot be stratified and belong on the inference
        path, not in `reserve_test`.  Also if a test roster matches no
        corpus row, or combines with `test_split=0`.
    """
    n = len(dataset)
    test_split = config.get("test_split", 0.20)
    seed = config.get("seed", 0)
    roster_path = config.get("test_roster_path")

    # stratify: hand a class label of every row and sklearn will balance the split
    stratify_labels = _strat_labels_for_indices(dataset, range(n))
    n_null = sum(label is None for label in stratify_labels)
    if n_null:
        raise ValueError(
            f"reserve_test needs stratifiable labels, but {n_null} of "
            f"{n} rows are None. No-label corpora (require_qap_name=False "
            f"without strat_labels) are an inference path: run the trained "
            f"encoder over all indices (extract_latents), don't split."
        )

    # test_split=0 short-circuit -- sklearn rejects test_size=0, and the
    # downstream evaluate_on_test handles an empty test_indices already.
    if test_split == 0:
        if roster_path:
            raise ValueError(
                "test_roster_path with test_split=0 is contradictory: "
                "pinned rows would land in trainval"
            )
        return {"trainval": list(range(n)), "test": []}

    if roster_path:
        roster = pd.read_csv(roster_path)["sample_id"].to_numpy(dtype=np.int64)
        pinned_mask = np.isin(dataset.sample_ids, roster)
        pinned_idx = np.flatnonzero(pinned_mask).tolist()
        if not pinned_idx:
            raise ValueError(
                f"test roster {roster_path} matches no corpus row; "
                f"wrong roster or wrong corpus"
            )
        remaining = np.flatnonzero(~pinned_mask).tolist()
        top_up = round(n * test_split) - len(pinned_idx)
        if top_up <= 0:
            trainval_idx, test_idx = remaining, pinned_idx
        else:
            trainval_idx, extra_idx = train_test_split(
                remaining, test_size=top_up, random_state=seed,
                stratify=_strat_labels_for_indices(dataset, remaining),
            )
            test_idx = pinned_idx + extra_idx
    else:
        trainval_idx, test_idx = train_test_split(
            list(range(n)), test_size=test_split, random_state=seed,
            stratify=stratify_labels,
        )
    if not trainval_idx:
        raise ValueError(
            f"reserve_test produced empty trainval pool: n={n}, "
            f"test_split={test_split}"
        )
    return {"trainval": trainval_idx, "test": test_idx}


def _strat_labels_for_indices(dataset: Dataset, indices) -> list:
    """Stratification labels indexed by `indices`.

    Prefers `dataset.strat_labels` (composite qap-else-rock_group) and
    falls back to `dataset.qap_names` for datasets that don't carry the
    composite, so `train_test_split(stratify=...)` can preserve
    per-class proportions either way.
    """
    labels = getattr(dataset, "strat_labels", None)
    if labels is None:
        labels = dataset.qap_names
    return [labels[i] for i in indices]

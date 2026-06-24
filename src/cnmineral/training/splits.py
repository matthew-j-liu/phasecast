"""
Train/val/test split logic.

`reserve_test` is the outer 80/20 that peels off the test indices
before any training touches the data, stratified by `qap_name`.  It is
the labeled-training door: it requires every row to carry a stratifiable
`qap_name` and rejects a corpus with null labels.  No-label corpora
(`require_qap_name=False`) are an inference-only path -- run the frozen
encoder over all rows (`extract_latents`), don't split.

The inner train/val 80/20 lives in `ssl_pretrain.py::run_pretraining` because
it depends on per-trial config and runs many times during HPO.
"""

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def reserve_test(dataset: Dataset, config: dict) -> dict[str, list[int]]:
    """Outer 80/20 split: peel off the test set, keep the trainval pool.

    The test indices are reserved before any training touches the data.
    `run_pretraining` receives `splits["trainval"]` and structurally
    cannot reach the held-out test rows.  The test evaluation happens
    once at the very end via `evaluate_on_test`.

    Splits are stratified by `dataset.qap_names` so per-class
    proportions are preserved on both sides.

    Parameters
    ----------
    dataset
        Already-constructed dataset; ownership lives at the entry
        point (`run_single` / `run_hpo`) so the corpus is built once
        per invocation, not once per call into this helper.
    config
        Required: `seed`.  Optional: `test_split` (default 0.20).

    Returns
    -------
    dict[str, list[int]]
        `{"trainval": [int, ...], "test": [int, ...]}` indexing into
        `dataset`.

    Raises
    ------
    ValueError
        If any row's `qap_name` is None (a no-label corpus built with
        `require_qap_name=False`).  Such corpora cannot be stratified and
        belong on the inference path, not in `reserve_test`.
    """
    n = len(dataset)
    test_split = config.get("test_split", 0.20)
    seed = config.get("seed", 0)

    # stratify: hand a class label of every row and sklearn will balance the split
    stratify_labels = _qap_labels_for_indices(dataset, range(n))
    n_null = sum(label is None for label in stratify_labels)
    if n_null:
        raise ValueError(
            f"reserve_test needs stratifiable qap_name labels, but {n_null} of "
            f"{n} rows are None. No-label corpora (require_qap_name=False) are an "
            f"inference path: run the trained encoder over all indices "
            f"(extract_latents), don't split. Build with require_qap_name=True to "
            f"train."
        )

    # test_split=0 short-circuit -- sklearn rejects test_size=0, and the
    # downstream evaluate_on_test handles an empty test_indices already.
    if test_split == 0:
        return {"trainval": list(range(n)), "test": []}

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


def _qap_labels_for_indices(dataset: Dataset, indices) -> list:
    """Stratification labels indexed by `indices`.

    Returns `dataset.qap_names[i]` for each i in `indices` so
    `train_test_split(stratify=...)` can preserve per-class proportions.
    """
    return [dataset.qap_names[i] for i in indices]

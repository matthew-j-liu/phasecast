"""
Masked-element reconstruction MSE on the held-out test split.

`evaluate_on_test` is the final step of a training run: after
`reserve_test` peels off the test indices and `run_pretraining`
trains on the rest, the entry points (`run_single` / `run_hpo`)
load the headline checkpoint and call this once to score the model
on rows it structurally never saw.  Checkpoint persistence itself
lives in `cnmineral.training.checkpoint::Checkpoint`.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead


def evaluate_on_test(
    encoder: Encoder,
    head: PretrainHead,
    dataset: Dataset,
    test_indices: list[int],
    config: dict,
    device: torch.device,
) -> float:
    """One-shot test-set evaluation on an already-trained, in-memory model.

    Builds a test loader from `test_indices` (the output of
    `reserve_test(dataset, config)["test"]`) and returns the mean
    masked-element MSE.  Called from `run_single` and `run_hpo`
    immediately after the headline checkpoint is loaded.

    Returns
    -------
    float
        Mean reconstruction MSE over the test subset.  Returns `nan`
        if `test_indices` is empty (e.g. `test_frac: 0` in config) --
        callers should translate that to `None` for downstream JSON.
    """
    # imported here to avoid a circular import: ssl_pretrain.py imports this
    # module for evaluate_on_test, and _forward_loss is the only
    # ssl_pretrain.py symbol this module needs.
    from cnmineral.training.ssl_pretrain import _forward_loss

    if not test_indices:
        print("warning: test_indices is empty; returning nan")
        return float("nan")

    encoder.eval()
    head.eval()

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
    )

    mask_prob = config["mask_prob"]
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            total += _forward_loss(
                batch, encoder, head, mask_prob, device,
            ).item()
            n_batches += 1
    assert n_batches > 0, "no test batches processed"
    return float(total / n_batches)

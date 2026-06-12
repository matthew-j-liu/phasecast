"""Test-set evaluation pass.

After the Checkpoint refactor this module only carries
`evaluate_on_test`.  The previous `save_checkpoint` and
`load_checkpoint` functions moved into
`cnmineral.training.checkpoint::Checkpoint` (their work is now
`Checkpoint.save` / `Checkpoint.load` / `Checkpoint.apply_to`).

`evaluate_on_test` stays here because it's not really an "IO"
operation -- it runs a forward pass under no_grad and returns a
loss number.  The name is historical; living next to the
checkpoint module keeps the discovery story together.
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
    # imported here to avoid a circular import: loop.py imports
    # io.py for evaluate_on_test, and _forward_loss is the only
    # loop.py symbol io.py needs.
    from cnmineral.training.loop import _forward_loss

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

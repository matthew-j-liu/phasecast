"""
The training loop + single-run entry point.

`run_pretraining` is the workhorse: trainval indices + config in, a
result dict out (val_loss, train_loss, encoder, head, monitor,
final_epoch, best_epoch, best_val_loss).  It has no idea whether
it's being called once or 50 times -- the only signal it gets is
whether `trial` is None (single-run / production) or a real Optuna
`Trial` (HPO sweep).

Checkpoint saves are dependency-injected via the `store` parameter
(`CheckpointStore | None`).  Passing `store=None` runs the loop
without any disk writes -- HPO trials use this path; production
runs pass a real store.  `run_pretraining` itself reads nothing
from `config["checkpoint_dir"]`.

`run_single` wraps `run_pretraining` for the single-config CLI path:
load YAML, build dataset, split, train, evaluate on test, stamp
test_loss onto the store's best checkpoint.

`write_run_results` lands the per-run human-facing artifacts
(`run_summary.json` + `loss_curve.png`) under `RESULTS_ROOT/<tag>/`;
shared by `run_single` and `run_hpo`'s production run.
"""

import json
import math
from time import time

import matplotlib.pyplot as plt
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from cnmineral.data.tokens import assert_padding_convention, mask_for_ssl
from cnmineral.eval.training_curves import plot_run_loss
from cnmineral.losses.pretrain_loss import masked_element_mse
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead
from cnmineral.training.checkpoint import (
    REPO_ROOT,
    Checkpoint,
    CheckpointStore,
    checkpoint_tag,
)
from cnmineral.training.build_from_config import create_dataset, create_encoder_and_head
from cnmineral.training.reconstruction_mse import evaluate_on_test
from cnmineral.training.splits import _qap_labels_for_indices, reserve_test

# yaml + Path are still needed (run_single reads the YAML; pathlib
# coerces config strings for CheckpointStore).
import yaml
from pathlib import Path

# Single-mode run artifacts land under a per-run tag here.  Module-level
# so tests can monkeypatch it away from the real repo results/ tree.
RESULTS_ROOT = REPO_ROOT / "results" / "pretrain"


def _forward_loss(
    batch: dict,
    encoder: Encoder,
    head: PretrainHead,
    mask_prob: float,
    device: torch.device,
) -> torch.Tensor:
    """One forward pass: mask -> encoder -> head -> masked-element MSE.

    Shared by the train and val loops so the train pass is `forward +
    backward + step` and the val pass is `forward` (under no_grad).
    """
    src = batch["src"].to(device)
    frac = batch["frac"].to(device)

    # 1. Create a MaskedBatch
    masked_batch = mask_for_ssl(src=src, frac=frac, mask_prob=mask_prob)

    # 2. Pass `src_corrupted`, `frac_corrupted` to Encoder.forward
    pooled_latent = encoder(
        src=masked_batch.src_corrupted, frac=masked_batch.frac_corrupted,
    )

    # 3. Pass `pooled_latent` to the PretrainHead.forward
    predicted_frac = head(pooled_latent=pooled_latent)

    # 4. Calculate the loss (torch.Tensor object)
    return masked_element_mse(
        predicted_frac=predicted_frac,
        target_frac=masked_batch.target_frac,
        mask_positions=masked_batch.mask_positions,
    )


def run_pretraining(
    dataset: Dataset,
    trainval_indices: list[int],
    config: dict,
    device: torch.device,
    trial: optuna.trial.Trial | None = None,
    *,
    store: CheckpointStore | None = None,
) -> dict:
    """Run SSL pretraining for `config` and return a result dict.

    `trainval_indices` is the output of
    `reserve_test(dataset, config)["trainval"]`.  The function further
    splits these indices 80/20 (configurable via `val_split`) into
    train + val, runs the training loop on train, and evaluates per
    epoch on val.  Test indices never reach this function.

    The encoder and head are returned in-memory so the caller can
    hand them straight to `evaluate_on_test` without a save/load
    roundtrip.

    Checkpoint policy: when `store` is not None, a best-val
    `Checkpoint` is written via `store.write_best` on each new val
    minimum, and a final-epoch `Checkpoint` is written via
    `store.write_last` once after the loop ends.  When `store` is
    None, the loop runs without any disk writes (HPO trials use
    this path).

    Returns
    -------
    dict
        Keys: val_loss, train_loss, encoder, head, monitor,
        final_epoch, best_epoch, best_val_loss.
    """
    seed = config.get("seed", 0)
    torch.manual_seed(seed)

    encoder, head = create_encoder_and_head(config)
    encoder.to(device)
    head.to(device)

    trainable_params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    # inner 80/20: trainval -> train + val.  Stratify by qap_names when
    # available so per-class proportions match the outer split.
    val_split = config.get("val_split", 0.20)
    if val_split <= 0:
        raise ValueError(
            f"val_split must be > 0 (HPO needs a val curve); got {val_split}"
        )
    train_indices, val_indices = train_test_split(
        trainval_indices, test_size=val_split, random_state=seed,
        stratify=_qap_labels_for_indices(dataset, trainval_indices),
    )
    if not train_indices or not val_indices:
        raise ValueError(
            f"train_val split produced empty subset: "
            f"trainval={len(trainval_indices)}, val_split={val_split}"
        )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
    )

    # single-batch sanity check
    first_batch = next(iter(train_loader))
    assert_padding_convention(first_batch["src"], first_batch["frac"])

    mask_prob = config["mask_prob"]

    print(
        f"run_pretraining: device={device} "
        f"n_train={len(train_indices)} n_val={len(val_indices)} "
        f"epochs={config['epochs']} batch_size={config['batch_size']} "
        f"lr={config['lr']} mask_prob={mask_prob}"
    )
    print()

    # conceptually, our training loop will always be the trio of forward, backward, and step
    # what changes between models is: (1) how the forward is built and
    # (2) what the loss is. in our case, the forward is defined in 3 places
    #   1. cnmineral.data.tokens.mask_for_ssl: takes (src, frac) from the batch and returns a MaskedBatch
    #   2. cnmineral.models.encoder.Encoder: the `forward` method takes (src_corrupted, frac_corrupted) from the MaskedBatch and returns a `pooled_latent`
    #   3. cnmineral.models.pretrain_head.PretrainHead: the `forward` method takes the `pooled_latent` and returns `predicted` (one frac value for every position in the 22-token vocab)
    # as for the loss, we employ `masked_element_mse` in cnmineral.losses.pretrain_loss, which takes `predicted` at the `mask_positions` and compares to `target_frac`.

    monitor: dict[str, list[float]] = {"train": [], "val": []}
    train_mean = 0.0
    val_mean = 0.0
    final_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(config["epochs"]):
        # ---- train pass ----
        # toggle training mode: nonzero dropout will randomly zero out activations
        encoder.train()
        head.train()
        train_total, n_train_batches = 0.0, 0
        epoch_start_time = time()
        for batch in train_loader:
            loss = _forward_loss(batch, encoder, head, mask_prob, device) # see function for comments #1-4

            # Note by this point, by virtue of calling `forward`, PyTorch has constructed a graph data structure
            # that stores how to compute gradients. So we have built a graph connecting
            # parameters (weights and biases), predictions, and loss

            # 5. Standard PyTorch learning mechanism
            # we now compute gradients and use them to update parameters
            #   (i). Clear out gradients from the previous batch
            #   (ii). Compute gradients
            #   (iii). Update parameters
            optimizer.zero_grad()
            loss.backward()
            # cap global grad norm at 1.0 to prevent spikes
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # 6. Track and aggregate loss
            train_total += loss.item()
            n_train_batches += 1
        assert n_train_batches > 0, "no train batches processed"
        train_mean = train_total / n_train_batches
        monitor["train"].append(train_mean)

        # ---- val pass ----
        encoder.eval()
        head.eval()
        val_total, n_val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                val_total += _forward_loss(
                    batch, encoder, head, mask_prob, device,
                ).item()
                n_val_batches += 1
        assert n_val_batches > 0, "no val batches processed"
        val_mean = val_total / n_val_batches
        monitor["val"].append(val_mean)

        print(
            f"epoch {epoch + 1}/{config['epochs']} "
            f"train={train_mean:.4f} val={val_mean:.4f} "
            f"elapsed={(time() - epoch_start_time) * 1000:.0f}ms"
        )

        if trial is not None:
            # 1-indexed to match the printed `epoch 1/N` lines and the
            # loss-curve figures; optuna only needs the index to be
            # consistent across trials for pruner alignment.
            trial.report(val_mean, epoch + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # The best Checkpoint is the only one written DURING the loop,
        # and only on a new val minimum.  The last Checkpoint lands
        # once after the loop ends (see below) -- mid-training
        # overwrites were disk-IO waste since we never resume from a
        # mid-run snapshot.
        if store is not None and val_mean < best_val_loss:
            best_val_loss = val_mean
            best_epoch = epoch + 1
            store.write_best(
                Checkpoint.from_models(
                    encoder, head, optimizer, best_epoch, config,
                    train_mean, val_mean, monitor,
                )
            )

        final_epoch = epoch + 1

    # Single epoch_last.pt write after the loop completes -- captures
    # the post-final-epoch state (encoder + head + optimizer) without
    # the per-epoch churn.  test_loss stays None on this file; the
    # headline test number gets stamped onto epoch_best.pt by the
    # caller (run_single / run_hpo) after reloading the best model.
    if store is not None and final_epoch > 0:
        store.write_last(
            Checkpoint.from_models(
                encoder, head, optimizer, final_epoch, config,
                train_mean, val_mean, monitor,
            )
        )

    return {
        "val_loss": float(val_mean),
        "train_loss": float(train_mean),
        "encoder": encoder,
        "head": head,
        "monitor": monitor,
        "final_epoch": final_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
    }


def write_run_results(result: dict, store: CheckpointStore) -> Path:
    """Write the per-run artifacts for a finished production run.

    Reads the stamped best checkpoint from `store` and lands
    `run_summary.json` + `loss_curve.png` under `RESULTS_ROOT/<tag>/`.
    Headline fields come off the best checkpoint; the monitor and the
    figure come from `result` because the best checkpoint's monitor
    copy is truncated at the best epoch.  Returns the results dir.

    Callers: `run_single` and `run_hpo`'s post-sweep production run.
    """
    best = store.load_best()
    tag = checkpoint_tag(store.best_path)
    results_dir = RESULTS_ROOT / tag
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "epoch": best.epoch,
        "train_loss": best.train_loss,
        "val_loss": best.val_loss,
        "test_loss": best.test_loss,
        "monitor": result["monitor"],
        "git_sha": best.git_sha,
        "saved_at": best.saved_at,
        "config": best.config,
    }
    summary_path = results_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {summary_path}")

    fig, ax = plt.subplots()
    plot_run_loss(
        result["monitor"], ax=ax, title=tag, ylabel="masked-element MSE",
    )
    curve_path = results_dir / "loss_curve.png"
    fig.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {curve_path}")
    return results_dir


def run_single(config_path: Path, device: torch.device) -> dict:
    """Full single-run pipeline: split, train+val, test eval.

    Returns a dict with val_loss, train_loss, test_loss (None when
    `test_split=0`).  Stamps test_loss + test_indices onto the
    store's best checkpoint when `config["checkpoint_dir"]` is set,
    and writes the headline numbers to
    `RESULTS_ROOT/<run_tag>/run_summary.json`.

    This is the only function in the codebase that reads
    `config["checkpoint_dir"]`.  Every downstream consumer
    (`run_pretraining`, `run_hpo`, tests) gets a `CheckpointStore`
    passed in directly.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # build the dataset once at the entry point; reserve_test,
    # run_pretraining, and evaluate_on_test all index into this same
    # instance instead of reconstructing per call.
    dataset = create_dataset(config)
    splits = reserve_test(dataset, config)
    print(
        f"split: trainval={len(splits['trainval'])} test={len(splits['test'])}"
    )

    # Construct the store once.  None when no checkpoint_dir in YAML.
    # When checkpoint_dir IS set, both filename keys are required --
    # a missing key raises KeyError immediately rather than silently
    # falling back to a default that would overwrite an earlier run.
    store = (
        CheckpointStore(
            config["checkpoint_dir"],
            best_filename=config["checkpoint_best_filename"],
            last_filename=config["checkpoint_last_filename"],
        )
        if config.get("checkpoint_dir") else None
    )

    result = run_pretraining(
        dataset, splits["trainval"], config, device, store=store,
    )

    # Compute the headline test_loss against the BEST-VAL model when
    # the store has one on disk -- not the in-memory final-epoch
    # model, which may sit in a worse basin if val overshot.
    if store is not None and store.best_exists():
        best_encoder, best_head = create_encoder_and_head(config)
        store.load_best().apply_to(best_encoder, best_head)
        best_encoder.to(device)
        best_head.to(device)
        test_loss_raw = evaluate_on_test(
            best_encoder, best_head,
            dataset, splits["test"], config, device,
        )
    else:
        # No store or no best -- fall back to in-memory final-epoch.
        test_loss_raw = evaluate_on_test(
            result["encoder"], result["head"],
            dataset, splits["test"], config, device,
        )

    # NaN from evaluate_on_test (empty test) becomes None at the API surface.
    test_loss: float | None = (
        None if math.isnan(test_loss_raw) else test_loss_raw
    )

    # Stamp test_loss + test_indices onto the store's best.  The last
    # snapshot stays test-loss-free (it's the resume snapshot, not
    # the headline).
    if store is not None and store.best_exists():
        store.stamp_test_results_on_best(test_loss, list(splits["test"]))
        write_run_results(result, store)

    return {
        "val_loss": result["val_loss"],
        "train_loss": result["train_loss"],
        "test_loss": test_loss,
    }

"""
Two modes, one CLI flag (`--config PATH`):

  Single-run:
      python scripts/pretrain.py --config configs/pretrain/pretrain_production.yaml

  Optuna HPO sweep (--hpo switches the interpretation of --config):
      python scripts/pretrain.py --hpo \
          --config configs/pretrain/pretrain_hpo.yaml --n-trials 20

How the three top-level functions fit together:

                            USER CLI ENTRY
                                  |
              +-------------------+--------------------+
              |                                        |
     --config pretrain_production.yaml  --hpo --config pretrain_hpo.yaml
     (single-run mode)                  (sweep mode)
              |                                        |
              v                                        v
        run_single(...)                          run_hpo(...)
              |                                        |
              v                                        v
     splits = reserve_test(config)        splits = reserve_test(base_config)
        (outer 80/20: peels off test;       (test reserved once at study scope;
         test indices are held aside         all trials share train/val split
         and never enter training)           via seed determinism)
              |                                        |
              |                                        v
              |                        1. set up JournalStorage at
              |                           cfg["storage_path"]
              |                        2. create study(
              |                              study_name=cfg["study_name"],
              |                              sampler=TPESampler(seed=...),
              |                              pruner=MedianPruner(...),
              |                              direction="minimize",
              |                              load_if_exists=True)
              |                        3. study.optimize(objective, n_trials=N)
              |                                        |
              |                                        v
              |                        +----- objective(trial) -----+
              |                        |                            |
              |                        |  trial_hyperparameters = { |
              |                        |    name: _sample_hyperparameter(trial, name, spec)
              |                        |    for name, spec in       |
              |                        |    cfg["search_space"].items()
              |                        |  }                         |
              |                        |  merged = {**base, **trial_hyperparameters}
              |                        |                            |
              |                        |  return run_pretraining(   |
              |                        |          splits["trainval"],
              |                        |          merged, device, trial=trial)
              |                        |                            |
              |                        +-----------|----------------+
              |                                    |
              +------------------------------------+
                                  |
                                  v
       run_pretraining(trainval_indices, config, device, trial=None|Trial)
                                  |
                +-----------------+-----------------+
                |                                   |
                v                                   v
       inner 80/20 of trainval     epoch loop:
        -> train + val               (train pass) forward (mask -> enc -> head -> loss)
        seed RNG                                  backward -> clip_grad_norm -> step
        build encoder + head                      accumulate train loss
        build AdamW                    (val pass) eval mode + torch.no_grad
        build train/val loaders                   forward (no backward)
                                                  accumulate val loss
                                       monitor["train"].append, monitor["val"].append
                                       if trial is not None:
                                          trial.report(val_loss, epoch)
                                          maybe TrialPruned
                                       save_checkpoint per epoch
                                  |
                                  v
        returns: dict {val_loss, train_loss, encoder, head, monitor, final_epoch}
                                  |
                +-----------------+-----------------+
                |                                   |
     single-run: feed             HPO: objective returns result["val_loss"];
     (encoder, head) into         Optuna uses it to (a) update TPE posterior,
     evaluate_on_test on          (b) decide best trial, (c) seed the next
     splits["test"]               trial's suggestions.
                                  After study.optimize: ONE FINAL PRODUCTION
                                  RUN with study.best_params at full epochs,
                                  then evaluate_on_test on splits["test"].
                |                                   |
                v                                   v
       evaluate_on_test(encoder, head, test_indices, config, device) -> float
          In-memory eval -- no disk roundtrip.  Reads forward + masked MSE
          on splits["test"] in eval mode under torch.no_grad.  Returns nan
          when test_indices is empty (test_frac=0); callers translate to None.
                |                                   |
                v                                   v
      print + save final ckpt        write summary JSON:
      with test_loss field           best_trial_val_loss, best_params,
      populated                      final_train_loss, final_val_loss,
                                     final_test_loss

Role of each function:

- `reserve_test` peels off test indices from the full dataset using
  sklearn's `train_test_split`, seeded for determinism.  Test indices
  are returned alongside `trainval` but structurally do not flow into
  the training machinery.

- `run_pretraining` is the workhorse.  Trainval indices + config in,
  a result dict out (`val_loss`, `train_loss`, `encoder`, `head`,
  `monitor`, `final_epoch`).  It has no idea whether it's being called
  once or 50 times -- the only signal it gets is whether `trial` is
  `None` (single-run / production) or a real `Trial` (HPO sweep).
  Optuna's objective extracts `result["val_loss"]`; pruning is gated by
  `trial.report` inside the epoch loop.

- `evaluate_on_test` runs forward + masked MSE on the held-out test
  indices against an in-memory encoder + head.  No disk roundtrip --
  the caller hands it the trained model directly from
  `run_pretraining`'s result dict.  Called exactly once per pipeline
  invocation: at the end of `run_single` or at the end of `run_hpo`'s
  post-sweep production run.

- `_sample_hyperparameter` draws one value from a `search_space:` entry:
  given a YAML spec like `{type: float, low: 1e-5, high: 1e-3, log: true}`,
  it calls the matching `trial.suggest_*` method.  The search space lives
  entirely in `pretrain_hpo.yaml::search_space:` -- one place to edit when
  you want to widen, narrow, or freeze a knob.

- `run_hpo` orchestrates the sweep + the production run.  It owns:
  storage setup (filesystem journal), study creation (TPE sampler +
  median pruner + load-if-exists for resume), the trial loop, the
  final production run with `study.best_params` at full epoch count,
  the test eval, and the summary JSON write.  `reserve_test` is called
  once at study scope so every trial AND the final run all use the same
  train/val/test split.

The dataset defaults to the real `BulkChemistryDataset` (imported lazily
so the script still runs without it); set `data_source: synthetic` in the
YAML to swap in `SyntheticBulkChemistryDataset`.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import datetime
import json
import math
import subprocess
import sys
import optuna
import torch
import yaml
from pathlib import Path
from time import time
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnmineral.data.synthetic import SyntheticBulkChemistryDataset
from cnmineral.data.tokens import assert_padding_convention, mask_for_ssl
from cnmineral.losses.pretrain_loss import masked_element_mse
from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead


def _git_sha() -> str | None:
    """Return the current git HEAD SHA, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return torch.device("mps")
    return torch.device("cpu")


# some notes on parsing yaml files
# yaml files are just easy ways of representing python dictionaries
# you'll load a yaml file like so: config = yaml.safe_load(f)
# in theory, we can now tap in like config["model"]["layers"]
# but in practice, the more robust way is using .get(), which is a safe dictionary lookup
# (it returns None is the key doesn't exist, or returns a default_value specified in the 2nd argument) 
def create_dataset(config: dict) -> Dataset:
    """
    Build the SSL pretrain dataset per `config['data_source']`.

    `synthetic` always works; `real` tries `cnmineral.data.bulk_chemistry`
    and falls back to synthetic with a printed warning if the module is
    unavailable
    """

    source = config.get("data_source", "real")
    if source == "synthetic":
        return SyntheticBulkChemistryDataset(
            n=config.get("synthetic_n", 256),
            seed=config.get("synthetic_seed", 0),
        )
    try:
        from cnmineral.data.bulk_chemistry import BulkChemistryDataset  # type: ignore
    except ImportError as exc:
        print(
            f"warning: BulkChemistryDataset unavailable ({exc}); "
            f"falling back to SyntheticBulkChemistryDataset"
        )
        return SyntheticBulkChemistryDataset(
            n=config.get("synthetic_n", 256),
            seed=config.get("synthetic_seed", 0),
        )

    # grab dictionary of parameters from config and pass them into the constructor
    # as keyword arguments
    # the config.get("data_kwargs", {}) means we either return the value for the key "data_kwargs" or {}
    # the ** means that we take the resulting dictionary and unpack it into keyword arguments
    return BulkChemistryDataset(**config.get("data_kwargs", {}))


def reserve_test(dataset: Dataset, config: dict) -> dict[str, list[int]]:
    """
    Outer 80/20 split: peel off the test set, keep the trainval pool.

    The test indices are reserved before any training touches the data.
    `run_pretraining` receives `splits["trainval"]` and structurally
    cannot reach the held-out test rows.  The test evaluation happens
    once at the very end via `evaluate_on_test`.

    Splits are stratified by `dataset.qap_names` when that attribute
    exists (the real corpus); synthetic datasets without rock-class
    labels fall through to unstratified random split.

    Parameters
    ----------
    dataset
        Already-constructed dataset; ownership lives at the entry
        point (`run_single` / `run_hpo`) so the corpus is built once
        per invocation, not once per call into this helper.
    config
        Required: `seed`.  Optional: `test_frac` (default 0.20).

    Returns
    -------
    dict[str, list[int]]
        `{"trainval": [int, ...], "test": [int, ...]}` indexing into
        `dataset`.
    """
    n = len(dataset)
    test_frac = config.get("test_frac", 0.20)
    seed = config.get("seed", 0)

    # test_frac=0 short-circuit -- sklearn rejects test_size=0, and the
    # downstream evaluate_on_test handles an empty test_indices already.
    if test_frac == 0:
        return {"trainval": list(range(n)), "test": []}

    stratify_labels = _qap_labels_for_indices(dataset, range(n))
    trainval_idx, test_idx = train_test_split(
        list(range(n)), test_size=test_frac, random_state=seed,
        stratify=stratify_labels,
    )
    if not trainval_idx:
        raise ValueError(
            f"reserve_test produced empty trainval pool: n={n}, "
            f"test_frac={test_frac}"
        )
    return {"trainval": trainval_idx, "test": test_idx}


def _qap_labels_for_indices(
    dataset: Dataset, indices,
) -> list | None:
    """Stratification labels indexed by `indices`, or None.

    Returns None when `dataset` has no `qap_names` attribute (synthetic
    datasets fall through to unstratified split).  When present, returns
    a list of `dataset.qap_names[i]` for each i in `indices` so
    `train_test_split(stratify=...)` can preserve per-class proportions.
    """
    if not hasattr(dataset, "qap_names"):
        return None
    return [dataset.qap_names[i] for i in indices]


def create_encoder_and_head(config: dict) -> tuple[Encoder, PretrainHead]:
    """
    Construct the encoder + SSL pretrain head from a config dict.

    Logic for lookup style: `config[X]` for keys whose absence is
    almost certainly a bug (no safe default exists, e.g. `d_model`,
    `n_heads`). Meanwhile, `config.get(X, default)` for keys whose default is the
    common case (e.g. `dropout=0.0`, `frac_resolution=5000`).  Required
    keys raise `KeyError` at the misspelled YAML line; optional
    keys inherit the default.
    """
    
    encoder = Encoder(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=config.get("dropout", 0.0),
        frac_resolution=config.get("frac_resolution", 5000),
    )

    head = PretrainHead(d_model=config["d_model"])

    return encoder, head


def save_checkpoint(
    path: Path,
    encoder: Encoder,
    head: PretrainHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    train_loss: float,
    val_loss: float,
    monitor: dict[str, list[float]],
    test_loss: float | None = None,
    test_indices: list[int] | None = None,
) -> None:

    # create all missing parent directories if needed
    # If the directory already exists, don’t throw FileExistsError
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "encoder": encoder.state_dict(), # every learnable parameter in the encoder
            "head": head.state_dict(), # the two Linear layers in PretrainHead
            "optimizer": optimizer.state_dict(), # per-parameter AdamW state (e.g., step count)
            "epoch": epoch,
            "config": config,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss, # None for per-epoch saves; populated on the final post-test save
            "monitor": monitor, # {"train": [per-epoch], "val": [per-epoch]}
            "test_indices": test_indices, # None for per-epoch saves; populated on the final post-test save
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "git_sha": _git_sha(),
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        path,
    )


def load_checkpoint(
    path: Path, encoder: Encoder, head: PretrainHead,
) -> dict:
    """Load encoder + head state dicts in place; return the full checkpoint."""

    # load to CPU (safe) and would move to target device (e.g., MPS) downstream 
    # since our checkpoint contains things beyond tensors, like config dict and strings,
    # these would trip the safe-mode on torch.load unless we change weights_only=False  
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    head.load_state_dict(checkpoint["head"])
    return checkpoint


def evaluate_on_test(
    encoder: Encoder,
    head: PretrainHead,
    dataset: Dataset,
    test_indices: list[int],
    config: dict,
    device: torch.device,
) -> float:
    """
    One-shot test-set evaluation on an already-trained, in-memory model.

    Builds a test loader from `test_indices` (the output of
    `reserve_test(dataset, config)["test"]`) and returns the mean
    masked-element MSE.  Called directly from `run_single` and `run_hpo`
    immediately after `run_pretraining`, so the encoder + head are still
    resident on `device` -- no disk roundtrip.

    This is the headline test number -- the unbiased estimate of model
    quality.  Touched exactly once per pipeline invocation.

    Returns
    -------
    float
        Mean reconstruction MSE over the test subset.  Returns `nan`
        if `test_indices` is empty (e.g. `test_frac: 0` in config) --
        callers should translate that to `None` for downstream JSON.
    """
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
) -> dict:
    """
    Run SSL pretraining for `config` and return a result dict carrying the
    final-epoch train + val losses, the in-memory encoder + head, the
    per-epoch monitor, and the epoch count.

    `trainval_indices` is the output of
    `reserve_test(dataset, config)["trainval"]`.  The function further
    splits these indices 80/20 (configurable via `val_frac`) into train
    + val, runs the training loop on train, and evaluates per epoch on
    val.  Test indices never reach this function.

    The encoder and head are returned in-memory so the caller can hand
    them straight to `evaluate_on_test` without a save/load roundtrip.

    Parameters
    ----------
    dataset
        Already-constructed corpus, passed down from the entry point so
        the underlying file is read once per `run_single` / `run_hpo`
        invocation rather than once per trial.
    trainval_indices
        Indices into `dataset`.  The held-out test indices live
        elsewhere and must not be passed here.
    config
        Flat dict.  Required keys: `epochs`, `batch_size`, `lr`,
        `mask_prob`, `weight_decay`, `d_model`, `n_layers`, `n_heads`.
        Optional: `dropout`, `frac_resolution`, `checkpoint_dir`
        (skip saving if absent), `num_workers`, `seed`, `val_frac`
        (default 0.20).
    device
        Where to place the model and batches.
    trial
        Optional Optuna trial.  When present, the per-epoch training
        step calls `trial.report(val_loss, epoch)` and
        `trial.should_prune()` so MedianPruner can bail bad trials.

    Returns
    -------
    dict
        Keys:
        - `val_loss` (float): final-epoch mean val loss; HPO objective.
        - `train_loss` (float): final-epoch mean train loss.
        - `encoder` (Encoder), `head` (PretrainHead): trained models,
          still on `device`.
        - `monitor` (dict[str, list[float]]): per-epoch curves
          `{"train": [...], "val": [...]}`.
        - `final_epoch` (int): number of epochs actually run.
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

    # inner 80/20: trainval -> train + val.  Stratify by qap_names
    # when available so per-class proportions match the outer split.
    val_frac = config.get("val_frac", 0.20)
    if val_frac <= 0:
        raise ValueError(
            f"val_frac must be > 0 (HPO needs a val curve); got {val_frac}"
        )
    train_indices, val_indices = train_test_split(
        trainval_indices, test_size=val_frac, random_state=seed,
        stratify=_qap_labels_for_indices(dataset, trainval_indices),
    )
    if not train_indices or not val_indices:
        raise ValueError(
            f"train_val split produced empty subset: "
            f"trainval={len(trainval_indices)}, val_frac={val_frac}"
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

    # single batch sanity check
    first_batch = next(iter(train_loader))
    assert_padding_convention(first_batch["src"], first_batch["frac"])

    mask_prob = config["mask_prob"]
    checkpoint_dir = (
        Path(config["checkpoint_dir"])
        if config.get("checkpoint_dir")
        else None
    )

    print(
        f"run_pretraining: device={device} "
        f"n_train={len(train_indices)} n_val={len(val_indices)} "
        f"epochs={config['epochs']} batch_size={config['batch_size']} "
        f"lr={config['lr']} mask_prob={mask_prob}"
    )

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
            f"elapsed={time() - epoch_start_time:.1f}s"
        )

        if trial is not None:
            trial.report(val_mean, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if checkpoint_dir is not None:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
                encoder, head, optimizer, epoch + 1, config,
                train_mean,
                val_mean,
                monitor,
            )

        final_epoch = epoch + 1

    return {
        "val_loss": float(val_mean),
        "train_loss": float(train_mean),
        "encoder": encoder,
        "head": head,
        "monitor": monitor,
        "final_epoch": final_epoch,
    }


def _sample_hyperparameter(
    trial: optuna.trial.Trial, name: str, spec: dict,
):
    """Sample one hyperparameter from a `search_space:` entry via `trial.suggest_*`."""
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False),
        )
    if kind == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"], log=spec.get("log", False),
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(
        f"unknown search_space type {kind!r} for {name!r}; "
        f"expected one of: float, int, categorical"
    )


def run_hpo(
    config_path: Path,
    device: torch.device,
    n_trials_override: int | None = None,
) -> None:
    """Orchestrate an Optuna HPO sweep + a final production run on best params.

    Pipeline:
      1. Set up filesystem `JournalStorage` and create / reload the study.
         Resumable: re-running the same config picks up where it left off.
      2. Call `reserve_test(base_config)` once at study scope -- all trials
         share the same train/val/test split via seed determinism.
      3. Run N trials.  Each trial samples HPs from `cfg["search_space"]`
         via `_sample_hyperparameter`, merges them onto `cfg["base"]`, and
         runs `run_pretraining` for `cfg["hpo_run_epochs"]` epochs.  The
         trial's val loss is reported per epoch so MedianPruner can bail
         bad runs.
      4. After `study.optimize` returns, do ONE final production run with
         `study.best_params` merged onto base AT FULL `base["epochs"]` (not
         the abbreviated `hpo_run_epochs`).  Then evaluate that model on
         the held-out test set.  This is the headline result.
      5. Write a summary JSON next to the storage file carrying:
         - `best_trial_val_loss` (the sweep's best, at hpo_run_epochs).
         - `best_params` (the HPs chosen).
         - `final_train_loss`, `final_val_loss`, `final_test_loss` (from
           the production run; `final_test_loss` is `None` when
           `test_frac=0` or no trials completed).

    Test discipline: each trial only sees train+val.  Test is touched
    exactly once, in step 4, on the chosen model.  Per-trial test
    evaluation would tempt cherry-picking HPs based on test scores,
    defeating the held-out set's purpose.

    Parameters
    ----------
    config_path
        YAML config.  Expected keys: `study_name`, `storage_path`,
        `sampler_seed`, `n_trials`, `n_startup_trials`,
        `pruner_warmup_steps`, `hpo_run_epochs`, `base` (locked config
        keys), `search_space` (per-knob spec dict consumed by
        `_sample_hyperparameter`).  `base` and `search_space` are
        disjoint -- a key lives in exactly one section.
    device
        Where to place models and batches.
    n_trials_override
        If set, replaces `cfg["n_trials"]` for this invocation.
    """

    # load configs and setup storage
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if n_trials_override is not None:
        cfg["n_trials"] = n_trials_override

    storage_path = Path(cfg["storage_path"])
    if not storage_path.is_absolute():
        storage_path = REPO_ROOT / storage_path
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(storage_path)))

    # setup Optuna study 
    sampler = TPESampler(
        seed=cfg["sampler_seed"],
        n_startup_trials=cfg["n_startup_trials"],
    )
    pruner = MedianPruner(n_warmup_steps=cfg["pruner_warmup_steps"])

    study = optuna.create_study(
        study_name=cfg["study_name"],
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    n_done = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )

    trials_left = max(0, cfg["n_trials"] - n_done)
    print(
        f"resuming study {cfg['study_name']!r}: {n_done} complete, "
        f"running {trials_left} more to reach n_trials={cfg['n_trials']}"
    )

    base_config = dict(cfg["base"])
    base_config["epochs"] = cfg.get("hpo_run_epochs", 3)
    search_space = cfg.get("search_space", {})

    # build the dataset once at study scope: every trial, the
    # production run, and the test eval below all index into this
    # same instance.  `trial_hyperparameters` fills in the search-space
    # dims that `base_config` deliberately leaves out -- the two are
    # disjoint, so merge is set union, not override.
    dataset = create_dataset(base_config)
    splits = reserve_test(dataset, base_config)
    print(
        f"split: trainval={len(splits['trainval'])} test={len(splits['test'])}"
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_hyperparameters = {
            name: _sample_hyperparameter(trial, name, spec)
            for name, spec in search_space.items()
        }
        merged = {**base_config, **trial_hyperparameters}
        result = run_pretraining(
            dataset, splits["trainval"], merged, device, trial=trial,
        )
        return result["val_loss"]

    # run HPO
    if trials_left > 0:
        study.optimize(objective, n_trials=trials_left, gc_after_trial=True)

    # post-sweep production run + test eval on the best params.  If no
    # trials completed (e.g. all pruned, or n_trials=0), skip.
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    best_trial_val_loss: float | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    final_test_loss: float | None = None
    best_params: dict = {}
    if completed:
        best_trial_val_loss = float(study.best_value)
        best_params = study.best_params

        # full-epoch production config: restore base["epochs"] over the
        # hpo_run_epochs substitution we did at the top.
        production_config = {**base_config, **best_params}
        production_config["epochs"] = cfg["base"].get(
            "epochs", cfg.get("hpo_run_epochs", 3),
        )

        print(
            f"production run with best params: "
            f"epochs={production_config['epochs']} "
            f"best_trial_val_loss={best_trial_val_loss:.4f}"
        )
        prod_result = run_pretraining(
            dataset, splits["trainval"], production_config, device,
        )
        test_loss_raw = evaluate_on_test(
            prod_result["encoder"], prod_result["head"],
            dataset, splits["test"], production_config, device,
        )
        final_train_loss = prod_result["train_loss"]
        final_val_loss = prod_result["val_loss"]
        final_test_loss = (
            None if math.isnan(test_loss_raw) else test_loss_raw
        )

        # final checkpoint: rewrite the per-epoch save with test fields
        checkpoint_dir = production_config.get("checkpoint_dir")
        if checkpoint_dir:
            final_path = (
                Path(checkpoint_dir)
                / f"epoch_{prod_result['final_epoch']:03d}.pt"
            )
            if final_path.exists():
                payload = torch.load(
                    final_path, map_location="cpu", weights_only=False,
                )
                payload["test_loss"] = final_test_loss
                payload["test_indices"] = list(splits["test"])
                torch.save(payload, final_path)

    # save results
    summary = {
        "study_name": cfg["study_name"],
        "storage_path": str(storage_path),
        "n_trials_complete": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ),
        "n_trials_pruned": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ),
        "best_trial_val_loss": best_trial_val_loss,
        "best_params": best_params,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_test_loss": final_test_loss,
        "device": str(device),
        "config": cfg,
    }

    summary_path = storage_path.parent / f"{cfg['study_name']}_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"summary -> {summary_path}")

    if completed:
        print(f"best trial val:   {best_trial_val_loss:.4f}")
        print(f"best params:      {best_params}")
        test_str = (
            f"{final_test_loss:.4f}"
            if final_test_loss is not None else "n/a"
        )
        print(
            f"final production: train={final_train_loss:.4f} "
            f"val={final_val_loss:.4f} test={test_str}"
        )


def run_single(config_path: Path, device: torch.device) -> dict:
    """Full single-run pipeline: split, train+val, test eval.

    Returns a dict with `val_loss`, `train_loss`, `test_loss` (None when
    `test_frac=0`).  Saves a final checkpoint that bundles the test
    number, if `checkpoint_dir` is set in the config.
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

    result = run_pretraining(dataset, splits["trainval"], config, device)
    test_loss_raw = evaluate_on_test(
        result["encoder"], result["head"],
        dataset, splits["test"], config, device,
    )
    # NaN from evaluate_on_test (empty test) becomes None at the API surface.
    test_loss: float | None = (
        None if math.isnan(test_loss_raw) else test_loss_raw
    )

    # final checkpoint: re-save the final epoch with test fields populated
    checkpoint_dir = config.get("checkpoint_dir")
    if checkpoint_dir:
        # reconstruct an optimizer-state-less save would lose resume; instead,
        # rewrite the existing final epoch checkpoint with test fields added.
        final_path = (
            Path(checkpoint_dir) / f"epoch_{result['final_epoch']:03d}.pt"
        )
        if final_path.exists():
            payload = torch.load(
                final_path, map_location="cpu", weights_only=False,
            )
            payload["test_loss"] = test_loss
            payload["test_indices"] = list(splits["test"])
            torch.save(payload, final_path)

    return {
        "val_loss": result["val_loss"],
        "train_loss": result["train_loss"],
        "test_loss": test_loss,
        "monitor": result["monitor"],
        "final_epoch": result["final_epoch"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="YAML config path; structure depends on --hpo",
    )
    parser.add_argument(
        "--hpo", action="store_true",
        help="treat --config as an HPO sweep config (configs/pretrain/pretrain_hpo.yaml)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="override n_trials from the HPO config",
    )

    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")

    if args.hpo:
        run_hpo(args.config, device, n_trials_override=args.n_trials)
    else:
        if args.n_trials is not None:
            parser.error("--n-trials only applies in --hpo mode")
        result = run_single(args.config, device)
        test_str = (
            f"{result['test_loss']:.4f}"
            if result["test_loss"] is not None else "n/a"
        )
        print(
            f"final: train={result['train_loss']:.4f} "
            f"val={result['val_loss']:.4f} test={test_str}"
        )


if __name__ == "__main__":
    main()

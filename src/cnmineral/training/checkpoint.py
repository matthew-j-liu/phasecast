"""
Checkpoint payload + directory-shaped store.

- `Checkpoint` -- frozen dataclas1s holding one serialized payload.
  Has `Checkpoint.from_models`, `.load(path)`, `.save(path)`,
  `.apply_to(encoder, head)`, `.with_test_results(...)`.  Caller
  supplies the path; the class has no opinion on filename conventions.

- `CheckpointStore` -- a directory + best/last filename triple
  passed in at construction.  Callers use `store.best_exists()`,
  `store.load_best()`, etc. instead of building `.pt` paths
  themselves.  Both filenames are required -- the store has no
  default convention; the YAML driving a run is the single source
  of truth for what gets written where.
"""

import datetime
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from cnmineral.models.encoder import Encoder
from cnmineral.models.pretrain_head import PretrainHead


REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_sha() -> str | None:
    """Return the current git HEAD SHA, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def checkpoint_tag(checkpoint_path: Path) -> str:
    """Filesystem-safe tag derived from `<parent_dir>_<filename_stem>`.

    Used to namespace per-checkpoint artifacts (probe HPO journals,
    single-mode diagnostics directories, summary JSON filenames) so
    multiple checkpoints' outputs share one parent directory without
    colliding.  E.g. `checkpoints/pretrain_v3_smoke/epoch_001.pt` ->
    `pretrain_v3_smoke_epoch_001`.
    """
    stem = checkpoint_path.stem
    parent = checkpoint_path.parent.name
    raw = f"{parent}_{stem}"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """In-memory representation of one serialized checkpoint payload.

    Mutable container caveat: the `*_state` fields and `config` /
    `monitor` hold dicts the dataclass cannot make immutable.  The
    class is `frozen=True` to advertise immutability of the
    *Checkpoint identity* (you cannot reassign fields).  Do NOT
    mutate the inner state-dict / config / monitor objects from
    outside -- the dataclass advertises immutability and external
    mutation silently corrupts the in-memory copy.  Consume via
    `apply_to()` (loads weights into a model) or
    `with_test_results()` (returns a new Checkpoint).
    """

    encoder_state: dict
    head_state: dict
    optimizer_state: dict
    epoch: int
    config: dict
    train_loss: float
    val_loss: float
    monitor: dict
    test_loss: float | None = None
    test_indices: list[int] | None = None
    # durable identity of the test rows: corpus-order indices go stale
    # the moment the filter chain changes, sample_ids don't.  This is
    # what lets a later corpus pin these rows into its own test split
    # (splits.reserve_test `test_roster_path`).
    test_sample_ids: list[int] | None = None
    git_sha: str | None = None
    saved_at: str | None = None

    @classmethod
    def from_models(
        cls,
        encoder: Encoder,
        head: PretrainHead,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        config: dict,
        train_loss: float,
        val_loss: float,
        monitor: dict,
        test_loss: float | None = None,
        test_indices: list[int] | None = None,
    ) -> "Checkpoint":
        """Capture current model + optimizer state into a Checkpoint."""
        return cls(
            encoder_state=encoder.state_dict(),
            head_state=head.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            config=config,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            monitor=monitor,
            test_loss=test_loss,
            test_indices=test_indices,
            git_sha=_git_sha(),
            saved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    @classmethod
    def load(cls, path: Path | str) -> "Checkpoint":
        """Load from disk.

        Reads with `weights_only=False` because the payload carries
        the config dict, ISO timestamp, git SHA, etc. alongside
        tensor state dicts.
        """
        payload = torch.load(
            Path(path), map_location="cpu", weights_only=False,
        )
        return cls(
            encoder_state=payload["encoder"],
            head_state=payload["head"],
            optimizer_state=payload["optimizer"],
            epoch=payload["epoch"],
            config=payload["config"],
            train_loss=payload["train_loss"],
            val_loss=payload["val_loss"],
            monitor=payload["monitor"],
            test_loss=payload.get("test_loss"),
            test_indices=payload.get("test_indices"),
            test_sample_ids=payload.get("test_sample_ids"),
            git_sha=payload.get("git_sha"),
            saved_at=payload.get("saved_at"),
        )

    def save(self, path: Path | str) -> None:
        """Serialize to disk; creates parent dirs as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # On-disk keys are "encoder" / "head" / "optimizer" (not the
        # *_state field names); `load` and any external tool reading
        # a payload via raw `torch.load` expect these.
        torch.save(
            {
                "encoder": self.encoder_state,
                "head": self.head_state,
                "optimizer": self.optimizer_state,
                "epoch": self.epoch,
                "config": self.config,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "monitor": self.monitor,
                "test_loss": self.test_loss,
                "test_indices": self.test_indices,
                "test_sample_ids": self.test_sample_ids,
                "git_sha": self.git_sha,
                "saved_at": self.saved_at,
            },
            path,
        )

    def apply_to(self, encoder: Encoder, head: PretrainHead) -> None:
        """Load encoder + head state dicts into the given models in place."""
        encoder.load_state_dict(self.encoder_state)
        head.load_state_dict(self.head_state)

    def with_test_results(
        self,
        test_loss: float | None,
        test_indices: list[int] | None,
        test_sample_ids: list[int] | None = None,
    ) -> "Checkpoint":
        """Return a new Checkpoint with the test fields populated.

        All other fields are shared with the original (no deep copy
        of state dicts) -- callers that subsequently `save()` the
        returned instance get an updated on-disk payload without
        round-tripping the model state through CPU memory twice.
        """
        return Checkpoint(
            encoder_state=self.encoder_state,
            head_state=self.head_state,
            optimizer_state=self.optimizer_state,
            epoch=self.epoch,
            config=self.config,
            train_loss=self.train_loss,
            val_loss=self.val_loss,
            monitor=self.monitor,
            test_loss=test_loss,
            test_indices=test_indices,
            test_sample_ids=test_sample_ids,
            git_sha=self.git_sha,
            saved_at=self.saved_at,
        )

    def summary(self, name: str | None = None) -> str:
        """Return a human-readable, multi-line summary of this Checkpoint.

        Designed for terminal inspection and side-by-side comparison
        of multiple checkpoints.  Prints architecture + training knobs,
        the headline losses, a compact val-curve view, and provenance.

        `name` -- optional header line (e.g., the file path).
        """
        lines: list[str] = []
        if name is not None:
            lines.append(f"=== {name} ===")

        val_curve = self.monitor.get("val", [])
        n_epochs = len(val_curve)
        if val_curve:
            best_val = min(val_curve)
            best_idx = val_curve.index(best_val) + 1
            best_marker = f"  (best in run; reached at epoch {best_idx})"
        else:
            best_val = None
            best_marker = ""

        lines.append(f"epoch         {self.epoch} (of {n_epochs} trained)")
        lines.append(f"val_loss      {self.val_loss:.4e}{best_marker if self.val_loss == best_val else ''}")
        if self.test_loss is None:
            lines.append("test_loss     None  (not stamped; this is the resume snapshot)")
        else:
            lines.append(f"test_loss     {self.test_loss:.4e}")
        lines.append(f"train_loss    {self.train_loss:.4e}")
        lines.append("")

        cfg = self.config
        lines.append(
            f"architecture  d_model={cfg.get('d_model')}  "
            f"n_layers={cfg.get('n_layers')}  n_heads={cfg.get('n_heads')}"
        )
        lr = cfg.get("lr")
        wd = cfg.get("weight_decay")
        mp = cfg.get("mask_prob")
        dr = cfg.get("dropout")
        lines.append(
            f"training      batch_size={cfg.get('batch_size')}  "
            f"lr={lr:.3e}  mask_prob={mp:.3f}  "
            f"weight_decay={wd:.3e}  dropout={dr:.3f}"
        )

        n_enc = sum(v.numel() for v in self.encoder_state.values())
        n_head = sum(v.numel() for v in self.head_state.values())
        lines.append(
            f"params        encoder={_fmt_count(n_enc)}  head={_fmt_count(n_head)}"
        )
        lines.append("")

        lines.append(f"val curve     {_compact_curve(val_curve)}")
        if val_curve:
            lines.append(
                f"              min={best_val:.4e} at epoch {best_idx}  |  "
                f"final={val_curve[-1]:.4e} at epoch {n_epochs}"
            )

        lines.append(f"saved_at      {self.saved_at}")
        lines.append(f"git_sha       {self.git_sha}")
        return "\n".join(lines)


def _fmt_count(n: int) -> str:
    """Format a parameter count as 4.2M / 131K / 250 with a fixed width."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _compact_curve(curve: list[float], head_tail: int = 3) -> str:
    """Compact one-line view of a per-epoch val curve.

    Short curves (<= 2*head_tail + 1) print in full; longer ones show
    `head_tail` from the start, then `...`, then `head_tail` from the
    end.  Each value renders as `epoch:value` so the reader sees both.
    """
    if not curve:
        return "(empty)"
    if len(curve) <= 2 * head_tail + 1:
        return "  ".join(f"{i + 1}:{v:.4e}" for i, v in enumerate(curve))
    head = [f"{i + 1}:{v:.4e}" for i, v in enumerate(curve[:head_tail])]
    tail = [
        f"{i + 1}:{v:.4e}"
        for i, v in enumerate(curve[-head_tail:], start=len(curve) - head_tail)
    ]
    return "  ".join(head) + "  ...  " + "  ".join(tail)


class CheckpointStore:
    """A directory plus best-val and last-epoch filenames.

    The store is the only definition of "the canonical best
    model" in this codebase.  External processes that want to
    designate a checkpoint as canonical do so by writing into a
    store, not by picking a name.

    Both filenames are required at construction -- the store has
    no default convention.  Forcing the YAML to name both files
    explicitly means two runs sharing one `checkpoint_dir` cannot
    silently overwrite each other.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        best_filename: str,
        last_filename: str,
    ) -> None:
        self.root = Path(root)
        self._best_filename = best_filename
        self._last_filename = last_filename

    @property
    def best_path(self) -> Path:
        return self.root / self._best_filename

    @property
    def last_path(self) -> Path:
        return self.root / self._last_filename

    def best_exists(self) -> bool:
        return self.best_path.exists()

    def last_exists(self) -> bool:
        return self.last_path.exists()

    def load_best(self) -> Checkpoint:
        return Checkpoint.load(self.best_path)

    def load_last(self) -> Checkpoint:
        return Checkpoint.load(self.last_path)

    def write_best(self, ckpt: Checkpoint) -> None:
        ckpt.save(self.best_path)

    def write_last(self, ckpt: Checkpoint) -> None:
        ckpt.save(self.last_path)

    def stamp_test_results_on_best(
        self,
        test_loss: float | None,
        test_indices: list[int] | None,
        test_sample_ids: list[int] | None = None,
    ) -> None:
        """Load best, update test fields, save back.

        Caller responsibility: `test_loss` was computed by running
        `evaluate_on_test` against the model loaded from this same
        best checkpoint (use `load_best().apply_to(encoder, head)`
        first).  This method only handles the disk round-trip; it
        does not re-evaluate.
        """
        self.write_best(
            self.load_best().with_test_results(
                test_loss, test_indices, test_sample_ids,
            )
        )

"""Dataset joining mineralogy-corpus labels to encoder input tokens.

Each item is one (bulk, draw, rung) corpus row: the bulk's 22-token
cation+trace input (built from the original gard chemistry via
`BulkChemistryDataset` -- never the per-draw renormalized oxides, whose scale
factor leaks the sampled H2O; see ADR 008) plus the ModeHead hurdle targets
from `mineralogy_labels`. All rows of a bulk share one input vector, exposed
per-bulk (`bulk_src` / `bulk_frac` + `row_bulk_idx`) so a frozen-encoder
trainer can cache one latent per bulk. Splits are bulk-level; see
`provisional_bulk_split`.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .bulk_chemistry import BulkChemistryDataset
from .mineralogy_labels import hurdle_targets, load_mode_labels
from .phase_vocab import PRESENCE_CUTOFF


class ModeHeadDataset(Dataset):
    """(bulk, draw, rung) rows with hurdle targets over the phase vocabulary."""

    def __init__(
        self,
        corpus_dir: str | Path,
        rungs: tuple[int, ...] = (50, 30),
        presence_cutoff: float = PRESENCE_CUTOFF,
        data_kwargs: dict | None = None,
    ) -> None:
        self.presence_cutoff = presence_cutoff
        labels = load_mode_labels(corpus_dir, rungs)

        chem_kwargs = {"require_all_majors": False, **(data_kwargs or {})}
        chem = BulkChemistryDataset(**chem_kwargs)
        id_to_row = {int(s): i for i, s in enumerate(chem.sample_ids)}
        corpus_ids = np.sort(labels["bulk"].unique())
        resolved = np.array([b for b in corpus_ids if int(b) in id_to_row])
        n_missing = len(corpus_ids) - len(resolved)
        if n_missing:
            print(f"dropping {n_missing} corpus bulks without resolvable input chemistry")
            labels = labels[labels["bulk"].isin(set(resolved.tolist()))]

        chem_rows = [id_to_row[int(b)] for b in resolved]
        items = [chem[i] for i in chem_rows]
        self.bulk_ids = resolved
        self.bulk_src = torch.stack([it["src"] for it in items])
        self.bulk_frac = torch.stack([it["frac"] for it in items])
        self.bulk_qap_names = chem.qap_names[chem_rows]

        bulk_pos = {int(b): i for i, b in enumerate(resolved)}
        self.row_bulk_idx = torch.tensor(
            [bulk_pos[int(b)] for b in labels["bulk"]], dtype=torch.long
        )
        self.rung = torch.tensor(labels["rung"].to_numpy(), dtype=torch.long)

        modes = labels.drop(columns=["bulk", "draw", "rung"]).to_numpy(dtype=np.float64)
        presence, magnitude = hurdle_targets(modes, presence_cutoff)
        self.presence = torch.tensor(presence, dtype=torch.float32)
        self.magnitude = torch.tensor(magnitude, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.row_bulk_idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        b = self.row_bulk_idx[idx]
        return {
            "src": self.bulk_src[b].clone(),
            "frac": self.bulk_frac[b].clone(),
            "presence": self.presence[idx].clone(),
            "magnitude": self.magnitude[idx].clone(),
            "bulk_idx": b.clone(),
        }


def provisional_bulk_split(
    bulk_ids: np.ndarray, val_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic bulk-level train/val membership, stable as chunks accumulate.

    Each bulk hashes independently to train or val, so membership never
    changes as more corpus chunks land -- at corpus scale the split is
    class-representative without explicit stratification. PROVISIONAL until
    the high-precision valtest tier exists: this carves validation out of the
    train tier and must be discarded once `data/corpus/valtest/` lands. Never
    use it as the final 64/16/20 split.
    """
    digests = np.array([
        int(hashlib.md5(f"{seed}:{int(b)}".encode()).hexdigest()[:8], 16)
        for b in bulk_ids
    ])
    is_val = (digests % 10_000) < int(round(val_fraction * 10_000))
    return bulk_ids[~is_val], bulk_ids[is_val]

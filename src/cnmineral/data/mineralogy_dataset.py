"""Dataset joining mineralogy-corpus labels to encoder input tokens.

Each item is one (bulk, draw, rung) corpus row: the bulk's 22-token
cation+trace input (built from the original gard chemistry via
`BulkChemistryDataset` -- never the per-draw renormalized oxides, whose scale
factor leaks the sampled H2O; see ADR 008) plus the ModeHead hurdle targets
from `mineralogy_labels`. All rows of a bulk share one input vector, exposed
per-bulk (`bulk_src` / `bulk_frac` + `row_bulk_idx`) so a frozen-encoder
trainer can cache one latent per bulk. Splits are bulk-level: the valtest
tier's val/test roster is the committed `valtest_split.csv`, read via
`load_valtest_split`; the train tier trains whole.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .bulk_chemistry import BulkChemistryDataset
from .mineralogy_labels import hurdle_targets, load_mode_labels
from .phase_vocab import PRESENCE_CUTOFF

VALTEST_SPLIT_PATH = Path(__file__).resolve().parent / "valtest_split.csv"


def load_valtest_split(subset: str) -> np.ndarray:
    """Sorted bulk ids of the requested valtest roster subset (`val` | `test`).

    The roster is the committed `valtest_split.csv`, generated once by
    `scripts/build/build_valtest_split.py` and never recomputed at load time:
    the test subset is sealed, so the assignment must not drift with library
    versions. Roster membership is nominal -- a bulk with no usable corpus
    rows contributes nothing downstream.
    """
    roster = pd.read_csv(VALTEST_SPLIT_PATH)
    if subset not in ("val", "test"):
        raise ValueError(f"unknown valtest subset {subset!r}; expected 'val' or 'test'")
    ids = roster.loc[roster["subset"] == subset, "sample_id"].to_numpy(dtype=np.int64)
    return np.sort(ids)


class ModeHeadDataset(Dataset):
    """(bulk, draw, rung) rows with hurdle targets over the phase vocabulary.

    `bulk_ids` restricts label rows to those bulks (how the trainer and eval
    harness select a valtest roster subset); default keeps every corpus bulk.
    """

    def __init__(
        self,
        corpus_dir: str | Path,
        rungs: tuple[int, ...] = (50, 30),
        presence_cutoff: float = PRESENCE_CUTOFF,
        data_kwargs: dict | None = None,
        bulk_ids: np.ndarray | None = None,
    ) -> None:
        self.presence_cutoff = presence_cutoff
        labels = load_mode_labels(corpus_dir, rungs)
        if bulk_ids is not None:
            labels = labels[labels["bulk"].isin({int(b) for b in bulk_ids})]
            if labels.empty:
                raise ValueError(
                    f"no label rows for the requested bulk_ids under {corpus_dir}"
                )

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

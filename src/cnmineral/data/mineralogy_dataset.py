"""Dataset joining mineralogy-corpus labels to encoder input tokens.

Each item is one (bulk, draw, rung) corpus row: the bulk's 22-token
cation+trace input (built from the original gard chemistry via
`BulkChemistryDataset` -- never the per-draw renormalized oxides, whose scale
factor leaks the sampled H2O; see ADR 008) plus the head targets from
`mineralogy_labels` (ModeHead hurdle targets, comp, cond, and mass-balance
fixings). All rows of a bulk share one input vector, exposed
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
from .mineralogy_labels import hurdle_targets, load_joint_labels
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


def _resolve_bulk_inputs(corpus_ids: np.ndarray, data_kwargs: dict | None):
    """Join corpus bulk ids to input chemistry; drops unresolvable bulks.

    Returns (resolved ids, src [n, 22], frac [n, 22], qap names).
    """
    chem_kwargs = {"require_all_majors": False, **(data_kwargs or {})}
    chem = BulkChemistryDataset(**chem_kwargs)
    id_to_row = {int(s): i for i, s in enumerate(chem.sample_ids)}
    resolved = np.array([b for b in corpus_ids if int(b) in id_to_row])
    n_missing = len(corpus_ids) - len(resolved)
    if n_missing:
        print(f"dropping {n_missing} corpus bulks without resolvable input chemistry")
    chem_rows = [id_to_row[int(b)] for b in resolved]
    items = [chem[i] for i in chem_rows]
    src = torch.stack([it["src"] for it in items])
    frac = torch.stack([it["frac"] for it in items])
    return resolved, src, frac, chem.qap_names[chem_rows]


class JointHeadDataset(Dataset):
    """(bulk, draw, rung) rows with mode, comp, cond, and closure targets.

    Hurdle targets over the phase vocabulary plus the CompHead oxide targets,
    the CondHead (T_end, P_kbar, H2O) targets, and the teacher-forced
    mass-balance fixings from `load_joint_labels`. `bulk_ids` restricts label
    rows to those bulks (how the trainer and eval harness select a valtest
    roster subset); `draw_limit` keeps only draws with id below it (draw ids
    are 0..n-1 per bulk, so this is a per-bulk draw subsample); default keeps
    every corpus row.
    """

    def __init__(
        self,
        corpus_dir: str | Path,
        rungs: tuple[int, ...] = (50, 30),
        presence_cutoff: float = PRESENCE_CUTOFF,
        comp_eps: float = 1e-4,
        data_kwargs: dict | None = None,
        bulk_ids: np.ndarray | None = None,
        draw_limit: int | None = None,
    ) -> None:
        self.presence_cutoff = presence_cutoff
        self.comp_eps = comp_eps
        labels = load_joint_labels(corpus_dir, rungs, comp_eps)
        index = labels["index"]
        # keep is our own buffer; in-place &= on a pandas .to_numpy() result
        # can raise "output array is read-only"
        keep = np.ones(len(index), dtype=bool)
        if bulk_ids is not None:
            keep &= index["bulk"].isin({int(b) for b in bulk_ids}).to_numpy()
            if not keep.any():
                raise ValueError(
                    f"no label rows for the requested bulk_ids under {corpus_dir}"
                )
        if draw_limit is not None:
            keep &= (index["draw"] < int(draw_limit)).to_numpy()
            if not keep.any():
                raise ValueError(
                    f"no label rows left under draw_limit={draw_limit} in {corpus_dir}"
                )

        corpus_ids = np.sort(index["bulk"][keep].unique())
        resolved, self.bulk_src, self.bulk_frac, self.bulk_qap_names = (
            _resolve_bulk_inputs(corpus_ids, data_kwargs)
        )
        keep &= index["bulk"].isin(set(resolved.tolist())).to_numpy()
        self.bulk_ids = resolved
        self.draws_per_bulk = float(index[keep].groupby("bulk")["draw"].nunique().mean())

        bulk_pos = {int(b): i for i, b in enumerate(resolved)}
        self.row_bulk_idx = torch.tensor(
            [bulk_pos[int(b)] for b in index["bulk"][keep]], dtype=torch.long
        )
        self.rung = torch.tensor(index["rung"][keep].to_numpy(), dtype=torch.long)

        presence, magnitude = hurdle_targets(labels["modes"][keep], presence_cutoff)
        self.presence = torch.tensor(presence, dtype=torch.float32)
        self.magnitude = torch.tensor(magnitude, dtype=torch.float32)
        self.comp_target = torch.tensor(labels["comp"][keep], dtype=torch.float32)
        self.cond_target = torch.tensor(labels["cond"][keep], dtype=torch.float32)
        self.f_liq = torch.tensor(labels["f_liq"][keep], dtype=torch.float32)
        self.f_fl = torch.tensor(labels["f_fl"][keep], dtype=torch.float32)
        self.c_liq = torch.tensor(labels["c_liq"][keep], dtype=torch.float32)
        self.c_fl = torch.tensor(labels["c_fl"][keep], dtype=torch.float32)
        self.bulk_11 = torch.tensor(labels["bulk_11"][keep], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.row_bulk_idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        b = self.row_bulk_idx[idx]
        return {
            "src": self.bulk_src[b].clone(),
            "frac": self.bulk_frac[b].clone(),
            "presence": self.presence[idx].clone(),
            "magnitude": self.magnitude[idx].clone(),
            "comp_target": self.comp_target[idx].clone(),
            "cond_target": self.cond_target[idx].clone(),
            "f_liq": self.f_liq[idx].clone(),
            "f_fl": self.f_fl[idx].clone(),
            "c_liq": self.c_liq[idx].clone(),
            "c_fl": self.c_fl[idx].clone(),
            "bulk_11": self.bulk_11[idx].clone(),
            "bulk_idx": b.clone(),
        }

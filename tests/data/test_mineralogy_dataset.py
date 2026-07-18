import numpy as np
import pytest
import torch

from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.data.mineralogy_dataset import ModeHeadDataset
from cnmineral.data.phase_vocab import N_PHASES

_DATA_KWARGS = {"min_class_count": 5}


def _dataset(corpus_subset_path, synthetic_mineralogy_corpus, tmp_path, n_bulks=2):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, **_DATA_KWARGS
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:n_bulks]]
    corpus_dir = synthetic_mineralogy_corpus(tmp_path / "corpus", bulk_ids)
    ds = ModeHeadDataset(
        corpus_dir,
        data_kwargs={"csv_path": str(corpus_subset_path), **_DATA_KWARGS},
    )
    return ds, chem, bulk_ids


def test_rows_and_target_shapes(corpus_subset_path, synthetic_mineralogy_corpus, tmp_path):
    ds, _, bulk_ids = _dataset(corpus_subset_path, synthetic_mineralogy_corpus, tmp_path)
    # 3 label rows per bulk: draw 0 at rungs 50+30, draw 1 at rung 50
    assert len(ds) == 3 * len(bulk_ids)
    item = ds[0]
    assert item["presence"].shape == (N_PHASES,)
    assert item["magnitude"].shape == (N_PHASES,)
    assert item["src"].shape == (22,)
    present = item["presence"] > 0
    assert torch.isclose(item["magnitude"][present].sum(), torch.tensor(1.0))
    assert torch.all(item["magnitude"][~present] == 0)


def test_input_tokens_match_bulk_chemistry(
    corpus_subset_path, synthetic_mineralogy_corpus, tmp_path
):
    ds, chem, _ = _dataset(corpus_subset_path, synthetic_mineralogy_corpus, tmp_path)
    item = ds[0]
    bulk_id = int(ds.bulk_ids[item["bulk_idx"]])
    chem_row = int(np.flatnonzero(chem.sample_ids == bulk_id)[0])
    ref = chem[chem_row]
    assert torch.equal(item["src"], ref["src"])
    assert torch.equal(item["frac"], ref["frac"])


def test_unresolvable_bulk_dropped(
    corpus_subset_path, synthetic_mineralogy_corpus, tmp_path, capsys
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, **_DATA_KWARGS
    )
    bulk_ids = [int(chem.sample_ids[0]), 999_999_999]
    corpus_dir = synthetic_mineralogy_corpus(tmp_path / "corpus", bulk_ids)
    ds = ModeHeadDataset(
        corpus_dir,
        data_kwargs={"csv_path": str(corpus_subset_path), **_DATA_KWARGS},
    )
    assert len(ds.bulk_ids) == 1
    assert len(ds) == 3
    assert "dropping 1 corpus bulks" in capsys.readouterr().out


def test_bulk_ids_filter_restricts_rows(
    corpus_subset_path, synthetic_mineralogy_corpus, tmp_path
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, **_DATA_KWARGS
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:3]]
    corpus_dir = synthetic_mineralogy_corpus(tmp_path / "corpus", bulk_ids)
    data_kwargs = {"csv_path": str(corpus_subset_path), **_DATA_KWARGS}
    ds = ModeHeadDataset(
        corpus_dir, data_kwargs=data_kwargs, bulk_ids=np.array(bulk_ids[:2])
    )
    assert sorted(ds.bulk_ids.tolist()) == sorted(bulk_ids[:2])
    assert len(ds) == 3 * 2
    with pytest.raises(ValueError, match="no label rows"):
        ModeHeadDataset(
            corpus_dir, data_kwargs=data_kwargs, bulk_ids=np.array([999_999_999])
        )

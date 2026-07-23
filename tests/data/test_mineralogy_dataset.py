import numpy as np
import pytest
import torch

from cnmineral.data.bulk_chemistry import BulkChemistryDataset
from cnmineral.data.mineralogy_dataset import JointHeadDataset
from cnmineral.data.phase_vocab import N_PHASES
from cnmineral.models.comp_head import N_OXIDES

_DATA_KWARGS = {"min_class_count": 5}


def _dataset(corpus_subset_path, synthetic_joint_corpus, tmp_path, n_bulks=2):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, **_DATA_KWARGS
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:n_bulks]]
    corpus_dir = synthetic_joint_corpus(tmp_path / "corpus", bulk_ids)
    ds = JointHeadDataset(
        corpus_dir,
        data_kwargs={"csv_path": str(corpus_subset_path), **_DATA_KWARGS},
    )
    return ds, chem, bulk_ids


def test_rows_and_target_shapes(corpus_subset_path, synthetic_joint_corpus, tmp_path):
    ds, _, bulk_ids = _dataset(corpus_subset_path, synthetic_joint_corpus, tmp_path)
    # 3 label rows per bulk: draw 0 at rungs 50+30, draw 1 at rung 50
    assert len(ds) == 3 * len(bulk_ids)
    assert ds.draws_per_bulk == 2.0
    item = ds[0]
    assert item["presence"].shape == (N_PHASES,)
    assert item["magnitude"].shape == (N_PHASES,)
    assert item["src"].shape == (22,)
    present = item["presence"] > 0
    assert torch.isclose(item["magnitude"][present].sum(), torch.tensor(1.0))
    assert torch.all(item["magnitude"][~present] == 0)


def test_input_tokens_match_bulk_chemistry(
    corpus_subset_path, synthetic_joint_corpus, tmp_path
):
    ds, chem, _ = _dataset(corpus_subset_path, synthetic_joint_corpus, tmp_path)
    item = ds[0]
    bulk_id = int(ds.bulk_ids[item["bulk_idx"]])
    chem_row = int(np.flatnonzero(chem.sample_ids == bulk_id)[0])
    ref = chem[chem_row]
    assert torch.equal(item["src"], ref["src"])
    assert torch.equal(item["frac"], ref["frac"])


def test_unresolvable_bulk_dropped(
    corpus_subset_path, synthetic_joint_corpus, tmp_path, capsys
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, **_DATA_KWARGS
    )
    bulk_ids = [int(chem.sample_ids[0]), 999_999_999]
    corpus_dir = synthetic_joint_corpus(tmp_path / "corpus", bulk_ids)
    ds = JointHeadDataset(
        corpus_dir,
        data_kwargs={"csv_path": str(corpus_subset_path), **_DATA_KWARGS},
    )
    assert len(ds.bulk_ids) == 1
    assert len(ds) == 3
    assert "dropping 1 corpus bulks" in capsys.readouterr().out


def test_bulk_ids_filter_restricts_rows(
    corpus_subset_path, synthetic_joint_corpus, tmp_path
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, **_DATA_KWARGS
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:3]]
    corpus_dir = synthetic_joint_corpus(tmp_path / "corpus", bulk_ids)
    data_kwargs = {"csv_path": str(corpus_subset_path), **_DATA_KWARGS}
    ds = JointHeadDataset(
        corpus_dir, data_kwargs=data_kwargs, bulk_ids=np.array(bulk_ids[:2])
    )
    assert sorted(ds.bulk_ids.tolist()) == sorted(bulk_ids[:2])
    assert len(ds) == 3 * 2
    with pytest.raises(ValueError, match="no label rows"):
        JointHeadDataset(
            corpus_dir, data_kwargs=data_kwargs, bulk_ids=np.array([999_999_999])
        )


def test_draw_limit_filter_restricts_rows(
    corpus_subset_path, synthetic_joint_corpus, tmp_path
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, **_DATA_KWARGS
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:2]]
    corpus_dir = synthetic_joint_corpus(tmp_path / "corpus", bulk_ids)
    data_kwargs = {"csv_path": str(corpus_subset_path), **_DATA_KWARGS}
    # draw 0 carries rungs 50+30; draw 1 only rung 50 -- limit keeps 2 rows/bulk
    ds = JointHeadDataset(corpus_dir, data_kwargs=data_kwargs, draw_limit=1)
    assert len(ds) == 2 * len(bulk_ids)
    assert sorted(ds.bulk_ids.tolist()) == sorted(bulk_ids)
    with pytest.raises(ValueError, match="draw_limit"):
        JointHeadDataset(corpus_dir, data_kwargs=data_kwargs, draw_limit=0)


def test_joint_dataset_shapes_and_dtypes(
    corpus_subset_path, synthetic_joint_corpus, tmp_path
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, **_DATA_KWARGS
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:2]]
    corpus_dir = synthetic_joint_corpus(tmp_path / "corpus", bulk_ids)
    ds = JointHeadDataset(
        corpus_dir,
        data_kwargs={"csv_path": str(corpus_subset_path), **_DATA_KWARGS},
    )
    assert len(ds) == 3 * len(bulk_ids)
    item = ds[0]
    assert item["comp_target"].shape == (N_PHASES, N_OXIDES)
    assert item["cond_target"].shape == (3,)
    assert item["c_liq"].shape == item["c_fl"].shape == item["bulk_11"].shape == (N_OXIDES,)
    assert item["f_liq"].ndim == 0 and item["f_fl"].ndim == 0
    for key in ("comp_target", "cond_target", "bulk_11", "f_liq"):
        assert item[key].dtype == torch.float32
    assert torch.allclose(item["comp_target"].sum(-1), torch.ones(N_PHASES))


def test_joint_dataset_cond_join_matches_draws(
    corpus_subset_path, synthetic_joint_corpus, tmp_path
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, **_DATA_KWARGS
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:2]]
    corpus_dir = synthetic_joint_corpus(tmp_path / "corpus", bulk_ids)
    ds = JointHeadDataset(
        corpus_dir,
        data_kwargs={"csv_path": str(corpus_subset_path), **_DATA_KWARGS},
    )
    # conftest _SYNTH_GROUPS/_SYNTH_DRAW_CONDITIONS: draw 0 -> rungs 50 (T 900)
    # and 30 (T 800) at P 6.0; draw 1 -> rung 50 (T 950) at P 2.0
    t_end = ds.cond_target[:, 0]
    p_kbar = ds.cond_target[:, 1]
    assert set(t_end.tolist()) == {900.0, 800.0, 950.0}
    assert torch.all(p_kbar[t_end == 950.0] == 2.0)
    assert torch.all(p_kbar[t_end != 950.0] == 6.0)


def test_joint_dataset_rung50_closure_identity(
    corpus_subset_path, synthetic_joint_corpus, tmp_path
):
    chem = BulkChemistryDataset(
        csv_path=str(corpus_subset_path), require_all_majors=False, **_DATA_KWARGS
    )
    bulk_ids = [int(s) for s in chem.sample_ids[:2]]
    corpus_dir = synthetic_joint_corpus(tmp_path / "corpus", bulk_ids)
    ds = JointHeadDataset(
        corpus_dir,
        data_kwargs={"csv_path": str(corpus_subset_path), **_DATA_KWARGS},
    )
    r50 = ds.rung == 50
    solids = torch.einsum("rp,rpo->ro", ds.magnitude[r50], ds.comp_target[r50])
    recon = (
        (1 - ds.f_liq[r50] - ds.f_fl[r50]).unsqueeze(-1) * solids
        + ds.f_liq[r50].unsqueeze(-1) * ds.c_liq[r50]
        + ds.f_fl[r50].unsqueeze(-1) * ds.c_fl[r50]
    )
    # eps-clip renorm perturbs comps at the 1e-3 level; identity holds to that
    assert (recon - ds.bulk_11[r50]).abs().max() < 5e-3

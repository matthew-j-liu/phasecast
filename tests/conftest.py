"""pytest auto-loaded configuration.

pytest auto-loads this file before any test module imports run, which
makes it the right place to set the ``KMP_DUPLICATE_LIB_OK=TRUE`` env
var.  pytest's own collection machinery imports numpy, which in turn
loads libomp; setting the env var inside ``cnmineral/__init__.py`` is
too late because the test collector pulls numpy in before any
``cnmineral`` import.

Background: macOS's PyTorch and PetThermoTools (via MAGEMin) both link
against libomp.  Concurrent loading aborts with "OpenMP already
initialized" unless this env var is set.  See ``docs/MODEL.md`` macOS
gotcha section.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def pretrain_results_root(monkeypatch, tmp_path: Path) -> Path:
    """Redirect `run_single`'s artifact writes into the test's tmp dir.

    Autouse so no test (including fixtures that drive `run_single` as
    setup) can litter the real `results/pretrain/` tree.  Tests
    asserting on the summary artifact request this fixture by name to
    learn the patched root.
    """
    from cnmineral.training import loop

    root = tmp_path / "results"
    monkeypatch.setattr(loop, "RESULTS_ROOT", root)
    return root


@pytest.fixture(scope="session")
def corpus_subset_path() -> Path:
    """Path to the 2000-row deterministic parquet subset of the bulk corpus.

    Built by ``scripts/build/build_test_fixture.py`` (seed=42).  Tests
    that want a built ``BulkChemistryDataset`` instantiate it themselves
    with the filter combo they need.
    """
    path = REPO_ROOT / "tests" / "fixtures" / "complete_subset.parquet"
    if not path.exists():
        pytest.fail(
            f"test fixture missing: {path}\n"
            f"rebuild with: python scripts/build/build_test_fixture.py"
        )
    return path

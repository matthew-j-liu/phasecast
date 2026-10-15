"""The inference tutorial executes end to end; a renamed library symbol or a
moved checkpoint turns the suite red before a reader hits it."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = Path(__file__).resolve().parents[2] / "docs" / "tutorial" / "inference.ipynb"


def test_inference_notebook_executes():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(
        nb, timeout=300, kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()

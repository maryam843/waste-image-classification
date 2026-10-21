"""T13: the 2026 notebook runs headless (smoke config). T5: the README commands run. Notebook hygiene."""

import json
import os
import shlex
import subprocess
import sys

import nbformat
import pytest


@pytest.mark.slow
def test_2026_notebook_executes_headless_smoke(root, monkeypatch):
    from nbclient import NotebookClient

    monkeypatch.setenv("WASTE_CONFIG", "smoke")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = nbformat.read(root / "notebooks" / "waste_classification_2026.ipynb", as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(root / "notebooks")}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors
    res = json.loads((root / "runs" / "smoke" / "results.json").read_text())
    assert res["status"] == "smoke"
    assert (root / "runs" / "smoke" / "hf_space_build" / "model.pt").exists()


def test_notebook_is_clean(root):
    nb = json.loads((root / "notebooks" / "waste_classification_2026.ipynb").read_text())
    assert "colab" not in nb["metadata"]
    assert all(c.get("metadata", {}) == {} for c in nb["cells"])
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code)


def readme_commands(root) -> list[str]:
    text = (root / "README.md").read_text()
    block = text.split("<!-- readme-commands -->", 1)[1].split("```bash", 1)[1].split("```", 1)[0]
    return [line.split("#", 1)[0].strip() for line in block.strip().splitlines() if line.strip()]


def test_readme_has_expected_commands(root):
    cmds = readme_commands(root)
    assert cmds[0].startswith("python -m venv") and "pytest" in cmds
    assert any(c.startswith("python -m waste_cls.cli run --config smoke") for c in cmds)


@pytest.mark.slow
def test_readme_python_commands_run(root):
    """Runs every README command that does not create environments or install packages (those run in T6)."""
    ran = 0
    for cmd in readme_commands(root):
        if cmd.startswith(("python -m venv", "source", "pip ", "pytest")):
            continue
        args = shlex.split(cmd)
        assert args[0] == "python"
        r = subprocess.run([sys.executable, *args[1:]], cwd=root, capture_output=True, text=True,
                           env={**os.environ, "MPLBACKEND": "Agg"}, timeout=900)
        assert r.returncode == 0, f"{cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        ran += 1
    assert ran == 3

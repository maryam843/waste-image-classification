"""Tests for the CLI, the helper scripts (README claims, PII scan, HF Space build) and the demo app."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

import check_readme_claims as CRC
import pii_scan
import prepare_hf_space
from waste_cls import cli
from waste_cls.config import IMAGENET_MEAN, IMAGENET_STD, get_config
from waste_cls.models import build_model, export_model
from waste_cls.pipeline import run_pipeline

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def smoke_results(tmp_path_factory):
    d = tmp_path_factory.mktemp("smoke")
    cfg = get_config("smoke", image_size=16, synthetic_per_class=12, max_epochs=1)
    res = run_pipeline(cfg, d / "results.json", d / "fig", d / "demo.pt", log=None)
    return res, d


@pytest.fixture()
def exported(tmp_path):
    m = build_model("smallcnn").eval()
    p = tmp_path / "m.pt"
    export_model(m, "smallcnn", p, 32, ("glass", "metal", "paper", "plastic"))
    img = tmp_path / "x.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (50, 40, 3), dtype=np.uint8)).save(img)
    return p, img


# ------------------------------------------------------------------ CLI


def test_cli_predict_with_cam(exported, tmp_path, capsys):
    model, img = exported
    assert cli.main(["predict", "--model", str(model), "--image", str(img), "--cam", str(tmp_path / "c.png")]) == 0
    out = capsys.readouterr().out
    top3 = json.loads(out[: out.rindex("}") + 1])["top3"]
    assert len(top3) == 3 and top3[0][1] >= top3[1][1]
    assert (tmp_path / "c.png").exists()


def test_cli_prepare_data_smoke(capsys):
    assert cli.main(["prepare-data", "--config", "smoke"]) == 0
    assert json.loads(capsys.readouterr().out)["source"] == "synthetic"


def test_cli_run_smoke(tmp_path, monkeypatch, capsys):
    import waste_cls.config as C
    monkeypatch.setitem(C.CONFIGS, "smoke", get_config("smoke", image_size=16, synthetic_per_class=12,
                                                       max_epochs=1, models=("dnn",)))
    monkeypatch.setattr("waste_cls.experiment.GRADCAM_CAPABLE", ("dnn",))
    monkeypatch.setattr("waste_cls.pipeline.GRADCAM_CAPABLE", ())
    assert cli.main(["run", "--config", "smoke", "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "results.json").exists()
    assert (tmp_path / "o" / "figures" / "results_bar.png").exists()


@pytest.mark.slow
def test_module_entry_point_help():
    r = subprocess.run([sys.executable, "-m", "waste_cls.cli", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "predict" in r.stdout


# ------------------------------------------------------------------ README claims


README_TODO = "# x\n<!-- RESULTS:START -->\nTODO: after the Colab run\n<!-- RESULTS:END -->\n2025: F1 0.6988\n"


def _full(res):
    r = copy.deepcopy(res)
    r["status"] = "full"
    return r


def test_claims_skip_while_todo():
    errors, msgs = CRC.check(README_TODO, None)
    assert errors == 0 and any("SKIPPED" in m for m in msgs)


def test_claims_pass_with_rendered_table(smoke_results):
    res = _full(smoke_results[0])
    table = CRC.render_table(res)
    readme = f"# x\n<!-- RESULTS:START -->\n{table}\n<!-- RESULTS:END -->\n2025 notebook: 0.6988 on 100 images\n"
    errors, msgs = CRC.check(readme, res)
    assert errors == 0, msgs


def test_claims_fail_on_edited_number_and_stray_number(smoke_results):
    res = _full(smoke_results[0])
    table = CRC.render_table(res).replace("| n/a | n/a |", "| n/a | n/a |", 1)
    tampered = table.replace(CRC.fmt(res["baselines"]["majority"]["accuracy"]), "0.999", 1)
    readme = f"<!-- RESULTS:START -->\n{tampered}\n<!-- RESULTS:END -->\nThe CNN reached 0.987 accuracy.\n"
    errors, msgs = CRC.check(readme, res)
    assert errors >= 2
    assert any("differs" in m for m in msgs) and any("0.987" in m for m in msgs)


def test_claims_reject_smoke_results_and_missing_results(smoke_results):
    readme = "<!-- RESULTS:START -->\n| table |\n<!-- RESULTS:END -->\n"
    assert CRC.check(readme, smoke_results[0])[0] == 1
    assert CRC.check(readme, None)[0] == 1
    assert CRC.check("no block here", None)[0] == 1


def test_claims_forbidden_phrases_and_improvement():
    bad = README_TODO + "A real-time system deployed for sorting.\nThe 2025 F1 0.70 improved to 0.95.\n"
    errors, msgs = CRC.check(bad, None)
    assert errors == 3
    ok = README_TODO + "This is not a real-time system and was never deployed for sorting.\n"
    assert CRC.check(ok, None)[0] == 0


def test_claims_main_and_render(tmp_path, smoke_results, capsys):
    res, _ = smoke_results
    readme = tmp_path / "README.md"
    readme.write_text(README_TODO)
    assert CRC.main(["--readme", str(readme), "--results", str(tmp_path / "none.json")]) == 0
    assert CRC.main(["--render", "--results", str(tmp_path / "none.json")]) == 1
    full = tmp_path / "r.json"
    full.write_text(json.dumps(_full(res)))
    assert CRC.main(["--render", "--results", str(full)]) == 0
    assert "| Model |" in capsys.readouterr().out


def test_repo_readme_claims(root):
    r = subprocess.run([sys.executable, "scripts/check_readme_claims.py"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------ PII scan


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path, email="185722545+maryam843@users.noreply.github.com"):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "user.email", email)
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def test_pii_scan_clean_repo(tmp_path, capsys):
    repo = _repo(tmp_path)
    (repo / "a.py").write_text("print('hello')\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    assert pii_scan.main(["--repo", str(repo)]) == 0
    assert "0 findings" in capsys.readouterr().out


def test_pii_scan_finds_planted_items_in_history(tmp_path, capsys):
    repo = _repo(tmp_path)
    planted = {
        "t.txt": "token " + "hf_" + "a" * 34,
        "k.txt": "AKIA" + "ABCDEFGHIJKLMNOP",
        "n.ipynb": json.dumps({"cells": [{"cell_type": "code", "source": ["# " + "6" * 8 + "\n"],
                                          "metadata": {"execution" + "Info": {"user" + "Id": "1" * 20}},
                                          "outputs": [{"data": {"image/png": "AKIA" + "Z" * 16}}]}]}),
        "m.md": "mail me at someone" + "@" + "example.com, see github.com/user-" + "attachments/assets/x",
        "p.md": "/Users/" + "someone/file",
        "s.txt": "Student " + "ID: " + "1234" + "5678",
    }
    for name, text in planted.items():
        (repo / name).write_text(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add")
    for name in planted:
        (repo / name).unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "remove")
    assert pii_scan.main(["--repo", str(repo), "--no-history"]) == 0
    capsys.readouterr()
    assert pii_scan.main(["--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    for kind in ("huggingface-token", "aws-access-key", "student-id-line", "colab-metadata", "google-user-id",
                 "email", "demo-video-link", "personal-path", "student-id-labelled"):
        assert kind in out, kind
    assert out.count("aws-access-key") == 1  # the base64 image payload is ignored
    assert "a" * 34 not in out  # values are masked


def test_pii_scan_identity_and_denylist(tmp_path, capsys):
    repo = _repo(tmp_path, email="someone" + "@" + "example.com")
    (repo / "a.txt").write_text("value SECRETVALUE123 here")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "x")
    deny = tmp_path / "deny.txt"
    deny.write_text("SECRETVALUE123\n")
    assert pii_scan.main(["--repo", str(repo), "--denylist", str(deny)]) == 1
    out = capsys.readouterr().out
    assert "git-identity" in out and "denylist" in out


def test_pii_scan_empty_repo(tmp_path):
    assert pii_scan.main(["--repo", str(_repo(tmp_path))]) == 0


def test_pii_scan_this_repo_full_history(root):
    r = subprocess.run([sys.executable, "scripts/pii_scan.py"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


# ------------------------------------------------------------------ Hugging Face demo


def test_hf_app_predicts_with_dummy_model(exported):
    sys.path.insert(0, str(prepare_hf_space.ROOT / "hf_space"))
    try:
        import app
    finally:
        sys.path.pop(0)
    from waste_cls.models import load_exported

    assert app.MEAN == IMAGENET_MEAN and app.STD == IMAGENET_STD
    model, blob = load_exported(exported[0])
    img = Image.open(exported[1])
    probs, cam = app.predict(img, model, blob)
    assert len(probs) == 3 and all(0 <= v <= 1 for v in probs.values())
    assert cam.shape == (32, 32, 3)
    probs2, _ = app.predict(np.asarray(img), model, blob)
    assert probs2 == pytest.approx(probs)
    with pytest.raises(ValueError):
        app.predict(None, model, blob)
    gr = pytest.importorskip("gradio")
    demo = app.build_demo(model, blob)
    assert isinstance(demo, gr.Interface)


@pytest.mark.slow
def test_hf_space_build_runs_standalone(exported, tmp_path):
    """The Space folder must work WITHOUT the waste_cls package (imports from copied files)."""
    out = prepare_hf_space.build(exported[0], tmp_path / "space")
    assert prepare_hf_space.main(["--model", str(exported[0]), "--out", str(tmp_path / "space2")]) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "README.md", "app.py", "gradcam.py", "model.pt", "models.py", "requirements.txt"]
    code = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "def block(name, *a, **k):\n"
        "    if name.startswith('waste_cls'): raise ImportError('blocked')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = block\n"
        "import app\n"
        "from PIL import Image\n"
        "m, b = app.load_exported(app.MODEL_PATH)\n"
        f"p, cam = app.predict(Image.open(r'{exported[1]}'), m, b)\n"
        "assert len(p) == 3 and cam.shape[2] == 3\n"
        "print('ok', p)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=out, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ok")


def test_torch_load_is_weights_only(exported):
    blob = torch.load(exported[0], weights_only=True)
    assert blob["arch"] == "smallcnn"

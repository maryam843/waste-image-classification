"""T4 integration, overfit sanity check and T8 reproducibility, all on CPU with tiny synthetic images."""

import json

import pytest
import torch
from torch.utils.data import TensorDataset

from waste_cls.config import get_config
from waste_cls.data import make_split, prepare_data
from waste_cls.experiment import run_single
from waste_cls.models import build_model
from waste_cls.pipeline import run_pipeline
from waste_cls.train import make_loader, run_epoch, seed_everything


@pytest.mark.parametrize("name", ["dnn", "smallcnn", "resnet18"])
def test_one_epoch_train_eval_loop(name, tiny_cfg, tiny_data):
    record, model = run_single(name, 0, tiny_data, tiny_cfg, log=None)
    assert record["epochs_run"] == 1
    assert 0.0 <= record["test"]["accuracy"] <= 1.0
    assert len(record["test"]["y_pred"]) == len(tiny_data.y["test"])


@pytest.mark.parametrize("name", ["dnn", "smallcnn", "resnet18"])
def test_each_model_can_overfit_16_images(name):
    """Proves the training loop works: loss on a fixed batch of 16 must drop clearly."""
    seed_everything(0)
    size = 32
    x = torch.randn(16, 3, size, size)
    y = torch.arange(16) % 4
    loader = make_loader(TensorDataset(x, y), 16, shuffle=False)
    model = build_model(name, image_size=size, pretrained=False)
    lr = 1e-2 if name == "resnet18" else 1e-3
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    first = run_epoch(model, loader, crit, "cpu")["loss"]
    for _ in range(60):
        run_epoch(model, loader, crit, "cpu", opt)
    last = run_epoch(model, loader, crit, "cpu")["loss"]
    assert last < 0.5 * first, (first, last)


def test_reproducible_split_and_first_epoch_loss(tiny_cfg):
    d1, d2 = prepare_data(tiny_cfg), prepare_data(tiny_cfg)
    assert d1.split_hash == d2.split_hash
    losses = []
    for d in (d1, d2):
        for name in ("dnn", "smallcnn"):
            rec, _ = run_single(name, 7, d, tiny_cfg, log=None)
            losses.append((name, rec["history"]["train_loss"][0]))
    assert losses[:2] == losses[2:]


def test_different_seed_changes_split():
    from waste_cls.data import Record
    recs = [Record("", f"{c}/{i}", c, f"{c}{i}") for c in ("glass", "metal", "paper", "plastic")
            for i in range(20)]
    assert make_split(recs, seed=1).rows != make_split(recs, seed=2).rows


def test_full_pipeline_smoke_writes_results_figures_and_export(tmp_path):
    cfg = get_config("smoke", image_size=16, synthetic_per_class=12, max_epochs=1)
    res = run_pipeline(cfg, tmp_path / "results.json", tmp_path / "fig", tmp_path / "exp" / "demo.pt",
                       log=None)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["status"] == "smoke" and set(saved["models"]) == {"dnn", "smallcnn", "resnet18"}
    assert saved["data"]["split_csv_sha256"] == res["data"]["split_csv_sha256"]
    assert saved["environment"]["torch"] == torch.__version__
    for f in ("learning_curves", "confusion_matrices", "errors_dnn", "errors_smallcnn", "errors_resnet18",
              "gradcam", "results_bar"):
        assert (tmp_path / "fig" / f"{f}.png").stat().st_size > 1000
    assert (tmp_path / "exp" / "demo.pt").exists()
    assert saved["demo_model"]["arch"] in ("smallcnn", "resnet18")

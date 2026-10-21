import inspect

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from waste_cls.baselines import logistic_regression_baseline, majority_baseline, pixels
from waste_cls.config import get_config
from waste_cls.evaluate import compute_metrics, evaluate_model, predict, summarise, wilson_ci
from waste_cls.models import build_model
from waste_cls.train import make_loader, run_epoch, seed_everything, train_model


def test_wilson_ci_matches_known_value_for_70_of_100():
    lo, hi = wilson_ci(70, 100)
    assert round(lo, 3) == 0.604 and round(hi, 3) == 0.781


def test_wilson_ci_edges():
    assert wilson_ci(0, 10)[0] == 0.0
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson_ci(0, 0)


def test_compute_metrics():
    m = compute_metrics([0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 1, 1])
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["correct"] == 5 and m["n"] == 6
    assert m["confusion_matrix"][0] == [1, 1, 0, 0]
    assert m["per_class"]["glass"]["recall"] == 0.5
    assert 0 < m["macro_f1"] < 1


def test_summarise():
    s = summarise([0.5, 0.7, 0.9])
    assert s["mean"] == pytest.approx(0.7) and s["sd"] == pytest.approx(0.2)
    assert summarise([0.4])["sd"] == 0.0


def _toy_loaders(n=32, seed=0, size=8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, size, size, generator=g)
    y = torch.arange(n) % 4
    x[:, 0] += y.view(-1, 1, 1).float()  # learnable signal
    ds = TensorDataset(x, y)
    return make_loader(ds, 8, True, seed), make_loader(ds, 8, False, seed)


def test_train_model_signature_has_no_test_loader():
    params = inspect.signature(train_model).parameters
    assert list(params)[:3] == ["model", "train_loader", "val_loader"]
    assert not any("test" in p for p in params)


def test_train_model_early_stops_and_returns_best_val_state(tmp_path):
    cfg = get_config("smoke", max_epochs=30, early_stop_patience=2, lr_patience=1)
    tr, va = _toy_loaders()
    seed_everything(0)
    model = build_model("dnn", image_size=8)
    res = train_model(model, tr, va, cfg, checkpoint_path=tmp_path / "ck" / "best.pt", log=None)
    h = res.history
    assert res.epochs_run == len(h["val_f1"]) <= 30
    assert res.best_val_f1 == max(h["val_f1"])
    assert res.best_epoch == int(np.argmax(h["val_f1"])) + 1
    assert (tmp_path / "ck" / "best.pt").exists()
    saved = torch.load(tmp_path / "ck" / "best.pt", weights_only=True)
    assert all(torch.equal(saved[k], res.best_state[k]) for k in saved)


def test_lr_scheduler_is_stepped_with_validation_f1(monkeypatch):
    seen = []
    real_step = torch.optim.lr_scheduler.ReduceLROnPlateau.step

    def spy(self, metrics, *a, **kw):
        seen.append(float(metrics))
        return real_step(self, metrics, *a, **kw)

    monkeypatch.setattr(torch.optim.lr_scheduler.ReduceLROnPlateau, "step", spy)
    cfg = get_config("smoke", max_epochs=4, early_stop_patience=10)
    tr, va = _toy_loaders()
    res = train_model(build_model("dnn", image_size=8), tr, va, cfg, log=None)
    assert seen == res.history["val_f1"]


def test_run_epoch_eval_does_not_change_weights():
    tr, va = _toy_loaders(size=16)
    m = build_model("smallcnn")
    before = {k: v.clone() for k, v in m.state_dict().items()}
    run_epoch(m, va, torch.nn.CrossEntropyLoss(), "cpu")
    after = m.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if "running" not in k and "num_batches" not in k)


def test_predict_and_evaluate_model():
    _, va = _toy_loaders()
    m = build_model("dnn", image_size=8)
    yt, yp, p = predict(m, va)
    assert p.shape == (32, 4) and np.allclose(p.sum(1), 1, atol=1e-5)
    out = evaluate_model(m, va)
    assert out["n"] == 32 and len(out["y_pred"]) == 32


def test_majority_baseline():
    out = majority_baseline([0, 1, 1, 2], [1, 1, 0, 3])
    assert out["predicted_class"] == 1 and out["accuracy"] == 0.5


def test_logreg_baseline_chooses_c_on_val(tiny_data):
    out = logistic_regression_baseline(tiny_data.x, {k: v.numpy() for k, v in tiny_data.y.items()},
                                       size=8, c_grid=(0.01, 1.0))
    assert out["chosen_C"] in (0.01, 1.0)
    assert set(out["val_macro_f1_by_C"]) == {"0.01", "1.0"}
    assert out["n"] == len(tiny_data.y["test"])
    assert pixels(tiny_data.x["train"], 8).shape == (len(tiny_data.y["train"]), 192)

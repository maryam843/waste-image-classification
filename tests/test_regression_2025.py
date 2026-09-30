"""Regression tests for the problems found in the 2025 notebook."""

from waste_cls.experiment import run_all, run_single, select_demo_model
from waste_cls.models import build_model


def test_test_loader_is_used_exactly_once_per_model_and_after_training(tiny_cfg, tiny_data):
    calls = []

    def spy_evaluator(model, loader, device):
        calls.append(len(loader.dataset))
        from waste_cls.evaluate import evaluate_model
        return evaluate_model(model, loader, device)

    cfg = tiny_cfg.__class__(**{**tiny_cfg.to_dict(), "models": ("dnn", "smallcnn"), "seeds": (0, 1)})
    runs, _ = run_all(cfg, tiny_data, evaluator=spy_evaluator, log=None)
    assert len(calls) == 4  # 2 models x 2 seeds, once each
    assert all(n == len(tiny_data.y["test"]) for n in calls)


def test_model_selection_never_sees_test_data(tiny_cfg, tiny_data, monkeypatch):
    """train_model only gets train and val loaders; neither contains the test images."""
    import waste_cls.experiment as E

    seen = {}
    real = E.train_model

    def spy(model, train_loader, val_loader, cfg, device, ckpt, log=None):
        seen["train"] = train_loader.dataset.images
        seen["val"] = val_loader.dataset.images
        return real(model, train_loader, val_loader, cfg, device, ckpt, log=None)

    monkeypatch.setattr(E, "train_model", spy)
    run_single("dnn", 0, tiny_data, tiny_cfg, log=None)
    test_imgs = {bytes(t.numpy().tobytes()) for t in tiny_data.x["test"]}
    for split in ("train", "val"):
        assert not test_imgs & {bytes(t.numpy().tobytes()) for t in seen[split]}


def test_fresh_model_is_created_for_every_experiment(tiny_cfg, tiny_data):
    built = []

    def factory(name, **kw):
        m = build_model(name, **kw)
        built.append(m)
        return m

    cfg = tiny_cfg.__class__(**{**tiny_cfg.to_dict(), "models": ("dnn",), "seeds": (0, 0)})
    run_all(cfg, tiny_data, model_factory=factory, log=None)
    assert len(built) == 2 and built[0] is not built[1]


def test_same_seed_gives_same_start_weights_so_runs_do_not_continue_each_other(tiny_cfg, tiny_data):
    r1, m1 = run_single("smallcnn", 0, tiny_data, tiny_cfg, log=None)
    r2, m2 = run_single("smallcnn", 0, tiny_data, tiny_cfg, log=None)
    # a continued run would start from trained weights and give a different first-epoch loss
    assert r1["history"]["train_loss"][0] == r2["history"]["train_loss"][0]


def test_reported_architecture_is_the_trained_one(tiny_cfg, tiny_data):
    """2025 wrote [1024,512,256,128] in the conclusion but trained [512,256,128]."""
    record, model = run_single("dnn", 0, tiny_data, tiny_cfg, log=None)
    widths = [m.out_features for m in model.network if hasattr(m, "out_features")]
    assert widths[:-1] == list(tiny_cfg.dnn_hidden) == [512, 256, 128]
    assert record["params_total"] == sum(p.numel() for p in model.parameters())


def test_demo_model_selected_on_validation_not_test():
    runs = [
        {"model": "smallcnn", "seed": 0, "best_val_macro_f1": 0.9, "test": {"macro_f1": 0.1}},
        {"model": "resnet18", "seed": 0, "best_val_macro_f1": 0.8, "test": {"macro_f1": 0.99}},
        {"model": "dnn", "seed": 0, "best_val_macro_f1": 0.95, "test": {"macro_f1": 0.99}},
    ]
    pick = select_demo_model(runs)
    assert pick["arch"] == "smallcnn" and pick["val_macro_f1"] == 0.9
    import pytest
    with pytest.raises(ValueError):
        select_demo_model(runs[2:])

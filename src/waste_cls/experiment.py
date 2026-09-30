"""Experiment runner: one fresh model per (architecture, seed), test evaluated exactly once per model.

In 2025 the "60-epoch experiment" silently continued from an already trained model because
cells ran out of order. Here every run builds its model through the factory, so that cannot happen.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision

from .baselines import logistic_regression_baseline, majority_baseline
from .config import CLASSES, Config
from .data import DataBundle, ImageTensorDataset, eval_transform, train_transform
from .evaluate import evaluate_model, summarise
from .models import DISPLAY_NAMES, build_model, count_parameters
from .train import make_loader, seed_everything, train_model

GRADCAM_CAPABLE = ("smallcnn", "resnet18")


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def environment_info(device: str) -> dict:
    gpu = torch.cuda.get_device_name(0) if (device.startswith("cuda") and torch.cuda.is_available()) else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device": device,
        "gpu_name": gpu,
    }


def run_baselines(data: DataBundle, cfg: Config) -> dict:
    maj = majority_baseline(data.y["train"].numpy(), data.y["test"].numpy())
    logreg = logistic_regression_baseline(data.x, {k: v.numpy() for k, v in data.y.items()},
                                          size=cfg.logreg_size, c_grid=cfg.logreg_c_grid)
    return {"majority": maj, "logreg_pixels": logreg}


def run_single(name: str, seed: int, data: DataBundle, cfg: Config, device: str = "cpu",
               model_factory=build_model, evaluator=evaluate_model, checkpoint_dir: str | Path | None = None,
               log=print) -> tuple[dict, torch.nn.Module]:
    """Train one fresh model on train, select on val, then evaluate on test once."""
    seed_everything(seed)
    model = model_factory(name, image_size=cfg.image_size, pretrained=cfg.pretrained,
                          dnn_hidden=cfg.dnn_hidden, dnn_dropout=cfg.dnn_dropout)
    augment = name in cfg.augment_models
    train_ds = ImageTensorDataset(data.x["train"], data.y["train"], train_transform(augment))
    val_ds = ImageTensorDataset(data.x["val"], data.y["val"], eval_transform())
    test_ds = ImageTensorDataset(data.x["test"], data.y["test"], eval_transform())
    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, seed=seed, num_workers=cfg.num_workers)
    val_loader = make_loader(val_ds, cfg.batch_size, shuffle=False, seed=seed, num_workers=cfg.num_workers)

    ckpt = Path(checkpoint_dir) / f"{name}_seed{seed}.pt" if checkpoint_dir else None
    if log:
        log(f"[{name} seed {seed}] training (augmentation={'on' if augment else 'off'})")
    result = train_model(model, train_loader, val_loader, cfg, device, ckpt, log=log)
    model.load_state_dict(result.best_state)

    # the only place the test set is touched
    test_loader = make_loader(test_ds, cfg.batch_size, shuffle=False, seed=seed, num_workers=0)
    test = evaluator(model, test_loader, device)
    model.to("cpu").eval()
    record = {
        "model": name,
        "seed": seed,
        "augmentation": augment,
        "params_total": count_parameters(model),
        "params_trainable": count_parameters(model, trainable_only=True),
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "best_val_macro_f1": result.best_val_f1,
        "train_time_s": result.train_time_s,
        "history": result.history,
        "test": test,
    }
    if log:
        log(f"[{name} seed {seed}] best epoch {result.best_epoch} val F1 {result.best_val_f1:.3f} | "
            f"test acc {test['accuracy']:.3f} macro F1 {test['macro_f1']:.3f} | {result.train_time_s:.0f}s")
    return record, model


def run_all(cfg: Config, data: DataBundle, device: str = "cpu", checkpoint_dir: str | Path | None = None,
            log=print, model_factory=build_model, evaluator=evaluate_model) -> tuple[list[dict], dict]:
    """All architectures x seeds. Returns (run records, best-on-validation model per architecture)."""
    runs, best_models, best_scores = [], {}, {}
    for name in cfg.models:
        for seed in cfg.seeds:
            record, model = run_single(name, seed, data, cfg, device, model_factory, evaluator,
                                       checkpoint_dir, log)
            runs.append(record)
            if record["best_val_macro_f1"] > best_scores.get(name, -1.0):
                best_scores[name] = record["best_val_macro_f1"]
                best_models[name] = (seed, model)
    return runs, best_models


def summarise_models(runs: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for name in dict.fromkeys(r["model"] for r in runs):
        rs = [r for r in runs if r["model"] == name]
        out[name] = {
            "display_name": DISPLAY_NAMES.get(name, name),
            "params_total": rs[0]["params_total"],
            "params_trainable": rs[0]["params_trainable"],
            "augmentation": rs[0]["augmentation"],
            "n_seeds": len(rs),
            "test_accuracy": summarise([r["test"]["accuracy"] for r in rs]),
            "test_macro_f1": summarise([r["test"]["macro_f1"] for r in rs]),
            "val_macro_f1": summarise([r["best_val_macro_f1"] for r in rs]),
            "train_time_s": summarise([r["train_time_s"] for r in rs]),
            "best_epoch": [r["best_epoch"] for r in rs],
            "test_accuracy_wilson_95_per_seed": [r["test"]["accuracy_wilson_95"] for r in rs],
        }
    return out


def select_demo_model(runs: list[dict]) -> dict:
    """Best single run by VALIDATION macro F1 among models that support Grad-CAM."""
    candidates = [r for r in runs if r["model"] in GRADCAM_CAPABLE]
    if not candidates:
        raise ValueError("no Grad-CAM capable model was trained")
    best = max(candidates, key=lambda r: r["best_val_macro_f1"])
    return {"arch": best["model"], "seed": best["seed"], "val_macro_f1": best["best_val_macro_f1"],
            "selected_by": "highest validation macro F1 among Grad-CAM capable models"}


def build_results(cfg: Config, data: DataBundle, baselines: dict, runs: list[dict], device: str) -> dict:
    return {
        "schema_version": 1,
        "status": cfg.name,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": cfg.to_dict(),
        "environment": environment_info(device),
        "data": {
            "source": data.source,
            "classes": list(CLASSES),
            "split_csv_sha256": data.split_hash,
            "counts": data.counts,
            "n_images": sum(sum(v.values()) for v in data.counts.values()),
            "dropped_label_conflicts": data.dropped,
            "test_y_true": data.y["test"].tolist(),
            "test_source_names": data.names["test"],
        },
        "baselines": baselines,
        "models": summarise_models(runs),
        "runs": runs,
        "demo_model": select_demo_model(runs),
    }


def save_results(results: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return path

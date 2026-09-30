"""Training loop. Model selection, early stopping and the LR schedule use the VALIDATION set only.

train_model() does not accept a test loader on purpose: in 2025 the test set was used for
early stopping, LR scheduling and checkpoint choice, which made the reported score optimistic.
"""

from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import Config


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int = 0,
                num_workers: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=num_workers, worker_init_fn=_seed_worker,
                      persistent_workers=num_workers > 0, pin_memory=torch.cuda.is_available())


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str | torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> dict:
    """One pass over loader. Trains if an optimizer is given, otherwise evaluates without gradients."""
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            y_true.append(y.cpu())
            y_pred.append(logits.argmax(1).cpu())
    yt, yp = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
    return {
        "loss": total_loss / max(n, 1),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "accuracy": float((yt == yp).mean()) if n else 0.0,
    }


@dataclass
class TrainResult:
    best_state: dict
    best_epoch: int
    best_val_f1: float
    epochs_run: int
    train_time_s: float
    history: dict = field(default_factory=dict)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: Config,
                device: str | torch.device = "cpu", checkpoint_path: str | Path | None = None,
                log=print) -> TrainResult:
    """Adam + ReduceLROnPlateau on val macro F1 + early stopping on val macro F1.

    The best-on-validation weights are returned (and optionally written to checkpoint_path,
    which must be in a git-ignored folder).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience
    )
    history: dict[str, list[float]] = {k: [] for k in
                                       ("train_loss", "train_f1", "val_loss", "val_f1", "val_acc", "lr")}
    best_f1, best_epoch, best_state, bad_epochs = -1.0, 0, None, 0
    start = time.perf_counter()
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va["macro_f1"])
        for k, v in (("train_loss", tr["loss"]), ("train_f1", tr["macro_f1"]), ("val_loss", va["loss"]),
                     ("val_f1", va["macro_f1"]), ("val_acc", va["accuracy"]), ("lr", lr_now)):
            history[k].append(float(v))
        improved = va["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = va["macro_f1"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            bad_epochs += 1
        if log:
            log(f"  epoch {epoch:3d}  train loss {tr['loss']:.4f} f1 {tr['macro_f1']:.3f} | "
                f"val loss {va['loss']:.4f} f1 {va['macro_f1']:.3f}{' *' if improved else ''}")
        if bad_epochs >= cfg.early_stop_patience:
            break
    return TrainResult(best_state=best_state, best_epoch=best_epoch, best_val_f1=float(best_f1),
                       epochs_run=epoch, train_time_s=time.perf_counter() - start, history=history)

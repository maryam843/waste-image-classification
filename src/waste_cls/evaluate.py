"""Test-set evaluation: accuracy, macro F1, per-class report, Wilson CI, confusion matrix."""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from .config import CLASSES


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion k/n."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str | torch.device = "cpu"):
    """Return (y_true, y_pred, probabilities) as numpy arrays."""
    model.to(device).eval()
    ys, probs = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        ys.append(y)
    p = torch.cat(probs).numpy()
    return torch.cat(ys).numpy(), p.argmax(1), p


def compute_metrics(y_true, y_pred, class_names: tuple[str, ...] = CLASSES) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    lo, hi = wilson_ci(correct, n)
    return {
        "n": n,
        "correct": correct,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy_wilson_95": [lo, hi],
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i, c in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: str | torch.device = "cpu",
                   class_names: tuple[str, ...] = CLASSES) -> dict:
    y_true, y_pred, probs = predict(model, loader, device)
    out = compute_metrics(y_true, y_pred, class_names)
    out["y_pred"] = y_pred.tolist()
    out["y_true"] = y_true.tolist()
    return out


def summarise(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "sd": sd, "values": [float(v) for v in arr]}

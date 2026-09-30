"""Baselines: majority class and logistic regression on small raw-pixel images."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import compute_metrics


def majority_baseline(y_train, y_test) -> dict:
    """Always predict the most common training class."""
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    majority = int(np.bincount(y_train).argmax())
    out = compute_metrics(y_test, np.full_like(y_test, majority))
    out["predicted_class"] = majority
    return out


def pixels(images_uint8: torch.Tensor, size: int) -> np.ndarray:
    """Downsample uint8 images [N,3,H,W] to size x size and flatten to floats in [0,1]."""
    x = images_uint8.float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return x.flatten(1).numpy()


def logistic_regression_baseline(x: dict, y: dict, size: int = 64,
                                 c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0), seed: int = 0) -> dict:
    """Pick C on the validation set, then evaluate the chosen model once on test."""
    xtr, xva, xte = (pixels(x[s], size) for s in ("train", "val", "test"))
    ytr, yva, yte = (np.asarray(y[s]) for s in ("train", "val", "test"))
    start = time.perf_counter()
    val_scores, best = {}, None
    for c in c_grid:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed))
        clf.fit(xtr, ytr)
        score = float(f1_score(yva, clf.predict(xva), average="macro", zero_division=0))
        val_scores[str(c)] = score
        if best is None or score > best[0]:
            best = (score, c, clf)
    _, best_c, best_clf = best
    out = compute_metrics(yte, best_clf.predict(xte))
    out.update({"chosen_C": best_c, "val_macro_f1_by_C": val_scores, "image_size": size,
                "fit_time_s": time.perf_counter() - start, "y_pred": best_clf.predict(xte).tolist()})
    return out

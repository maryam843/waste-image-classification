"""Figures, all computed from variables (never hand-typed numbers). Saved at dpi 150."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .config import CLASSES  # noqa: E402
from .data import eval_transform  # noqa: E402
from .gradcam import GradCAM, overlay  # noqa: E402
from .models import DISPLAY_NAMES  # noqa: E402

DPI = 150
C_TRAIN, C_VAL = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9,
    "axes.titlesize": 10, "legend.frameon": False,
})


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def learning_curves(runs: list[dict], path) -> Path:
    """Train vs validation loss and macro F1 per model; one thin line per seed."""
    names = list(dict.fromkeys(r["model"] for r in runs))
    fig, axes = plt.subplots(len(names), 2, figsize=(9, 2.6 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        for r in (r for r in runs if r["model"] == name):
            h = r["history"]
            ep = np.arange(1, len(h["train_loss"]) + 1)
            for ax, tr, va in ((axes[i, 0], h["train_loss"], h["val_loss"]),
                               (axes[i, 1], h["train_f1"], h["val_f1"])):
                ax.plot(ep, tr, color=C_TRAIN, lw=1.5, alpha=0.85)
                ax.plot(ep, va, color=C_VAL, lw=1.5, alpha=0.85)
                ax.axvline(r["best_epoch"], color=MUTED, lw=0.6, ls=":")
        axes[i, 0].set_title(f"{DISPLAY_NAMES.get(name, name)}: loss")
        axes[i, 1].set_title(f"{DISPLAY_NAMES.get(name, name)}: macro F1")
        axes[i, 1].set_ylim(0, 1)
        for ax in axes[i]:
            ax.set_xlabel("epoch")
    axes[0, 0].plot([], [], color=C_TRAIN, label="train")
    axes[0, 0].plot([], [], color=C_VAL, label="validation")
    axes[0, 0].plot([], [], color=MUTED, ls=":", label="best val epoch")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Learning curves (one line per seed)", y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def confusion_matrices(matrices: dict[str, list[list[int]]], path, class_names=CLASSES) -> Path:
    """matrices: title -> confusion matrix (rows = true, columns = predicted)."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.3), squeeze=False)
    for ax, (title, cm) in zip(axes[0], matrices.items(), strict=True):
        cm = np.asarray(cm)
        ax.imshow(cm, cmap="Blues", vmin=0)
        ax.grid(False)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > cm.max() * 0.6 else INK)
        ax.set_xticks(range(len(class_names)), class_names, rotation=45)
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def pick_errors(y_true, y_pred, per_class: int = 2, seed: int = 0) -> list[int]:
    """Sample misclassified indices spread across all true classes (not just the first ones)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    out = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero((y_true == c) & (y_pred != c))
        if len(idx):
            out += sorted(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    return out


def error_examples(images: torch.Tensor, y_true, y_pred, indices: list[int], path, title: str) -> Path:
    n = max(len(indices), 1)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices, strict=False):
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.set_title(f"true {CLASSES[y_true[i]]}\npred {CLASSES[y_pred[i]]}", fontsize=8)
    if not indices:
        axes[0, 0].text(0.5, 0.5, "no errors", ha="center")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def pick_gradcam_indices(y_true, n: int, seed: int = 0) -> list[int]:
    """n test images spread across the classes (round robin)."""
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    pools = [list(rng.permutation(np.flatnonzero(y_true == c))) for c in range(len(CLASSES))]
    out = []
    while len(out) < n and any(pools):
        for pool in pools:
            if pool and len(out) < n:
                out.append(int(pool.pop()))
    return out


def gradcam_grid(models: dict[str, torch.nn.Module], images: torch.Tensor, y_true, indices: list[int],
                 path) -> Path:
    """Row 1: images. One row of Grad-CAM overlays per model."""
    tf = eval_transform()
    rows = 1 + len(models)
    fig, axes = plt.subplots(rows, len(indices), figsize=(2.0 * len(indices), 2.6 * rows), squeeze=False)
    for j, i in enumerate(indices):
        img = images[i].permute(1, 2, 0).numpy()
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"true {CLASSES[y_true[i]]}", fontsize=8)
        for k, (name, model) in enumerate(models.items(), start=1):
            cam, pred, probs = GradCAM(model, model.gradcam_layer)(tf(images[i]).unsqueeze(0))
            axes[k, j].imshow(overlay(img, cam))
            axes[k, j].set_title(f"{DISPLAY_NAMES.get(name, name)}\n{CLASSES[pred]} {probs[pred]:.2f}",
                                 fontsize=7)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Grad-CAM on test images (red = regions that raised the predicted class score)")
    fig.tight_layout()
    return _save(fig, path)


def results_bar(results: dict, path) -> Path:
    """Test accuracy and macro F1 per model: mean with +/- 1 sd error bars over seeds."""
    labels, acc, acc_sd, f1, f1_sd = [], [], [], [], []
    base = results["baselines"]
    for key, label in (("majority", "Majority"), ("logreg_pixels", "LogReg pixels")):
        labels.append(label)
        acc.append(base[key]["accuracy"])
        f1.append(base[key]["macro_f1"])
        acc_sd.append(0.0)
        f1_sd.append(0.0)
    for name, m in results["models"].items():
        labels.append(DISPLAY_NAMES.get(name, name))
        acc.append(m["test_accuracy"]["mean"])
        acc_sd.append(m["test_accuracy"]["sd"])
        f1.append(m["test_macro_f1"]["mean"])
        f1_sd.append(m["test_macro_f1"]["sd"])
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.5 * len(labels) + 1, 3.6))
    b1 = ax.bar(x - w / 2 - 0.01, acc, w, yerr=acc_sd, color=C_TRAIN, label="accuracy", capsize=3)
    b2 = ax.bar(x + w / 2 + 0.01, f1, w, yerr=f1_sd, color=C_VAL, label="macro F1", capsize=3)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK, padding=2)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test score")
    n_seeds = max((m["n_seeds"] for m in results["models"].values()), default=1)
    ax.set_title(f"Test results (networks: mean +/- sd over {n_seeds} seed(s); baselines: single fit)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, path)

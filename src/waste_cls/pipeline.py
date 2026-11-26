"""End-to-end run used by the CLI and the notebook: data -> baselines -> models -> results -> figures."""

from __future__ import annotations

from pathlib import Path

from . import plots
from .config import CLASSES, Config
from .data import prepare_data
from .experiment import GRADCAM_CAPABLE, build_results, pick_device, run_all, run_baselines, save_results
from .models import DISPLAY_NAMES, export_model


def best_run(runs: list[dict], name: str) -> dict:
    return max((r for r in runs if r["model"] == name), key=lambda r: r["best_val_macro_f1"])


def make_figures(results: dict, data, best_models: dict, figures_dir, n_gradcam: int = 6) -> list[Path]:
    figures_dir = Path(figures_dir)
    runs = results["runs"]
    y_true = data.y["test"].numpy()
    out = [plots.learning_curves(runs, figures_dir / "learning_curves.png")]

    matrices = {"LogReg pixels": results["baselines"]["logreg_pixels"]["confusion_matrix"]}
    for name in results["models"]:
        r = best_run(runs, name)
        matrices[f"{DISPLAY_NAMES.get(name, name)}\n(seed {r['seed']}, best on val)"] = r["test"]["confusion_matrix"]
    out.append(plots.confusion_matrices(matrices, figures_dir / "confusion_matrices.png"))

    for name in results["models"]:
        r = best_run(runs, name)
        idx = plots.pick_errors(y_true, r["test"]["y_pred"], per_class=2, seed=0)
        out.append(plots.error_examples(data.x["test"], y_true, r["test"]["y_pred"], idx,
                                        figures_dir / f"errors_{name}.png",
                                        f"{DISPLAY_NAMES.get(name, name)}: misclassified test images"))

    cam_models = {n: m for n, (_, m) in best_models.items() if n in GRADCAM_CAPABLE}
    if cam_models:
        idx = plots.pick_gradcam_indices(y_true, n_gradcam, seed=0)
        out.append(plots.gradcam_grid(cam_models, data.x["test"], y_true, idx, figures_dir / "gradcam.png"))
    out.append(plots.results_bar(results, figures_dir / "results_bar.png"))
    return out


def export_demo_model(results: dict, best_models: dict, cfg: Config, path) -> Path:
    arch = results["demo_model"]["arch"]
    seed, model = best_models[arch]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_model(model, arch, path, cfg.image_size, CLASSES, cfg.dnn_hidden,
                 extra={"seed": seed, "val_macro_f1": results["demo_model"]["val_macro_f1"],
                        "split_csv_sha256": results["data"]["split_csv_sha256"], "status": cfg.name})
    return path


def run_pipeline(cfg: Config, results_path, figures_dir, export_path=None, data_dir="data",
                 split_csv="data_split/trashnet_split.csv", checkpoint_dir=None, device=None, log=print) -> dict:
    device = pick_device(device)
    data = prepare_data(cfg, data_dir, split_csv)
    if log:
        log(f"data: {data.source}, split hash {data.split_hash[:12]}, counts {data.counts}")
    baselines = run_baselines(data, cfg)
    runs, best_models = run_all(cfg, data, device, checkpoint_dir, log)
    results = build_results(cfg, data, baselines, runs, device)
    save_results(results, results_path)
    make_figures(results, data, best_models, figures_dir, cfg.n_gradcam)
    if export_path:
        export_demo_model(results, best_models, cfg, export_path)
    return results

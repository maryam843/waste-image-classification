"""Command line entry point.

    python -m waste_cls.cli run --config smoke --out runs/smoke
    python -m waste_cls.cli run --config full  --out results --figures figures --export exports/demo_model.pt
    python -m waste_cls.cli prepare-data
    python -m waste_cls.cli predict --model exports/demo_model.pt --image photo.jpg [--cam cam.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import get_config
from .data import eval_transform, prepare_data


def _cmd_run(args) -> int:
    from .pipeline import run_pipeline

    cfg = get_config(args.config)
    out = Path(args.out)
    figures = Path(args.figures) if args.figures else out / "figures"
    results = run_pipeline(cfg, out / "results.json", figures, args.export, args.data_dir, args.split_csv,
                           args.checkpoints, args.device)
    summary = {k: {"acc": round(v["test_accuracy"]["mean"], 4), "f1": round(v["test_macro_f1"]["mean"], 4)}
               for k, v in results["models"].items()}
    print(json.dumps({"status": results["status"], "models": summary}, indent=1))
    print(f"wrote {out / 'results.json'} and figures to {figures}")
    return 0


def _cmd_prepare(args) -> int:
    cfg = get_config(args.config)
    data = prepare_data(cfg, args.data_dir, args.split_csv)
    print(json.dumps({"source": data.source, "split_csv_sha256": data.split_hash, "counts": data.counts,
                      "dropped_label_conflicts": len(data.dropped)}, indent=1))
    return 0


def _cmd_predict(args) -> int:
    from .gradcam import GradCAM, overlay
    from .models import load_exported

    model, blob = load_exported(args.model)
    size = blob["image_size"]
    with Image.open(args.image) as im:
        rgb = np.array(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
    x = eval_transform()(torch.from_numpy(rgb).permute(2, 0, 1).contiguous()).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].numpy()
    ranked = sorted(zip(blob["class_names"], probs.tolist(), strict=True), key=lambda t: -t[1])
    print(json.dumps({"top3": ranked[:3]}, indent=1))
    if args.cam:
        cam, _, _ = GradCAM(model, model.gradcam_layer)(x)
        Image.fromarray(overlay(rgb, cam)).save(args.cam)
        print(f"wrote {args.cam}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="waste-cls", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="baselines + DNN/SmallCNN/ResNet-18, results.json and figures")
    run.add_argument("--config", default="smoke", choices=["smoke", "full"])
    run.add_argument("--out", default="runs/smoke", help="folder for results.json")
    run.add_argument("--figures", default=None, help="folder for figures (default: <out>/figures)")
    run.add_argument("--export", default=None, help="write the demo model here (git-ignored folder)")
    run.add_argument("--checkpoints", default=None, help="checkpoint folder (git-ignored)")
    run.add_argument("--device", default=None)
    run.add_argument("--data-dir", default="data")
    run.add_argument("--split-csv", default="data_split/trashnet_split.csv")
    run.set_defaults(func=_cmd_run)

    prep = sub.add_parser("prepare-data", help="download TrashNet, hash images, check/make the split")
    prep.add_argument("--config", default="full", choices=["smoke", "full"])
    prep.add_argument("--data-dir", default="data")
    prep.add_argument("--split-csv", default="data_split/trashnet_split.csv")
    prep.set_defaults(func=_cmd_prepare)

    pred = sub.add_parser("predict", help="top-3 classes for one image with an exported model")
    pred.add_argument("--model", required=True)
    pred.add_argument("--image", required=True)
    pred.add_argument("--cam", default=None, help="optional path for a Grad-CAM overlay PNG")
    pred.set_defaults(func=_cmd_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

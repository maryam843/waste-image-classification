"""Gradio demo: upload a photo of one waste item -> top-3 classes + Grad-CAM overlay.

Trained on the TrashNet dataset (studio-style photos, 4 classes).
Not for real sorting decisions.
"""

from __future__ import annotations

try:  # Hugging Face ZeroGPU (preinstalled on the Space); not installed locally or in CI
    import spaces
except ImportError:  # no-op stand-in so local runs and tests work without the package
    import types

    def _gpu(fn=None, **_kwargs):
        """Accept both @spaces.GPU and @spaces.GPU(duration=...) and return the function unchanged."""
        return fn if fn is not None else (lambda f: f)

    spaces = types.SimpleNamespace(GPU=_gpu)

from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:  # inside the project repo
    from waste_cls.gradcam import GradCAM, overlay
    from waste_cls.models import load_exported
except ImportError:  # inside the Hugging Face Space (files copied next to app.py)
    from gradcam import GradCAM, overlay
    from models import load_exported

MODEL_PATH = Path(__file__).with_name("model.pt")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Only these two decoders are used: the file's first bytes must match one of them.
SIGNATURES = {"JPEG": b"\xff\xd8\xff", "PNG": b"\x89PNG\r\n\x1a\n"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NOTE = ("Trained on TrashNet studio photos of glass, metal, paper and plastic. "
        "It is not reliable on real bins or sorting lines. "
        "Not for real sorting decisions.")


def to_input(image: Image.Image, size: int) -> tuple[np.ndarray, torch.Tensor]:
    rgb = np.array(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    x = (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    return rgb, x.unsqueeze(0)


def open_upload(path: str | Path) -> Image.Image:
    """Open an uploaded file only if it is a JPEG or PNG of at most 10 MB, checked before decoding."""
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise ValueError(f"the file is {size / 2**20:.1f} MB; the limit is 10 MB")
    with open(path, "rb") as fh:
        head = fh.read(8)
    if not any(head.startswith(sig) for sig in SIGNATURES.values()):
        raise ValueError("only JPEG and PNG photos are accepted")
    image = Image.open(path, formats=list(SIGNATURES))
    image.load()
    return image


def predict(image, model, blob) -> tuple[dict[str, float], np.ndarray]:
    if image is None:
        raise ValueError("please upload an image")
    if isinstance(image, (str, Path)):
        image = open_upload(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    rgb, x = to_input(image, blob["image_size"])
    cam, _, probs = GradCAM(model, model.gradcam_layer)(x)
    ranked = sorted(zip(blob["class_names"], probs.tolist(), strict=True), key=lambda t: -t[1])[:3]
    return dict(ranked), overlay(rgb, cam)


def build_demo(model, blob):
    import gradio as gr

    @spaces.GPU  # required on free ZeroGPU Spaces; no-op locally
    def classify(image):
        try:
            return predict(image, model, blob)
        except ValueError as err:  # show a message in the app instead of a generic error
            raise gr.Error(str(err)) from err
        except OSError as err:  # damaged or unreadable image file
            raise gr.Error("the image could not be read; please upload a JPEG or PNG photo") from err

    return gr.Interface(
        fn=classify,
        # filepath + image_mode=None: Gradio passes the raw file on, and open_upload() checks it before decoding
        inputs=gr.Image(type="filepath", image_mode=None, sources=["upload"], label="Photo of one item (JPEG or PNG)"),
        outputs=[gr.Label(num_top_classes=3, label="Top-3 classes"),
                 gr.Image(label="Grad-CAM (red = regions that drove the prediction)")],
        title="Waste material classifier",
        description=NOTE,
        flagging_mode="never",
    )


if __name__ == "__main__":
    loaded_model, loaded_blob = load_exported(MODEL_PATH)
    build_demo(loaded_model, loaded_blob).launch(max_file_size="10mb")

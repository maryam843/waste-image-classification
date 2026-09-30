import numpy as np
import pytest
import torch

from waste_cls import plots
from waste_cls.gradcam import GradCAM, colorize, overlay
from waste_cls.models import build_model


@pytest.mark.parametrize("name", ["smallcnn", "resnet18"])
def test_gradcam_shape_range_and_hook_cleanup(name):
    m = build_model(name, pretrained=False)
    m.train()
    cam, idx, probs = GradCAM(m, m.gradcam_layer)(torch.randn(1, 3, 32, 32))
    assert cam.shape == (32, 32) and cam.dtype == np.float32
    assert cam.min() >= 0 and cam.max() <= 1
    assert 0 <= idx < 4 and np.isclose(probs.sum(), 1, atol=1e-5)
    assert m.training  # mode restored
    assert not m.gradcam_layer._forward_hooks  # hooks removed
    cam2, idx2, _ = GradCAM(m, m.gradcam_layer)(torch.randn(1, 3, 32, 32), class_idx=2)
    assert idx2 == 2


def test_gradcam_rejects_batches():
    m = build_model("smallcnn")
    with pytest.raises(ValueError):
        GradCAM(m, m.gradcam_layer)(torch.randn(2, 3, 16, 16))


def test_gradcam_flat_map_is_zero():
    class Const(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, 1)
            self.fc = torch.nn.Linear(2, 4)

        def forward(self, x):
            return self.fc(self.conv(x).mean((2, 3))) * 0

    m = Const()
    cam, _, _ = GradCAM(m, m.conv)(torch.randn(1, 3, 8, 8))
    assert np.all(cam == 0)


def test_colorize_and_overlay():
    cam = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    rgb = colorize(cam)
    assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
    assert rgb[0, 0, 2] > rgb[0, 0, 0]  # low = blue
    assert rgb[-1, -1, 0] > rgb[-1, -1, 2]  # high = red
    img = np.full((8, 8, 3), 100, np.uint8)
    out = overlay(img, cam)
    assert out.shape == img.shape and out.dtype == np.uint8
    with pytest.raises(ValueError):
        overlay(img, cam[:4])


def test_pick_errors_covers_all_classes():
    y_true = np.repeat(np.arange(4), 10)
    y_pred = (y_true + 1) % 4
    idx = plots.pick_errors(y_true, y_pred, per_class=2)
    assert sorted({int(y_true[i]) for i in idx}) == [0, 1, 2, 3] and len(idx) == 8
    assert plots.pick_errors(y_true, y_true) == []


def test_pick_gradcam_indices_round_robin():
    y = np.repeat(np.arange(4), 3)
    idx = plots.pick_gradcam_indices(y, 6)
    assert len(idx) == 6 and len(set(idx)) == 6
    assert {int(y[i]) for i in idx[:4]} == {0, 1, 2, 3}


def test_error_examples_with_no_errors(tmp_path):
    imgs = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
    p = plots.error_examples(imgs, [0, 1], [0, 1], [], tmp_path / "e.png", "none")
    assert p.exists()

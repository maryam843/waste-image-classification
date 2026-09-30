import pytest
import torch

from waste_cls.models import (
    DNN,
    ResNet18Frozen,
    SmallCNN,
    build_model,
    count_parameters,
    export_model,
    load_exported,
)


def test_dnn_matches_2025_architecture_and_parameter_count():
    m = build_model("dnn", image_size=224)
    assert count_parameters(m) == 77_237_380  # same as the 2025 notebook's network
    linears = [layer for layer in m.network if isinstance(layer, torch.nn.Linear)]
    assert [(lin.in_features, lin.out_features) for lin in linears] == [
        (150528, 512), (512, 256), (256, 128), (128, 4)]
    dropouts = [layer.p for layer in m.network if isinstance(layer, torch.nn.Dropout)]
    assert dropouts == [0.5, 0.5, 0.5]
    assert sum(isinstance(layer, torch.nn.BatchNorm1d) for layer in m.network) == 3
    first = sum(p.numel() for p in m.network[0].parameters())
    assert f"{100 * first / count_parameters(m):.1f} %" == "99.8 %"  # stated in the README


@pytest.mark.parametrize("name,size", [("dnn", 16), ("smallcnn", 32), ("resnet18", 32)])
def test_forward_shapes(name, size):
    m = build_model(name, image_size=size, pretrained=False).eval()
    assert m(torch.randn(2, 3, size, size)).shape == (2, 4)


def test_smallcnn_is_small_and_has_gradcam_layer():
    m = SmallCNN()
    assert count_parameters(m) < 500_000
    assert isinstance(m.gradcam_layer, torch.nn.Conv2d)


def test_resnet_backbone_frozen_and_stays_in_eval_mode():
    m = ResNet18Frozen(pretrained=False)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"head.weight", "head.bias"}
    assert count_parameters(m, trainable_only=True) == 512 * 4 + 4
    m.train()
    assert m.training and m.head.training
    assert not m.backbone.training  # BatchNorm statistics of the frozen backbone are not updated
    before = m.backbone.bn1.running_mean.clone()
    m(torch.randn(4, 3, 32, 32))
    assert torch.equal(before, m.backbone.bn1.running_mean)
    assert m.gradcam_layer is m.backbone.layer4[-1]


def test_factory_returns_fresh_models():
    a, b = build_model("smallcnn"), build_model("smallcnn")
    assert a is not b
    with pytest.raises(ValueError):
        build_model("vit")


def test_dnn_custom_hidden():
    m = DNN(12, hidden_sizes=(8,), dropout=0.1)
    assert m(torch.randn(3, 3, 2, 2)).shape == (3, 4)


@pytest.mark.parametrize("arch", ["smallcnn", "dnn"])
def test_export_and_safe_load_roundtrip(tmp_path, arch):
    m = build_model(arch, image_size=16, pretrained=False).eval()
    path = tmp_path / "m.pt"
    export_model(m, arch, path, 16, ("glass", "metal", "paper", "plastic"), extra={"seed": 1})
    loaded, blob = load_exported(path)
    x = torch.randn(2, 3, 16, 16)
    assert torch.allclose(m(x), loaded(x))
    assert blob["arch"] == arch and blob["extra"]["seed"] == 1 and not loaded.training

import pytest

from waste_cls.config import CLASSES, FULL, SMOKE, get_config


def test_classes_are_the_four_2025_classes():
    assert CLASSES == ("glass", "metal", "paper", "plastic")


def test_full_config_matches_study_design():
    assert FULL.image_size == 224
    assert FULL.seeds == (0, 1, 2)
    assert FULL.models == ("dnn", "smallcnn", "resnet18")
    assert FULL.dnn_hidden == (512, 256, 128)
    assert FULL.dnn_dropout == 0.5
    assert FULL.lr == 1e-3
    assert FULL.split_fractions == (0.70, 0.15, 0.15)
    assert FULL.pretrained and FULL.data_source == "trashnet"
    assert "dnn" not in FULL.augment_models


def test_smoke_config_is_cpu_friendly():
    assert SMOKE.data_source == "synthetic"
    assert SMOKE.pretrained is False  # no downloads in CI
    assert SMOKE.max_epochs <= 2


def test_get_config_overrides_and_errors():
    cfg = get_config("smoke", max_epochs=5)
    assert cfg.max_epochs == 5 and cfg.name == "smoke"
    assert get_config("full") is FULL
    with pytest.raises(ValueError):
        get_config("nope")
    assert cfg.to_dict()["max_epochs"] == 5

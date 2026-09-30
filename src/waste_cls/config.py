"""Experiment configurations: FULL (Colab GPU run) and SMOKE (tiny CPU run for CI and tests)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

CLASSES: tuple[str, ...] = ("glass", "metal", "paper", "plastic")

# ImageNet statistics. Used for every model so the comparison is fair and the
# pretrained ResNet-18 sees inputs in the range it was trained on.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Config:
    name: str
    data_source: str  # "trashnet" or "synthetic"
    image_size: int
    batch_size: int
    max_epochs: int
    early_stop_patience: int
    lr: float
    weight_decay: float
    lr_factor: float
    lr_patience: int
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    split_seed: int = 42
    split_fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    dnn_hidden: tuple[int, ...] = (512, 256, 128)
    dnn_dropout: float = 0.5
    pretrained: bool = True
    logreg_size: int = 64
    logreg_c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    num_workers: int = 0
    synthetic_per_class: int = 0
    augment_models: tuple[str, ...] = ("smallcnn", "resnet18")
    n_gradcam: int = 6
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


FULL = Config(
    name="full",
    data_source="trashnet",
    image_size=224,
    batch_size=32,
    max_epochs=40,
    early_stop_patience=8,
    lr=1e-3,
    weight_decay=0.0,
    lr_factor=0.5,
    lr_patience=3,
    seeds=(0, 1, 2),
    models=("dnn", "smallcnn", "resnet18"),
    pretrained=True,
    num_workers=2,
)

SMOKE = Config(
    name="smoke",
    data_source="synthetic",
    image_size=32,
    batch_size=8,
    max_epochs=2,
    early_stop_patience=2,
    lr=1e-3,
    weight_decay=0.0,
    lr_factor=0.5,
    lr_patience=1,
    seeds=(0,),
    models=("dnn", "smallcnn", "resnet18"),
    pretrained=False,  # no weight download in CI
    logreg_size=16,
    logreg_c_grid=(0.1, 1.0),
    num_workers=0,
    synthetic_per_class=20,
    n_gradcam=4,
)

CONFIGS = {"full": FULL, "smoke": SMOKE}


def get_config(name: str, **overrides) -> Config:
    """Return a named config, optionally with some fields replaced."""
    try:
        cfg = CONFIGS[name]
    except KeyError as err:
        raise ValueError(f"unknown config {name!r}; choose from {sorted(CONFIGS)}") from err
    return replace(cfg, **overrides) if overrides else cfg

import sys
from pathlib import Path

import pytest
import torch

from waste_cls.config import get_config
from waste_cls.data import prepare_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def smoke_cfg():
    return get_config("smoke")


@pytest.fixture(scope="session")
def tiny_cfg():
    # 16x16 synthetic images, 1 epoch: fast enough for many tests
    return get_config("smoke", image_size=16, max_epochs=1, synthetic_per_class=12, batch_size=8)


@pytest.fixture(scope="session")
def tiny_data(tiny_cfg):
    return prepare_data(tiny_cfg)


def write_images(folder: Path, spec: dict[str, int], size: int = 20, duplicate: dict | None = None) -> Path:
    """Write small PNGs: spec = {class: count}. Each image differs (pixel value = running index)."""
    from PIL import Image

    k = 0
    for cls, n in spec.items():
        (folder / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            k += 1
            img = Image.new("RGB", (size, size), (k % 256, (k * 7) % 256, (k * 13) % 256))
            img.save(folder / cls / f"{cls}{i}.png")
    for src, dst in (duplicate or {}).items():
        (folder / dst).write_bytes((folder / src).read_bytes())
    return folder

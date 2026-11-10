"""Data: download TrashNet, hash every image, make a duplicate-safe stratified split, load tensors.

Images are never committed. Only the split (file hashes + labels) is saved as CSV so
anyone can rebuild exactly the same train/val/test sets.
"""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .config import CLASSES, IMAGENET_MEAN, IMAGENET_STD, Config

# TrashNet (Thung and Yang, 2016), MIT licence. Pinned to a commit so the bytes never change.
TRASHNET_COMMIT = "6fa2b878c6c1b4304b91109070ce0edf9279bb31"
TRASHNET_URL = (
    f"https://raw.githubusercontent.com/garythung/trashnet/{TRASHNET_COMMIT}/data/dataset-resized.zip"
)
TRASHNET_SHA256 = "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Record:
    path: str  # absolute path on disk (never saved)
    source_name: str  # "<class>/<file>" as in the TrashNet zip
    label: str
    sha256: str


@dataclass
class SplitResult:
    rows: list[dict]  # sha256, label, split, source_name
    dropped: list[dict] = field(default_factory=list)  # label-conflicting duplicates


# ---------------------------------------------------------------- download and scan


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    """Download url to dest (skipped if dest exists with the right hash)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_sha256 is None or sha256_file(dest) == expected_sha256):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)
    got = sha256_file(tmp)
    if expected_sha256 is not None and got != expected_sha256:
        tmp.unlink()
        raise ValueError(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {got}")
    tmp.replace(dest)
    return dest


def extract_classes(zip_path: Path, out_dir: Path, classes: tuple[str, ...] = CLASSES) -> Path:
    """Extract only the wanted class folders (skips __MACOSX and hidden files)."""
    out_dir = Path(out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) != 3 or parts[0].startswith("__MACOSX"):
                continue
            _, cls, fname = parts
            if cls not in classes or not fname or fname.startswith("."):
                continue
            if not fname.lower().endswith(IMAGE_SUFFIXES):
                continue
            target = out_dir / cls / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(zf.read(name))
    return out_dir


def download_trashnet(data_dir: str | Path = "data", url: str = TRASHNET_URL,
                      expected_sha256: str | None = TRASHNET_SHA256) -> Path:
    """Download the pinned TrashNet zip and extract glass/metal/paper/plastic. Returns the image root."""
    data_dir = Path(data_dir)
    zip_path = download_file(url, data_dir / "trashnet-dataset-resized.zip", expected_sha256)
    return extract_classes(zip_path, data_dir / "trashnet")


def scan_images(root: str | Path, classes: tuple[str, ...] = CLASSES) -> list[Record]:
    """List every image under root/<class>/ with its SHA-256, in a stable order."""
    root = Path(root)
    records = []
    for cls in classes:
        folder = root / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"missing class folder: {folder}")
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."):
                records.append(Record(str(p), f"{cls}/{p.name}", cls, sha256_file(p)))
    return records


# ---------------------------------------------------------------- split


def make_split(records: list[Record], seed: int = 42,
               fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> SplitResult:
    """Stratified train/val/test split on unique image hashes.

    - exact duplicates (same SHA-256, same label) always land in the same split
    - duplicates with conflicting labels are dropped (the true label is unknown)
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r.sha256].append(r)

    keep_hashes, keep_labels, dropped = [], [], []
    for h in sorted(groups):
        labels = {r.label for r in groups[h]}
        if len(labels) > 1:
            dropped += [{"sha256": h, "label": r.label, "source_name": r.source_name} for r in groups[h]]
            continue
        keep_hashes.append(h)
        keep_labels.append(groups[h][0].label)

    _, f_val, f_test = fractions
    rest_h, test_h, rest_y, _ = train_test_split(
        keep_hashes, keep_labels, test_size=f_test, stratify=keep_labels, random_state=seed
    )
    train_h, val_h = train_test_split(
        rest_h, test_size=f_val / (1.0 - f_test), stratify=rest_y, random_state=seed
    )
    assignment = {h: "train" for h in train_h} | {h: "val" for h in val_h} | {h: "test" for h in test_h}

    rows = []
    for h in keep_hashes:
        for r in groups[h]:
            rows.append({"sha256": h, "label": r.label, "split": assignment[h], "source_name": r.source_name})
    rows.sort(key=lambda d: (d["label"], d["source_name"]))
    return SplitResult(rows=rows, dropped=dropped)


def split_to_csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["sha256", "label", "split", "source_name"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def save_split_csv(rows: list[dict], path: str | Path) -> str:
    """Write the split CSV and return its SHA-256 (the 'split hash')."""
    text = split_to_csv_text(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8", newline="")
    return sha256_bytes(text.encode("utf-8"))


def load_split_csv(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def split_hash(rows: list[dict]) -> str:
    return sha256_bytes(split_to_csv_text(rows).encode("utf-8"))


def split_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {s: dict.fromkeys(CLASSES, 0) for s in SPLITS}
    for r in rows:
        counts[r["split"]][r["label"]] = counts[r["split"]].get(r["label"], 0) + 1
    return counts


def match_records_to_split(records: list[Record], rows: list[dict]) -> dict[str, list[Record]]:
    """Map a saved split onto local files by hash. Fails loudly if any file is missing or changed."""
    by_key = {(r.sha256, r.source_name): r for r in records}
    out: dict[str, list[Record]] = {s: [] for s in SPLITS}
    missing = []
    for row in rows:
        rec = by_key.get((row["sha256"], row["source_name"]))
        if rec is None:
            missing.append(row["source_name"])
            continue
        if rec.label != row["label"]:
            raise ValueError(f"label mismatch for {row['source_name']}")
        out[row["split"]].append(rec)
    if missing:
        raise ValueError(f"{len(missing)} images in the split CSV were not found locally, e.g. {missing[:3]}")
    return out


# ---------------------------------------------------------------- tensors and transforms


def load_images(paths: list[str], size: int) -> torch.Tensor:
    """Load images as a uint8 tensor [N, 3, size, size] (bilinear resize, same as the original version)."""
    arr = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def make_synthetic_images(n_per_class: int, size: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiny fake 4-class image set for CPU tests: each class has its own colour and shape, plus noise."""
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    colours = np.array([[40, 160, 60], [150, 150, 160], [230, 230, 210], [60, 110, 220]], dtype=np.float32)
    images, labels = [], []
    for c in range(len(CLASSES)):
        for _ in range(n_per_class):
            img = np.full((size, size, 3), 245.0, dtype=np.float32)
            cx, cy = g.integers(size // 4, 3 * size // 4, size=2)
            r = size // 4
            if c == 0:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
            elif c == 1:
                mask = (abs(xx - cx) < r) & (abs(yy - cy) < r)
            elif c == 2:
                mask = ((yy // max(2, size // 8)) % 2 == 0) & (abs(xx - cx) < r)
            else:
                mask = (yy > cy - r) & (abs(xx - cx) < (yy - cy + r) // 2 + 1) & (yy < cy + r)
            img[mask] = colours[c]
            img += g.normal(0, 12, size=img.shape)
            images.append(np.clip(img, 0, 255).astype(np.uint8))
            labels.append(c)
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    return x, torch.tensor(labels, dtype=torch.long)


def eval_transform() -> v2.Compose:
    return v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform(augment: bool) -> v2.Compose:
    """Light augmentation: flip, small rotation, colour jitter. Off = identical to eval."""
    if not augment:
        return eval_transform()
    return v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10, fill=255),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Inverse of Normalize, clipped to [0, 1]. Works on [3,H,W] or [N,3,H,W]."""
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


class ImageTensorDataset(Dataset):
    """uint8 image tensor + labels, with a per-item transform."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        if len(images) != len(labels):
            raise ValueError("images and labels differ in length")
        self.images = images
        self.labels = labels
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.transform(self.images[idx]), self.labels[idx]


# ---------------------------------------------------------------- one call for the notebook / CLI


@dataclass
class DataBundle:
    x: dict[str, torch.Tensor]  # split -> uint8 images
    y: dict[str, torch.Tensor]  # split -> labels
    names: dict[str, list[str]]  # split -> source names
    split_hash: str
    counts: dict[str, dict[str, int]]
    dropped: list[dict]
    source: str


def prepare_data(cfg: Config, data_dir: str | Path = "data",
                 split_csv: str | Path = "data_split/trashnet_split.csv") -> DataBundle:
    """Return train/val/test tensors for a config.

    TrashNet: download + hash; if the committed split CSV exists it is used (and checked),
    otherwise a new split is made and written.
    Synthetic: generated in memory and split with the same function.
    """
    if cfg.data_source == "synthetic":
        x, y = make_synthetic_images(cfg.synthetic_per_class, cfg.image_size, seed=cfg.split_seed)
        records = [Record("", f"{CLASSES[int(lab)]}/synthetic{i}.png", CLASSES[int(lab)],
                          sha256_bytes(x[i].numpy().tobytes())) for i, lab in enumerate(y)]
        result = make_split(records, cfg.split_seed, cfg.split_fractions)
        index = {r.source_name: i for i, r in enumerate(records)}
        xs, ys, names = {}, {}, {}
        for s in SPLITS:
            sel = [r["source_name"] for r in result.rows if r["split"] == s]
            idx = torch.tensor([index[n] for n in sel], dtype=torch.long)
            xs[s], ys[s], names[s] = x[idx], y[idx], sel
        return DataBundle(xs, ys, names, split_hash(result.rows), split_counts(result.rows),
                          result.dropped, "synthetic")

    if cfg.data_source != "trashnet":
        raise ValueError(f"unknown data_source {cfg.data_source!r}")
    root = download_trashnet(data_dir)
    records = scan_images(root)
    split_csv = Path(split_csv)
    fresh = make_split(records, cfg.split_seed, cfg.split_fractions)
    if split_csv.exists():
        rows = load_split_csv(split_csv)
        if split_hash(rows) != split_hash(fresh.rows):
            raise ValueError("the committed split CSV differs from the split rebuilt with the same seed")
    else:
        save_split_csv(fresh.rows, split_csv)
        rows = fresh.rows
    by_split = match_records_to_split(records, rows)
    label_ix = {c: i for i, c in enumerate(CLASSES)}
    xs, ys, names = {}, {}, {}
    for s in SPLITS:
        recs = by_split[s]
        xs[s] = load_images([r.path for r in recs], cfg.image_size)
        ys[s] = torch.tensor([label_ix[r.label] for r in recs], dtype=torch.long)
        names[s] = [r.source_name for r in recs]
    return DataBundle(xs, ys, names, split_hash(rows), split_counts(rows), fresh.dropped, "trashnet")

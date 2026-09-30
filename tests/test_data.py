import zipfile
from collections import Counter

import pytest
import torch

from conftest import write_images
from waste_cls import data as D
from waste_cls.config import get_config


def test_sha256_file_and_bytes_agree(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    assert D.sha256_file(p) == D.sha256_bytes(b"hello")


def _fake_zip(tmp_path):
    src = write_images(tmp_path / "src", {"glass": 3, "metal": 3, "paper": 3, "plastic": 3, "trash": 2})
    zpath = tmp_path / "fake.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in sorted(src.rglob("*.png")):
            zf.write(p, f"dataset-resized/{p.parent.name}/{p.name}")
        zf.writestr("dataset-resized/.DS_Store", b"x")
        zf.writestr("__MACOSX/dataset-resized/glass/._glass0.png", b"x")
        zf.writestr("dataset-resized/glass/notes.txt", b"x")
    return zpath


def test_download_verifies_hash_and_extracts_only_four_classes(tmp_path):
    zpath = _fake_zip(tmp_path)
    url = zpath.as_uri()
    good = D.sha256_file(zpath)
    root = D.download_trashnet(tmp_path / "data", url=url, expected_sha256=good)
    assert sorted(p.name for p in root.iterdir()) == ["glass", "metal", "paper", "plastic"]
    assert len(list(root.rglob("*.png"))) == 12
    # second call re-uses the verified file
    assert D.download_trashnet(tmp_path / "data", url=url, expected_sha256=good) == root


def test_download_rejects_wrong_hash(tmp_path):
    zpath = _fake_zip(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        D.download_file(zpath.as_uri(), tmp_path / "out" / "x.zip", expected_sha256="0" * 64)
    assert not (tmp_path / "out" / "x.zip").exists()


def test_scan_images_is_sorted_and_hashed(tmp_path):
    root = write_images(tmp_path, {"glass": 2, "metal": 2, "paper": 2, "plastic": 2})
    (root / "glass" / ".hidden.png").write_bytes(b"x")
    recs = D.scan_images(root)
    assert len(recs) == 8
    assert [r.source_name for r in recs][:2] == ["glass/glass0.png", "glass/glass1.png"]
    assert all(len(r.sha256) == 64 for r in recs)
    with pytest.raises(FileNotFoundError):
        D.scan_images(tmp_path / "missing")


def _records(n_per_class=40):
    recs = []
    for c in ("glass", "metal", "paper", "plastic"):
        for i in range(n_per_class):
            recs.append(D.Record("", f"{c}/{c}{i}.jpg", c, f"{c}-{i}"))
    return recs


def test_split_fractions_and_stratification():
    res = D.make_split(_records(40), seed=1)
    counts = Counter((r["split"], r["label"]) for r in res.rows)
    for c in ("glass", "metal", "paper", "plastic"):
        assert counts[("train", c)] == 28 and counts[("val", c)] == 6 and counts[("test", c)] == 6
    assert res.dropped == []


def test_split_keeps_duplicates_together_and_drops_label_conflicts():
    recs = _records(20)
    recs.append(D.Record("", "glass/copy.jpg", "glass", "glass-3"))  # same-label duplicate
    recs.append(D.Record("", "metal/clash.jpg", "metal", "paper-5"))  # conflicting labels
    res = D.make_split(recs, seed=0)
    split_of = {}
    for r in res.rows:
        split_of.setdefault(r["sha256"], set()).add(r["split"])
    assert all(len(s) == 1 for s in split_of.values())
    assert {d["source_name"] for d in res.dropped} == {"metal/clash.jpg", "paper/paper5.jpg"}
    assert "paper-5" not in split_of


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        D.make_split(_records(10), fractions=(0.5, 0.2, 0.2))


def test_split_csv_roundtrip_and_hash(tmp_path):
    res = D.make_split(_records(10), seed=3)
    h = D.save_split_csv(res.rows, tmp_path / "s.csv")
    rows = D.load_split_csv(tmp_path / "s.csv")
    assert rows == res.rows
    assert D.split_hash(rows) == h
    counts = D.split_counts(rows)
    assert sum(sum(v.values()) for v in counts.values()) == 40


def test_match_records_to_split(tmp_path):
    root = write_images(tmp_path, {"glass": 8, "metal": 8, "paper": 8, "plastic": 8})
    recs = D.scan_images(root)
    rows = D.make_split(recs, seed=0).rows
    by = D.match_records_to_split(recs, rows)
    assert sum(len(v) for v in by.values()) == 32
    with pytest.raises(ValueError, match="not found"):
        D.match_records_to_split(recs[:-1], rows)
    bad = [dict(r) for r in rows]
    bad[0]["label"] = "metal" if bad[0]["label"] != "metal" else "glass"
    with pytest.raises(ValueError, match="label mismatch"):
        D.match_records_to_split(recs, bad)


def test_load_images_shape_and_dtype(tmp_path):
    root = write_images(tmp_path, {"glass": 2}, size=30)
    x = D.load_images([str(p) for p in sorted((root / "glass").iterdir())], size=12)
    assert x.shape == (2, 3, 12, 12) and x.dtype == torch.uint8


def test_synthetic_images_are_balanced_and_deterministic():
    x1, y1 = D.make_synthetic_images(5, 16, seed=0)
    x2, y2 = D.make_synthetic_images(5, 16, seed=0)
    assert x1.shape == (20, 3, 16, 16) and torch.equal(x1, x2) and torch.equal(y1, y2)
    assert torch.bincount(y1).tolist() == [5, 5, 5, 5]


def test_dataset_and_denormalize():
    x, y = D.make_synthetic_images(2, 8)
    ds = D.ImageTensorDataset(x, y)
    img, lab = ds[0]
    assert img.dtype == torch.float32 and img.shape == (3, 8, 8) and len(ds) == 8
    back = D.denormalize(img)
    assert torch.allclose(back, x[0].float() / 255, atol=1e-5)
    with pytest.raises(ValueError):
        D.ImageTensorDataset(x, y[:-1])


def test_prepare_data_synthetic(tiny_data):
    assert set(tiny_data.x) == {"train", "val", "test"}
    n = sum(len(v) for v in tiny_data.y.values())
    assert n == 48
    assert tiny_data.source == "synthetic" and len(tiny_data.split_hash) == 64


def test_prepare_data_trashnet_uses_and_checks_committed_split(tmp_path, monkeypatch):
    root = write_images(tmp_path / "imgs", {"glass": 8, "metal": 8, "paper": 8, "plastic": 8})
    monkeypatch.setattr(D, "download_trashnet", lambda data_dir: root)
    cfg = get_config("full", image_size=8)
    csv_path = tmp_path / "split.csv"
    first = D.prepare_data(cfg, tmp_path, csv_path)  # writes the CSV
    assert csv_path.exists()
    second = D.prepare_data(cfg, tmp_path, csv_path)  # reads and checks it
    assert first.split_hash == second.split_hash
    assert second.x["train"].shape[1:] == (3, 8, 8)
    rows = D.load_split_csv(csv_path)
    rows[0]["split"] = "test" if rows[0]["split"] != "test" else "train"
    D.save_split_csv(rows, csv_path)
    with pytest.raises(ValueError, match="differs"):
        D.prepare_data(cfg, tmp_path, csv_path)


def test_prepare_data_unknown_source():
    with pytest.raises(ValueError):
        D.prepare_data(get_config("smoke", data_source="other"))


def test_committed_split_csv_is_consistent(root):
    rows = D.load_split_csv(root / "data_split" / "trashnet_split.csv")
    assert len(rows) == 1981
    counts = D.split_counts(rows)
    assert sum(counts["train"].values()) == 1386
    assert sum(counts["val"].values()) == 297
    assert sum(counts["test"].values()) == 298
    hashes = {}
    for r in rows:
        hashes.setdefault(r["sha256"], set()).add(r["split"])
    assert all(len(s) == 1 for s in hashes.values())

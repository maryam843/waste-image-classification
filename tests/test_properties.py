"""Hypothesis property tests: split safety under random duplicates, transform shape/range."""

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from waste_cls.config import CLASSES, IMAGENET_MEAN, IMAGENET_STD
from waste_cls.data import Record, eval_transform, make_split, train_transform


@st.composite
def records_with_duplicates(draw):
    per_class = draw(st.integers(min_value=10, max_value=30))
    recs = []
    for c in CLASSES:
        for i in range(per_class):
            recs.append(Record("", f"{c}/{c}{i}.jpg", c, f"h-{c}-{i}"))
    # random same-label duplicates
    for j in range(draw(st.integers(0, 15))):
        base = draw(st.sampled_from(recs[: 4 * per_class]))
        recs.append(Record("", f"{base.label}/dup{j}.jpg", base.label, base.sha256))
    # random label-conflicting duplicates
    for j in range(draw(st.integers(0, 3))):
        base = draw(st.sampled_from(recs[: 4 * per_class]))
        other = draw(st.sampled_from([c for c in CLASSES if c != base.label]))
        recs.append(Record("", f"{other}/clash{j}.jpg", other, base.sha256))
    return recs


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(records_with_duplicates(), st.integers(0, 10_000))
def test_split_never_puts_a_hash_in_two_splits(recs, seed):
    res = make_split(recs, seed=seed)
    where = {}
    for r in res.rows:
        where.setdefault(r["sha256"], set()).add(r["split"])
    assert all(len(v) == 1 for v in where.values())
    # every non-conflicting record is kept; conflicting ones are dropped completely
    dropped_hashes = {d["sha256"] for d in res.dropped}
    assert not dropped_hashes & set(where)
    assert len(res.rows) + len(res.dropped) == len(recs)
    # every class appears in every split
    for s in ("train", "val", "test"):
        assert {r["label"] for r in res.rows if r["split"] == s} == set(CLASSES)


@settings(max_examples=30, deadline=None)
@given(records_with_duplicates(), st.integers(0, 10_000))
def test_split_is_deterministic_for_a_seed(recs, seed):
    assert make_split(recs, seed=seed).rows == make_split(list(reversed(recs)), seed=seed).rows


LOW = min((0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True))
HIGH = max((1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True))


@settings(max_examples=40, deadline=None)
@given(st.integers(4, 40), st.booleans(), st.integers(0, 2**31 - 1))
def test_transforms_keep_shape_and_range(size, augment, seed):
    torch.manual_seed(seed)
    img = torch.randint(0, 256, (3, size, size), dtype=torch.uint8)
    out = train_transform(augment)(img)
    assert out.shape == (3, size, size) and out.dtype == torch.float32
    assert out.min() >= LOW - 1e-4 and out.max() <= HIGH + 1e-4
    ev = eval_transform()(img)
    assert ev.shape == img.shape and ev.min() >= LOW - 1e-4 and ev.max() <= HIGH + 1e-4

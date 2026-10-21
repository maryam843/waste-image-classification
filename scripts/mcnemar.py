"""Exact McNemar test per seed, from the saved test predictions in results/results.json.

For each seed, two models are compared on the same 298 test images. Only the images where exactly one
of the two models is right count: b = first model right and second wrong, c = the opposite.
The two-sided exact p-value is min(1, 2 * P(X <= min(b, c))) with X ~ Binomial(b + c, 0.5).
Standard library only, no training.

Usage:
    python scripts/mcnemar.py [--results results/results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

PAIRS = (("dnn", "smallcnn"), ("smallcnn", "resnet18"))


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def discordant(y_true: list[int], pred_a: list[int], pred_b: list[int]) -> tuple[int, int]:
    """(b, c): images only the first model got right, and images only the second model got right."""
    if not len(y_true) == len(pred_a) == len(pred_b):
        raise ValueError("predictions and labels must have the same length")
    b = sum(a == t != p for t, a, p in zip(y_true, pred_a, pred_b, strict=True))
    c = sum(p == t != a for t, a, p in zip(y_true, pred_a, pred_b, strict=True))
    return b, c


def paired_rows(results: dict, pairs=PAIRS) -> list[dict]:
    runs = {(r["model"], r["seed"]): r["test"] for r in results["runs"]}
    rows = []
    for first, second in pairs:
        seeds = sorted(s for m, s in runs if m == first and (second, s) in runs)
        for seed in seeds:
            ta, tb = runs[(first, seed)], runs[(second, seed)]
            if ta["y_true"] != tb["y_true"]:
                raise ValueError(f"seed {seed}: {first} and {second} were not scored on the same test images")
            b, c = discordant(ta["y_true"], ta["y_pred"], tb["y_pred"])
            rows.append({"first": first, "second": second, "seed": seed, "only_first_right": b,
                         "only_second_right": c, "p_value": exact_mcnemar(b, c)})
    return rows


def render(rows: list[dict]) -> str:
    lines = ["| Comparison | Seed | Only first right | Only second right | Exact McNemar p |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['first']} vs {r['second']} | {r['seed']} | {r['only_first_right']} | "
                     f"{r['only_second_right']} | {r['p_value']:.5f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/results.json")
    args = ap.parse_args(argv)
    path = Path(args.results)
    if not path.exists():
        print(f"{path} not found")
        return 1
    rows = paired_rows(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        print("no model pairs with matching seeds in", path)
        return 1
    print(render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

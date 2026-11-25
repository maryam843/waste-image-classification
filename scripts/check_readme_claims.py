"""Check that every result number in README.md comes from results/results.json.

- The results table lives between <!-- RESULTS:START --> and <!-- RESULTS:END -->.
  While that block says TODO, the number check is SKIPPED (exit 0) with a clear message.
  Otherwise the block must equal the table rendered from results.json (use --render to print it).
- Outside the block, any score-like number (0.xx or NN %) must appear in the rendered table,
  or be on a line that is explicitly about the earlier, test-selected version (marked by its year).
- Forbidden claims are always checked (real-time, deployed for sorting, ...), and a line may not
  present the earlier score and a new score as an improvement.
- results.json from a smoke run is rejected: only the full Colab run can be reported.

Usage:
    python scripts/check_readme_claims.py [--readme README.md] [--results results/results.json] [--render]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
FORBIDDEN = [
    r"real[- ]time", r"deployed for (?:real )?sorting", r"state[- ]of[- ]the[- ]art", r"production[- ]ready",
    r"\bSOTA\b",
]
# Non-result numbers the README may state anywhere: the CI level, and the share of DNN parameters in its
# first layer (150,528*512+512 of 77,237,380 = 99.8 %, checked in tests/test_models.py).
STATIC_FACTS = {"95 %", "99.8 %"}
SCORE = re.compile(r"(?<![\w.])(?:0\.\d{2,4}|\d{1,3}(?:\.\d+)?\s?%)(?![\w.])")
NAMES = {"dnn": "DNN [512,256,128] (original network)", "smallcnn": "SmallCNN (from scratch)",
         "resnet18": "ResNet-18, frozen backbone (reference)"}


def fmt(x: float) -> str:
    return f"{x:.3f}"


def render_table(results: dict) -> str:
    lines = [
        "| Model | Test accuracy | Test macro F1 | 95 % Wilson CI for accuracy | Parameters (trainable) "
        "| Train time per run |",
        "|---|---|---|---|---|---|",
    ]
    base = results["baselines"]
    maj, lr = base["majority"], base["logreg_pixels"]
    lr_label = f"Logistic regression, {lr['image_size']}x{lr['image_size']} pixels"
    for label, b in (("Majority class", maj), (lr_label, lr)):
        lo, hi = b["accuracy_wilson_95"]
        lines.append(f"| {label} | {fmt(b['accuracy'])} | {fmt(b['macro_f1'])} | {fmt(lo)} to {fmt(hi)} | n/a | n/a |")
    for name, m in results["models"].items():
        acc, f1 = m["test_accuracy"], m["test_macro_f1"]
        cis = m["test_accuracy_wilson_95_per_seed"]
        lo = min(c[0] for c in cis)
        hi = max(c[1] for c in cis)
        t = m["train_time_s"]["mean"]
        lines.append(
            f"| {NAMES.get(name, name)} | {fmt(acc['mean'])} +/- {fmt(acc['sd'])} | "
            f"{fmt(f1['mean'])} +/- {fmt(f1['sd'])} | {fmt(lo)} to {fmt(hi)} (widest over seeds) | "
            f"{m['params_total']:,} ({m['params_trainable']:,}) | {t:.0f} s |"
        )
    n_test = results["baselines"]["majority"]["n"]
    seeds = results["config"]["seeds"]
    lines.append("")
    env = results["environment"]
    lines.append(f"Test set: {n_test} images. Networks: mean +/- sd over seeds {', '.join(map(str, seeds))}. "
                 f"GPU: {env.get('gpu_name') or 'none'}. "
                 f"Split hash: `{results['data']['split_csv_sha256'][:12]}`.")
    if env.get("torch"):
        lines.append("")
        lines.append(f"Run environment (recorded in results.json): Python {env.get('python', '?')}, "
                     f"torch {env['torch']}, torchvision {env.get('torchvision', '?')}.")
    return "\n".join(lines)


def extract_block(readme: str) -> str | None:
    if START not in readme or END not in readme:
        return None
    return readme.split(START, 1)[1].split(END, 1)[0].strip()


def check(readme_text: str, results: dict | None) -> tuple[int, list[str]]:
    msgs: list[str] = []
    errors = 0
    lines = readme_text.splitlines()

    for i, line in enumerate(lines, 1):
        for pat in FORBIDDEN:
            if re.search(pat, line, re.I) and not re.search(r"\bnot\b|\bno\b|never|n't", line, re.I):
                errors += 1
                msgs.append(f"line {i}: forbidden claim /{pat}/: {line.strip()[:100]}")
        if "2025" in line and re.search(r"improv", line, re.I) and SCORE.search(line):
            errors += 1
            msgs.append(f"line {i}: presents an earlier score and a new score as an improvement")

    block = extract_block(readme_text)
    if block is None:
        return errors + 1, msgs + ["README has no RESULTS:START/END block"]

    if "TODO" in block:
        msgs.append("SKIPPED number check: the README results block is still TODO "
                    "(fill it in with --render after the Colab run).")
        return errors, msgs
    if results is None:
        return errors + 1, msgs + ["README shows results but results/results.json is missing"]
    if results.get("status") != "full":
        return errors + 1, msgs + [f"results.json status is {results.get('status')!r}; only a full run can be reported"]

    table = render_table(results)
    if block != table:
        errors += 1
        msgs.append("results block differs from the table rendered from results.json (run with --render)")
    allowed = set(SCORE.findall(table)) | STATIC_FACTS
    outside = readme_text.replace(block, "")
    for i, line in enumerate(outside.splitlines(), 1):
        if "2025" in line:
            continue
        for num in SCORE.findall(line):
            if num not in allowed:
                errors += 1
                msgs.append(f"outside-table line {i}: number {num!r} is not in results.json: {line.strip()[:90]}")
    if errors == 0:
        msgs.append(f"OK: results block matches results.json; {len(allowed)} numbers checked")
    return errors, msgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--render", action="store_true", help="print the table rendered from results.json")
    args = ap.parse_args(argv)

    results = None
    rpath = Path(args.results)
    if rpath.exists():
        results = json.loads(rpath.read_text(encoding="utf-8"))
    if args.render:
        if results is None:
            print(f"{rpath} not found")
            return 1
        print(render_table(results))
        return 0
    if results is None:
        print(f"note: {rpath} not found (results are still TODO)")
    errors, msgs = check(Path(args.readme).read_text(encoding="utf-8"), results)
    for m in msgs:
        print(m)
    print("README claims check:", "PASS" if errors == 0 else f"FAIL ({errors} problem(s))")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

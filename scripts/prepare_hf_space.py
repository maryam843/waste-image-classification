"""Build the folder you upload to the Hugging Face Space.

    python scripts/prepare_hf_space.py --model exports/demo_model.pt --out hf_space_build

Copies hf_space/app.py, requirements.txt, README.md, the two model files from src/waste_cls
(models.py, gradcam.py) and the exported model (as model.pt). The output folder is git-ignored.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(model: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(ROOT / "hf_space" / name, out / name)
    for name in ("models.py", "gradcam.py"):
        shutil.copy2(ROOT / "src" / "waste_cls" / name, out / name)
    shutil.copy2(model, out / "model.pt")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="file written by export_model (exports/demo_model.pt)")
    ap.add_argument("--out", default="hf_space_build")
    args = ap.parse_args(argv)
    out = build(Path(args.model), Path(args.out))
    print(f"Space files ready in {out}: {sorted(p.name for p in out.iterdir())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

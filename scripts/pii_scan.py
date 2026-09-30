"""Scan the repository (working tree and full git history) for personal data and secrets.

Checks:
- secrets: Hugging Face, GitHub, AWS, sk- style API keys, Kaggle keys, private keys
- Colab / Google account metadata left in notebooks
- student-ID style numbers, personal e-mail addresses, personal file paths
- old demo video links (GitHub user-attachments)
- git identity: every author/committer must use the GitHub no-reply address

Base64 image outputs inside notebooks are removed before scanning, because
random base64 text can look like a key.

Usage:
    python scripts/pii_scan.py                 # working tree + full history
    python scripts/pii_scan.py --no-history    # working tree only
    python scripts/pii_scan.py --denylist FILE # also look for exact private values
                                               # (one per line; keep FILE out of the repo)

Exit code 0 = no findings, 1 = findings (printed with the value masked).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ALLOWED_EMAIL_DOMAINS = ("users.noreply.github.com",)
ALLOWED_IDENTITY = "185722545+maryam843@users.noreply.github.com"

# Keys are joined at runtime so this file does not flag itself.
_COLAB_KEYS = ["execution" + "Info", "user" + "Id", "display" + "Name", "mount_" + "file_id",
               "authorship" + "_tag", "output" + "Id"]

PATTERNS: dict[str, re.Pattern[str]] = {
    "huggingface-token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "sk-style-key": re.compile(r"\bsk-[A-Za-z0-9_-]{32,}\b"),
    "slack-token": re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b"),
    "kaggle-key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "colab-metadata": re.compile(r'"(?:' + "|".join(_COLAB_KEYS) + r')"\s*:'),
    "google-user-id": re.compile(r"(?<![\w.])\d{20,21}(?![\w.])"),
    "student-id-line": re.compile(r'(?m)^\s*"?#\s*\d{8,10}\s*(?:\\n)?"?,?\s*$'),
    "student-id-labelled": re.compile(r"(?i)(?:student\s*(?:id|no|number)|\bID\s*[:#])\D{0,20}\d{6,10}"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "personal-path": re.compile(
        r"(?:/Users/[A-Za-z]|[A-Za-z]:\\Users\\|/content/" + r"drive/|/home/(?!runner\b|user\b)[a-z][\w-]*/)"
    ),
    "demo-video-link": re.compile(r"github\.com/user-attachments/"),
}

TEXT_SUFFIXES = {
    ".py", ".ipynb", ".md", ".txt", ".json", ".csv", ".toml", ".yml", ".yaml", ".cfg", ".ini",
    ".gitignore", ".sh", ".ps1", "",
}


def strip_notebook_images(text: str) -> str:
    """Drop base64 image payloads from a notebook's JSON text."""
    try:
        nb = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []) or []:
            data = out.get("data")
            if isinstance(data, dict):
                for key in list(data):
                    if key.startswith("image/"):
                        data[key] = "<image removed for scan>"
    return json.dumps(nb, indent=1, ensure_ascii=False)


def mask(value: str) -> str:
    value = value.strip()
    if len(value) <= 6:
        return "*" * len(value)
    return value[:3] + "*" * (len(value) - 5) + value[-2:]


def scan_text(text: str, where: str, denylist: list[str]) -> list[str]:
    if where.endswith(".ipynb"):
        text = strip_notebook_images(text)
    hits: list[str] = []
    for name, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            value = m.group(0)
            if name == "email" and value.split("@")[-1].lower().endswith(ALLOWED_EMAIL_DOMAINS):
                continue
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{where}:{line}: {name}: {mask(value)}")
    for value in denylist:
        if value and value in text:
            line = text.count("\n", 0, text.index(value)) + 1
            hits.append(f"{where}:{line}: denylist: {mask(value)}")
    return hits


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], input=input_bytes, capture_output=True, check=True
    ).stdout


def _is_text_path(path: str) -> bool:
    p = Path(path)
    return p.suffix.lower() in TEXT_SUFFIXES or p.name in {"LICENSE", "Makefile"}


def scan_working_tree(repo: Path, denylist: list[str]) -> list[str]:
    files = _git(repo, "ls-files", "-co", "--exclude-standard", "-z").decode().split("\0")
    hits: list[str] = []
    for rel in filter(None, files):
        path = repo / rel
        if not path.is_file():
            continue
        if not _is_text_path(rel):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits += scan_text(text, f"worktree:{rel}", denylist)
    return hits


def scan_history(repo: Path, denylist: list[str]) -> list[str]:
    try:
        _git(repo, "rev-parse", "--verify", "HEAD")
    except subprocess.CalledProcessError:
        return []  # no commits yet
    hits: list[str] = []
    listing = _git(repo, "rev-list", "--all", "--objects").decode().splitlines()
    blobs: dict[str, str] = {}
    for line in listing:
        parts = line.split(" ", 1)
        if len(parts) == 2 and _is_text_path(parts[1]):
            blobs.setdefault(parts[0], parts[1])
    for sha, rel in blobs.items():
        kind = _git(repo, "cat-file", "-t", sha).decode().strip()
        if kind != "blob":
            continue
        text = _git(repo, "cat-file", "blob", sha).decode("utf-8", errors="replace")
        hits += scan_text(text, f"history:{sha[:10]}:{rel}", denylist)

    log = _git(repo, "log", "--all", "--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e").decode()
    for record in filter(None, (r.strip() for r in log.split("\x1e"))):
        commit, _an, ae, _cn, ce, body = record.split("\x1f", 5)
        where = f"commit:{commit[:10]}"
        for role, email in (("author", ae), ("committer", ce)):
            if email != ALLOWED_IDENTITY:
                hits.append(f"{where}: git-identity: {role} e-mail {mask(email)} is not the no-reply address")
        hits += scan_text(body, f"{where}:message", denylist)
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=".", help="repository root (default: current directory)")
    parser.add_argument("--no-history", action="store_true", help="scan the working tree only")
    parser.add_argument("--denylist", help="file with exact private values to look for (never commit it)")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    denylist: list[str] = []
    if args.denylist:
        denylist = [v.strip() for v in Path(args.denylist).read_text().splitlines() if v.strip()]

    hits = scan_working_tree(repo, denylist)
    if not args.no_history:
        hits += scan_history(repo, denylist)

    if hits:
        print(f"PII/secret scan: {len(hits)} finding(s)")
        for hit in hits:
            print("  " + hit)
        return 1
    scope = "working tree" if args.no_history else "working tree + full git history"
    print(f"PII/secret scan: 0 findings ({scope}, {len(PATTERNS)} patterns, denylist={len(denylist)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

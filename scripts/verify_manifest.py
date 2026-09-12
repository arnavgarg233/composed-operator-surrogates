#!/usr/bin/env python3
"""Check MANIFEST.sha256 digests and package-file coverage.

The release also carries legacy research artifacts outside the package
manifest; those additional files are intentionally not required to be listed.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify() -> list[str]:
    entries: dict[str, str] = {}
    problems: list[str] = []
    for number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            problems.append(f"malformed manifest line {number}")
            continue
        if relative in entries:
            problems.append(f"duplicate manifest entry: {relative}")
        entries[relative] = expected

    listed = set(entries)
    existing = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    for relative in sorted(listed - existing):
        problems.append(f"manifest names absent file: {relative}")
    for relative in sorted(listed & existing):
        observed = digest(ROOT / relative)
        if observed != entries[relative]:
            problems.append(f"digest mismatch: {relative}")
    return problems


def main() -> int:
    problems = verify()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print(f"PASS  manifest: {len(MANIFEST.read_text().splitlines())} files covered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

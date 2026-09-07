"""Rebuild the figures and confirm the output is byte-identical.

A figure that changes between builds cannot be cited by hash, and a reader who rebuilds
it has no way to tell a real difference from a timestamp. This rebuilds and compares.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "figures"
RECEIPT = ROOT / "results" / "tables" / "FIGURE_RECEIPT.json"
PYTHON = "$HOME/.local/bin/python3.12"


def digests() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(OUT.iterdir()) if path.suffix == ".pdf"
    }


def main() -> None:
    first = digests()
    if not first:
        raise SystemExit("no figures to check; run make_figures.py first")

    result = subprocess.run(
        [PYTHON, str(HERE / "make_figures.py")],
        capture_output=True, text=True, cwd=HERE,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit("rebuild failed")

    second = digests()
    identical = first == second
    for name in sorted(first):
        same = first[name] == second.get(name)
        print(f"  {'same' if same else 'DIFFERS':>8}  {first[name][:16]}...  {name}")
    print(f"\nBYTE-IDENTICAL REBUILD: {identical}")

    receipt = json.loads(RECEIPT.read_text())
    receipt["byte_identical_rebuild"] = identical
    receipt["rebuild_digests"] = second
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()

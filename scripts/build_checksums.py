"""Write results/CHECKSUMS.txt over every published artifact in the repository.

One file answers "is this tree the tree that was released" for the results, the figures
and the learner artifacts together, so a reader checks one thing rather than several.
The list of covered roots lives in the package, so the builder and the checks in
`verify.py` cannot disagree about what is published.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from composed_operator_surrogates import CHECKSUMS, COVERED, ROOT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=CHECKSUMS,
        help="where to write; the replay writes elsewhere and compares, so the committed "
        "file is checked rather than overwritten",
    )
    out = parser.parse_args().output

    rows = []
    for name in COVERED:
        for path in sorted((ROOT / name).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path == CHECKSUMS:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(ROOT)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    print(f"{len(rows)} files hashed into {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

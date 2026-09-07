"""Run every artifact check and report what failed.

No network, no arguments. Exits 1 if any check finds a problem, so `reproduce.sh` and CI
both fail on the same condition a reader would see.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from composed_operator_surrogates.verify import run_all  # noqa: E402


def main() -> int:
    results = run_all()
    failed = 0
    for name, problems in results.items():
        print(f"{'PASS' if not problems else 'FAIL'}  {name}")
        for problem in problems[:10]:
            print(f"        {problem}")
        if len(problems) > 10:
            print(f"        ... and {len(problems) - 10} more")
        failed += bool(problems)
    print(f"\n{len(results) - failed} of {len(results)} checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

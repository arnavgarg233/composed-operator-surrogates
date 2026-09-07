"""Apply the equal-load criterion to a sampler timeline.

The thresholds live here and in the plan, both written before leg 5a was launched.
This reads only the sampler's own output and prints the verdict; it decides nothing that
was not already decided.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

MEDIAN_POINTS = 25.0
MEAN_POINTS = 50.0


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 6:
            continue
        try:
            rows.append({"t": f[0], "leg": float(f[2]), "other": float(f[3]),
                         "reps": f[5]})
        except ValueError:
            continue
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    rows = load(Path(sys.argv[1]))
    first = [r for r in rows if "repeat_01" not in r["reps"]]
    second = [r for r in rows if "repeat_01" in r["reps"]]
    if not first or not second:
        print("the timeline does not cover both repeats; no verdict")
        return 2

    stats = {}
    for name, series in (("repeat_00", first), ("repeat_01", second)):
        other = [r["other"] for r in series]
        stats[name] = {
            "n": len(series),
            "median": statistics.median(other),
            "mean": statistics.mean(other),
            "max": max(other),
            "leg_mean": statistics.mean([r["leg"] for r in series]),
        }
        s = stats[name]
        print(f"{name}  n={s['n']:3d}  other-process CPU  median {s['median']:7.1f}  "
              f"mean {s['mean']:7.1f}  max {s['max']:7.1f}   leg {s['leg_mean']:6.1f}")

    d_median = abs(stats["repeat_00"]["median"] - stats["repeat_01"]["median"])
    d_mean = abs(stats["repeat_00"]["mean"] - stats["repeat_01"]["mean"])
    ok_median = d_median <= MEDIAN_POINTS
    ok_mean = d_mean <= MEAN_POINTS
    print(f"\nmedian difference {d_median:.1f} against a limit of {MEDIAN_POINTS:.0f}: "
          f"{'within' if ok_median else 'OVER'}")
    print(f"mean   difference {d_mean:.1f} against a limit of {MEAN_POINTS:.0f}: "
          f"{'within' if ok_mean else 'OVER'}")
    met = ok_median and ok_mean
    print("\nVERDICT:", "the equal-load cell is filled by this leg" if met else
          "NOT equal load; this is another asymmetric leg and the cell stays empty")
    return 0 if met else 1


if __name__ == "__main__":
    sys.exit(main())

"""Hold a constant CPU background for the duration of a determinism leg.

Five legs in a row have failed the equal-load criterion because the background changed
across the repeat boundary, in both directions, from work nobody could schedule precisely.
Borrowing a steady background has not worked. This makes one.

Not a benchmark and not tuned: a tight integer loop in N processes, which occupies about N
cores and does nothing else. It is started before the harness and stopped after the report
appears, so both repeats see the same floor by construction rather than by hope.

The sampler counts these processes in its other-process CPU column, which is the point:
the spin IS the background, and the criterion judges it like any other.

    python3 constant_load.py start --cores 2 --tag leg5a
    python3 constant_load.py stop  --tag leg5a
    python3 constant_load.py status --tag leg5a
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

STATE = Path(__file__).resolve().parent / ".constant_load"

WORKER = (
    "import time\n"
    "x = 0\n"
    "while True:\n"
    "    for _ in range(2_000_000):\n"
    "        x = (x * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF\n"
)


def pidfile(tag: str) -> Path:
    return STATE / f"{tag}.pids"


def start(tag: str, cores: int) -> int:
    STATE.mkdir(exist_ok=True)
    if pidfile(tag).exists():
        print(f"a spin tagged {tag} is already recorded; stop it first")
        return 2
    pids = []
    for _ in range(cores):
        proc = subprocess.Popen([sys.executable, "-c", WORKER],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pids.append(proc.pid)
    pidfile(tag).write_text("\n".join(str(p) for p in pids) + "\n")
    print(f"started {cores} spin workers for {tag}: {pids}")
    print("start this BEFORE the harness and stop it AFTER the report appears")
    return 0


def stop(tag: str) -> int:
    path = pidfile(tag)
    if not path.exists():
        print(f"no spin recorded for {tag}")
        return 2
    stopped, missing = [], []
    for line in path.read_text().split():
        pid = int(line)
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            missing.append(pid)
    path.unlink()
    print(f"stopped {stopped}")
    if missing:
        # A worker that died on its own means the background was not constant, and the
        # leg's own sampler will show where it dropped. Say so rather than tidy it away.
        print(f"WARNING: {missing} were already gone; the background was NOT constant "
              "for the whole leg and the timeline should be read for when it fell")
    return 0


def status(tag: str) -> int:
    path = pidfile(tag)
    if not path.exists():
        print(f"no spin recorded for {tag}")
        return 2
    alive = []
    for line in path.read_text().split():
        pid = int(line)
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except ProcessLookupError:
            pass
    recorded = len(path.read_text().split())
    print(f"{len(alive)} of {recorded} spin workers alive for {tag}: {alive}")
    return 0 if len(alive) == recorded else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cores", type=int, default=2)
    args = parser.parse_args()
    if args.action == "start":
        return start(args.tag, args.cores)
    if args.action == "stop":
        return stop(args.tag)
    return status(args.tag)


if __name__ == "__main__":
    sys.exit(main())

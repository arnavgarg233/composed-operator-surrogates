#!/bin/zsh
set -e
RELEASE_ROOT="${RELEASE_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
export RELEASE_ROOT
export MPLCONFIGDIR="$RELEASE_ROOT/work/matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
printf '{"pid": %s, "status": "starting", "expected_wall_time": "approximately 3-4 hours"}\n' "$$" > \
  "$RELEASE_ROOT/work/family1_rerun/LAUNCH.json"
exec "${PYTHON:-python3}" -u \
  "$RELEASE_ROOT/scripts/family1_rerun/run_family1_rerun.py" driver

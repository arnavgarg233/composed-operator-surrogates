#!/bin/zsh
# Family 1b — truth-only pre-screen. One process at a time: the CPU is shared with two
# other workers and every leg here is single-threaded by design.
set -e
RELEASE_ROOT="${RELEASE_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
export RELEASE_ROOT
cd "$RELEASE_ROOT/scripts/family1b_pair_screen"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
PY="${PYTHON:-python3}"
L="$RELEASE_ROOT/work/family1b_pair_screen/logs"
mkdir -p "$L"

run () {   # run <logname> <args...>
  local name=$1; shift
  echo "=== $name start $(date -u +%FT%TZ) ===" >> $L/$name.log
  $PY screen_pairs.py "$@" >> $L/$name.log 2>&1
  echo "=== $name done  $(date -u +%FT%TZ) ===" >> $L/$name.log
}

run geom_stage1 geometry 1
run order_stage1 order 1
run geom_stage3 geometry 3
run order_stage3 order 3
run stage2       stage2
run order_stage2 order 2
run intervals    intervals
run report       report
echo "ALL DONE $(date -u +%FT%TZ)" >> $L/DONE

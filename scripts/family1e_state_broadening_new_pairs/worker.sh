#!/bin/zsh
# One BROAD-S fit: worker.sh <pair> <seed>. Single-threaded, private JAX cache.
set -u
RELEASE_ROOT="${RELEASE_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
export RELEASE_ROOT
HERE="$RELEASE_ROOT/scripts/family1e_state_broadening_new_pairs"
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export JAX_COMPILATION_CACHE_DIR="$RELEASE_ROOT/work/family1e/jax_cache"

pair=$1
seed=$2
out="$RELEASE_ROOT/results/family1e_state_broadening_new_pairs/broadS/${pair}_seed${seed}_pregate.json"
log="$RELEASE_ROOT/work/family1e/logs/train_${pair}_seed${seed}.log"
if [[ -f "$out" ]]; then
  echo "skip ${pair} seed ${seed}: $out exists" >> "$RELEASE_ROOT/work/family1e/logs/queue.log"
  exit 0
fi
echo "$(date -u +%FT%TZ) start ${pair} seed ${seed}" >> "$RELEASE_ROOT/work/family1e/logs/queue.log"
cd "$HERE"
"$PY" -u run_family1e.py train "$pair" "$seed" > "$log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) done  ${pair} seed ${seed} rc=${rc}" >> "$RELEASE_ROOT/work/family1e/logs/queue.log"
exit $rc

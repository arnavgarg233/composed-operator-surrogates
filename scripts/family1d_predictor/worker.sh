#!/bin/zsh
# Family 1d worker: runs a serial list of run_family1d.py invocations.
# usage: worker.sh <worker-name> <cmd...> ::: <cmd...> ::: ...
set -u
RELEASE_ROOT="${RELEASE_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
export RELEASE_ROOT
HERE="$RELEASE_ROOT/scripts/family1d_predictor"
# Private compilation cache (PREDECLARED.md section 4): the shared one in ENV.sh holds
# AOT entries written by another lane under different machine features, which XLA
# refuses with a SIGILL warning.
export JAX_COMPILATION_CACHE_DIR="$RELEASE_ROOT/work/family1d/jax_cache"
export MPLCONFIGDIR="$RELEASE_ROOT/work/matplotlib-cache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
PY="${PYTHON:-python3}"

name=$1; shift
echo "=== worker $name start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
args=()
for tok in "$@"; do
  if [[ "$tok" == ":::" ]]; then
    echo "--- $(date -u +%H:%M:%S) run: ${args[@]}"
    "$PY" "$HERE/run_family1d.py" "${args[@]}" || echo "!!! FAILED: ${args[@]}"
    args=()
  else
    args+=("$tok")
  fi
done
if (( ${#args[@]} > 0 )); then
  echo "--- $(date -u +%H:%M:%S) run: ${args[@]}"
  "$PY" "$HERE/run_family1d.py" "${args[@]}" || echo "!!! FAILED: ${args[@]}"
fi
echo "=== worker $name done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

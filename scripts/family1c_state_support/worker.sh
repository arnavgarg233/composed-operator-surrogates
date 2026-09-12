#!/bin/zsh
# Family 1c worker: runs a serial list of run_family1c.py invocations.
# usage: worker.sh <worker-name> <cmd...> ::: <cmd...> ::: ...
set -u
RELEASE_ROOT="${RELEASE_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
export RELEASE_ROOT
HERE="$RELEASE_ROOT/scripts/family1c_state_support"
# Own compilation cache: the shared one at $JAX_COMPILATION_CACHE_DIR holds AOT entries
# written by another lane on a different machine-feature setting, which XLA refuses with
# a SIGILL warning. A private cache keeps this family off that shared state entirely.
export JAX_COMPILATION_CACHE_DIR="$RELEASE_ROOT/work/family1c/jax_cache"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
PY="${PYTHON:-python3}"

name=$1; shift
echo "=== worker $name start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
args=()
for tok in "$@"; do
  if [[ "$tok" == ":::" ]]; then
    echo "--- $(date -u +%H:%M:%S) run: ${args[@]}"
    "$PY" "$HERE/run_family1c.py" "${args[@]}" || echo "!!! FAILED: ${args[@]}"
    args=()
  else
    args+=("$tok")
  fi
done
if (( ${#args[@]} > 0 )); then
  echo "--- $(date -u +%H:%M:%S) run: ${args[@]}"
  "$PY" "$HERE/run_family1c.py" "${args[@]}" || echo "!!! FAILED: ${args[@]}"
fi
echo "=== worker $name done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

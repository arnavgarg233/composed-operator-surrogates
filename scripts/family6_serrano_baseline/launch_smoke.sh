#!/bin/zsh
# Family 6 smoke: 2 seeds x 48 test units, N=256, equal budget (8000 steps/fit).
#
# Eight A1 FNO fits (2 conditions x 2 primitives x 2 seeds), run two at a time so
# that the concurrent family-1 worker on the same CPU is not starved, then one
# evaluation per seed, then the aggregate.
#
# usage:  nohup ./launch_smoke.sh > logs/smoke.log 2>&1 &
set -e
RELEASE_ROOT="${RELEASE_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
export RELEASE_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
D="$RELEASE_ROOT/scripts/family6_serrano_baseline"
PY="${PYTHON:-python3}"
mkdir -p "$RELEASE_ROOT/work/family6_serrano_baseline/logs" "$RELEASE_ROOT/work/family6_serrano_baseline/ckpt"

echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for cond in narrow broad; do
    for prim in diffusion reaction; do
      ck="$D/ckpt/${cond}_${prim}_seed${seed}.npz"
      if [[ -f "$ck" ]]; then echo "SKIP $cond/$prim seed $seed (exists)"; continue; fi
      "$PY" -u "$D/serrano_splitting.py" fit "$cond" "$prim" "$seed" \
        > "$D/logs/fit_${cond}_${prim}_seed${seed}.log" 2>&1 &
    done
    wait                      # two concurrent fits per wave
    echo "WAVE DONE seed $seed $cond $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
done
echo "FITS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

for seed in 0 1 2 3 4 5 6 7 8 9; do
  "$PY" -u "$D/serrano_splitting.py" evaluate "$seed" \
    > "$D/logs/evaluate_seed${seed}.log" 2>&1
  echo "EVAL DONE seed $seed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

"$PY" -u "$D/serrano_splitting.py" aggregate > "$D/logs/aggregate.log" 2>&1
echo "ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

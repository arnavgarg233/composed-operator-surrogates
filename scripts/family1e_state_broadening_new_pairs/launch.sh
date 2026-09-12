#!/bin/zsh
# Detached driver: 20 BROAD-S fits (2 pairs x 10 seeds), 5 concurrent single-threaded
# workers. Jobs are interleaved across pairs so both pairs advance together.
set -u
RELEASE_ROOT="${RELEASE_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
export RELEASE_ROOT
HERE="$RELEASE_ROOT/scripts/family1e_state_broadening_new_pairs"
cd "$HERE"
echo "$(date -u +%FT%TZ) driver start pid=$$" >> logs/queue.log
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for pair in S1_fkpp_r0.5 S1_grayscott_1sp; do
    echo "$pair $seed"
  done
done | xargs -P 5 -n 2 "$HERE/worker.sh"
echo "$(date -u +%FT%TZ) driver done rc=$?" >> logs/queue.log

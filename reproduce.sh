#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"

uv sync --locked

if [[ ! -x "$PY" ]]; then
  printf 'Interpreter not found: %s\nRun uv sync --locked before replay.\n' "$PY" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1

cd "$ROOT"

"$PY" -c 'import composed_operator_surrogates; print("package import: ok")'
"$PY" -m ruff check .
"$PY" -m ruff format --check .
"$PY" -m pytest
"$PY" scripts/verify_package.py

# CHECKSUMS.txt is regenerated and compared rather than trusted: a checksum file that is
# only ever written is a record of what was written, not a check on it. The rebuild goes to
# a scratch path, so the committed file is the thing under test and the replay leaves the
# tree as it found it.
REGENERATED="$(mktemp -t composed-checksums)"
trap 'rm -f "$REGENERATED"' EXIT
"$PY" scripts/build_checksums.py --output "$REGENERATED" > /dev/null
if ! cmp -s "$REGENERATED" results/CHECKSUMS.txt; then
  printf 'results/CHECKSUMS.txt does not match the tree it covers.\n' >&2
  diff -u results/CHECKSUMS.txt "$REGENERATED" >&2 || true
  exit 1
fi

printf '\nReplay complete. The published figures rebuild byte-identically offline;\n'
printf 'results/tables/FIGURE_RECEIPT.json records the digests and\n'
printf 'scripts/code/check_determinism.py reruns that comparison.\n'

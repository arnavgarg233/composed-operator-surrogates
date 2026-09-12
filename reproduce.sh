#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export RELEASE_ROOT
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1

cd "$RELEASE_ROOT"
python3 scripts/verify_manifest.py
python3 scripts/verify_headlines.py
python3 -m unittest discover -s tests -v

API_PATTERN='API[_-]?'"KEY"
USER_PATH='/''Users/'
VOLUME_PATH='/''Volumes/'
NAME_ONE='aksh''garg'
NAME_TWO='arnav''garg'
if grep -RInE --exclude=MANIFEST.sha256 --exclude-dir=__pycache__ --exclude-dir=.git \
  "${USER_PATH}|${VOLUME_PATH}|${NAME_ONE}|${NAME_TWO}|${API_PATTERN}|sk-[A-Za-z0-9]" .; then
  printf 'FAIL: private path, identity, or credential-like text found.\n' >&2
  exit 1
fi

printf 'PASS  release package verified offline; no fitting was run.\n'

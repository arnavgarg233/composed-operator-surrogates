#!/usr/bin/env python3
"""Run one command N times and prove, or disprove, that it wrote the same numbers.

Why this exists. `recovery/baseline/run_dissociation.py` was re-run on the same
seeds and did not reproduce its own first run: the median dissociation moved from
13.148 to 12.784 and seed 4 moved 27.222 to 22.287, about 18 percent. Nothing in
the script is stochastic across runs, so the difference is in the runtime, not in
the design. Before that can be claimed either way, somebody has to hold the command
fixed and compare the emitted files leaf by leaf. That is all this harness does.

It is also built so that it can fail. An audit that cannot fail is decoration, so
the harness refuses rather than reports BIT_IDENTICAL when:

  - `--cmd` carries no `{OUT}` token, because then two repeats would write to the
    same place and the second would overwrite the first, and on this project that
    place is a banked result file;
  - `--outputs` matched nothing, because comparing zero files always agrees;
  - `--outputs` reaches outside the scratch directory, or names a file type the
    harness has no comparator for;
  - `--report` already exists, because a receipt is not silently overwritten;
  - the command exited non-zero on any repeat.

Each of those exits 2 and writes the reason into the report. Only a real
comparison of at least one real file can return 0.

What it never touches. Every repeat writes into its own fresh scratch directory
and the harness reads only from there. It creates no file outside those
directories and the report path, and it deletes nothing at all, including its own
scratch, so a DIVERGENT verdict leaves both sets of artifacts on disk for
inspection.

Exit codes: 0 BIT_IDENTICAL, 1 DIVERGENT, 2 refused or the command failed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

SCHEMA = "pde-determinism-harness-v1"
OUT_TOKEN = "{OUT}"
TAIL_CHARS = 4000

# Exactly the environment the plan names, and nothing more. JAX_ENABLE_X64 is
# deliberately absent: the scripts under test choose their own precision, and a
# harness that changes precision is measuring a different program.
PINNED_ENV = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "JAX_PLATFORMS": "cpu",
}

# Recorded as inherited, before any override, because these are the variables that
# can move a float without moving a line of code. Reading them back off the report
# is how a later session tells a pinned run from an unpinned one.
NUMERICS_ENV_NAMES = (
    "XLA_FLAGS",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "JAX_DEFAULT_MATMUL_PRECISION",
    "JAX_DISABLE_JIT",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
)

COMPARABLE_SUFFIXES = (".json", ".npz", ".npy")

MISSING = object()  # not a JSON value, so it can never collide with one


class Refused(Exception):
    """A condition under which no verdict may be reported. Always exit 2."""


# --------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def env_digest(env: dict[str, str]) -> str:
    """Pin the whole child environment without printing it.

    The full environment carries credentials and is long, so the report records
    the variables the harness set plus this digest of everything the child
    actually received. Two runs with the same digest had the same environment.
    """
    body = "\n".join(f"{k}={env[k]}" for k in sorted(env))
    return sha256_bytes(body.encode("utf-8"))


# --------------------------------------------------------------------------
# JSON comparison
# --------------------------------------------------------------------------

def _json_leaves(node, path: str = "$"):
    """Every scalar in the document, with the key path that reaches it."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _json_leaves(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _json_leaves(value, f"{path}[{index}]")
    else:
        yield path, node


def _leaf_equal(a, b) -> bool:
    """Exact equality on floats, with two deliberate refinements.

    NaN != NaN in IEEE, but two runs that both wrote NaN into the same slot did
    not diverge from each other, and calling that a divergence would make every
    NaN-producing run unrepeatable by definition. True == 1 in Python, but a run
    that turned a gate flag into an integer did change the record, so that is
    reported.
    """
    if a is MISSING or b is MISSING:
        return False
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


def _relative_difference(a, b):
    """|a - b| / max(|a|, |b|), or None when the pair is not numeric.

    The denominator is the larger magnitude rather than the reference value, so
    it is never zero for a pair that actually differs: if both are zero they are
    equal and never reach here.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return None
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else 0.0


def compare_json(reference: Path, other: Path, ignore=()) -> dict:
    """Key by key, exact float equality. Report the first divergence and the worst.

    `ignore` holds fnmatch patterns over key paths. Every result file in this
    project records its own wall time, so a strict comparison of one can never
    return equal and the harness would be a check that cannot pass. An ignored
    leaf is still compared and still written into the report with both values; it
    is only kept out of the verdict, and the verdict says so in its own name.
    """
    left = dict(_json_leaves(json.loads(reference.read_text())))
    right = dict(_json_leaves(json.loads(other.read_text())))
    paths = list(left) + [p for p in right if p not in left]

    first = None
    divergent = 0
    worst = None
    ignored = []
    for path in paths:
        a = left.get(path, MISSING)
        b = right.get(path, MISSING)
        if _leaf_equal(a, b):
            continue
        relative = _relative_difference(a, b)
        record = {
            "kind": ("json_key_absent" if a is MISSING or b is MISSING
                     else "json_value"),
            "key_path": path,
            "reference_value": None if a is MISSING else a,
            "other_value": None if b is MISSING else b,
            "relative_difference": relative,
        }
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in ignore):
            ignored.append(record)
            continue
        divergent += 1
        if relative is not None and (worst is None or relative > worst):
            worst = relative
        if first is None:
            first = record
    return {
        "identical": divergent == 0,
        "divergent_leaves": divergent,
        "max_relative_difference": worst,
        "first_divergence": first,
        "ignored_divergences": ignored,
    }


# --------------------------------------------------------------------------
# array comparison
# --------------------------------------------------------------------------

def _numpy():
    """Imported only when an array file is actually being compared.

    The pinned interpreter for this project does not carry numpy in its base
    environment; the scripts under test reach it through the uv cache. A
    JSON-only comparison must not need it, and an array comparison that cannot
    have it must refuse rather than skip.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise Refused(f"comparing .npz/.npy needs numpy, which is not importable: {exc}")
    return np


def _scalar(value):
    """A numpy element in a form json.dumps will accept."""
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(as_float) or math.isinf(as_float):
        return str(value)
    return as_float


def _compare_arrays(np, name: str, a, b):
    """None when equal, else the first element that differs."""
    if a.dtype != b.dtype:
        return {"kind": "array_dtype", "array": name,
                "reference_value": str(a.dtype), "other_value": str(b.dtype),
                "index": None, "relative_difference": None}
    if a.shape != b.shape:
        return {"kind": "array_shape", "array": name,
                "reference_value": list(a.shape), "other_value": list(b.shape),
                "index": None, "relative_difference": None}
    if np.array_equal(a, b):
        return None

    unequal = a != b
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.complexfloating):
        # Same reason as _leaf_equal: NaN in the same slot of both runs agrees.
        unequal = unequal & ~(np.isnan(a) & np.isnan(b))
    flat = np.flatnonzero(np.asarray(unequal).reshape(-1))
    if flat.size == 0:
        return None
    position = int(flat[0])
    index = [int(i) for i in np.unravel_index(position, a.shape)] if a.shape else []
    va = a.reshape(-1)[position]
    vb = b.reshape(-1)[position]
    return {
        "kind": "array_value",
        "array": name,
        "index": index,
        "reference_value": _scalar(va),
        "other_value": _scalar(vb),
        "relative_difference": _relative_difference(_scalar(va), _scalar(vb)),
        "divergent_elements": int(np.count_nonzero(unequal)),
    }


def compare_npz(reference: Path, other: Path, ignore=()) -> dict:
    """--ignore-keys deliberately does not reach inside arrays.

    Excusing a key in a summary record is a bounded thing a reader can check. A
    pattern that silences elements of a field is not, so it is not offered.
    """
    np = _numpy()
    # allow_pickle stays off: a comparison must not execute what it is comparing.
    with np.load(reference, allow_pickle=False) as left, \
            np.load(other, allow_pickle=False) as right:
        names = list(left.files) + [n for n in right.files if n not in left.files]
        first = None
        divergent = 0
        worst = None
        for name in names:
            if name not in left.files or name not in right.files:
                divergent += 1
                if first is None:
                    first = {"kind": "array_absent", "array": name,
                             "index": None,
                             "reference_value": name in left.files,
                             "other_value": name in right.files,
                             "relative_difference": None}
                continue
            found = _compare_arrays(np, name, left[name], right[name])
            if found is not None:
                divergent += 1
                relative = found.get("relative_difference")
                if relative is not None and (worst is None or relative > worst):
                    worst = relative
                if first is None:
                    first = found
    return {"identical": divergent == 0, "divergent_leaves": divergent,
            "max_relative_difference": worst, "first_divergence": first,
            "ignored_divergences": []}


def compare_npy(reference: Path, other: Path, ignore=()) -> dict:
    np = _numpy()
    left = np.load(reference, allow_pickle=False)
    right = np.load(other, allow_pickle=False)
    found = _compare_arrays(np, reference.name, left, right)
    return {"identical": found is None, "divergent_leaves": 0 if found is None else 1,
            "max_relative_difference": None if found is None
            else found.get("relative_difference"),
            "first_divergence": found, "ignored_divergences": []}


COMPARATORS = {".json": compare_json, ".npz": compare_npz, ".npy": compare_npy}


# --------------------------------------------------------------------------
# collection and per-file comparison
# --------------------------------------------------------------------------

def collect(out_dir: Path, patterns) -> dict[str, Path]:
    """Resolve --outputs against one repeat's scratch directory.

    Patterns are relative by construction. An absolute pattern or one containing
    `..` is refused rather than normalised, because the harness's promise is that
    it reads nothing outside the directory it created.
    """
    found: dict[str, Path] = {}
    for pattern in patterns:
        parts = Path(pattern).parts
        if Path(pattern).is_absolute() or ".." in parts:
            raise Refused(
                f"--outputs must stay inside the scratch directory, got {pattern!r}")
        for path in sorted(out_dir.glob(pattern)):
            if not path.is_file():
                continue
            relative = str(path.relative_to(out_dir))
            if path.suffix.lower() not in COMPARABLE_SUFFIXES:
                raise Refused(
                    f"--outputs matched {relative!r}; the harness compares only "
                    f"{', '.join(COMPARABLE_SUFFIXES)}")
            found[relative] = path
    return found


def compare_file(relative: str, paths: list[Path], ignore=()) -> dict:
    """One file across every repeat: digests, then the typed comparison.

    Both are load-bearing. The digest catches a change the typed comparator is
    blind to, such as key order or trailing whitespace; the typed comparator says
    which number moved and by how much, which a digest never can.
    """
    digests = [sha256_file(p) for p in paths]
    comparator = COMPARATORS[Path(relative).suffix.lower()]

    first = None
    divergent = 0
    worst = None
    ignored = []
    for index, other in enumerate(paths[1:], start=1):
        result = comparator(paths[0], other, ignore)
        divergent += result["divergent_leaves"]
        for record in result["ignored_divergences"]:
            ignored.append({**record, "file": relative, "repeat": index})
        candidate = result["max_relative_difference"]
        if candidate is not None and (worst is None or candidate > worst):
            worst = candidate
        if not result["identical"] and first is None:
            first = dict(result["first_divergence"] or {})
            first.update({"file": relative, "repeat": index})

    content_identical = first is None
    bytes_identical = len(set(digests)) == 1
    if content_identical and not bytes_identical and not ignored:
        # Same numbers, different bytes, and nothing was excused. Real and worth
        # reporting: the file's layout is not stable even where its values are.
        differing = next(i for i, d in enumerate(digests) if d != digests[0])
        first = {"kind": "bytes", "file": relative, "repeat": differing,
                 "reference_value": digests[0], "other_value": digests[differing],
                 "index": None, "relative_difference": None}
        content_identical = False

    return {
        "sha256": digests,
        "bytes_identical": bytes_identical,
        "content_identical": content_identical,
        "divergent_leaves": divergent,
        "max_relative_difference": worst,
        "first_divergence": first,
        "ignored_divergences": ignored,
        # Strict: nothing differed at all, and nothing was excused.
        "identical": bytes_identical and content_identical and not ignored,
        # Everything that differed was on the --ignore-keys list.
        "identical_except_ignored": content_identical,
    }


def compare_all(names, per_repeat: list[dict[str, Path]],
                ignore=()) -> tuple[dict, dict | None]:
    files: dict[str, dict] = {}
    first = None
    for relative in sorted(names):
        result = compare_file(relative, [repeat[relative] for repeat in per_repeat],
                              ignore)
        files[relative] = result
        if not result["identical_except_ignored"] and first is None:
            first = result["first_divergence"]
    return files, first


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def build_env(pin_threads: bool,
              xla_flags_append: str = "") -> tuple[dict[str, str], dict]:
    inherited = {name: os.environ.get(name) for name in NUMERICS_ENV_NAMES}
    env = os.environ.copy()
    overrides: dict[str, str] = {}
    if pin_threads:
        overrides = dict(PINNED_ENV)
    if xla_flags_append:
        # The pinned recipe pins threads and says nothing about fast-math, so a run
        # that wants fast-math on the record has to append it. Appended rather than
        # replaced: dropping the thread flags while still calling the leg pinned would
        # be a different experiment wearing this one's name.
        base = overrides.get("XLA_FLAGS", os.environ.get("XLA_FLAGS") or "")
        overrides["XLA_FLAGS"] = f"{base} {xla_flags_append}".strip()
    if overrides:
        env.update(overrides)
    record = {
        "mode": "pinned" if pin_threads else "inherited",
        "overrides": overrides,
        "xla_flags_appended": xla_flags_append or None,
        "inherited_numerics": inherited,
        "child_env_sha256": env_digest(env),
        "note": ("JAX_ENABLE_X64 is never set by the harness; the script under test "
                 "chooses its own precision. Without --pin-threads nothing is "
                 "overridden, so the harness can show the nondeterminism as well as "
                 "the fix."),
    }
    return env, record


def tail(text: str) -> str:
    return text if len(text) <= TAIL_CHARS else "..." + text[-TAIL_CHARS:]


def run_repeats(command: str, repeats: int, scratch_root: Path,
                env: dict[str, str], cwd: Path):
    records = []
    directories = []
    for index in range(repeats):
        out_dir = scratch_root / f"repeat_{index:02d}"
        out_dir.mkdir(parents=True)
        resolved = command.replace(OUT_TOKEN, str(out_dir))
        started = time.perf_counter()
        completed = subprocess.run(resolved, shell=True, env=env, cwd=str(cwd),
                                   capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        records.append({
            "index": index,
            "scratch": str(out_dir),
            "resolved_command": resolved,
            "wall_seconds": elapsed,
            "returncode": completed.returncode,
            "stdout_tail": tail(completed.stdout),
            "stderr_tail": tail(completed.stderr),
        })
        directories.append(out_dir)
        print(f"  repeat {index}  {elapsed:8.2f} s  exit {completed.returncode}")
        if completed.returncode != 0:
            break
    return records, directories


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a command N times in fresh scratch directories and "
                    "compare every file it emits.")
    parser.add_argument("--cmd", required=True,
                        help="shell command; must contain the literal {OUT} token, "
                             "which is replaced by this repeat's scratch directory")
    parser.add_argument("--outputs", required=True, nargs="+",
                        help="globs, relative to {OUT}, naming the JSON/NPZ/NPY "
                             "files the command emits")
    parser.add_argument("--xla-flags-append", default="",
                        help="appended to XLA_FLAGS and recorded in the report; use it "
                             "to put --xla_cpu_enable_fast_math=false on the record, "
                             "which --pin-threads does not set")
    parser.add_argument("--pin-threads", action="store_true",
                        help="pin XLA/JAX and BLAS to one CPU thread")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ignore-keys", nargs="+", default=[], metavar="PATTERN",
                        help="fnmatch patterns over JSON key paths, e.g. "
                             "'$.elapsed_seconds', kept out of the verdict but still "
                             "reported with both values. Using this changes the "
                             "passing verdict to BIT_IDENTICAL_EXCEPT_IGNORED. "
                             "Scope each pattern to the key you mean: fnmatch '*' "
                             "crosses '.' and '[', so '$.*' would excuse the whole "
                             "document. Every exclusion the run actually used is "
                             "listed in the report, which is where to check that.")
    parser.add_argument("--report", required=True,
                        help="path for the report JSON; refused if it already exists")
    parser.add_argument("--scratch-root", default=None,
                        help="parent for the per-repeat scratch directories; "
                             "a fresh temporary directory by default")
    parser.add_argument("--cwd", default=None,
                        help="working directory for the command; the harness's own "
                             "by default")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = {
        "schema": SCHEMA,
        "generated": date.today().isoformat(),
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "verdict": "REFUSED",
        "exit_code": 2,
        "refusal": None,
        "command": {"template": args.cmd, "shell": True,
                    "cwd": str(Path(args.cwd).resolve() if args.cwd else Path.cwd())},
        "repeats_requested": args.repeats,
        "outputs_requested": list(args.outputs),
        "ignored_key_patterns": list(args.ignore_keys),
        "environment": None,
        "scratch_root": None,
        "repeats": [],
        "compared_file_count": 0,
        "files": {},
        "first_divergence": None,
    }
    report_path = Path(args.report).resolve()
    may_write_report = True

    try:
        if report_path.exists():
            # The one refusal that must not write the report: writing it is the
            # thing being refused.
            may_write_report = False
            raise Refused(f"--report already exists, refusing to overwrite it: "
                          f"{report_path}")
        if OUT_TOKEN not in args.cmd:
            raise Refused(
                f"--cmd carries no {OUT_TOKEN} token, so every repeat would write to "
                "the same place and the harness cannot guarantee it leaves existing "
                "result files alone")
        if args.repeats < 2:
            raise Refused("--repeats must be at least 2; one run compares with nothing")

        env, env_record = build_env(args.pin_threads, args.xla_flags_append)
        report["environment"] = env_record

        root = (Path(args.scratch_root).resolve() if args.scratch_root
                else Path(tempfile.mkdtemp(prefix="determinism_harness_")))
        root.mkdir(parents=True, exist_ok=True)
        report["scratch_root"] = str(root)
        print(f"scratch {root}")

        cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()
        records, directories = run_repeats(args.cmd, args.repeats, root, env, cwd)
        report["repeats"] = records

        failed = [r for r in records if r["returncode"] != 0]
        if failed or len(records) < args.repeats:
            report["verdict"] = "COMMAND_FAILED"
            report["refusal"] = (f"repeat {failed[0]['index']} exited "
                                 f"{failed[0]['returncode']}")
            raise Refused(report["refusal"])

        per_repeat = [collect(d, args.outputs) for d in directories]
        name_sets = [set(found) for found in per_repeat]
        if not set().union(*name_sets):
            raise Refused("--outputs matched no files in any repeat; a comparison of "
                          "nothing always agrees and must not be reported as a pass")

        if len(set(map(frozenset, name_sets))) != 1:
            union = sorted(set().union(*name_sets))
            missing = {str(i): sorted(set(union) - names)
                       for i, names in enumerate(name_sets) if set(union) - names}
            report["verdict"] = "DIVERGENT"
            report["exit_code"] = 1
            report["first_divergence"] = {"kind": "file_set", "file": None,
                                          "reference_value": union,
                                          "other_value": missing,
                                          "index": None,
                                          "relative_difference": None}
            report["compared_file_count"] = 0
            _write(report_path, report)
            print("\nDIVERGENT  the repeats did not emit the same file names")
            print(f"report {report_path}")
            return 1

        files, first = compare_all(name_sets[0], per_repeat, args.ignore_keys)
        report["files"] = files
        report["compared_file_count"] = len(files)
        report["first_divergence"] = first
        excused = sum(len(f["ignored_divergences"]) for f in files.values())
        report["ignored_divergence_count"] = excused
        if first is not None:
            report["verdict"], report["exit_code"] = "DIVERGENT", 1
        elif excused or not all(f["identical"] for f in files.values()):
            report["verdict"], report["exit_code"] = "BIT_IDENTICAL_EXCEPT_IGNORED", 0
        else:
            report["verdict"], report["exit_code"] = "BIT_IDENTICAL", 0
        _write(report_path, report)

        print(f"\n{report['verdict']}  {len(files)} file(s) compared across "
              f"{args.repeats} repeats")
        if excused:
            print(f"  {excused} divergence(s) excused by --ignore-keys:")
            for entry in (e for f in files.values() for e in f["ignored_divergences"]):
                print(f"    {entry['file']}  {entry['key_path']}  "
                      f"{entry['reference_value']} vs {entry['other_value']}")
        if first is not None:
            where = first.get("key_path") or first.get("array") or first.get("file")
            print(f"  first divergence  {first.get('file')}  {where}")
            print(f"    reference {first.get('reference_value')}")
            print(f"    other     {first.get('other_value')}")
        print(f"report {report_path}")
        return report["exit_code"]

    except Refused as exc:
        report["refusal"] = str(exc)
        if report["verdict"] != "COMMAND_FAILED":
            report["verdict"] = "REFUSED"
        report["exit_code"] = 2
        if may_write_report:
            try:
                _write(report_path, report)
            except OSError:
                pass
        print(f"\n{report['verdict']}  {exc}", file=sys.stderr)
        return 2


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

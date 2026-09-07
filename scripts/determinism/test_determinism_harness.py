#!/usr/bin/env python3
"""Prove that the determinism harness can fail, not only that it can pass.

A harness whose only demonstration is a green run is indistinguishable from a
harness that always says green. So every case here is paired: something the
harness must call BIT_IDENTICAL, several things it must call DIVERGENT, one thing
it must refuse outright, and a mutation that deletes the JSON comparison and shows
the DIVERGENT case then passes. The mutation is what makes the other cases mean
anything; without it, a comparator that never compared could produce this whole
suite's output.

Every fixture is synthetic and runs in about a second. Nothing here runs
run_dissociation.py or any other real script.

Read the exit code, not the printed lines: each case is caught individually, so a
crash inside one is recorded as a failure of that case rather than ending the run
after a row of PASS lines.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "determinism_harness.py"

# A constant array file alongside every JSON fixture. Its job is to still be there
# after the mutation deletes the JSON comparison, so the mutant compares something
# and reaches a verdict instead of tripping the harness's empty-comparison refusal.
_CONSTANT_ARRAY = 'np.savez(out / "arrays.npz", ' \
                  'field=np.arange(12, dtype=np.float64).reshape(3, 4))'

_HEADER = """import json, os, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
"""

FIXTURE_DETERMINISTIC = _HEADER + f"""
payload = {{"schema": "synthetic-v1",
           "observed_dissociation": 12.784256853723884,
           "per_seed": {{"3": {{"dissociation": 12.784256853723884}}}},
           "observed_range": [10.29007380434875, 22.286656121007212],
           "D1": {{"held": True}}}}
(out / "RESULT.json").write_text(json.dumps(payload, indent=2) + "\\n")
{_CONSTANT_ARRAY}
"""

# The nondeterminism the harness exists to catch: one float, in one leaf, drawn
# from the operating system rather than from a seed. 1e-12 keeps it in the last
# few bits, which is the size of the difference a thread-count change produces.
FIXTURE_JSON_DIVERGENT = _HEADER + f"""
noise = int.from_bytes(os.urandom(4), "little") * 1e-12
payload = {{"schema": "synthetic-v1",
           "observed_dissociation": 12.784256853723884,
           "per_seed": {{"3": {{"dissociation": 12.784256853723884 + noise}}}},
           "observed_range": [10.29007380434875, 22.286656121007212],
           "D1": {{"held": True}}}}
(out / "RESULT.json").write_text(json.dumps(payload, indent=2) + "\\n")
{_CONSTANT_ARRAY}
"""

FIXTURE_NPZ_DIVERGENT = _HEADER + """
payload = {"schema": "synthetic-v1", "observed_dissociation": 12.784256853723884}
(out / "RESULT.json").write_text(json.dumps(payload, indent=2) + "\\n")
field = np.arange(12, dtype=np.float64).reshape(3, 4)
field[1, 2] += int.from_bytes(os.urandom(4), "little") * 1e-12
np.savez(out / "arrays.npz", field=field)
"""

# Every result file in this project records its own wall time, so this is the
# shape a real re-run takes: one timing field moves and nothing else does.
FIXTURE_ELAPSED_ONLY = _HEADER + f"""
payload = {{"schema": "synthetic-v1",
           "observed_dissociation": 12.784256853723884,
           "per_seed": {{"3": {{"dissociation": 12.784256853723884}}}},
           "elapsed_seconds": 3754.0 + int.from_bytes(os.urandom(4), "little") * 1e-6}}
(out / "RESULT.json").write_text(json.dumps(payload, indent=2) + "\\n")
{_CONSTANT_ARRAY}
"""

FIXTURE_FAILS = _HEADER + """
print("the solver did not converge", file=sys.stderr)
sys.exit(3)
"""

# The mutation. This is the line in compare_all that hands each emitted file to
# its comparator; dropping .json from it is exactly the defect "we never compared
# the JSON", and nothing else in the harness changes.
MUTATION_FROM = "    for relative in sorted(names):"
MUTATION_TO = ("    for relative in sorted("
               "n for n in names if not n.endswith('.json')):")


def write_fixture(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body)
    return path


def command_for(script: Path) -> str:
    """The command template, with the {OUT} token the harness substitutes."""
    return f'"{sys.executable}" "{script}" "{{OUT}}"'


def run_harness(work: Path, script: Path, outputs, report_name: str,
                harness: Path = HARNESS, extra=()) -> tuple[int, dict | None]:
    report = work / report_name
    argv = [sys.executable, str(harness),
            "--cmd", command_for(script),
            "--outputs", *outputs,
            "--report", str(report),
            "--scratch-root", str(work / f"scratch_{report_name}"),
            *extra]
    completed = subprocess.run(argv, capture_output=True, text=True)
    payload = json.loads(report.read_text()) if report.exists() else None
    return completed.returncode, payload


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

def case_deterministic(work: Path) -> tuple[bool, str]:
    script = write_fixture(work, "fixture_deterministic.py", FIXTURE_DETERMINISTIC)
    code, report = run_harness(work, script, ["*.json", "*.npz"], "deterministic.json",
                               extra=["--pin-threads"])
    if code != 0 or report is None or report["verdict"] != "BIT_IDENTICAL":
        return False, f"exit {code}, verdict {report and report['verdict']}"
    if report["compared_file_count"] != 2:
        return False, f"compared {report['compared_file_count']} files, expected 2"
    if report["environment"]["mode"] != "pinned":
        return False, "--pin-threads did not reach the report"
    if report["environment"]["overrides"]["OMP_NUM_THREADS"] != "1":
        return False, "the pinned environment was not recorded"
    if not report["harness_sha256"]:
        return False, "no harness digest in the report"
    return True, (f"BIT_IDENTICAL over {report['compared_file_count']} files, "
                  f"env {report['environment']['child_env_sha256'][:12]}")


def case_json_divergent(work: Path) -> tuple[bool, str]:
    script = write_fixture(work, "fixture_json.py", FIXTURE_JSON_DIVERGENT)
    code, report = run_harness(work, script, ["*.json", "*.npz"], "json_divergent.json")
    if code != 1 or report is None or report["verdict"] != "DIVERGENT":
        return False, f"exit {code}, verdict {report and report['verdict']}"
    first = report["first_divergence"]
    if first["key_path"] != "$.per_seed.3.dissociation":
        return False, f"named {first['key_path']}, expected $.per_seed.3.dissociation"
    if first["reference_value"] == first["other_value"]:
        return False, "reported a divergence between two equal values"
    result = report["files"]["RESULT.json"]
    if result["divergent_leaves"] != 1:
        return False, f"{result['divergent_leaves']} divergent leaves, expected 1"
    # The fixture adds at most 2**32 * 1e-12 to 12.784, so the relative move is
    # bounded by 3.4e-4. Anything larger means the harness reported the wrong pair.
    if not (0 < result["max_relative_difference"] < 3.4e-4):
        return False, f"relative difference {result['max_relative_difference']}"
    if len(set(result["sha256"])) != 2:
        return False, "the two repeats wrote the same bytes; the fixture did not vary"
    if not report["files"]["arrays.npz"]["identical"]:
        return False, "the constant array file was reported as divergent"
    return True, (f"{first['key_path']} moved "
                  f"{result['max_relative_difference']:.3e} relative")


def case_npz_divergent(work: Path) -> tuple[bool, str]:
    script = write_fixture(work, "fixture_npz.py", FIXTURE_NPZ_DIVERGENT)
    code, report = run_harness(work, script, ["*.json", "*.npz"], "npz_divergent.json")
    if code != 1 or report is None or report["verdict"] != "DIVERGENT":
        return False, f"exit {code}, verdict {report and report['verdict']}"
    first = report["first_divergence"]
    if first["kind"] != "array_value":
        return False, f"kind {first['kind']}, expected array_value"
    if first["array"] != "field" or first["index"] != [1, 2]:
        return False, f"named {first['array']} at {first['index']}, expected field [1, 2]"
    if first["reference_value"] == first["other_value"]:
        return False, "reported a divergence between two equal values"
    if not report["files"]["RESULT.json"]["identical"]:
        return False, "the constant JSON was reported as divergent"
    return True, f"field{first['index']} {first['reference_value']!r} vs " \
                 f"{first['other_value']!r}"


def case_ignored_key_passes(work: Path) -> tuple[bool, str]:
    """A wall-time field alone must reach a pass, under a verdict that names it."""
    script = write_fixture(work, "fixture_elapsed.py", FIXTURE_ELAPSED_ONLY)
    code, report = run_harness(work, script, ["*.json", "*.npz"], "elapsed.json",
                               extra=["--ignore-keys", "$.elapsed_seconds"])
    if code != 0 or report is None:
        return False, f"exit {code}"
    if report["verdict"] != "BIT_IDENTICAL_EXCEPT_IGNORED":
        return False, f"verdict {report['verdict']}"
    result = report["files"]["RESULT.json"]
    if result["bytes_identical"]:
        return False, "the fixture did not vary, so nothing was excused"
    if result["divergent_leaves"] != 0:
        return False, f"{result['divergent_leaves']} unignored leaves, expected 0"
    excused = result["ignored_divergences"]
    if len(excused) != 1 or excused[0]["key_path"] != "$.elapsed_seconds":
        return False, f"excused {[e['key_path'] for e in excused]}"
    if excused[0]["reference_value"] == excused[0]["other_value"]:
        return False, "the excused entry does not record two different values"
    return True, (f"excused $.elapsed_seconds {excused[0]['reference_value']:.3f} vs "
                  f"{excused[0]['other_value']:.3f}, verdict names the exclusion")


def case_ignored_key_hides_nothing_else(work: Path) -> tuple[bool, str]:
    """The exclusion must not become a way to pass a run that really diverged."""
    script = write_fixture(work, "fixture_json_ignored.py", FIXTURE_JSON_DIVERGENT)
    code, report = run_harness(
        work, script, ["*.json", "*.npz"], "ignore_guard.json",
        extra=["--ignore-keys", "$.elapsed_seconds", "$.observed_range[*]",
               "$.per_seed.4.*"])
    if code != 1 or report is None or report["verdict"] != "DIVERGENT":
        return False, f"exit {code}, verdict {report and report['verdict']}"
    if report["first_divergence"]["key_path"] != "$.per_seed.3.dissociation":
        return False, f"named {report['first_divergence']['key_path']}"
    if report["files"]["RESULT.json"]["divergent_leaves"] != 1:
        return False, "the exclusions changed the divergent-leaf count"
    return True, ("three exclusions, including one over a neighbouring seed, and the "
                  "real divergence still lands")


def case_command_fails(work: Path) -> tuple[bool, str]:
    script = write_fixture(work, "fixture_fails.py", FIXTURE_FAILS)
    code, report = run_harness(work, script, ["*.json"], "command_fails.json")
    if code != 2:
        return False, f"exit {code}, expected 2"
    if report is None or report["verdict"] != "COMMAND_FAILED":
        return False, f"verdict {report and report['verdict']}"
    if report["repeats"][0]["returncode"] != 3:
        return False, "the command's own exit code was not recorded"
    if "did not converge" not in report["repeats"][0]["stderr_tail"]:
        return False, "the command's stderr was not captured"
    return True, f"exit 2, verdict COMMAND_FAILED, child exit 3"


def case_mutation_removes_json_comparison(work: Path) -> tuple[bool, str]:
    """Delete the JSON comparison and the DIVERGENT case must wrongly pass.

    If it does not, either the mutation missed or the JSON comparison was never
    deciding anything, and both readings invalidate case_json_divergent.
    """
    source = HARNESS.read_text()
    if source.count(MUTATION_FROM) != 1:
        return False, (f"the mutation target appears {source.count(MUTATION_FROM)} "
                       "times; it must appear exactly once")
    mutant_source = source.replace(MUTATION_FROM, MUTATION_TO)
    if mutant_source == source:
        return False, "the mutation changed nothing, so it proves nothing"

    mutant = work / "mutant_determinism_harness.py"
    mutant.write_text(mutant_source)
    script = write_fixture(work, "fixture_json_for_mutant.py", FIXTURE_JSON_DIVERGENT)
    code, report = run_harness(work, script, ["*.json", "*.npz"], "mutant.json",
                               harness=mutant)
    if code != 0 or report is None or report["verdict"] != "BIT_IDENTICAL":
        return False, (f"the mutant said {report and report['verdict']} (exit {code}); "
                       "expected it to wrongly pass")
    if report["compared_file_count"] != 1:
        return False, (f"the mutant compared {report['compared_file_count']} files; "
                       "expected 1, the array file alone")
    if "RESULT.json" in report["files"]:
        return False, "the mutant still compared the JSON"
    return True, "the mutant wrongly reported BIT_IDENTICAL, so the real comparison " \
                 "is load-bearing"


def case_refuses_without_out_token(work: Path) -> tuple[bool, str]:
    """Without {OUT} both repeats write to one place, and here that place is real."""
    script = write_fixture(work, "fixture_no_token.py", FIXTURE_DETERMINISTIC)
    report = work / "no_token.json"
    completed = subprocess.run(
        [sys.executable, str(HARNESS),
         "--cmd", f'"{sys.executable}" "{script}" "{work}"',
         "--outputs", "*.json", "--report", str(report)],
        capture_output=True, text=True)
    if completed.returncode != 2:
        return False, f"exit {completed.returncode}, expected 2"
    payload = json.loads(report.read_text())
    if payload["verdict"] != "REFUSED" or "{OUT}" not in payload["refusal"]:
        return False, f"refusal {payload['refusal']!r}"
    return True, "refused, exit 2"


def case_refuses_empty_comparison(work: Path) -> tuple[bool, str]:
    """Comparing zero files always agrees, so it must not be reported as a pass."""
    script = write_fixture(work, "fixture_empty.py", FIXTURE_DETERMINISTIC)
    code, report = run_harness(work, script, ["NOT_EMITTED_*.json"], "empty.json")
    if code != 2:
        return False, f"exit {code}, expected 2"
    if report is None or report["verdict"] != "REFUSED":
        return False, f"verdict {report and report['verdict']}"
    if "matched no files" not in report["refusal"]:
        return False, f"refusal {report['refusal']!r}"
    return True, "refused, exit 2"


CASES = (
    ("deterministic command is BIT_IDENTICAL", case_deterministic),
    ("os.urandom in a JSON leaf is DIVERGENT", case_json_divergent),
    ("a divergent NPZ array is DIVERGENT", case_npz_divergent),
    ("a wall-time key alone passes under its own verdict", case_ignored_key_passes),
    ("an exclusion hides nothing else", case_ignored_key_hides_nothing_else),
    ("a failing command exits 2", case_command_fails),
    ("removing the JSON comparison makes the DIVERGENT case pass",
     case_mutation_removes_json_comparison),
    ("no {OUT} token is refused", case_refuses_without_out_token),
    ("an empty comparison is refused", case_refuses_empty_comparison),
)


def main() -> int:
    try:
        import numpy  # noqa: F401
    except ImportError:
        print(f"FAIL  the interpreter running these tests ({sys.executable}) has no "
              "numpy, and the array fixtures need it", file=sys.stderr)
        return 2

    failures = []
    with tempfile.TemporaryDirectory(prefix="test_determinism_harness_") as tmp:
        work = Path(tmp)
        for name, case in CASES:
            try:
                ok, detail = case(work)
            except Exception as exc:  # a crash is a failure of that case, not the run
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            print(f"{'PASS' if ok else 'FAIL'}  {name}\n        {detail}")
            if not ok:
                failures.append(name)

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} passed")
    for name in failures:
        print(f"  FAILED  {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

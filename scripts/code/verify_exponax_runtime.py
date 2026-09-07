"""Verify the cached Exponax runtime against the source hashes the pilot pinned.

The destroyed pilot.py recorded EXPECTED_EXPONAX_SOURCE_SHA256, a per-module
digest of every load-bearing Exponax source file, together with the upstream tag
commit. Those constants survive in the transcript. The uv cache still holds an
unpacked exponax 0.2.0 alongside jax and jaxlib 0.8.1.

If the cached modules hash to the pinned values then the solver available today
is byte-identical to the one that produced the destroyed results, which upgrades
the solver from `reconstructed` to `verified` even though every result file is
gone. That is a real, checkable claim and it is worth establishing before any
regeneration is attempted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE = Path.home() / ".cache/uv/archive-v0"
OUT = (Path(__file__).resolve().parents[2] / "results" / "tables"
       / "runtime_receipt")

EXPECTED_TAG_COMMIT = "6bd49a7a241358badd16a894d91f71d725fd8e61"

# Recovered verbatim from S1 step 349, destroyed /tmp/pde_core_pilot_exponax_v1/pilot.py
# lines 33-45. The transcript truncated the dict after etdrk._base_etdrk, so this is
# the recoverable subset, not the full pinned set.
EXPECTED_SOURCE_SHA256 = {
    "exponax": "5129897ab44a03bf0d148004970e745f5aa0d2c87297c5976ed4914cb4b22ddb",
    "exponax._base_stepper": "83daa060d942470d8be7b7b18c1695efe7521f5869feb42211597f0fd1ffa240",
    "exponax._spectral": "92c5b3d21e0e859dcf66fc4b45fc3ce4697e8743ab446f446c114fe490e1b217",
    "exponax.stepper": "da888e5716fb16f36477ebfb15fd78a0810861e6f26041e38136d687e50f5c59",
    "exponax.stepper._diffusion": "13eb546bcdd7d97f492ded2ec9cb7b091edb4d01772559487b23d5cff6b740eb",
    "exponax.stepper.reaction": "869d0e99c4175910484ea2ee4a94c05c19def8f903721671238a376a73008b05",
    "exponax.stepper.reaction._fisher_kpp": "d51da0ca542a66138b4eb2f360885a372c608743334579e72f586fb155ad8729",
    "exponax.nonlin_fun._base": "fa7c46aa926da4848697aedc75e622e7226d41556bddf6502c8c9b929a7edb95",
    "exponax.nonlin_fun._polynomial": "7b58f9279c51c3bff4d892673b0f11939ed457ad3ed754eb77e5c91cc96c8067",
    "exponax.nonlin_fun._zero": "805b5dd17dbdbcbbf2148b4bf5668d3d4aa3334a7b5a39bd137066ee3746346d",
    "exponax.etdrk._base_etdrk": "239a711c97c991babf46689619b66114cdebb42dc0cd12da8b128024a72a2fad",
}

PACKAGES = {
    "exponax": "0.2.0",
    "jax": "0.8.1",
    "jaxlib": "0.8.1",
    "equinox": "0.13.8",
    "jaxtyping": "0.3.11",
    "ml_dtypes": "0.6.0",
    "opt_einsum": "3.4.0",
    "scipy": "1.17.1",
    "numpy": "2.5.2",
    "typing_extensions": "4.16.0",
    "wadler_lindig": "0.1.7",
}


def resolve_paths() -> dict[str, str]:
    """Find the unpacked archive directory for each pinned package version."""
    found: dict[str, str] = {}
    for entry in CACHE.iterdir():
        for info in entry.glob("*.dist-info"):
            raw_name, raw_version = info.name.rsplit("-", 2)[:2]
            name = raw_name.lower().replace("-", "_")
            version = raw_version.removesuffix(".dist")
            if PACKAGES.get(name) == version and name not in found:
                found[name] = str(entry)
    return found


def module_path(root: Path, module: str) -> Path:
    parts = module.split(".")
    candidate = root.joinpath(*parts)
    return candidate / "__init__.py" if candidate.is_dir() else candidate.with_suffix(".py")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths()
    missing = sorted(set(PACKAGES) - set(paths))

    exponax_root = Path(paths["exponax"]) if "exponax" in paths else None
    checks = []
    if exponax_root is not None:
        for module, expected in EXPECTED_SOURCE_SHA256.items():
            source = module_path(exponax_root, module)
            if not source.exists():
                checks.append({"module": module, "status": "SOURCE_MISSING",
                               "expected_sha256": expected})
                continue
            observed = hashlib.sha256(source.read_bytes()).hexdigest()
            checks.append(
                {
                    "module": module,
                    "file": source.name,
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "matches": observed == expected,
                }
            )

    matched = sum(1 for c in checks if c.get("matches"))
    all_match = bool(checks) and matched == len(checks)

    payload = {
        "schema": "pde-exponax-runtime-receipt-v1",
        "generated": "2026-09-05",
        "purpose": (
            "Check the uv-cached Exponax 0.2.0 against the per-module SHA-256 values "
            "pinned by the destroyed pilot.py, recovered from S1 step 349."
        ),
        "expected_tag_commit": EXPECTED_TAG_COMMIT,
        "pinned_versions": PACKAGES,
        "resolved_archive_paths": paths,
        "missing_packages": missing,
        "modules_checked": len(checks),
        "modules_matching": matched,
        "all_load_bearing_sources_match": all_match,
        "note": (
            "The recovered hash dict is the subset the transcript rendered before "
            "truncating; the original pinned more modules. A full match on this "
            "subset is strong but not complete evidence of byte identity."
        ),
        "checks": checks,
    }
    (OUT / "EXPONAX_RUNTIME_RECEIPT.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"packages resolved from uv cache : {len(paths)} of {len(PACKAGES)}")
    if missing:
        print(f"  MISSING: {missing}")
    print(f"exponax modules checked         : {len(checks)}")
    print(f"exponax modules matching pin    : {matched}")
    print()
    for check in checks:
        mark = "OK  " if check.get("matches") else "FAIL"
        print(f"  {mark} {check['module']}")
    print()
    print(f"ALL LOAD-BEARING SOURCES MATCH  : {all_match}")


if __name__ == "__main__":
    main()

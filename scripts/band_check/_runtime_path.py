"""Put the pinned Exponax runtime on sys.path from the uv cache, offline.

The pinned scientific runtime `/tmp/pde_real_runtime_v1/venv` was destroyed. The uv
cache still holds every pinned package unpacked, including exponax 0.2.0 whose source
files hash to the values the destroyed pilot pinned (see
recovery/verify_exponax_runtime.py). Importing from the cache avoids any network
access and any new resolution.

Import this module before importing exponax or jax.
"""

from __future__ import annotations

import sys
from pathlib import Path

CACHE = Path.home() / ".cache/uv/archive-v0"

# The destroyed pilot's requirements.txt pinned exactly three packages. Those are
# the scientific runtime and must match exactly.
PINNED = {
    "exponax": "0.2.0",
    "jax": "0.8.1",
    "jaxlib": "0.8.1",
}

# Transitive support packages. The pilot did not pin these, and the cache does not
# hold every version for every ABI, so any cached version built for this interpreter
# is acceptable.
SUPPORT = (
    "equinox",
    "jaxtyping",
    "ml_dtypes",
    "opt_einsum",
    "scipy",
    "numpy",
    "typing_extensions",
    "wadler_lindig",
    # exponax/__init__ imports its viz subpackage unconditionally, which needs
    # matplotlib even though nothing here plots.
    "matplotlib",
    "contourpy",
    "cycler",
    "fonttools",
    "kiwisolver",
    "pyparsing",
    "packaging",
    "python_dateutil",
    "six",
    "pillow",
)


def _abi_ok(info: Path) -> bool:
    """Accept pure-Python wheels, or extension wheels built for this interpreter.

    jaxlib 0.8.1 ships only cp312 for macOS arm64, and several packages are cached
    at the same version for more than one ABI, so the archive has to be chosen by
    tag rather than by version alone.
    """
    wheel = info / "WHEEL"
    if not wheel.exists():
        return True
    tags = [
        line.split(":", 1)[1].strip()
        for line in wheel.read_text().splitlines()
        if line.lower().startswith("tag:")
    ]
    if not tags:
        return True
    this = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return any("none-any" in tag or tag.startswith(this) for tag in tags)


def resolve() -> tuple[dict[str, str], dict[str, str]]:
    """Return (exact-version pinned archives, best-available support archives)."""
    pinned: dict[str, str] = {}
    support: dict[str, tuple[str, str]] = {}
    for entry in CACHE.iterdir():
        if not entry.is_dir():
            continue
        for info in entry.glob("*.dist-info"):
            raw_name, raw_version = info.name.rsplit("-", 2)[:2]
            name = raw_name.lower().replace("-", "_")
            version = raw_version.removesuffix(".dist")
            if not _abi_ok(info):
                continue
            if PINNED.get(name) == version and name not in pinned:
                pinned[name] = str(entry)
            elif name in SUPPORT:
                current = support.get(name)
                key = tuple(int(p) for p in version.split(".") if p.isdigit())
                if current is None or key > current[0]:
                    support[name] = (key, str(entry))
    return pinned, {name: path for name, (_, path) in support.items()}


def activate() -> dict[str, str]:
    pinned, support = resolve()
    missing = sorted(set(PINNED) - set(pinned))
    if missing:
        raise RuntimeError(f"pinned packages absent from the uv cache: {missing}")
    missing_support = sorted(set(SUPPORT) - set(support))
    if missing_support:
        raise RuntimeError(f"support packages absent for this ABI: {missing_support}")
    found = {**support, **pinned}
    # Prepend so the cached pinned versions win over anything in the ambient env.
    for path in found.values():
        if path not in sys.path:
            sys.path.insert(0, path)
    return found


ACTIVE = activate()

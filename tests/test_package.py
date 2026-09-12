from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PackageTest(unittest.TestCase):
    def test_headlines(self) -> None:
        self.assertEqual(load_script("verify_headlines").verify(), [])

    def test_manifest_coverage_and_digests(self) -> None:
        self.assertEqual(load_script("verify_manifest").verify(), [])

    def test_no_private_path_or_identity_leaks(self) -> None:
        forbidden = (
            b"/" + b"Users" + b"/",
            b"/" + b"Volumes" + b"/",
            b"aksh" + b"garg",
            b"arnav" + b"garg",
        )
        for path in ROOT.rglob("*"):
            if (
                path.is_file()
                and path.name != "MANIFEST.sha256"
                and "__pycache__" not in path.parts
                and ".git" not in path.parts
            ):
                data = path.read_bytes()
                for needle in forbidden:
                    self.assertNotIn(needle, data, f"{needle!r} leaked in {path}")


if __name__ == "__main__":
    unittest.main()

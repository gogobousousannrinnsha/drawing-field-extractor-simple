"""Initial setup failures must not write outside the app or accept corrupt assets."""
import importlib.util
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

SCRIPT = Path(__file__).resolve().parents[2]/"scripts/bootstrap.py"
spec = importlib.util.spec_from_file_location("tested_bootstrap", SCRIPT)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def test_pip_environment_ignores_external_install_locations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.dict(os.environ, {"PIP_TARGET": "outside-fixture", "PIP_PREFIX": "outside-fixture",
                                         "PIP_USER": "1", "PIP_CONFIG_FILE": "outside.ini"}):
                env = bootstrap.pip_environment(root)
                self.assertEqual(os.environ["PIP_TARGET"], "outside-fixture")
            self.assertNotIn("PIP_TARGET", env)
            self.assertNotIn("PIP_PREFIX", env)
            self.assertNotIn("PIP_USER", env)
            self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
            self.assertEqual(Path(env["PIP_CACHE_DIR"]), root/"temp/pip")

    def test_reuse_rejects_modified_or_missing_installed_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            site = root/"runtime/Lib/site-packages"
            info = site/"demo-1.0.dist-info"
            info.mkdir(parents=True)
            code = site/"demo.py"
            code.write_bytes(b"hello")
            metadata = info/"METADATA"
            metadata.write_text("Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n", encoding="utf-8")
            rows = []
            for path in (code, metadata):
                digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()
                rows.append(f"{path.relative_to(site).as_posix()},sha256={digest},{path.stat().st_size}")
            rows.append("demo-1.0.dist-info/RECORD,,")
            (info/"RECORD").write_text("\n".join(rows)+"\n", encoding="utf-8")
            packages = [{"name": "demo", "version": "1.0"}]
            self.assertTrue(bootstrap.installed_wheels_ok(root, packages))
            code.write_bytes(b"wrong")
            self.assertFalse(bootstrap.installed_wheels_ok(root, packages))
            code.unlink()
            self.assertFalse(bootstrap.installed_wheels_ok(root, packages))

    def test_capacity_preflight_is_not_repeated_after_partial_acquisition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"assets").mkdir()
            (root/"assets/python-packages.json").write_text("{}")
            with patch.object(bootstrap.shutil, "disk_usage", return_value=SimpleNamespace(free=30*1024**3)):
                bootstrap.check_capacity(root)
            with patch.object(bootstrap.shutil, "disk_usage", return_value=SimpleNamespace(free=8*1024**3)):
                bootstrap.check_capacity(root)
            with patch.object(bootstrap.shutil, "disk_usage", return_value=SimpleNamespace(free=1000)), self.assertRaises(RuntimeError):
                bootstrap.check_capacity(root)
    def test_untrusted_and_parent_filenames_are_rejected(self):
        base = {"filename": "good.whl", "url": "https://files.pythonhosted.org/good.whl", "sha256": "a"*64}
        for updates in ({"filename": ".."}, {"filename": "../outside"}, {"filename": "a:b"},
                        {"url": "http://files.pythonhosted.org/good.whl"}, {"url": "https://example.com/a"},
                        {"sha256": "G"*64}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                bootstrap.validate_asset(base | updates)

    def test_valid_cached_download_does_not_use_network(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"valid.whl"
            path.write_bytes(b"valid")
            asset = {"filename": path.name, "url": "https://files.pythonhosted.org/valid.whl", "sha256": bootstrap.sha(path)}
            with patch.object(bootstrap.subprocess, "run", side_effect=AssertionError("network")):
                self.assertEqual(bootstrap.download(asset, folder, offline=True), path)

    def test_bad_cached_download_is_not_used_offline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"valid.whl"
            path.write_bytes(b"invalid")
            asset = {"filename": path.name, "url": "https://files.pythonhosted.org/valid.whl", "sha256": "a"*64}
            with self.assertRaises(RuntimeError):
                bootstrap.download(asset, folder, offline=True)
            self.assertFalse(path.exists())
            self.assertEqual(len(list(Path(folder).glob("*.bad-sha-*"))), 1)

    def test_zip_traversal_and_symbolic_links_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for entry in ("../outside", "C:/outside", "dir\\..\\..\\outside"):
                archive = root/"bad.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(entry, b"bad")
                with self.assertRaises(ValueError):
                    bootstrap.extract(archive, root/"out")
                self.assertFalse((root/"outside").exists())
            archive = root/"link.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                member = zipfile.ZipInfo("link")
                member.external_attr = 0o120777 << 16
                bundle.writestr(member, "outside")
            with self.assertRaises(ValueError):
                bootstrap.extract(archive, root/"out")

    def test_tar_symbolic_links_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root/"link.tar"
            with tarfile.open(archive, "w") as bundle:
                member = tarfile.TarInfo("link")
                member.type = tarfile.SYMTYPE
                member.linkname = "../outside"
                bundle.addfile(member)
            with self.assertRaises(ValueError):
                bootstrap.extract(archive, root/"out")

    def test_atomic_record_and_safe_archive_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bootstrap.write(root/"record.json", {"a": 1})
            bootstrap.write(root/"record.json", {"a": 2})
            self.assertEqual(bootstrap.read(root/"record.json"), {"a": 2})
            self.assertFalse((root/"record.json.building").exists())
            archive = root/"good.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("nested/good.txt", "good")
            bootstrap.extract(archive, root/"out")
            self.assertEqual((root/"out/nested/good.txt").read_text(), "good")

    def test_all_locked_packages_have_unique_wheel_hashes_and_exact_versions(self):
        root = Path(__file__).resolve().parents[2]
        packages = bootstrap.read(root/"assets/python-packages.json")["packages"]
        self.assertEqual(len(packages), len({item["name"].lower() for item in packages}))
        for item in packages:
            bootstrap.validate_asset(item)
            self.assertTrue(item["version"])
            self.assertTrue(item["filename"].endswith(".whl"))
        self.assertFalse(any(item["name"].startswith("docuworks") for item in packages))


if __name__ == "__main__":
    unittest.main()

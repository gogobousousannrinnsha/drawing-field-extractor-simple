"""Bundled Core must install locally and must never trust changed installed code."""
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import sys
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("core_setup_bootstrap", ROOT/"scripts/bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def make_bundle(root):
    for name in ("assets/bundled-core.json", "licenses/core/LICENSE.txt",
                 "bundled/core/docuworks_ctypes-1.0.1-py3-none-any.whl"):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, path)


class CoreSetupTests(unittest.TestCase):
    def test_changed_bundle_is_rejected_before_any_install_or_download(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_bundle(root)
            wheel = root/"bundled/core/docuworks_ctypes-1.0.1-py3-none-any.whl"
            wheel.write_bytes(wheel.read_bytes() + b"changed")
            with patch.object(bootstrap.subprocess, "run", side_effect=AssertionError("process")), \
                    patch.object(bootstrap, "download", side_effect=AssertionError("download")):
                with self.assertRaises(ValueError):
                    bootstrap.setup_bundled_core(root)
            self.assertEqual(bootstrap.read(root/"data/setup-core.json")["status"], "failed")

    def test_install_command_is_local_hash_pinned_and_does_not_resolve_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_bundle(root)
            with patch.object(bootstrap, "installed_core_ok", side_effect=[False, True]), \
                    patch.object(bootstrap.subprocess, "run") as run, \
                    patch.object(bootstrap, "download", side_effect=AssertionError("download")):
                bootstrap.setup_bundled_core(root)
            command = run.call_args_list[0].args[0]
            for option in ("--no-index", "--no-deps", "--require-hashes", "--isolated", "--no-user"):
                self.assertIn(option, command)
            self.assertEqual(command[command.index("--prefix") + 1], str(root/"runtime"))
            self.assertEqual(command[command.index("--find-links") + 1], str(root/"bundled/core"))
            self.assertEqual(run.call_count, 2)
            receipt = bootstrap.read(root/"data/setup-core.json")
            self.assertEqual(receipt["version"], "1.0.1")
            self.assertEqual(receipt["source"], "bundled-local-wheel")

    def test_valid_installed_core_skips_reinstall(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_bundle(root)
            with patch.object(bootstrap, "installed_core_ok", return_value=True), \
                    patch.object(bootstrap.subprocess, "run") as run:
                bootstrap.setup_bundled_core(root)
            self.assertEqual(run.call_count, 1)
            self.assertIn("check", run.call_args.args[0])

    def test_installed_record_cannot_hide_a_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_bundle(root)
            item = bootstrap.bundled_core_manifest(root)
            site = root/"runtime/Lib/site-packages"
            with zipfile.ZipFile(root/item["path"]) as archive:
                archive.extractall(site)  # Exact pinned wheel, validated above.
            self.assertTrue(bootstrap.installed_core_ok(root, item))
            source = site/"docuworks_ctypes/__init__.py"
            source.write_bytes(source.read_bytes() + b"\n# changed\n")
            record = site/"docuworks_ctypes-1.0.1.dist-info/RECORD"
            record.write_text("docuworks_ctypes/__init__.py,,\n", encoding="utf-8")
            self.assertFalse(bootstrap.installed_core_ok(root, item))

    def test_unreviewed_package_file_is_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_bundle(root)
            item = bootstrap.bundled_core_manifest(root)
            site = root/"runtime/Lib/site-packages"
            with zipfile.ZipFile(root/item["path"]) as archive:
                archive.extractall(site)
            (site/"docuworks_ctypes/extra.dll").write_bytes(b"synthetic fixture")
            self.assertFalse(bootstrap.installed_core_ok(root, item))

    def test_failed_install_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_bundle(root)
            with patch.object(bootstrap, "installed_core_ok", return_value=False), \
                    patch.object(bootstrap.subprocess, "run"):
                with self.assertRaises(RuntimeError):
                    bootstrap.setup_bundled_core(root)
            self.assertEqual(bootstrap.read(root/"data/setup-core.json")["status"], "failed")

    def test_failed_repair_invalidates_an_earlier_success_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_bundle(root)
            bootstrap.write(root/"data/setup-core.json", {"status": "complete", "version": "1.0.1"})
            with patch.object(bootstrap, "installed_core_ok", return_value=False), \
                    patch.object(bootstrap.subprocess, "run", side_effect=RuntimeError("install failed")):
                with self.assertRaises(RuntimeError):
                    bootstrap.setup_bundled_core(root)
            receipt = bootstrap.read(root/"data/setup-core.json")
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["reason"], "install failed")

    @unittest.skipUnless(os.name == "nt", "Windows setup lock")
    def test_new_setup_attempt_invalidates_previous_whole_setup_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bootstrap.write(root/"data/setup-complete.json", {"status": "complete", "app_version": "0.1.2"})
            with patch.object(sys, "argv", ["bootstrap.py", "--root", str(root)]), \
                    patch.object(bootstrap, "bundled_core_manifest", side_effect=ValueError("corrupt bundle")), \
                    patch.object(bootstrap, "setup_dependencies", side_effect=AssertionError("no download")):
                with self.assertRaises(ValueError):
                    bootstrap.main()
            self.assertEqual(bootstrap.read(root/"data/setup-complete.json")["status"], "in_progress")


if __name__ == "__main__":
    unittest.main()

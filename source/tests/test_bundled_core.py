"""Only the reviewed Core artifact may enter a source distribution or setup."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tested_bundled_core", ROOT / "scripts/bundled_core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class BundledCoreTests(unittest.TestCase):
    def modified_wheel(self, folder, updates=None, *, extra=None, remove=None,
                       recompute_record=False, symlink=None, duplicate=None):
        with zipfile.ZipFile(ROOT / core.CORE_PATH) as source:
            entries = {name: source.read(name) for name in source.namelist()}
        entries.update(updates or {})
        if remove:
            entries.pop(remove)
        if extra:
            entries[extra] = b"unreviewed"
        if recompute_record:
            rows = []
            for name, data in entries.items():
                if name == core.RECORD_MEMBER:
                    rows.append([name, "", ""])
                else:
                    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                    rows.append([name, "sha256=" + digest, str(len(data))])
            output = io.StringIO(newline="")
            csv.writer(output).writerows(rows)
            entries[core.RECORD_MEMBER] = output.getvalue().encode()
        path = Path(folder) / "changed.whl"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for name, data in entries.items():
                info = zipfile.ZipInfo(name)
                info.external_attr = ((stat.S_IFLNK if name == symlink else stat.S_IFREG) | 0o644) << 16
                target.writestr(info, data)
            if duplicate:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    target.writestr(duplicate, entries[duplicate])
        return path

    def test_real_public_artifact_and_manifest(self):
        result = core.audit_core_wheel(ROOT / core.CORE_PATH)
        self.assertEqual(result["member_count"], 28)
        self.assertEqual(result["python_source_count"], 23)
        self.assertTrue(result["record_verified"])
        self.assertFalse(result["proprietary_binaries_present"])
        self.assertEqual(core.validate_core_bundle(ROOT)["sha256"], core.CORE_SHA256)

    def test_unknown_dll_and_unsafe_members_rejected(self):
        for name in ("docuworks_ctypes/xdwapi.dll", "docuworks_ctypes/xdwapi.h", "sample.xdw",
                     "../outside", "dir\\outside", "/outside", "name:stream", "trailing. /file"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                path = self.modified_wheel(folder, extra=name)
                with self.assertRaises(ValueError):
                    core.audit_core_wheel(path)

    def test_duplicate_and_symbolic_link_rejected(self):
        for option in ({"duplicate": "docuworks_ctypes/api.py"}, {"symlink": "docuworks_ctypes/api.py"}):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as folder:
                path = self.modified_wheel(folder, **option)
                with self.assertRaisesRegex(ValueError, "Duplicate|Non-regular"):
                    core.audit_core_wheel(path)

    def test_missing_member_and_invalid_utf8_rejected(self):
        cases = ({"remove": "docuworks_ctypes/api.py"},
                 {"updates": {"docuworks_ctypes/api.py": b"\xff"}})
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(ValueError, "missing|UTF-8"):
                    core.audit_core_wheel(self.modified_wheel(folder, **case))

    def test_record_mismatch_and_missing_coverage_rejected(self):
        with zipfile.ZipFile(ROOT / core.CORE_PATH) as source:
            record = source.read(core.RECORD_MEMBER)
        cases = ({"docuworks_ctypes/api.py": b"# modified\n"},
                 {core.RECORD_MEMBER: b"\n".join(record.splitlines()[1:]) + b"\n"})
        for update in cases:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(ValueError, "RECORD"):
                    core.audit_core_wheel(self.modified_wheel(folder, updates=update))

    def test_mit_notice_and_metadata_must_be_preserved(self):
        with zipfile.ZipFile(ROOT / core.CORE_PATH) as source:
            metadata = source.read(core.DIST_INFO + "METADATA")
        cases = ({core.LICENSE_MEMBER: b"MIT License\n"},
                 {core.DIST_INFO + "METADATA": metadata.replace(b"License-Expression: MIT", b"License-Expression: Proprietary")},
                 {core.DIST_INFO + "METADATA": metadata.replace(b"Version: 1.0.1", b"Version: 1.0.2")},
                 {core.DIST_INFO + "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp313-win_amd64\n"})
        for update in cases:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(ValueError, "license|metadata|pure Python"):
                    core.audit_core_wheel(self.modified_wheel(folder, updates=update, recompute_record=True))

    def test_repackaged_bytes_cannot_replace_pinned_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "pinned public artifact"):
                core.audit_core_wheel(self.modified_wheel(folder))

    def test_secret_like_text_rejected_even_with_record_updated(self):
        with tempfile.TemporaryDirectory() as folder:
            value = "# ghp_" + "a" * 40 + "\n"
            with self.assertRaisesRegex(ValueError, "Secret-like"):
                core.audit_core_wheel(self.modified_wheel(folder,
                    updates={"docuworks_ctypes/api.py": value.encode()}, recompute_record=True))

    def copy_bundle(self, root):
        for relative in (core.CORE_PATH, "assets/bundled-core.json", "licenses/core/LICENSE.txt"):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)

    def test_manifest_cannot_redirect_or_change_version_or_digest(self):
        for update in ({"path": "../outside"}, {"version": "1.0.2"}, {"sha256": "a" * 64},
                       {"schema": True}, {"size_bytes": 1}, {"url": "https://example.com/core.whl"}):
            with self.subTest(update=update), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.copy_bundle(root)
                path = root / "assets/bundled-core.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                path.write_text(json.dumps(manifest | update), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest"):
                    core.validate_core_bundle(root)

    def test_changed_copied_license_and_missing_wheel_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.copy_bundle(root)
            (root / "licenses/core/LICENSE.txt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Copied Core license"):
                core.validate_core_bundle(root)
            (root / core.CORE_PATH).unlink()
            with self.assertRaisesRegex(ValueError, "regular file"):
                core.validate_core_bundle(root)


if __name__ == "__main__":
    unittest.main()

"""Publication may include exactly the reviewed Core wheel alongside text source."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("source_archive_builder", ROOT/"scripts/build_source_zip.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

MARKER_PUBLIC_INPUTS = {
    "黄色マーカー追加.bat", "黄色マーカー_使い方.txt", "source/marker_run.py",
    "source/simple_drawing/marker_excel.py", "source/simple_drawing/xdw_marker.py",
    "source/tests/test_marker_excel.py", "source/tests/test_marker_writer.py",
    "licenses/黄色マーカー-MIT-LICENSE.txt", "docs/VERIFICATION_JA.md",
}


class SourceArchiveTests(unittest.TestCase):
    def copy_public_inputs(self, root):
        for name, content in builder.source_files(ROOT).items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def test_archive_contains_exact_core_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"source"
            self.copy_public_inputs(root)
            (root/"bundled/core/unreviewed.whl").write_bytes(b"must not publish")
            (root/"bundled/core/xdwapi.dll").write_bytes(b"must not publish")
            first = builder.build(root, Path(folder)/"first.zip")
            second = builder.build(root, Path(folder)/"second.zip")
            self.assertEqual(first["sha256"], second["sha256"])
            with zipfile.ZipFile(Path(folder)/"first.zip") as archive:
                wheels = [name for name in archive.namelist() if name.endswith(".whl")]
                self.assertEqual(wheels, [builder.BUNDLED_CORE_PATH])
                self.assertFalse(any(name.endswith(".dll") for name in archive.namelist()))
                self.assertEqual(archive.read(builder.BUNDLED_CORE_PATH),
                                 (ROOT/builder.BUNDLED_CORE_PATH).read_bytes())
                self.assertIn("licenses/core/LICENSE.txt", archive.namelist())
                self.assertTrue(MARKER_PUBLIC_INPUTS.issubset(set(archive.namelist())))

    def test_missing_marker_public_inputs_cannot_be_published(self):
        for name in sorted(MARKER_PUBLIC_INPUTS):
            with self.subTest(missing=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.copy_public_inputs(root)
                (root/name).unlink()
                with self.assertRaisesRegex(ValueError, "Required public inputs are missing"):
                    builder.build(root, check_only=True)

    def test_missing_core_license_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.copy_public_inputs(root)
            (root/"licenses/core/LICENSE.txt").unlink()
            with self.assertRaises((FileNotFoundError, ValueError)):
                builder.build(root, check_only=True)

    def test_altered_core_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.copy_public_inputs(root)
            wheel = root/builder.BUNDLED_CORE_PATH
            wheel.write_bytes(wheel.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                builder.build(root, check_only=True)


if __name__ == "__main__":
    unittest.main()

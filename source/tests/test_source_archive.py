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

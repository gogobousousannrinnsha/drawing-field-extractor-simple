"""Atomic JSON replacement retries only transient Windows sharing failures."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

from simple_drawing import storage


def windows_error(code, text):
    error = OSError(text)
    error.winerror = code
    return error


class AtomicJsonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "record.json"
        self.old = {"state": "old", "value": "00017"}
        self.new = {"state": "new", "value": "こんにちは"}
        storage.write_json(self.path, self.old)
        self.before = self.path.read_bytes()

    def test_two_sharing_violations_then_complete_atomic_replacement(self):
        original_replace = storage.os.replace
        attempts = []
        def replace(source, destination):
            # Both failed attempts and the final successful attempt still see
            # the old destination and a fully written, readable staging JSON.
            self.assertEqual(Path(destination).read_bytes(), self.before)
            self.assertEqual(storage.read_json(source), self.new)
            attempts.append(Path(source))
            if len(attempts) < 3:
                raise windows_error(32 if len(attempts) == 1 else 33, "synthetic sharing violation")
            original_replace(source, destination)
        with patch.object(storage.os, "replace", side_effect=replace), \
             patch.object(storage.time, "sleep") as sleep:
            storage.write_json(self.path, self.new)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_args_list, [call(.1), call(.1)])
        self.assertEqual(storage.read_json(self.path), self.new)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_access_denied_and_disk_full_propagate_without_retry_or_old_json_change(self):
        for code in (5, 112):
            error = windows_error(code, "synthetic non-transient failure")
            with patch.object(storage.os, "replace", side_effect=error) as replace, \
                 patch.object(storage.time, "sleep") as sleep:
                with self.assertRaises(OSError) as caught:
                    storage.write_json(self.path, self.new)
            self.assertIs(caught.exception, error)
            replace.assert_called_once()
            sleep.assert_not_called()
            self.assertEqual(self.path.read_bytes(), self.before)
            self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_cleanup_failure_preserves_original_replace_exception(self):
        original_error = windows_error(112, "original disk-full failure")
        cleanup_error = windows_error(32, "cleanup file is held")
        with patch.object(storage.os, "replace", side_effect=original_error), \
             patch.object(Path, "unlink", side_effect=cleanup_error), \
             patch.object(storage.time, "sleep") as sleep:
            with self.assertRaises(OSError) as caught:
                storage.write_json(self.path, self.new)
        self.assertIs(caught.exception, original_error)
        self.assertEqual(self.path.read_bytes(), self.before)
        self.assertEqual(len(list(self.root.glob("*.tmp"))), 1)
        sleep.assert_not_called()

    def test_persistent_sharing_failure_is_bounded_and_preserves_old_json(self):
        error = windows_error(32, "persistent sharing violation")
        with patch.object(storage.os, "replace", side_effect=error) as replace, \
             patch.object(storage.time, "sleep") as sleep:
            with self.assertRaises(OSError) as caught:
                storage.write_json(self.path, self.new)
        self.assertIs(caught.exception, error)
        self.assertEqual(replace.call_count, 21)
        self.assertEqual(sleep.call_args_list, [call(.1)]*20)
        self.assertEqual(self.path.read_bytes(), self.before)
        self.assertEqual(list(self.root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

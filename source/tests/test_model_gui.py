import threading
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from simple_drawing.model_gui import ModelManagerWindow, OperationWorker, launch, perform_action, progress_message


class ProgressDisplayTests(unittest.TestCase):
    def test_download_status_is_japanese_without_changing_manager_record(self):
        for status in ("pulling manifest", "pulling abc123", "verifying sha256 digest", "writing manifest", "success"):
            value = {"stage": "download", "status": status, "digest": "sha256:original"}
            original = dict(value)
            text = progress_message(value)
            self.assertIn("ダウンロード", text)
            self.assertNotIn(status, text)
            self.assertNotIn("sha256:original", text)
            self.assertEqual(value, original)

    def test_known_stage_names_are_not_shown_as_english(self):
        for stage in ("download", "verify", "check", "complete"):
            text = progress_message({"stage": stage, "status": "確認しています。"})
            self.assertNotIn(stage, text)
            self.assertIn("確認しています。", text)


class ManagerDispatchTests(unittest.TestCase):
    def setUp(self):
        self.manager = Mock()
        self.cancel = threading.Event()
        self.progress = Mock()

    def test_refresh_and_selection_use_manager_without_gui_lock(self):
        self.manager.list_profiles.return_value = [{"profile_id": "gemma4-q8"}]
        self.assertEqual(perform_action(self.manager, "refresh", None, self.cancel, self.progress),
                         [{"profile_id": "gemma4-q8"}])
        perform_action(self.manager, "select", "gemma3-4b", self.cancel, self.progress)
        self.manager.select.assert_called_once_with("gemma3-4b")

    def test_long_operations_receive_cancel_and_progress(self):
        for action in ("install", "check"):
            perform_action(self.manager, action, "gemma3-4b", self.cancel, self.progress)
            getattr(self.manager, action).assert_called_once_with(
                "gemma3-4b", cancel=self.cancel, progress=self.progress)

    def test_unknown_action_and_missing_selection_do_not_run_manager(self):
        for action, profile_id in (("delete", "gemma4-q8"), ("install", None), ("check", None), ("select", None)):
            with self.assertRaises(ValueError):
                perform_action(self.manager, action, profile_id, self.cancel, self.progress)
        self.assertEqual(self.manager.mock_calls, [])


class WorkerTests(unittest.TestCase):
    def test_completion_is_not_announced_before_cleanup(self):
        worker = OperationWorker()
        started = threading.Event()
        cleanup_released = threading.Event()
        cleaned = []

        class Cancelled(Exception):
            code = "CANCELLED"

        def operation(cancel, progress):
            started.set()
            self.assertTrue(cancel.wait(3))
            self.assertTrue(cleanup_released.wait(3))
            cleaned.append(True)
            raise Cancelled("中止")

        worker.start("install", operation)
        self.assertTrue(started.wait(3))
        worker.request_cancel()
        self.assertFalse(worker.finished)
        self.assertEqual(worker.read_events(), [])
        cleanup_released.set()
        worker.thread.join(3)
        events = worker.read_events()
        self.assertEqual(cleaned, [True])
        self.assertEqual(events[0]["kind"], "complete")
        self.assertEqual(events[0]["code"], "CANCELLED")
        self.assertTrue(worker.finished)

    def test_completed_but_unreceived_operation_blocks_second_start(self):
        worker = OperationWorker()
        worker.start("refresh", lambda cancel, progress: [{"profile_id": "gemma4-q8"}])
        worker.thread.join(3)
        with self.assertRaises(RuntimeError):
            worker.start("select", lambda cancel, progress: None)
        events = worker.read_events()
        self.assertEqual(events[0]["result"], [{"profile_id": "gemma4-q8"}])
        self.assertTrue(worker.finished)

    def test_progress_is_copied_and_emitted_before_completion(self):
        worker = OperationWorker()

        def operation(cancel, progress):
            value = {"stage": "download", "completed": 10, "total": 100}
            progress(value)
            value["completed"] = 99
            return {"installed": True}

        worker.start("install", operation)
        worker.thread.join(3)
        events = worker.read_events()
        self.assertEqual([event["kind"] for event in events], ["progress", "complete"])
        self.assertEqual(events[0]["value"]["completed"], 10)
        self.assertEqual(events[1]["result"], {"installed": True})

    def test_operation_failure_is_preserved_for_ui_and_next_job(self):
        worker = OperationWorker()

        class Failure(Exception):
            code = "CHECK_FAILED"

        def failing(cancel, progress):
            raise Failure("モデルの回答が一致しません。")

        worker.start("check", failing)
        worker.thread.join(3)
        event = worker.read_events()[0]
        self.assertEqual(event["code"], "CHECK_FAILED")
        self.assertIn("回答", event["error"])
        self.assertIn("Failure", event["traceback"])
        worker.start("refresh", lambda cancel, progress: [])
        worker.thread.join(3)
        self.assertEqual(worker.read_events()[0]["result"], [])

    def test_cancellation_does_not_leak_into_next_operation(self):
        worker = OperationWorker()
        worker.start("check", lambda cancel, progress: None)
        worker.request_cancel()
        worker.thread.join(3)
        worker.read_events()
        previous = worker.cancel_event
        worker.start("refresh", lambda cancel, progress: cancel.is_set())
        worker.thread.join(3)
        self.assertIsNot(worker.cancel_event, previous)
        self.assertFalse(worker.read_events()[0]["result"])


class WindowLifecycleTests(unittest.TestCase):
    def test_download_total_does_not_stay_at_100_percent_during_verification(self):
        app = ModelManagerWindow.__new__(ModelManagerWindow)
        app.closing = False
        app.worker = Mock(cancel_event=threading.Event())
        app.status_text = Mock()
        state = {"mode": "indeterminate", "value": 0}
        app.progress = Mock()
        app.progress.configure.side_effect = lambda **kwargs: state.update(kwargs)
        app.progress.cget.side_effect = lambda key: state[key]
        app._handle_progress({"stage": "download", "status": "success", "completed": 100, "total": 100})
        self.assertEqual(state, {"mode": "determinate", "value": 100})
        app._handle_progress({"stage": "verify", "status": "確認しています。"})
        self.assertEqual(state, {"mode": "indeterminate", "value": 0})
        app.progress.start.assert_called_once_with(12)
        app._handle_progress({"stage": "check", "status": "動作を確認しています。"})
        self.assertEqual(state, {"mode": "indeterminate", "value": 0})
        app.progress.start.assert_called_once_with(12)

    def test_window_close_waits_for_owned_cleanup_without_blocking_ui(self):
        worker = OperationWorker()
        started, release_cleanup = threading.Event(), threading.Event()

        def operation(cancel, progress):
            started.set()
            cancel.wait(3)
            release_cleanup.wait(3)
            return {"cleanup_complete": True}

        worker.start("install", operation)
        self.assertTrue(started.wait(3))
        app = ModelManagerWindow.__new__(ModelManagerWindow)
        app.window, app.status_text = Mock(), Mock()
        app.worker, app.closing = worker, False
        app._refresh_pending = True
        app._update_buttons = Mock()
        app._handle_complete = Mock()
        app.close()
        self.assertTrue(worker.cancel_event.is_set())
        app.window.destroy.assert_not_called()
        app.poll()
        app.window.after.assert_called_once_with(80, app.poll)
        app.window.destroy.assert_not_called()
        release_cleanup.set()
        worker.thread.join(3)
        app.poll()
        app.window.destroy.assert_called_once_with()

    def test_startup_failure_leaves_diagnostic_in_owned_logs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            root.resolve().relative_to(Path(__file__).resolve().parent)
            with patch("tkinter.Tk", side_effect=RuntimeError("Tk cannot start")), \
                 patch("simple_drawing.model_gui.ctypes.windll", create=True), \
                 patch("simple_drawing.model_gui.sys.stderr", None):
                self.assertEqual(launch(root, manager=Mock()), 1)
            logs = list((root / "logs").glob("model-gui-startup-*.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("Tk cannot start", logs[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from filelock import FileLock
import httpx

from simple_drawing import model_manager as manager
from simple_drawing.model_manager import ModelManager, ModelManagerError, ModelOperationCancelled, check_fingerprint
from simple_drawing.profiles import get_profile
from simple_drawing.storage import digest, read_json, sha, write_json

TEST_ROOT = Path(__file__).resolve().parent


class FakeServer:
    capabilities = ["completion", "vision"]
    remote_host = None
    requires = None
    instances = []
    def __init__(self, root, *, profile=None, management=False):
        self.root = Path(root)
        self.port = 11449
        self.profile = profile
        self.management = management
        self.contract = {}
        self.shutdown_report = {}
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def stop(self):
        self.shutdown_report = {"process_stopped": True, "model_unloaded": True, "errors": []}
        return self.shutdown_report

    def api(self, path, body=None, timeout=900):
        if path == "/api/tags":
            models = []
            base = self.root / "models/ollama/manifests/registry.ollama.ai/library"
            for file in base.glob("*/*"):
                data = read_json(file)
                models.append({"name": f"{file.parent.name}:{file.name}", "digest": sha(file),
                               "size": sum(r["size"] for r in [data["config"], *data["layers"]])})
            return {"models": models}
        if path == "/api/show":
            return {"capabilities": self.capabilities, "remote_host": self.remote_host, "requires": self.requires}
        return {"models": []}


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.joinpath("data").mkdir()
        write_json(self.root / "実行設定.json", {"input_folder": "input with spaces", "llm_backend": "saluki",
                                              "saluki_thinking": True, "saluki_batch_size": 512})
        self.service = ModelManager(self.root)
        FakeServer.capabilities = ["completion", "vision"]
        FakeServer.remote_host = None
        FakeServer.requires = None
        FakeServer.instances = []
        self.addCleanup(patch.stopall)
        patch.object(manager.llm, "OllamaServer", FakeServer).start()
        patch.object(manager.llm, "model_contract", return_value={"ready": True}).start()
        # These legacy cases isolate asset/probe behavior. Terms have their own
        # temporary-fixture tests in test_models_setup; no real consent is saved.
        patch.object(manager, "require_model_terms", return_value=None).start()
        patch.object(manager.shutil, "disk_usage", return_value=type("Disk", (), {"free": 20*1024**3})()).start()

    def assets(self, profile_id="gemma3-4b", blob=b"tiny model"):
        profile = get_profile(profile_id)
        name, tag = profile.model.split(":")
        hash_value = hashlib.sha256(blob).hexdigest()
        path = self.root / f"models/ollama/blobs/sha256-{hash_value}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        manifest = self.root / f"models/ollama/manifests/registry.ollama.ai/library/{name}/{tag}"
        write_json(manifest, {"config": {"digest": "sha256:" + hash_value, "size": len(blob)}, "layers": []})
        return manifest, path

    def good_check(self):
        def extract(server, image, page, regions, settings):
            value = ({"status": "found", "results": [{"value": "001-020", "source_ids": ["p1:r000003"]}], "note": ""}
                     if settings["item"] == "Part number" else {"status": "not_found", "results": [], "note": ""})
            return {"parsed": value, "response": {"done": True}, "timings": {"wall_seconds": .1}}
        return patch.object(manager.llm, "extract_page", side_effect=extract)

    def test_install_existing_does_not_pull_or_select(self):
        self.assets()
        before = (self.root / "実行設定.json").read_bytes()
        with patch.object(manager.httpx, "Client") as client:
            record = self.service.install("gemma3-4b")
        client.assert_not_called()
        self.assertTrue(record["full_hash_verified"])
        self.assertIsNone(record["check"])
        self.assertEqual((self.root / "実行設定.json").read_bytes(), before)

    def test_existing_same_identity_preserves_check_and_readiness(self):
        self.assets()
        with self.good_check():
            checked = self.service.check("gemma3-4b")
        verified = self.service.install("gemma3-4b")
        self.assertEqual(verified["check"], checked["check"])
        self.assertEqual(self.service.ensure_usable("gemma3-4b")["digest"], checked["digest"])

    def test_changed_adapter_or_probe_source_requires_new_check(self):
        self.assets()
        with self.good_check():
            checked = self.service.check("gemma3-4b")
        sources = manager._code_fingerprints()
        self.assertEqual(set(sources), {"llm.py", "model_manager.py", "model_download.py", "model_probe.py", "profiles.py", "model_terms.py"})
        with patch.object(manager, "_code_fingerprints", return_value={**sources, "llm.py": "0"*64}):
            with self.assertRaises(ModelManagerError) as caught:
                self.service.ensure_usable("gemma3-4b")
        self.assertEqual(caught.exception.code, "UNCHECKED_MODEL")
        self.assertEqual(self.service.ensure_usable("gemma3-4b")["digest"], checked["digest"])

    def test_code_changed_during_check_is_not_certified(self):
        self.assets()
        fingerprints = manager._code_fingerprints()
        reads = 0
        def source_identity():
            nonlocal reads
            reads += 1
            return fingerprints if reads == 1 else {**fingerprints, "llm.py": "f"*64}
        with self.good_check(), patch.object(manager, "_code_fingerprints", side_effect=source_identity):
            with self.assertRaises(ModelManagerError) as caught:
                self.service.check("gemma3-4b")
        self.assertEqual(caught.exception.code, "PROFILE_CHANGED")
        self.assertFalse(self.service.registry_path.exists())

    def test_generation_limit_answer_is_retained_and_never_passes_check(self):
        self.assets()
        original_api = FakeServer.api
        def api(server, path, body=None, timeout=900):
            if path == "/api/chat":
                return {"done": True, "done_reason": "length", "message": {"content": json.dumps({
                    "status": "found", "results": [{"value": "001-020", "source_ids": ["p1:r000003"]}], "note": ""})}}
            return original_api(server, path, body, timeout)
        with patch.object(FakeServer, "api", api), patch.object(manager.llm, "_gpu_sample", return_value={}):
            with self.assertRaises(ModelManagerError) as caught:
                self.service.check("gemma3-4b")
        self.assertEqual(caught.exception.code, "CHECK_FAILED")
        report = next(self.root.glob("data/model-checks/*/check.json"))
        raw = read_json(report.parent / "value-and-evidence.json")
        self.assertEqual(raw["error"]["code"], "INCOMPLETE_RESPONSE")
        self.assertEqual(raw["response"]["done_reason"], "length")
        self.assertIsNone(raw["parsed"])

    def test_missing_check_evidence_cannot_be_selected(self):
        self.assets()
        with self.good_check():
            checked = self.service.check("gemma3-4b")
        (self.root / checked["check"]["report_path"]).unlink()
        with self.assertRaises(ModelManagerError) as caught:
            self.service.select("gemma3-4b")
        self.assertEqual(caught.exception.code, "UNCHECKED_MODEL")
        card = self.service.list_profiles()[1]
        self.assertFalse(card["checked"])
        self.assertEqual(card["check_status"], "再確認が必要")

    def test_changed_probe_input_or_raw_record_or_report_requires_new_check(self):
        for artifact in ("input/page.png", "value-and-evidence.json", "check.json"):
            self.assets()
            with self.good_check():
                checked = self.service.check("gemma3-4b")
            folder = (self.root / checked["check"]["report_path"]).parent
            target = folder / artifact
            target.write_bytes(target.read_bytes()+b" ")
            with self.assertRaises(ModelManagerError) as caught:
                self.service.ensure_usable("gemma3-4b")
            self.assertEqual(caught.exception.code, "UNCHECKED_MODEL")

    def test_failed_shutdown_does_not_commit_a_successful_check(self):
        self.assets()
        def failed_stop(server):
            server.shutdown_report = {"model_unloaded": False, "process_stopped": False, "errors": ["identity changed"]}
        with self.good_check(), patch.object(FakeServer, "stop", failed_stop):
            with self.assertRaises(ModelManagerError) as caught:
                self.service.check("gemma3-4b")
        self.assertEqual(caught.exception.code, "CLEANUP_FAILED")
        self.assertFalse(self.service.registry_path.exists())
        report = next(self.root.glob("data/model-checks/*/check.json"))
        self.assertEqual(read_json(report)["status"], "failed")

    def test_missing_assets_and_thinking_and_unknown_identity_are_refused(self):
        with self.assertRaises(ModelManagerError) as caught:
            self.service.ensure_usable("gemma3-4b")
        self.assertEqual(caught.exception.code, "NOT_INSTALLED")
        with self.assertRaises(ModelManagerError) as caught:
            self.service._capabilities(FakeServer(self.root), get_profile("gemma4-q8"))
        self.assertEqual(caught.exception.code, "UNSUPPORTED_THINKING")
        with self.assertRaises(ModelManagerError) as caught:
            self.service.ensure_usable(replace(get_profile("gemma3-4b"), model="gemma3:1b"))
        self.assertEqual(caught.exception.code, "UNKNOWN_PROFILE")

    def test_malformed_registry_record_cannot_fake_a_check(self):
        self.assets()
        identity = self.service._asset_identity("gemma3-4b")
        write_json(self.service.registry_path, {"version": 1, "models": {"gemma3-4b": {
            **identity, "full_hash_verified": True, "check": "passed"}}})
        with self.assertRaises(ModelManagerError) as caught:
            self.service.ensure_usable("gemma3-4b")
        self.assertEqual(caught.exception.code, "INVALID_REGISTRY")

    def test_bad_blob_equal_size_fails_hash_verification(self):
        _, blob = self.assets()
        blob.write_bytes(b"evil model")
        with self.assertRaises(ModelManagerError) as caught:
            self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "CORRUPT_MODEL")
        self.assertFalse(self.service.registry_path.exists())

    def test_same_size_blob_change_after_check_blocks_run_and_selection(self):
        _, blob = self.assets()
        with self.good_check():
            self.service.check("gemma3-4b")
        selection_before = (self.root / "実行設定.json").read_bytes()
        registry_before = self.service.registry_path.read_bytes()
        old_size = blob.stat().st_size
        blob.write_bytes(b"evil model")
        self.assertEqual(blob.stat().st_size, old_size)
        # This is the same helper the real workflow calls once at run startup.
        with FileLock(str(self.root / "data/operation.lock"), timeout=0):
            with self.assertRaises(ModelManagerError) as run_error:
                manager.ensure_usable(self.root, "gemma3-4b")
        self.assertEqual(run_error.exception.code, "CORRUPT_MODEL")
        with self.assertRaises(ModelManagerError) as selection_error:
            self.service.select("gemma3-4b")
        self.assertEqual(selection_error.exception.code, "CORRUPT_MODEL")
        self.assertEqual((self.root / "実行設定.json").read_bytes(), selection_before)
        self.assertEqual(self.service.registry_path.read_bytes(), registry_before)

    def test_list_uses_fast_identity_but_selection_hashes_all_blobs(self):
        self.assets()
        with self.good_check():
            self.service.check("gemma3-4b")
        with patch.object(self.service, "_asset_identity", wraps=self.service._asset_identity) as identity:
            cards = self.service.list_profiles()
        self.assertTrue(cards[1]["checked"])
        self.assertTrue(identity.call_args_list)
        self.assertTrue(all(not call.kwargs.get("full_hash", False) for call in identity.call_args_list))
        with patch.object(self.service, "_asset_identity", wraps=self.service._asset_identity) as identity:
            self.service.select("gemma3-4b")
        self.assertEqual(len(identity.call_args_list), 1)
        self.assertTrue(identity.call_args.kwargs["full_hash"])

    def test_remote_and_nonvision_and_requires_are_refused(self):
        self.assets()
        scenarios = [(None, ["completion"], None, "UNSUPPORTED_MODEL"),
                     ("https://ollama.com", ["completion", "vision"], None, "REMOTE_MODEL"),
                     (None, ["completion", "vision"], "99.0.0", "UNSUPPORTED_RUNTIME")]
        for remote, caps, requires, code in scenarios:
            FakeServer.remote_host, FakeServer.capabilities, FakeServer.requires = remote, caps, requires
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
            self.assertEqual(caught.exception.code, code)
        self.assertFalse(self.service.registry_path.exists())

    def test_successful_check_saves_raw_and_select_preserves_other_settings(self):
        self.assets()
        with self.good_check():
            result = self.service.check("gemma3-4b")
        report = read_json(self.root / result["check"]["report_path"])
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["shutdown"]["process_stopped"])
        self.assertEqual(len(report["cases"]), 2)
        self.assertEqual(read_json((self.root / result["check"]["report_path"]).parent / "value-and-evidence.json")["timings"]["wall_seconds"], .1)
        selected = self.service.select("gemma3-4b")
        self.assertTrue(selected["selected"])
        self.assertEqual(read_json(self.root / "実行設定.json"), {
            "input_folder": "input with spaces", "llm_backend": "gemma", "ollama_profile": "gemma3-4b",
            "saluki_thinking": True, "saluki_batch_size": 512})
        cards = self.service.list_profiles()
        self.assertTrue(cards[1]["selected"])
        self.assertTrue(cards[1]["checked"])

    def test_unchecked_or_changed_digest_or_settings_cannot_select(self):
        manifest, _ = self.assets()
        with self.assertRaises(ModelManagerError) as caught:
            self.service.select("gemma3-4b")
        self.assertEqual(caught.exception.code, "UNCHECKED_MODEL")
        with self.good_check():
            self.service.check("gemma3-4b")
        profile = get_profile("gemma3-4b")
        changed = replace(profile, options={**profile.options, "num_ctx": 8192})
        with self.assertRaises(ModelManagerError) as caught:
            self.service.ensure_usable(changed)
        self.assertEqual(caught.exception.code, "UNCHECKED_MODEL")
        manifest.write_text(manifest.read_text()+"\n", encoding="utf-8")
        with self.assertRaises(ModelManagerError) as caught:
            self.service.select("gemma3-4b")
        self.assertEqual(caught.exception.code, "MODEL_CHANGED")

    def test_bad_registry_is_not_overwritten(self):
        self.service.registry_path.write_text("broken", encoding="utf-8")
        with self.assertRaises(ModelManagerError) as caught:
            self.service.list_profiles()
        self.assertEqual(caught.exception.code, "INVALID_REGISTRY")
        self.assertEqual(self.service.registry_path.read_text(), "broken")

    def test_failed_value_or_evidence_cannot_be_selected(self):
        self.assets()
        before = (self.root / "実行設定.json").read_bytes()
        wrong = {"parsed": {"status": "found", "results": [{"value": "001-020", "source_ids": []}], "note": "image"}}
        with patch.object(manager.llm, "extract_page", return_value=wrong):
            with self.assertRaises(ModelManagerError) as caught:
                self.service.check("gemma3-4b")
        self.assertEqual(caught.exception.code, "CHECK_FAILED")
        self.assertEqual((self.root / "実行設定.json").read_bytes(), before)
        report = next(self.root.glob("data/model-checks/*/check.json"))
        self.assertEqual(read_json(report)["status"], "failed")
        with self.assertRaises(ModelManagerError):
            self.service.select("gemma3-4b")

    def test_cancel_check_keeps_selection_and_saves_raw(self):
        self.assets()
        before = (self.root / "実行設定.json").read_bytes()
        cancel = threading.Event()
        def extract(*args):
            cancel.set()
            return {"parsed": None, "error": {"code": "API_ERROR"}, "response": {"raw": "retained"}}
        with patch.object(manager.llm, "extract_page", side_effect=extract):
            with self.assertRaises(ModelOperationCancelled):
                self.service.check("gemma3-4b", cancel=cancel)
        report = next(self.root.glob("data/model-checks/*/check.json"))
        self.assertEqual(read_json(report)["status"], "cancelled")
        self.assertEqual(read_json(report.parent / "value-and-evidence.json")["response"], {"raw": "retained"})
        self.assertEqual((self.root / "実行設定.json").read_bytes(), before)
        self.assertTrue(FakeServer.instances[-1].shutdown_report["process_stopped"])

    def test_lock_and_unknown_profile_and_pre_cancel(self):
        with FileLock(str(self.root / "data/operation.lock"), timeout=0):
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "BUSY")
        with self.assertRaises(ModelManagerError) as caught:
            self.service.install("unregistered")
        self.assertEqual(caught.exception.code, "UNKNOWN_PROFILE")
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(ModelOperationCancelled):
            self.service.install("gemma3-4b", cancel=cancel)
        self.assertFalse(FakeServer.instances)

    def test_stream_download_checks_terminal_success_and_hashes(self):
        requests = []
        def handle(request):
            requests.append(request)
            self.assets()
            return httpx.Response(200, text='{"status":"pulling model","completed":10,"total":10}\n{"status":"success"}\n')
        original_client = httpx.Client
        client = original_client(transport=httpx.MockTransport(handle), trust_env=False)
        with patch.object(manager.httpx, "Client", return_value=client):
            result = self.service.install("gemma3-4b")
        self.assertEqual(json.loads(requests[0].content), {"model": "gemma3:4b", "stream": True})
        self.assertTrue(result["full_hash_verified"])
        self.assertTrue(client.is_closed)

    def test_stream_error_or_no_success_keeps_selection(self):
        before = (self.root / "実行設定.json").read_bytes()
        for text, code in [('{"status":"pulling"}\n{"error":"network failed"}\n', "MODEL_DOWNLOAD_FAILED"),
                           ('{"status":"pulling","completed":10,"total":10}\n', "MODEL_DOWNLOAD_FAILED")]:
            client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=text)))
            with patch.object(manager.httpx, "Client", return_value=client):
                with self.assertRaises(ModelManagerError) as caught:
                    self.service.install("gemma3-4b")
            self.assertEqual(caught.exception.code, code)
            self.assertTrue(client.is_closed)
        self.assertEqual((self.root / "実行設定.json").read_bytes(), before)
        self.assertFalse(self.service.registry_path.exists())

    def test_cancel_download_closes_client_and_preserves_selection(self):
        cancel = threading.Event()
        before = (self.root / "実行設定.json").read_bytes()
        def progress(item):
            if item["stage"] == "download":
                cancel.set()
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200,
            text='{"status":"pulling","completed":1,"total":10}\n{"status":"pulling","completed":2,"total":10}\n')))
        with patch.object(manager.httpx, "Client", return_value=client):
            with self.assertRaises(ModelOperationCancelled):
                self.service.install("gemma3-4b", cancel=cancel, progress=progress)
        self.assertTrue(client.is_closed)
        self.assertEqual((self.root / "実行設定.json").read_bytes(), before)
        self.assertFalse(self.service.registry_path.exists())

    def test_disk_full_does_not_begin_download(self):
        with patch.object(manager.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()), patch.object(manager.httpx, "Client") as client:
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "DISK_FULL")
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()

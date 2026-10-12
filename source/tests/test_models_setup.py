"""New setup paths use tiny fake assets; these tests never accept real user terms."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import httpx

from simple_drawing import model_gui, model_manager, model_terms
from simple_drawing.model_manager import ModelManager, ModelManagerError, ModelOperationCancelled
from simple_drawing.model_terms import ModelTerms, ModelTermsError
from simple_drawing.profiles import GEMMA4_DIGEST, get_profile
from simple_drawing.storage import read_json, sha, write_json


class SetupFakeServer:
    instances = []

    def __init__(self, root, *, profile=None, management=False):
        self.root, self.profile, self.port = Path(root), profile, 11449
        self.shutdown_report = {}
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def stop(self):
        self.shutdown_report = {"process_stopped": True, "model_unloaded": True, "errors": []}

    def api(self, path, body=None, timeout=900):
        if path == "/api/tags":
            models = []
            base = self.root / "models/ollama/manifests/registry.ollama.ai/library"
            for manifest in base.glob("*/*"):
                models.append({"name": f"{manifest.parent.name}:{manifest.name}", "digest": sha(manifest)})
            return {"models": models}
        if path == "/api/show":
            return {"capabilities": ["completion", "vision", "thinking"]}
        return {}


class FirstModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = ModelManager(self.root)
        write_json(self.root / "実行設定.json", {"input_folder": "input", "llm_backend": "gemma",
                                              "ollama_profile": "gemma4-q8"})
        self.selection = (self.root / "実行設定.json").read_bytes()
        SetupFakeServer.instances = []
        self.addCleanup(patch.stopall)
        patch.object(model_manager.llm, "OllamaServer", SetupFakeServer).start()
        patch.object(model_manager.shutil, "disk_usage", return_value=Mock(free=20*1024**3)).start()

    def fake_assets(self):
        profile = get_profile("gemma4-q8")
        blob = b"synthetic Gemma4 test fixture, never a real model"
        blob_digest = hashlib.sha256(blob).hexdigest()
        path = self.root / f"models/ollama/blobs/sha256-{blob_digest}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        name, tag = profile.model.split(":")
        manifest = self.root / f"models/ollama/manifests/registry.ollama.ai/library/{name}/{tag}"
        write_json(manifest, {"config": {"digest": "sha256:"+blob_digest, "size": len(blob)}, "layers": []})
        return replace(profile, expected_digest=sha(manifest))

    def test_first_gemma4_download_records_verified_assets_without_selecting(self):
        # Override only the expected identity for a small, explicitly synthetic
        # model. A separate case tests the unchanged production digest gate.
        profile = self.fake_assets()
        manifest = next(self.root.glob("models/ollama/manifests/*/library/*/*"))
        saved_manifest = manifest.read_bytes()
        manifest.unlink()
        requests = []
        def handle(request):
            requests.append(request)
            manifest.write_bytes(saved_manifest)
            return httpx.Response(200, text='{"status":"pulling manifest"}\n{"status":"success"}\n')
        client = httpx.Client(transport=httpx.MockTransport(handle))
        original_profile = model_manager._profile
        with patch.object(model_manager, "_profile", side_effect=lambda value: profile if value in ("gemma4-q8", profile) else original_profile(value)), \
             patch.object(model_manager.httpx, "Client", return_value=client):
            record = self.service.install("gemma4-q8")
        self.assertEqual(json.loads(requests[0].content), {"model": "gemma4:e4b-it-q8_0", "stream": True})
        self.assertTrue(record["full_hash_verified"])
        self.assertIsNone(record["check"])
        self.assertEqual(read_json(self.service.registry_path)["models"]["gemma4-q8"]["digest"], sha(manifest))
        self.assertEqual((self.root / "実行設定.json").read_bytes(), self.selection)
        self.assertTrue(client.is_closed)

    def test_wrong_download_digest_is_rejected_without_registry_or_selection_change(self):
        self.assertEqual(get_profile().expected_digest, GEMMA4_DIGEST)
        self.assertEqual(GEMMA4_DIGEST, "3721c8128dc2e6cabed512039c22f63629c0ad6786b0fc852a9e4da47a6953f3")
        def handle(request):
            self.fake_assets()
            return httpx.Response(200, text='{"status":"success"}\n')
        client = httpx.Client(transport=httpx.MockTransport(handle))
        with patch.object(model_manager.httpx, "Client", return_value=client):
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma4-q8")
        self.assertEqual(caught.exception.code, "MODEL_CHANGED")
        self.assertFalse(self.service.registry_path.exists())
        self.assertEqual((self.root / "実行設定.json").read_bytes(), self.selection)
        self.assertTrue(SetupFakeServer.instances[-1].shutdown_report["process_stopped"])

    def test_gemma4_capacity_gate_uses_its_larger_size(self):
        with patch.object(model_manager.shutil, "disk_usage", return_value=Mock(free=5*1024**3)), \
             patch.object(model_manager.httpx, "Client") as client:
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma4-q8")
        self.assertEqual(caught.exception.code, "DISK_FULL")
        client.assert_not_called()
        self.assertFalse(self.service.registry_path.exists())

    def test_cancelled_gemma4_download_never_records_or_selects(self):
        cancel = threading.Event()
        def progress(event):
            if event["stage"] == "download":
                cancel.set()
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200,
            text='{"status":"pulling model","completed":1,"total":10}\n{"status":"success"}\n')))
        with patch.object(model_manager.httpx, "Client", return_value=client):
            with self.assertRaises(ModelOperationCancelled):
                self.service.install("gemma4-q8", cancel=cancel, progress=progress)
        self.assertFalse(self.service.registry_path.exists())
        self.assertEqual((self.root / "実行設定.json").read_bytes(), self.selection)
        self.assertTrue(client.is_closed)


class SyntheticTermsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.documents = []
        for index, definition in enumerate(model_terms.DOCUMENTS):
            path = self.root / definition["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"SYNTHETIC TEST DOCUMENT {index}. No human consent or real license agreement.", encoding="utf-8")
            self.documents.append({**definition, "sha256": sha(path)})
        self.addCleanup(patch.stopall)
        patch.object(model_terms, "DOCUMENTS", tuple(self.documents)).start()
        self.terms = ModelTerms(self.root)
        self.hashes = {item["path"]: item["sha256"] for item in self.documents}

    def test_no_acceptance_is_implied_by_viewing_or_requiring_terms(self):
        self.assertFalse(self.terms.status("gemma3-4b")["accepted"])
        with self.assertRaises(ModelTermsError) as caught:
            self.terms.require("gemma3-4b")
        self.assertEqual(caught.exception.code, "TERMS_NOT_ACCEPTED")
        self.assertFalse(self.terms.path.exists())
        self.assertTrue(self.terms.status("gemma4-q8")["accepted"])

    def test_synthetic_explicit_acceptance_records_document_identity_and_survives_move(self):
        record = self.terms.accept("gemma3-4b", self.hashes, accepted=True)
        self.assertEqual({item["path"]: item["sha256"] for item in record["documents"]}, self.hashes)
        self.assertEqual(record["acceptance_source"], "interactive-gui")
        self.assertTrue(record["explicit_acceptance"])
        self.assertEqual(self.terms.require("gemma3-4b"), record)
        self.assertNotIn(str(self.root), self.terms.path.read_text(encoding="utf-8"))

    def test_rejected_or_stale_dialog_does_not_save_acceptance(self):
        for accepted, hashes, expected in [(False, self.hashes, "TERMS_NOT_ACCEPTED"),
                                          (True, {}, "TERMS_CHANGED")]:
            with self.assertRaises(ModelTermsError) as caught:
                self.terms.accept("gemma3-4b", hashes, accepted=accepted)
            self.assertEqual(caught.exception.code, expected)
        self.assertFalse(self.terms.path.exists())

    def test_missing_or_tampered_document_blocks_acceptance(self):
        path = self.root / self.documents[0]["path"]
        path.write_text("tampered", encoding="utf-8")
        with self.assertRaises(ModelTermsError) as caught:
            self.terms.accept("gemma3-4b", self.hashes, accepted=True)
        self.assertEqual(caught.exception.code, "TERMS_UNAVAILABLE")
        self.assertFalse(self.terms.path.exists())

    def test_changed_terms_version_requires_new_explicit_acceptance(self):
        self.terms.accept("gemma3-4b", self.hashes, accepted=True)
        original = self.terms.path.read_bytes()
        with patch.object(model_terms, "TERMS_VERSION", "synthetic-next-version"):
            self.assertFalse(self.terms.status("gemma3-4b")["accepted"])
            with self.assertRaises(ModelTermsError):
                self.terms.require("gemma3-4b")
        self.assertEqual(self.terms.path.read_bytes(), original)

    def test_broken_acceptance_record_is_not_rewritten(self):
        self.terms.path.parent.mkdir(parents=True, exist_ok=True)
        self.terms.path.write_text("broken", encoding="utf-8")
        with self.assertRaises(ModelTermsError) as caught:
            self.terms.status("gemma3-4b")
        self.assertEqual(caught.exception.code, "INVALID_TERMS_RECORD")
        self.assertEqual(self.terms.path.read_text(), "broken")

    def test_unaccepted_gemma3_install_check_select_and_extract_start_no_server(self):
        service = ModelManager(self.root)
        with patch.object(model_manager.llm, "OllamaServer") as server:
            for operation in (service.install, service.check, service.select, service.ensure_usable):
                with self.assertRaises(ModelManagerError) as caught:
                    operation("gemma3-4b")
                self.assertEqual(caught.exception.code, "TERMS_NOT_ACCEPTED")
        server.assert_not_called()
        self.assertFalse(service.registry_path.exists())
        self.assertFalse(self.terms.path.exists())
        self.assertFalse((self.root / "実行設定.json").exists())


class TermsDispatchTests(unittest.TestCase):
    def setUp(self):
        self.manager = Mock()
        self.status = {"required": True, "accepted": False, "terms_version": "synthetic",
                       "documents": [{"path": "synthetic.txt", "sha256": "0"*64, "text": "test"}]}
        self.manager.terms_status.return_value = self.status

    def test_cancelled_dialog_never_records_consent(self):
        with patch.object(model_gui, "present_model_terms", return_value=False):
            self.assertFalse(model_gui.confirm_model_terms(Mock(), self.manager, "gemma3-4b"))
        self.manager.accept_terms.assert_not_called()
        self.manager.install.assert_not_called()

    def test_explicit_synthetic_dialog_result_passes_only_displayed_hashes(self):
        with patch.object(model_gui, "present_model_terms", return_value=True):
            self.assertTrue(model_gui.confirm_model_terms(Mock(), self.manager, "gemma3-4b"))
        self.manager.accept_terms.assert_called_once_with("gemma3-4b", {"synthetic.txt": "0"*64}, accepted=True)

    def test_already_accepted_or_gemma4_does_not_prompt_again(self):
        for status in ({"required": True, "accepted": True}, {"required": False, "accepted": True}):
            self.manager.terms_status.return_value = status
            with patch.object(model_gui, "present_model_terms") as dialog:
                self.assertTrue(model_gui.confirm_model_terms(Mock(), self.manager, "gemma4-q8"))
            dialog.assert_not_called()
        self.manager.accept_terms.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Manager fallback integration with tiny fixtures; no real network or models."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from filelock import FileLock
import httpx

from simple_drawing import model_manager as manager
from simple_drawing import profiles
from simple_drawing.model_manager import ModelManager, ModelManagerError, ModelOperationCancelled
from simple_drawing.model_terms import ModelTermsError
from simple_drawing.profiles import get_profile
from simple_drawing.storage import read_json, sha, write_json


class FallbackFakeServer:
    instances = []
    capabilities = ["completion", "vision"]
    wrong_api_digest = False
    cleanup_error = False

    def __init__(self, root, **kwargs):
        self.root = Path(root)
        self.port = 11449
        self.shutdown_report = {}
        self.api_calls = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def stop(self):
        self.shutdown_report = {"process_stopped": not self.cleanup_error,
                                "model_unloaded": not self.cleanup_error,
                                "errors": ["synthetic cleanup failure"] if self.cleanup_error else []}

    def api(self, path, body=None, timeout=900):
        self.api_calls.append(path)
        if path == "/api/tags":
            models = []
            base = self.root / "models/ollama/manifests/registry.ollama.ai/library"
            for manifest in base.glob("*/*"):
                identity = sha(manifest)
                if self.wrong_api_digest and manifest.parent.name == "gemma3":
                    identity = "0" * 64
                models.append({"name": f"{manifest.parent.name}:{manifest.name}", "digest": identity})
            return {"models": models}
        if path == "/api/show":
            return {"capabilities": self.capabilities}
        raise AssertionError(f"Unexpected API call: {path}")


class ModelDownloadFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = ModelManager(self.root)
        write_json(self.root / "実行設定.json", {
            "input_folder": "synthetic input with spaces", "llm_backend": "gemma",
            "ollama_profile": "gemma4-q8", "saluki_batch_size": 512})
        self.active_manifest, self.active_blob = self.assets("gemma4-q8", blob=b"synthetic previous active model")
        write_json(self.service.registry_path, {"version": 1, "models": {"gemma4-q8": {
            "profile_id": "gemma4-q8", "model_tag": get_profile("gemma4-q8").model,
            "digest": sha(self.active_manifest), "full_hash_verified": True,
            "check": {"status": "passed", "fingerprint": "synthetic-prior-check"}}}})
        self.protected = {path: path.read_bytes() for path in (
            self.root / "実行設定.json", self.service.registry_path, self.active_manifest, self.active_blob)}
        FallbackFakeServer.instances = []
        FallbackFakeServer.capabilities = ["completion", "vision"]
        FallbackFakeServer.wrong_api_digest = False
        FallbackFakeServer.cleanup_error = False
        self.addCleanup(patch.stopall)
        patch.object(manager.llm, "OllamaServer", FallbackFakeServer).start()
        patch.object(manager, "require_model_terms", return_value=None).start()
        patch.object(manager.shutil, "disk_usage", return_value=Mock(free=20 * 1024 ** 3)).start()
        patch.object(socket.socket, "connect", side_effect=AssertionError("Real network is forbidden")).start()

    def assets(self, profile_id="gemma3-4b", *, blob=b"synthetic tiny downloaded model"):
        profile = get_profile(profile_id)
        blob_digest = hashlib.sha256(blob).hexdigest()
        blob_path = self.root / f"models/ollama/blobs/sha256-{blob_digest}"
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(blob)
        name, tag = profile.model.split(":", 1)
        manifest = self.root / f"models/ollama/manifests/registry.ollama.ai/library/{name}/{tag}"
        write_json(manifest, {"config": {"digest": "sha256:" + blob_digest, "size": len(blob)}, "layers": []})
        return manifest, blob_path

    def assert_protected(self, *, registry=True):
        for path, previous in self.protected.items():
            if not registry and path == self.service.registry_path:
                continue
            self.assertEqual(path.read_bytes(), previous, str(path))
        with FileLock(str(self.root / "data/operation.lock"), timeout=0):
            pass

    @contextmanager
    def primary(self, mode="stream_error"):
        requests = []
        def handle(request):
            requests.append(request)
            if mode == "network_error":
                raise httpx.ConnectError("SYNTHETIC Ollama proxy connection error", request=request)
            if mode == "http_error":
                return httpx.Response(502, text="SYNTHETIC Ollama proxy HTTP 502")
            if mode == "success":
                self.assets()
                content = '{"status":"pulling model"}\n{"status":"success"}\n'
            elif mode == "incomplete":
                content = '{"status":"pulling model","completed":10,"total":10}\n'
            elif mode == "cancel":
                content = '{"status":"pulling model","completed":1,"total":10}\n{"error":"SYNTHETIC Ollama network failure"}\n'
            else:
                content = '{"status":"pulling manifest"}\n{"error":"SYNTHETIC Ollama proxyconnect tcp: EOF"}\n'
            return httpx.Response(200, text=content)
        client = httpx.Client(transport=httpx.MockTransport(handle), trust_env=False)
        with patch.object(manager.httpx, "Client", return_value=client):
            yield client, requests
        self.assertTrue(client.is_closed)

    @contextmanager
    def successful_fallback(self, *, client=None):
        def fallback(root, profile, *, cancel, progress):
            self.assertEqual(Path(root), self.root)
            self.assertEqual(profile.id, "gemma3-4b")
            if client is not None:
                self.assertTrue(client.is_closed, "Primary request must close before HF transfer")
            self.assets()
            return {"source": "synthetic huggingface fixture"}
        with patch.object(manager.model_download, "install_from_huggingface", side_effect=fallback) as helper:
            yield helper

    def assert_success(self, result):
        self.assertEqual(result["profile_id"], "gemma3-4b")
        self.assertTrue(result["full_hash_verified"])
        self.assertIsNone(result["check"], "Installing does not certify extraction readiness")
        saved = read_json(self.service.registry_path)
        self.assertEqual(saved["models"]["gemma3-4b"]["digest"], result["digest"])
        self.assertEqual(saved["models"]["gemma4-q8"], json.loads(self.protected[self.service.registry_path])["models"]["gemma4-q8"])
        self.assert_protected(registry=False)
        self.assertIn("/api/show", FallbackFakeServer.instances[-1].api_calls)
        self.assertTrue(FallbackFakeServer.instances[-1].shutdown_report["process_stopped"])

    def test_pull_http_failure_uses_hf_once_and_commits_after_existing_gates(self):
        self.verify_fallback_success("http_error")

    def test_pull_stream_network_failure_uses_hf_once_and_commits_after_existing_gates(self):
        self.verify_fallback_success("stream_error")

    def test_pull_transport_failure_uses_hf_once_and_commits_after_existing_gates(self):
        self.verify_fallback_success("network_error")

    def test_pull_missing_terminal_success_uses_hf_once_and_commits_after_existing_gates(self):
        self.verify_fallback_success("incomplete")

    def verify_fallback_success(self, mode):
        events = []
        with self.primary(mode) as (client, requests), self.successful_fallback(client=client) as helper, \
             patch.object(self.service, "_asset_identity", wraps=self.service._asset_identity) as identity:
            result = self.service.install("gemma3-4b", progress=events.append)
        helper.assert_called_once()
        self.assertEqual(len(requests), 1)
        self.assertEqual(json.loads(requests[0].content), {"model": "gemma3:4b", "stream": True})
        identity.assert_called_once()
        self.assertTrue(identity.call_args.kwargs["full_hash"])
        self.assertTrue(any(event["stage"] == "complete" for event in events))
        self.assert_success(result)

    def test_primary_success_does_not_attempt_hf(self):
        with self.primary("success"), patch.object(manager.model_download, "install_from_huggingface") as helper:
            result = self.service.install("gemma3-4b")
        helper.assert_not_called()
        self.assert_success(result)

    def test_existing_installed_model_skips_both_download_routes(self):
        self.assets()
        with patch.object(manager.httpx, "Client") as primary, \
             patch.object(manager.model_download, "install_from_huggingface") as helper:
            result = self.service.install("gemma3-4b")
        primary.assert_not_called()
        helper.assert_not_called()
        self.assert_success(result)

    def test_both_sources_fail_without_committing_or_selecting(self):
        error = manager.model_download.HuggingFaceDownloadError("SYNTHETIC HF unavailable")
        events = []
        with self.primary(), patch.object(manager.model_download, "install_from_huggingface", side_effect=error) as helper:
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b", progress=events.append)
        self.assertEqual(caught.exception.code, "MODEL_DOWNLOAD_FAILED")
        self.assertIn("SYNTHETIC Ollama proxyconnect tcp: EOF", str(caught.exception))
        self.assertIn("SYNTHETIC HF unavailable", str(caught.exception))
        helper.assert_called_once()
        self.assertFalse(any(event["stage"] == "complete" for event in events))
        self.assert_protected()

    def test_hf_wrong_blob_hash_does_not_commit_or_replace_active_model(self):
        def bad_assets(*args, **kwargs):
            _, blob = self.assets(blob=b"correct synthetic model")
            blob.write_bytes(b"invalid synthetic model")
        with self.primary(), patch.object(manager.model_download, "install_from_huggingface", side_effect=bad_assets):
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "CORRUPT_MODEL")
        self.assert_protected()

    def test_hf_api_manifest_mismatch_does_not_commit_or_replace_active_model(self):
        FallbackFakeServer.wrong_api_digest = True
        with self.primary(), self.successful_fallback():
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "MODEL_CHANGED")
        self.assert_protected()

    def test_hf_missing_vision_capability_does_not_commit_or_replace_active_model(self):
        FallbackFakeServer.capabilities = ["completion"]
        with self.primary(), self.successful_fallback():
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "UNSUPPORTED_MODEL")
        self.assert_protected()

    def test_hf_success_with_failed_server_cleanup_does_not_commit(self):
        FallbackFakeServer.cleanup_error = True
        with self.primary(), self.successful_fallback():
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "CLEANUP_FAILED")
        self.assert_protected()

    def test_pre_cancel_does_not_start_either_download_or_server(self):
        cancel = threading.Event()
        cancel.set()
        with patch.object(manager.httpx, "Client") as primary, \
             patch.object(manager.model_download, "install_from_huggingface") as helper:
            with self.assertRaises(ModelOperationCancelled):
                self.service.install("gemma3-4b", cancel=cancel)
        primary.assert_not_called()
        helper.assert_not_called()
        self.assertFalse(FallbackFakeServer.instances)
        self.assert_protected()

    def test_cancel_during_primary_stream_does_not_attempt_hf(self):
        cancel = threading.Event()
        def progress(event):
            if event["stage"] == "download":
                cancel.set()
        with self.primary("cancel"), patch.object(manager.model_download, "install_from_huggingface") as helper:
            with self.assertRaises(ModelOperationCancelled):
                self.service.install("gemma3-4b", cancel=cancel, progress=progress)
        helper.assert_not_called()
        self.assert_protected()

    def test_cancel_during_fallback_stays_cancelled_and_does_not_commit(self):
        cancel = threading.Event()
        def cancelled_fallback(*args, **kwargs):
            cancel.set()
            raise manager.model_download.HuggingFaceDownloadError("SYNTHETIC transfer interrupted")
        with self.primary(), patch.object(manager.model_download, "install_from_huggingface", side_effect=cancelled_fallback) as helper:
            with self.assertRaises(ModelOperationCancelled):
                self.service.install("gemma3-4b", cancel=cancel)
        helper.assert_called_once()
        self.assert_protected()

    def test_cancellation_after_hf_transfer_prevents_identity_and_registry_commit(self):
        cancel = threading.Event()
        def completed_then_cancelled(*args, **kwargs):
            self.assets()
            cancel.set()
            return {"source": "synthetic huggingface fixture"}
        with self.primary(), patch.object(manager.model_download, "install_from_huggingface", side_effect=completed_then_cancelled):
            with self.assertRaises(ModelOperationCancelled):
                self.service.install("gemma3-4b", cancel=cancel)
        self.assert_protected()

    def test_primary_integrity_failure_is_not_repaired_by_hf(self):
        original_identity = self.service._asset_identity
        def corrupt_before_hash(*args, **kwargs):
            manifest = self.root / "models/ollama/manifests/registry.ollama.ai/library/gemma3/4b"
            data = read_json(manifest)
            blob = self.root / "models/ollama/blobs" / data["config"]["digest"].replace(":", "-")
            blob.write_bytes(b"x" * blob.stat().st_size)
            return original_identity(*args, **kwargs)
        with self.primary("success"), patch.object(self.service, "_asset_identity", side_effect=corrupt_before_hash), \
             patch.object(manager.model_download, "install_from_huggingface") as helper:
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "CORRUPT_MODEL")
        helper.assert_not_called()
        self.assert_protected()

    def test_busy_operation_does_not_start_download_or_server(self):
        with FileLock(str(self.root / "data/operation.lock"), timeout=0), \
             patch.object(manager.httpx, "Client") as primary, \
             patch.object(manager.model_download, "install_from_huggingface") as helper:
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "BUSY")
        primary.assert_not_called()
        helper.assert_not_called()
        self.assertFalse(FallbackFakeServer.instances)
        self.assert_protected()

    def test_disk_full_does_not_attempt_either_download(self):
        with patch.object(manager.shutil, "disk_usage", return_value=Mock(free=0)), \
             patch.object(manager.httpx, "Client") as primary, \
             patch.object(manager.model_download, "install_from_huggingface") as helper:
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "DISK_FULL")
        primary.assert_not_called()
        helper.assert_not_called()
        self.assert_protected()

    def test_terms_refusal_does_not_attempt_download_or_start_server(self):
        error = ModelTermsError("TERMS_NOT_ACCEPTED", "Synthetic terms are not accepted")
        with patch.object(manager, "require_model_terms", side_effect=error), \
             patch.object(manager.httpx, "Client") as primary, \
             patch.object(manager.model_download, "install_from_huggingface") as helper:
            with self.assertRaises(ModelManagerError) as caught:
                self.service.install("gemma3-4b")
        self.assertEqual(caught.exception.code, "TERMS_NOT_ACCEPTED")
        primary.assert_not_called()
        helper.assert_not_called()
        self.assertFalse(FallbackFakeServer.instances)
        self.assert_protected()


class Gemma4DeclaredVariantTests(unittest.TestCase):
    """Exercise declared alternate identities using a tiny production-shaped catalog."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = ModelManager(self.root)
        write_json(self.root / "実行設定.json", {"input_folder": "synthetic input", "llm_backend": "saluki"})
        self.selection = (self.root / "実行設定.json").read_bytes()
        candidate = Path(__file__).resolve().parents[2]
        catalog = read_json(candidate / "assets/huggingface-ollama.json")
        definition = deepcopy(catalog["profiles"]["gemma4-q8"])
        production_profile = get_profile("gemma4-q8")
        self.assertEqual(definition["manifest_sha256"], production_profile.fallback_digest)
        self.assertEqual(definition["asset_variant"], "huggingface")
        self.assertNotEqual(production_profile.expected_digest, production_profile.fallback_digest)
        manifest = json.loads(definition["manifest_text"])
        old_blobs = definition["blobs"]
        tiny_blobs = {}
        self.remote_payloads = {}
        self.inline_payloads = {}
        for index, ref in enumerate([manifest["config"], *manifest["layers"]]):
            asset = deepcopy(old_blobs[ref["digest"]])
            if "text" in asset:
                payload = asset["text"].encode("utf-8")
            else:
                payload = ("SYNTHETIC fixed HF variant asset " + str(index)).encode()
                self.remote_payloads[asset["url"]] = payload
            digest = hashlib.sha256(payload).hexdigest()
            asset.update(bytes=len(payload), sha256=digest)
            ref.update(digest="sha256:" + digest, size=len(payload))
            tiny_blobs[ref["digest"]] = asset
            self.inline_payloads[digest] = payload
        hf_bytes = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
        self.hf_digest = hashlib.sha256(hf_bytes).hexdigest()
        definition.update(manifest_text=hf_bytes.decode(), manifest_sha256=self.hf_digest, blobs=tiny_blobs)
        # A separate primary fixture represents the old declared identity. It is
        # deliberately different from the exact alternate HF fixture above.
        primary_manifest = deepcopy(manifest)
        primary_manifest["synthetic_primary_identity"] = True
        self.primary_bytes = json.dumps(primary_manifest, ensure_ascii=False, separators=(",", ":")).encode()
        self.primary_digest = hashlib.sha256(self.primary_bytes).hexdigest()
        self.hf_bytes = hf_bytes
        definition["origin_ollama_manifest_sha256"] = self.primary_digest
        write_json(self.root / "assets/huggingface-ollama.json", {
            "schema_version": catalog["schema_version"], "profiles": {"gemma4-q8": definition}})
        self.profile = replace(production_profile, expected_digest=self.primary_digest,
                               fallback_digest=self.hf_digest, download_size_bytes=100_000)
        registry = {profile.id: profile for profile in profiles.list_profiles()}
        registry[self.profile.id] = self.profile
        self.addCleanup(patch.stopall)
        patch.object(profiles, "_PROFILES", registry).start()
        patch.object(manager, "require_model_terms", return_value=None).start()
        patch.object(manager.shutil, "disk_usage", return_value=Mock(free=20 * 1024 ** 3)).start()
        patch.object(manager.llm, "OllamaServer", FallbackFakeServer).start()
        patch.object(socket.socket, "connect", side_effect=AssertionError("Real network is forbidden")).start()
        FallbackFakeServer.instances = []
        FallbackFakeServer.capabilities = ["completion", "vision", "thinking"]
        FallbackFakeServer.wrong_api_digest = False
        FallbackFakeServer.cleanup_error = False
        executable = self.root / "ollama/ollama.exe"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"SYNTHETIC runtime, never executed")
        patch.object(manager.llm, "EXPECTED_EXE", sha(executable)).start()
        patch.object(manager.llm, "EXPECTED_MANIFEST", self.primary_digest).start()
        name, tag = self.profile.model.split(":", 1)
        self.manifest_path = self.root / f"models/ollama/manifests/registry.ollama.ai/library/{name}/{tag}"

    def write_variant(self, encoded):
        for digest, payload in self.inline_payloads.items():
            path = self.root / "models/ollama/blobs" / ("sha256-" + digest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_bytes(encoded)

    def test_declared_hf_gemma4_identity_uses_real_helper_and_runtime_contract(self):
        requests = []
        def primary(request):
            requests.append(request)
            return httpx.Response(502, text="SYNTHETIC Ollama proxy failure")
        client = httpx.Client(transport=httpx.MockTransport(primary), trust_env=False)
        def tiny_download(asset, partial, cancel=None, progress=None):
            self.assertTrue(client.is_closed)
            Path(partial).write_bytes(self.remote_payloads[asset["url"]])
        real_helper = manager.model_download.install_from_huggingface
        with patch.object(manager.httpx, "Client", return_value=client), \
             patch.object(manager.model_download, "_curl_download", side_effect=tiny_download) as transfer, \
             patch.object(manager.model_download, "install_from_huggingface", wraps=real_helper) as helper, \
             patch.object(self.service, "_asset_identity", wraps=self.service._asset_identity) as identity:
            record = self.service.install("gemma4-q8")
        helper.assert_called_once()
        self.assertEqual(len(requests), 1)
        self.assertEqual(transfer.call_count, len(self.remote_payloads))
        self.assertTrue(identity.call_args.kwargs["full_hash"])
        self.assertEqual(record["digest"], self.hf_digest)
        self.assertEqual(record["asset_variant"], "huggingface")
        self.assertTrue(record["full_hash_verified"])
        self.assertIsNone(record["check"])
        self.assertEqual(record["acquisition"]["source"], "huggingface")
        contract = manager.llm.model_contract(self.root, self.profile)
        self.assertTrue(contract["ready"], contract["issues"])
        self.assertEqual(contract["expected_manifest_sha256"], self.hf_digest)
        self.assertEqual(contract["manifest_sha256"], self.hf_digest)
        self.assertNotEqual(contract["expected_manifest_sha256"], self.primary_digest)
        cards = self.service.list_profiles()
        card = next(card for card in cards if card["profile_id"] == "gemma4-q8")
        self.assertEqual(card["asset_variant"], "huggingface")
        self.assertIn("Hugging Face", card["label"])
        self.assertFalse(card["checked"])
        self.assertEqual((self.root / "実行設定.json").read_bytes(), self.selection)

    def test_undeclared_third_gemma4_manifest_is_refused_by_identity_and_runtime(self):
        third = json.loads(self.hf_bytes)
        third["synthetic_undeclared_identity"] = True
        encoded = json.dumps(third, separators=(",", ":")).encode()
        self.write_variant(encoded)
        with self.assertRaises(ModelManagerError) as caught:
            self.service._asset_identity(self.profile, full_hash=True)
        self.assertEqual(caught.exception.code, "MODEL_CHANGED")
        contract = manager.llm.model_contract(self.root, self.profile)
        self.assertFalse(contract["ready"])
        self.assertEqual(contract["expected_manifest_sha256"], self.primary_digest)
        self.assertNotEqual(contract["manifest_sha256"], self.hf_digest)
        self.assertFalse(self.service.registry_path.exists())
        self.assertEqual((self.root / "実行設定.json").read_bytes(), self.selection)

    def test_primary_passed_check_cannot_validate_declared_hf_variant(self):
        self.write_variant(self.primary_bytes)
        primary_identity = self.service._asset_identity(self.profile, full_hash=True)
        primary_fingerprint = manager.check_fingerprint(self.profile, self.primary_digest)
        primary_record = self.service._record_identity(primary_identity, {"version": 1, "models": {}},
            check={"status": "passed", "fingerprint": primary_fingerprint})
        saved = {"version": 1, "models": {self.profile.id: primary_record}}
        write_json(self.service.registry_path, saved)
        self.write_variant(self.hf_bytes)
        hf_identity = self.service._asset_identity(self.profile, full_hash=True)
        self.assertNotEqual(primary_identity["digest"], hf_identity["digest"])
        self.assertNotEqual(primary_fingerprint, manager.check_fingerprint(self.profile, self.hf_digest))
        with self.assertRaises(ModelManagerError) as caught:
            self.service.ensure_usable(self.profile)
        self.assertEqual(caught.exception.code, "MODEL_CHANGED")
        replacement = self.service._record_identity(hf_identity, saved)
        self.assertIsNone(replacement["check"], "Primary certification is invalidated for the HF assets")
        saved["models"][self.profile.id] = {**replacement, "check": primary_record["check"]}
        write_json(self.service.registry_path, saved)
        with self.assertRaises(ModelManagerError) as caught:
            self.service.ensure_usable(self.profile)
        self.assertEqual(caught.exception.code, "UNCHECKED_MODEL")
        self.assertEqual((self.root / "実行設定.json").read_bytes(), self.selection)

    def test_declared_hf_install_cannot_be_selected_before_its_own_check(self):
        self.write_variant(self.hf_bytes)
        record = self.service.install("gemma4-q8")
        self.assertEqual(record["asset_variant"], "huggingface")
        with self.assertRaises(ModelManagerError) as caught:
            self.service.select("gemma4-q8")
        self.assertEqual(caught.exception.code, "UNCHECKED_MODEL")
        self.assertEqual((self.root / "実行設定.json").read_bytes(), self.selection)

    def test_historical_check_remains_for_primary_and_cannot_certify_hf_assets(self):
        self.write_variant(self.primary_bytes)
        report_path = "data/synthetic-original-model-report.json"
        write_json(self.root / report_path, {"status": "passed", "digest": self.primary_digest,
                                           "synthetic_only": "legacy report, no real inference"})
        original = self.service.register_historical_check("gemma4-q8", report_path=report_path)
        self.assertEqual(original["digest"], self.primary_digest)
        self.assertEqual(original["check"]["kind"], "historical-regression")
        self.assertEqual(original["check"]["status"], "passed")
        registry_before = self.service.registry_path.read_bytes()
        self.write_variant(self.hf_bytes)
        with self.assertRaises(ModelManagerError) as caught:
            self.service.register_historical_check("gemma4-q8", report_path=report_path)
        self.assertIn(caught.exception.code, ("INVALID_HISTORICAL_CHECK", "CHECK_FAILED"))
        self.assertEqual(self.service.registry_path.read_bytes(), registry_before,
                         "An old report must not create a passed check for the HF variant")
        saved = read_json(self.service.registry_path)["models"]["gemma4-q8"]
        self.assertEqual(saved["digest"], self.primary_digest)
        self.assertNotEqual(saved["check"]["fingerprint"], manager.check_fingerprint(self.profile, self.hf_digest))
        self.assertEqual((self.root / "実行設定.json").read_bytes(), self.selection)


class BootstrapFallbackStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_json(self.root / "実行設定.json", {"input_folder": "synthetic input", "llm_backend": "saluki"})
        write_json(self.root / "data/model-registry.json", {"version": 1, "models": {}})
        self.selection = (self.root / "実行設定.json").read_bytes()
        self.registry = (self.root / "data/model-registry.json").read_bytes()
        script = Path(__file__).resolve().parents[2] / "scripts/bootstrap.py"
        spec = importlib.util.spec_from_file_location("fallback_test_bootstrap", script)
        self.bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bootstrap)
        self.addCleanup(patch.stopall)
        FallbackFakeServer.instances = []
        FallbackFakeServer.cleanup_error = False
        patch.object(manager.llm, "OllamaServer", FallbackFakeServer).start()
        patch.object(manager, "require_model_terms", return_value=None).start()
        patch.object(manager.shutil, "disk_usage", return_value=Mock(free=20 * 1024 ** 3)).start()
        patch.object(socket.socket, "connect", side_effect=AssertionError("Real network is forbidden")).start()

    @contextmanager
    def run_context(self, *, offline=False):
        argv = ["synthetic-bootstrap", "--root", str(self.root)] + (["--offline"] if offline else [])
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(redirect_stdout(io.StringIO()))
            for name in ("bundled_core_manifest", "check_capacity", "setup_dependencies", "setup_bundled_core",
                         "setup_ocr", "setup_ollama", "check_runtime"):
                stack.enter_context(patch.object(self.bootstrap, name))
            yield

    def assert_preserved(self):
        self.assertEqual((self.root / "実行設定.json").read_bytes(), self.selection)
        self.assertEqual((self.root / "data/model-registry.json").read_bytes(), self.registry)
        with FileLock(str(self.root / "data/operation.lock"), timeout=0):
            pass

    def test_double_download_failure_marks_setup_failed_and_never_checks_model(self):
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200,
            text='{"status":"pulling manifest"}\n{"error":"SYNTHETIC Ollama proxy failure"}\n')))
        hf_error = manager.model_download.HuggingFaceDownloadError("SYNTHETIC Hugging Face unavailable")
        with self.run_context(), patch.object(manager.httpx, "Client", return_value=client), \
             patch.object(manager.model_download, "install_from_huggingface", side_effect=hf_error) as helper, \
             patch.object(ModelManager, "check") as check, patch.object(ModelManager, "ensure_usable") as usable:
            with self.assertRaises(ModelManagerError) as caught:
                self.bootstrap.main()
        self.assertEqual(caught.exception.code, "MODEL_DOWNLOAD_FAILED")
        helper.assert_called_once()
        check.assert_not_called()
        usable.assert_not_called()
        marker = read_json(self.root / "data/setup-complete.json")
        self.assertEqual(marker["status"], "failed")
        self.assertEqual(marker["failed_stage"], "model")
        self.assertEqual(marker["error_code"], "MODEL_DOWNLOAD_FAILED")
        self.assertIn("SYNTHETIC Ollama proxy failure", marker["reason"])
        self.assertIn("SYNTHETIC Hugging Face unavailable", marker["reason"])
        self.assertTrue(client.is_closed)
        self.assert_preserved()

    def test_failed_model_stage_does_not_overwrite_another_run_marker(self):
        another_run = {"status": "complete", "app_version": "synthetic-other", "setup_id": "other-run"}
        def fail_after_another_run(*args):
            write_json(self.root / "data/setup-complete.json", another_run)
            raise ModelManagerError("MODEL_DOWNLOAD_FAILED", "Synthetic failure in old run")
        with self.run_context(), patch.object(self.bootstrap, "setup_model", side_effect=fail_after_another_run):
            with self.assertRaises(ModelManagerError):
                self.bootstrap.main()
        self.assertEqual(read_json(self.root / "data/setup-complete.json"), another_run)
        self.assert_preserved()

    def test_busy_model_stage_is_not_recorded_as_completed_or_failed(self):
        error = ModelManagerError("BUSY", "Synthetic competing model operation")
        with self.run_context(), patch.object(self.bootstrap, "setup_model", side_effect=error):
            with self.assertRaises(ModelManagerError):
                self.bootstrap.main()
        marker = read_json(self.root / "data/setup-complete.json")
        self.assertEqual(marker["status"], "in_progress")
        self.assertNotIn("failed_stage", marker)
        self.assert_preserved()

    def test_offline_missing_model_does_not_attempt_either_download(self):
        with self.run_context(offline=True), patch.object(manager.httpx, "Client") as primary, \
             patch.object(manager.model_download, "install_from_huggingface") as helper:
            with self.assertRaises(ModelManagerError) as caught:
                self.bootstrap.main()
        self.assertEqual(caught.exception.code, "NOT_INSTALLED")
        primary.assert_not_called()
        helper.assert_not_called()
        self.assertFalse(FallbackFakeServer.instances)
        marker = read_json(self.root / "data/setup-complete.json")
        self.assertEqual(marker["status"], "failed")
        self.assertEqual(marker["error_code"], "NOT_INSTALLED")
        self.assert_preserved()

    def test_keyboard_interrupt_marks_model_stage_cancelled(self):
        with self.run_context(), patch.object(self.bootstrap, "setup_model", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.bootstrap.main()
        marker = read_json(self.root / "data/setup-complete.json")
        self.assertEqual(marker["status"], "cancelled")
        self.assertEqual(marker["failed_stage"], "model")
        self.assert_preserved()

    def test_model_cancel_marks_model_stage_cancelled(self):
        with self.run_context(), patch.object(self.bootstrap, "setup_model", side_effect=ModelOperationCancelled()):
            with self.assertRaises(ModelOperationCancelled):
                self.bootstrap.main()
        marker = read_json(self.root / "data/setup-complete.json")
        self.assertEqual(marker["status"], "cancelled")
        self.assertEqual(marker["error_code"], "CANCELLED")
        self.assert_preserved()


if __name__ == "__main__":
    unittest.main()

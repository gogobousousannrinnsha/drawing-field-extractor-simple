import base64
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, call, patch

from simple_drawing import llm
from simple_drawing.profiles import DEFAULT_PROFILE_ID, GEMMA3_REQUEST_ADAPTER, get_profile, list_profiles
from simple_drawing.settings import ExecutionSettings
from simple_drawing.storage import read_json, write_json
from simple_drawing.workflow import resume_run


TEST_ROOT = Path(__file__).resolve().parent


class ModelProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.root = Path(self.temp.name)
        self.exe = self.root / "ollama/ollama.exe"
        self.exe.parent.mkdir()
        self.exe.write_bytes(b"local frozen runtime")
        self.runtime_hash = hashlib.sha256(self.exe.read_bytes()).hexdigest()
        self.page = {"number": 1, "width_mm": 297., "height_mm": 210.,
                     "width_px": 2970, "height_px": 2100}
        self.regions = [{"id": "p1:r1", "text": " 0007 mm ",
                         "bbox_mm": {"x": 1., "y": 2., "width": 3., "height": 4.}}]
        self.settings = {"item": "番号", "region_hint_mm": None, "take": "one", "instruction": ""}

    def tearDown(self):
        self.temp.cleanup()

    def install_fixture(self, profile_id):
        profile = get_profile(profile_id)
        blob = b"model content " + profile.model.encode()
        ident = hashlib.sha256(blob).hexdigest()
        path = self.root / ("models/ollama/blobs/sha256-" + ident)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        manifest = {"config": {"digest": "sha256:" + ident, "size": len(blob)}, "layers": []}
        manifest_path = self.root / llm.manifest_relative(profile)
        write_json(manifest_path, manifest)
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return profile, digest, path

    def request(self, profile_id, *, settings=None, regions=None, results=None):
        settings = self.settings if settings is None else settings
        regions = self.regions if regions is None else regions
        results = [{"value": " 0007 mm ", "source_ids": ["p1:r1"]}] if results is None else results
        server = Mock(spec=llm.OllamaServer)
        server.profile = get_profile(profile_id)
        server.root = self.root
        server.contract = {}
        server.loaded_model = None
        server.api.side_effect = [
            {"done": True, "done_reason": "stop", "message": {"content": json.dumps({
                "status": "found", "results": results,
                "note": ""})}}, {"models": []},
        ]
        image = self.root / "page.png"
        image.write_bytes(b"full original image bytes")
        with patch("simple_drawing.llm._gpu_sample", return_value={}):
            record = llm.extract_page(server, image, self.page, regions, settings)
        return server, server.api.call_args_list[0].args[1], record

    def test_catalog_is_fixed_and_profile_is_immutable(self):
        self.assertEqual([p.id for p in list_profiles()], ["gemma4-q8", "gemma3-4b"])
        profile = get_profile()
        self.assertEqual(profile.id, DEFAULT_PROFILE_ID)
        with self.assertRaises(FrozenInstanceError):
            profile.model = "another"
        with self.assertRaises(TypeError):
            profile.options["num_ctx"] = 1
        snapshot = profile.snapshot()
        snapshot["options"]["num_ctx"] = 1
        self.assertEqual(profile.options["num_ctx"], 32768)
        with self.assertRaises(ValueError):
            get_profile("../unknown")

    def test_legacy_settings_and_saluki_backend_are_preserved(self):
        self.assertEqual(ExecutionSettings.model_validate({"input_folder": "input"}).ollama_profile,
                         DEFAULT_PROFILE_ID)
        settings = ExecutionSettings.model_validate({"llm_backend": "saluki", "saluki_thinking": True,
                                                       "ollama_profile": "gemma3-4b"})
        self.assertEqual(settings.llm_backend, "saluki")
        self.assertTrue(settings.saluki_thinking)
        with self.assertRaises(ValueError):
            ExecutionSettings.model_validate({"ollama_profile": "not-supported"})

    def test_gemma4_entire_request_matches_frozen_baseline(self):
        server, request, record = self.request(DEFAULT_PROFILE_ID)
        payload = {"page": self.page, "ocr_regions": self.regions, "settings": self.settings}
        expected = {"model": "gemma4:e4b-it-q8_0", "think": True, "stream": False,
                    "format": llm.SCHEMA, "keep_alive": "30m", "truncate": False, "shift": False,
                    "options": {"temperature": 0, "top_p": 1, "top_k": 64, "repeat_penalty": 1,
                                "presence_penalty": 0, "seed": 42, "num_ctx": 32768,
                                "num_predict": 16384, "draft_num_predict": 0},
                    "messages": [{"role": "system", "content": llm.SYSTEM},
                                 {"role": "user", "content": "次の設定で画像の項目を抽出してください。\n" +
                                  json.dumps(payload, ensure_ascii=False),
                                  "images": [base64.b64encode(b"full original image bytes").decode("ascii")]}]}
        self.assertEqual(json.dumps(request, ensure_ascii=False), json.dumps(expected, ensure_ascii=False))
        self.assertEqual(hashlib.sha256(llm.SYSTEM.encode()).hexdigest(),
                         "0db3a9092f42d19d403c1801528cfede36e064731b346fab6eea80879dc4345c")
        schema_bytes = json.dumps(llm.SCHEMA, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(schema_bytes).hexdigest(),
                         "6b9b5a1c4a375c774212bf35ed936a1bb960b3817ee39d8f6a8f08c6bb942d6d")
        self.assertEqual(record["parsed"]["results"][0]["value"], " 0007 mm ")
        self.assertEqual(server.loaded_model, "gemma4:e4b-it-q8_0")
        self.assertEqual(set(get_profile().snapshot()), {"id", "model", "options", "think", "expected_digest"})
        self.assertEqual(llm.request_supplement(DEFAULT_PROFILE_ID, self.settings), "")
        for field in ("request_supplement", "request_supplement_sha256", "user_content_sha256"):
            self.assertNotIn(field, record["request_metadata"])

    def test_gemma3_omits_thinking_and_retains_full_evidence(self):
        server, request, record = self.request("gemma3-4b")
        self.assertEqual(request["model"], "gemma3:4b")
        self.assertNotIn("think", request)
        self.assertFalse(request["truncate"])
        self.assertFalse(request["shift"])
        self.assertEqual(record["request_metadata"]["input_payload"]["ocr_regions"], self.regions)
        self.assertEqual(server.loaded_model, "gemma3:4b")

    def test_gemma3_one_uses_exact_short_supplement_after_unchanged_full_json(self):
        settings = {**self.settings, "region_hint_mm": {"x": 12., "y": 34., "width": 56., "height": 78.},
                    "instruction": "対象部品の番号。元の文字を保持。"}
        _, request, record = self.request("gemma3-4b", settings=settings)
        supplement = ("\n\nExtract only the field 番号. Return at most one matching value. "
                      "If that field is absent, return not_found with empty results. Do not list other text.")
        payload = {"page": self.page, "ocr_regions": self.regions, "settings": settings}
        expected_content = "次の設定で画像の項目を抽出してください。\n" + json.dumps(payload, ensure_ascii=False) + supplement
        self.assertEqual(request["messages"][1]["content"], expected_content)
        self.assertEqual(request["messages"][0], {"role": "system", "content": llm.SYSTEM})
        self.assertEqual(request["messages"][1]["images"],
                         [base64.b64encode(b"full original image bytes").decode("ascii")])
        self.assertEqual(request["format"], llm.SCHEMA)
        self.assertNotIn("think", request)
        metadata = record["request_metadata"]
        self.assertEqual(metadata["input_payload"], payload)
        self.assertEqual(metadata["request_supplement"], supplement)
        self.assertEqual(metadata["request_supplement_sha256"], hashlib.sha256(supplement.encode()).hexdigest())
        self.assertEqual(metadata["user_content_sha256"], hashlib.sha256(expected_content.encode()).hexdigest())
        self.assertEqual(get_profile("gemma3-4b").snapshot()["request_adapter"], GEMMA3_REQUEST_ADAPTER)

    def test_gemma3_all_keeps_repeated_occurrences_and_full_ocr(self):
        settings = {**self.settings, "item": "PART NUMBER", "take": "all"}
        regions = self.regions + [{"id": "p1:r2", "text": " 0007 mm ",
                                  "bbox_mm": {"x": 10., "y": 20., "width": 3., "height": 4.}}]
        results = [{"value": " 0007 mm ", "source_ids": [region["id"]]} for region in regions]
        _, request, record = self.request("gemma3-4b", settings=settings, regions=regions, results=results)
        supplement = ("\n\nExtract only the field PART NUMBER. "
                      "Return every matching occurrence, including repeated values at different locations. "
                      "If that field is absent, return not_found with empty results. Do not list other text.")
        payload = {"page": self.page, "ocr_regions": regions, "settings": settings}
        self.assertEqual(request["messages"][1]["content"],
                         "次の設定で画像の項目を抽出してください。\n" + json.dumps(payload, ensure_ascii=False) + supplement)
        self.assertEqual(record["parsed"]["results"], results)
        self.assertEqual(record["request_metadata"]["input_payload"], payload)
        self.assertEqual(record["request_metadata"]["request_supplement"], supplement)
        self.assertNotIn("think", request)

    def test_unknown_request_adapter_is_rejected(self):
        with self.assertRaises(llm.OutputError):
            llm.request_supplement(replace(get_profile("gemma3-4b"), request_adapter="unknown"), self.settings)

    def test_changed_gemma3_request_adapter_preserves_old_interrupted_run(self):
        current = get_profile("gemma3-4b").snapshot()
        previous = {key: value for key, value in current.items() if key != "request_adapter"}
        old, manifest, _ = resume_run(self.root, {"model": {"profile": previous, "digest": "1" * 64}})
        manifest["status"] = "interrupted"
        write_json(old / "run.json", manifest)
        new, _, _ = resume_run(self.root, {"model": {"profile": current, "digest": "1" * 64}})
        self.assertNotEqual(old, new)
        self.assertEqual(read_json(old / "run.json")["status"], "interrupted")

    def test_management_runtime_does_not_require_any_model(self):
        with patch("simple_drawing.llm.EXPECTED_EXE", self.runtime_hash):
            runtime = llm.runtime_contract(self.root)
            model = llm.model_contract(self.root)
        self.assertTrue(runtime["ready"])
        self.assertFalse(model["ready"])
        self.assertNotIn("manifest", runtime)
        server = llm.OllamaServer(self.root, management=True)
        self.assertIsNone(server.loaded_model)

    def test_gemma3_assets_are_bound_to_install_record_and_exact_digest(self):
        profile, digest, asset = self.install_fixture("gemma3-4b")
        with patch("simple_drawing.llm.EXPECTED_EXE", self.runtime_hash):
            self.assertFalse(llm.model_contract(self.root, profile)["ready"])
            entry = {"model_tag": profile.model, "digest": digest}
            write_json(self.root / "data/model-registry.json", {"version": 1, "models": {profile.id: entry}})
            contract = llm.model_contract(self.root, profile)
            self.assertTrue(contract["ready"])
            self.assertEqual(contract["model_identity"]["digest"], digest)
            self.assertEqual(contract["expected_manifest_sha256"], digest)
            asset.write_bytes(b"broken")
            self.assertFalse(llm.model_contract(self.root, profile)["ready"])
            entry["digest"] = "0" * 64
            write_json(self.root / "data/model-registry.json", {"version": 1, "models": {profile.id: entry}})
            self.assertFalse(llm.model_contract(self.root, profile)["ready"])

    def test_selected_model_is_unloaded_before_owned_process_termination(self):
        server = llm.OllamaServer(self.root, profile="gemma3-4b")
        process = Mock()
        process.children.return_value = []
        api = Mock(side_effect=[{"done": True}, {"models": []}])
        with patch.object(server, "_owned_process", return_value=process), patch.object(server, "api", api), \
                patch("simple_drawing.llm.psutil.wait_procs", return_value=([], [])):
            report = server.stop()
        self.assertEqual(api.call_args_list, [call("/api/generate", {"model": "gemma3:4b", "keep_alive": 0}, timeout=10),
                                              call("/api/ps", timeout=3)])
        self.assertTrue(report["model_unloaded"])
        self.assertTrue(report["process_stopped"])

    def test_management_without_loaded_model_does_not_trigger_loading_for_unload(self):
        server = llm.OllamaServer(self.root, management=True)
        process = Mock()
        process.children.return_value = []
        with patch.object(server, "_owned_process", return_value=process), \
                patch.object(server, "api", return_value={"models": []}) as api, \
                patch("simple_drawing.llm.psutil.wait_procs", return_value=([], [])):
            report = server.stop()
        api.assert_called_once_with("/api/ps", timeout=3)
        self.assertTrue(report["model_unloaded"])

    def test_repeated_concurrent_shutdown_preserves_one_completed_report(self):
        server = llm.OllamaServer(self.root, profile="gemma3-4b")
        process = Mock()
        process.children.return_value = []
        api = Mock(side_effect=[{"done": True}, {"models": []}])
        reports = []
        barrier = threading.Barrier(3)
        def shutdown():
            barrier.wait(timeout=5)
            reports.append(server.stop())
        with patch.object(server, "_owned_process", return_value=process) as owned, \
                patch.object(server, "api", api), \
                patch("simple_drawing.llm.psutil.wait_procs", return_value=([], [])):
            workers = [threading.Thread(target=shutdown) for _ in range(2)]
            for worker in workers:
                worker.start()
            barrier.wait(timeout=5)
            for worker in workers:
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
            reports.append(server.stop())
        self.assertEqual(owned.call_count, 1)
        self.assertEqual(api.call_count, 2)
        process.terminate.assert_called_once_with()
        self.assertEqual(reports, [{"model_unloaded": True, "process_stopped": True, "errors": []}] * 3)
        reports[0]["errors"].append("caller mutation")
        self.assertEqual(server.shutdown_report["errors"], [])
        self.assertEqual(reports[1]["errors"], [])

    def test_second_shutdown_does_not_erase_first_unload_error(self):
        server = llm.OllamaServer(self.root)
        process = Mock()
        process.children.return_value = []
        with patch.object(server, "_owned_process", return_value=process), \
                patch.object(server, "api", side_effect=RuntimeError("unload failed")) as api, \
                patch("simple_drawing.llm.psutil.wait_procs", return_value=([], [])):
            first = server.stop()
            second = server.stop()
        self.assertEqual(first, second)
        self.assertTrue(first["process_stopped"])
        self.assertFalse(first["model_unloaded"])
        self.assertEqual(first["errors"], ["Model unload: unload failed"])
        api.assert_called_once()

    def test_changed_profile_starts_new_run_but_unchanged_profile_resumes(self):
        first_contract = {"model": {"profile": get_profile().snapshot(), "digest": "1" * 64}}
        folder, manifest, _ = resume_run(self.root, first_contract)
        manifest["status"] = "interrupted"
        write_json(folder / "run.json", manifest)
        resumed, _, _ = resume_run(self.root, first_contract)
        self.assertEqual(folder, resumed)
        changed = {"model": {"profile": get_profile("gemma3-4b").snapshot(), "digest": "2" * 64}}
        new, _, _ = resume_run(self.root, changed)
        self.assertNotEqual(new, folder)
        self.assertEqual(read_json(folder / "run.json")["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()

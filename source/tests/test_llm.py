import hashlib
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from simple_drawing.llm import OPTIONS, SCHEMA, MODEL, OllamaAPIError, OllamaServer, OutputError, build_payload, extract_page, model_contract, validate_output

TEST_ROOT = Path(__file__).resolve().parent


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.page = {"number": 1, "width_mm": 297.0, "height_mm": 210.0, "width_px": 2970, "height_px": 2100}
        self.regions = [{"id": "p1:value", "text": "001", "bbox_mm": {"x": 1.1, "y": 2.2, "width": 3.3, "height": 4.4}},
                        {"id": "p1:other", "text": "材質", "bbox_mm": {"x": 5.5, "y": 6.6, "width": 7.7, "height": 8.8}}]
        self.settings = {"item": "部品番号", "region_hint_mm": None, "take": "one", "instruction": ""}

    def output(self, **updates):
        result = {"status": "found", "results": [{"value": "001", "source_ids": ["p1:value"]}], "note": ""}
        result.update(updates)
        return json.dumps(result, ensure_ascii=False)

    def test_value_is_preserved(self):
        result = validate_output(self.output(results=[{"value": " 001-020 mm ", "source_ids": ["p1:value"]}]), self.settings, self.regions)
        self.assertEqual(result["results"][0]["value"], " 001-020 mm ")

    def test_found_empty_values_are_rejected(self):
        for value in ("", " \t\n "):
            with self.assertRaises(OutputError) as caught:
                validate_output(self.output(results=[{"value": value, "source_ids": ["p1:value"]}]), self.settings, self.regions)
            self.assertEqual(caught.exception.code, "INVALID_VALUE")

    def test_strict_types_and_extra_keys(self):
        for content in (self.output(note=1), self.output(extra=True), self.output(results=[{"value": 1, "source_ids": []}])):
            with self.assertRaises(OutputError) as caught:
                validate_output(content, self.settings, self.regions)
            self.assertEqual(caught.exception.code, "INVALID_SCHEMA")

    def test_unknown_page_id_rejected(self):
        with self.assertRaises(OutputError) as caught:
            validate_output(self.output(results=[{"value": "001", "source_ids": ["p2:value"]}]), self.settings, self.regions)
        self.assertEqual(caught.exception.code, "UNKNOWN_SOURCE_ID")

    def test_empty_source_ids_require_note(self):
        content = self.output(results=[{"value": "001", "source_ids": []}])
        with self.assertRaises(OutputError):
            validate_output(content, self.settings, self.regions)
        self.assertEqual(validate_output(self.output(results=[{"value": "001", "source_ids": []}], note="画像から読み取った。"), self.settings, self.regions)["results"][0]["source_ids"], [])

    def test_status_and_one_count(self):
        for content in (self.output(status="not_found"), self.output(results=[]), self.output(results=[{"value": "001", "source_ids": ["p1:value"]}] * 2)):
            with self.assertRaises(OutputError):
                validate_output(content, self.settings, self.regions)
        result = validate_output(self.output(status="ambiguous", results=[]), self.settings, self.regions)
        self.assertEqual(result["status"], "ambiguous")

    def test_all_preserves_duplicate_occurrences(self):
        settings = {**self.settings, "take": "all"}
        result = validate_output(self.output(results=[{"value": "001", "source_ids": ["p1:value"]}] * 2), settings, self.regions)
        self.assertEqual(len(result["results"]), 2)

    def test_existence_does_not_claim_semantic_correctness(self):
        result = validate_output(self.output(results=[{"value": "001", "source_ids": ["p1:other"]}]), self.settings, self.regions)
        self.assertEqual(result["results"][0]["source_ids"], ["p1:other"])

    def test_request_is_full_page_and_input_only(self):
        class FakeServer:
            contract = {}
            root = TEST_ROOT
            calls = []
            def api(inner, path, body=None, timeout=900):
                inner.calls.append((path, body))
                if path == "/api/ps":
                    return {"models": []}
                return {"done": True, "done_reason": "stop", "message": {"content": self.output()}, "total_duration": 1_000_000_000}
        server = FakeServer()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(TEST_ROOT))
            image = Path(folder) / "page.png"
            image.write_bytes(b"unmodified-image")
            with patch("simple_drawing.llm._gpu_sample", return_value={}):
                record = extract_page(server, image, self.page, self.regions, self.settings)
        request = server.calls[0][1]
        self.assertEqual(request["options"], OPTIONS)
        self.assertEqual(request["format"], SCHEMA)
        self.assertEqual(request["model"], MODEL)
        self.assertFalse(request["truncate"])
        self.assertFalse(request["shift"])
        self.assertEqual(record["request_metadata"]["input_payload"]["ocr_regions"], self.regions)
        self.assertEqual(record["parsed"]["results"][0]["value"], "001")
        self.assertEqual(record["timings"]["total_seconds"], 1)
        self.assertFalse(Path(record["request_metadata"]["image_path"]).is_absolute())

    def test_incomplete_raw_answer_is_retained(self):
        class FakeServer:
            contract = {}
            def api(inner, *args, **kwargs):
                return {"done": True, "done_reason": "length", "message": {"content": self.output()}}
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(TEST_ROOT))
            image = Path(folder) / "page.png"
            image.write_bytes(b"image")
            with patch("simple_drawing.llm._gpu_sample", return_value={}):
                record = extract_page(FakeServer(), image, self.page, self.regions, self.settings)
        self.assertIsNone(record["parsed"])
        self.assertEqual(record["error"]["code"], "INCOMPLETE_RESPONSE")
        self.assertEqual(record["response"]["done_reason"], "length")

    def test_port_collision_does_not_reuse_or_stop_server(self):
        server = OllamaServer(TEST_ROOT)
        probe = Mock()
        probe.connect_ex.return_value = 0
        context = Mock()
        context.__enter__ = Mock(return_value=probe)
        context.__exit__ = Mock(return_value=None)
        with patch("simple_drawing.llm.model_contract", return_value={"ready": True}), patch("simple_drawing.llm.socket.socket", return_value=context), patch("simple_drawing.llm.subprocess.Popen") as start:
            with self.assertRaises(RuntimeError):
                server.start()
            start.assert_not_called()
        self.assertIsNone(server.process)

    def test_startup_keyboard_interrupt_stops_the_owned_server(self):
        probe = Mock()
        probe.connect_ex.return_value = 1
        context = Mock()
        context.__enter__ = Mock(return_value=probe)
        context.__exit__ = Mock(return_value=None)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            server = OllamaServer(Path(folder))
            child_process = Mock(pid=777)
            child_process.poll.return_value = None
            owned = Mock()
            owned.create_time.return_value = 1
            owned.exe.return_value = str(Path(folder) / "ollama/ollama.exe")
            owned.children.return_value = []
            with patch("simple_drawing.llm.model_contract", return_value={"ready": True, "executable": "ollama/ollama.exe"}), patch("simple_drawing.llm.socket.socket", return_value=context), patch("simple_drawing.llm.subprocess.Popen", return_value=child_process), patch("simple_drawing.llm.psutil.Process", return_value=owned), patch("simple_drawing.llm.psutil.wait_procs", return_value=([], [])), patch.object(server, "api", side_effect=[KeyboardInterrupt, {"done": True}, {"models": []}]):
                with self.assertRaises(KeyboardInterrupt):
                    server.start()
            owned.terminate.assert_called_once_with()
            self.assertTrue(server.shutdown_report["model_unloaded"])
            self.assertTrue(server.shutdown_report["process_stopped"])
            self.assertEqual(server._log_files, [])

    def test_changed_process_identity_is_never_stopped(self):
        server = OllamaServer(TEST_ROOT)
        server.process = Mock(pid=123)
        server._process_created = 1
        server.contract = {"executable": str(TEST_ROOT / "ollama.exe")}
        process = Mock()
        process.create_time.return_value = 2
        with patch("simple_drawing.llm.psutil.Process", return_value=process):
            report = server.stop()
        process.terminate.assert_not_called()
        process.children.assert_not_called()
        self.assertFalse(report["process_stopped"])
        self.assertTrue(report["errors"])

    def test_shutdown_unloads_model_before_stopping_owned_descendants(self):
        server = OllamaServer(TEST_ROOT)
        process, child = Mock(), Mock()
        process.children.return_value = [child]
        events = Mock()
        events.attach_mock(process, "owned_process")
        events.attach_mock(child, "owned_child")
        api = Mock(side_effect=[{"done": True}, {"models": []}])
        events.attach_mock(api, "api")
        with patch.object(server, "_owned_process", return_value=process), patch.object(server, "api", api), patch("simple_drawing.llm.psutil.wait_procs", return_value=([], [])):
            report = server.stop()
        self.assertEqual(api.call_args_list, [call("/api/generate", {"model": MODEL, "keep_alive": 0}, timeout=10), call("/api/ps", timeout=3)])
        self.assertEqual(events.mock_calls[0], call.api("/api/generate", {"model": MODEL, "keep_alive": 0}, timeout=10))
        child.terminate.assert_called_once_with()
        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        self.assertEqual(report, {"model_unloaded": True, "process_stopped": True, "errors": []})

    def test_model_contract_survives_package_relocation(self):
        executable = b"frozen executable"
        blob = b"frozen model"
        digest = hashlib.sha256(blob).hexdigest()
        manifest = json.dumps({"config": {"digest": "sha256:" + digest, "size": len(blob)}, "layers": []}).encode()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            contracts = []
            for name in ("original", "移動先 with spaces"):
                root = Path(folder) / name
                exe = root / "ollama/ollama.exe"
                definition = root / "models/ollama/manifests/registry.ollama.ai/library/gemma4/e4b-it-q8_0"
                asset = root / "models/ollama/blobs" / ("sha256-" + digest)
                for path, value in ((exe, executable), (definition, manifest), (asset, blob)):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(value)
                with patch("simple_drawing.llm.EXPECTED_EXE", hashlib.sha256(executable).hexdigest()), patch("simple_drawing.llm.EXPECTED_MANIFEST", hashlib.sha256(manifest).hexdigest()):
                    contracts.append(model_contract(root))
            self.assertEqual(contracts[0], contracts[1])
            self.assertTrue(contracts[0]["ready"])
            for key in ("executable", "manifest", "models_path"):
                self.assertFalse(Path(contracts[0][key]).is_absolute())

    def test_http_client_ignores_environment_proxy_and_retains_error_body(self):
        client = Mock()
        client.request.return_value = Mock(status_code=400, content=b'{"error":"context overflow"}')
        context = Mock()
        context.__enter__ = Mock(return_value=client)
        context.__exit__ = Mock(return_value=None)
        with patch("simple_drawing.llm.httpx.Client", return_value=context) as constructor:
            with self.assertRaises(OllamaAPIError) as caught:
                OllamaServer(TEST_ROOT).api("/api/chat", {"model": MODEL}, timeout=17)
        constructor.assert_called_once_with(trust_env=False, timeout=17)
        self.assertEqual(caught.exception.response["http_status"], 400)
        self.assertEqual(caught.exception.response["body"], '{"error":"context overflow"}')

    def test_api_error_body_is_saved_in_extraction_record(self):
        class FakeServer:
            contract = {}
            def api(inner, *args, **kwargs):
                raise OllamaAPIError("context overflow", {"http_status": 400, "body": "context overflow"})
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            image = Path(folder) / "page.png"
            image.write_bytes(b"image")
            with patch("simple_drawing.llm._gpu_sample", return_value={}):
                record = extract_page(FakeServer(), image, self.page, self.regions, self.settings)
        self.assertEqual(record["error"]["code"], "API_ERROR")
        self.assertEqual(record["api_error_response"]["body"], "context overflow")
        self.assertIsNone(record["parsed"])

    def test_projection_preserves_all_text_ids_boxes_and_order(self):
        page = {**self.page, "rotation": 0, "source_rotation": 90,
                "dimension_basis": {"kind": "pdf_visible_page", "diagnostics": list(range(50))}}
        box = {"x": 1.123456789012345, "y": 2.123456789012345, "width": 3.123456789012345, "height": 4.123456789012345}
        regions = [{"id": f"p1:r{i:06d}", "text": " 0007 =SUM(A1)\n材質 mm ",
                    "bbox_mm": deepcopy(box), "confidence": .99999, "bbox_px": box,
                    "polygon_mm": [[1, 2]] * 4, "polygon_px": [[10, 20]] * 4} for i in range(1, 101)]
        originals = deepcopy((page, regions, self.settings))
        payload = build_payload(page, regions, self.settings)
        self.assertEqual(len(payload["ocr_regions"]), 100)
        self.assertEqual([(r["id"], r["text"], r["bbox_mm"]) for r in payload["ocr_regions"]],
                         [(r["id"], r["text"], r["bbox_mm"]) for r in regions])
        self.assertEqual(page, originals[0])
        self.assertEqual(regions, originals[1])
        self.assertEqual(self.settings, originals[2])
        self.assertEqual(payload["page"], self.page)
        self.assertTrue(all(set(r) == {"id", "text", "bbox_mm"} for r in payload["ocr_regions"]))
        payload["ocr_regions"][0]["bbox_mm"]["x"] = 99
        payload["settings"]["item"] = "changed"
        self.assertEqual(regions[0]["bbox_mm"], box)
        self.assertEqual(self.settings, originals[2])

    def test_projection_keeps_frozen_description_and_serialized_bytes(self):
        page = {**self.page, "dimension_basis": "authored evaluation page"}
        before = {"page": page, "ocr_regions": self.regions, "settings": self.settings}
        after = build_payload(page, self.regions, self.settings)
        self.assertEqual(json.dumps(before, ensure_ascii=False).encode(), json.dumps(after, ensure_ascii=False).encode())

    def test_projection_rejects_missing_evidence_geometry(self):
        with self.assertRaises(OutputError) as caught:
            build_payload(self.page, [{"id": "p1:value", "text": "001"}], self.settings)
        self.assertEqual(caught.exception.code, "INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()

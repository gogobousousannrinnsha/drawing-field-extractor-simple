import json
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import xml.etree.ElementTree as ET

from simple_drawing.storage import write_json, read_json, sha, local_path
from simple_drawing.workflow import prepare, extract, cache_valid
from simple_drawing.excel import write_excel
from simple_drawing.settings import ExtractionSettings

class FakeServer:
    starts = 0
    def __init__(self, root): pass
    def __enter__(self):
        FakeServer.starts += 1
        return self
    def __exit__(self, *args):
        self.shutdown_report = {"model_unloaded": True, "process_stopped": True, "errors": []}

def fake_inspect(path):
    if path.read_bytes() == b"broken":
        raise ValueError("broken PDF")
    return [{"number": n, "width_px": 10, "height_px": 10, "width_mm": 1.0, "height_mm": 1.0} for n in (1, 2)]

def fake_render(path, number, image):
    from PIL import Image
    Image.new("RGB", (10, 10), "white").save(image)
    return fake_inspect(path)[number - 1]

def fake_ocr(image, page, models, scratch):
    return [{"id": f"p{page['number']}:r000001", "text": "001 =SUM(A1) ",
             "bbox_mm": {"x": 0, "y": 0, "width": 1, "height": 1}}], {"raw": "original"}

def fake_extract(server, image, page, regions, settings):
    if page["number"] == 2:
        return {"parsed": None, "error": {"code": "http", "message": "timeout"}}
    return {"parsed": {"status": "found", "results": [{"value": "001", "source_ids": [regions[0]["id"]]},
        {"value": "001", "source_ids": [regions[0]["id"]]}], "note": ""}, "response": {"raw": True}}

def successful_extract(server, image, page, regions, settings):
    return {"parsed": {"status": "found", "results": [{"value": "001", "source_ids": [regions[0]["id"]]}],
        "note": ""}, "response": {"raw": True}}

def reporting_server(report):
    class Server(FakeServer):
        def __exit__(self, *args):
            self.shutdown_report = report
    return Server

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        artifacts = Path(__file__).resolve().parent.parent / "test-artifacts"
        artifacts.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=artifacts)
        self.root = Path(self.temp.name)
        (self.root / "input").mkdir()
        (self.root / "input" / "a.pdf").write_bytes(b"same")
        (self.root / "input" / "b.pdf").write_bytes(b"same")
        (self.root / "input" / "bad.pdf").write_bytes(b"broken")
        (self.root / "input" / "nested").mkdir()
        (self.root / "input" / "nested" / "ignored.pdf").write_bytes(b"same")
        write_json(self.root / "実行設定.json", {"input_folder": "input"})
        write_json(self.root / "抽出設定.json", {"item": "部品番号", "take": "all", "region_hint_mm": None, "instruction": ""})
        source = self.root / "source" / "simple_drawing"
        source.mkdir(parents=True)
        (source / "llm.py").write_text("test", encoding="utf-8")
    def tearDown(self): self.temp.cleanup()
    def prepare(self, producer=None):
        return prepare(self.root, inspector=fake_inspect, renderer=fake_render, recognizer=fake_ocr, producer=producer or {"v": 1})
    def extract(self, **kwargs):
        return extract(self.root, server_factory=FakeServer, extractor=fake_extract, model={"model": "fake"}, **kwargs)

    def test_aliases_cache_and_failures(self):
        code, batch = self.prepare()
        self.assertEqual(code, 2)
        self.assertEqual([d["name"] for d in batch["documents"]], ["a.pdf", "b.pdf", "bad.pdf"])
        self.assertEqual(batch["ocr_pages"], 2)
        self.assertEqual(batch["reused_pages"], 2)
        code, run = self.extract()
        self.assertEqual(code, 2)
        manifest = read_json(run / "run.json")
        self.assertEqual(manifest["failed_pages"], 2)
        self.assertEqual(len(manifest["records"]), 4)
        self.assertTrue((run / "抽出結果.xlsx").exists())
        previous = sha(run / "抽出結果.xlsx")
        _, next_run = self.extract()
        self.assertNotEqual(run, next_run)
        self.assertEqual(sha(run / "抽出結果.xlsx"), previous)

    def prepare_successful_batch(self):
        (self.root / "input" / "b.pdf").unlink()
        (self.root / "input" / "bad.pdf").unlink()
        self.assertEqual(self.prepare()[0], 0)

    def test_cleanup_outcome_does_not_change_successful_pages_or_excel(self):
        self.prepare_successful_batch()
        reports = [
            ({"model_unloaded": True, "process_stopped": True, "errors": []}, 0),
            ({"model_unloaded": False, "process_stopped": True, "errors": []}, 2),
            ({"model_unloaded": True, "process_stopped": False, "errors": []}, 2),
            ({"model_unloaded": True, "process_stopped": True, "errors": ["synthetic cleanup failure"]}, 2),
            ({}, 0),  # Legacy injected providers do not necessarily report cleanup.
        ]
        for report, expected_code in reports:
            with self.subTest(report=report):
                output = io.StringIO()
                with redirect_stdout(output):
                    code, run = extract(self.root, server_factory=reporting_server(report),
                        extractor=successful_extract, model={"model": "fake"})
                manifest = read_json(run / "run.json")
                self.assertEqual(code, expected_code)
                self.assertEqual(manifest["status"], "partial" if expected_code else "complete")
                self.assertEqual(manifest["completed_pages"], 2)
                self.assertEqual(manifest["failed_pages"], 0)
                self.assertEqual(manifest["failed_documents"], 0)
                self.assertEqual(manifest["server_shutdown"], report)
                self.assertEqual(manifest["excel_sha256"], sha(run / "抽出結果.xlsx"))
                for number in (1, 2):
                    record = read_json(run / "pages" / f"0-{number}.json")
                    self.assertFalse(record.get("error"))
                    self.assertEqual(record["parsed"]["results"],
                        [{"value": "001", "source_ids": [f"p{number}:r000001"]}])
                with zipfile.ZipFile(run / "抽出結果.xlsx") as archive:
                    self.assertIn("001", archive.read("xl/sharedStrings.xml").decode("utf-8"))
                if expected_code:
                    self.assertEqual([e["code"] for e in manifest["run_errors"]], ["server_shutdown"])
                    self.assertIn("終了処理失敗", output.getvalue())
                    self.assertIn("一部失敗", output.getvalue())
                else:
                    self.assertEqual(manifest["run_errors"], [])
                    self.assertNotIn("終了処理失敗", output.getvalue())

    def test_cleanup_failure_after_excel_publication_keeps_output_and_exit_two(self):
        self.prepare_successful_batch()
        from simple_drawing import workflow
        original = workflow.write_excel
        def publish_then_interrupt(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt()
        with patch.object(workflow, "write_excel", publish_then_interrupt):
            code, run = extract(self.root, server_factory=reporting_server(
                {"model_unloaded": False, "process_stopped": False, "errors": ["synthetic stop failure"]}),
                extractor=successful_extract, model={"model": "fake"})
        manifest = read_json(run / "run.json")
        self.assertEqual(code, 2)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual((manifest["completed_pages"], manifest["failed_pages"]), (2, 0))
        self.assertEqual(manifest["run_errors"][0]["code"], "server_shutdown")
        self.assertEqual(manifest["excel_sha256"], sha(run / "抽出結果.xlsx"))

    def test_interrupted_cleanup_failure_keeps_cli_130_and_clears_on_successful_resume(self):
        self.prepare_successful_batch()
        from simple_drawing import workflow
        def stop_after_saved(key, record):
            raise KeyboardInterrupt()
        original_extract = workflow.extract
        def interrupted_extract(root):
            return original_extract(root, server_factory=reporting_server(
                {"model_unloaded": False, "process_stopped": False, "errors": ["synthetic stop failure"]}),
                extractor=successful_extract, model={"model": "fake"}, on_saved=stop_after_saved)
        entry = Path(__file__).resolve().parents[1] / "run.py"
        output = io.StringIO()
        with patch.object(workflow, "extract", interrupted_extract), patch.object(sys, "argv",
                [str(entry), "extract", "--root", str(self.root)]), patch.object(sys, "path", sys.path.copy()), \
                patch.dict(os.environ), redirect_stdout(output), self.assertRaises(SystemExit) as exit_result:
            runpy.run_path(str(entry), run_name="__main__")
        self.assertEqual(exit_result.exception.code, 130)
        run = sorted((self.root / "data" / "runs").iterdir())[-1]
        interrupted = read_json(run / "run.json")
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(list(interrupted["records"]), ["0:1"])
        self.assertEqual(interrupted["run_errors"][0]["code"], "server_shutdown")
        self.assertFalse((run / "抽出結果.xlsx").exists())
        self.assertIn("終了処理失敗", output.getvalue())
        saved_sha = sha(run / "pages" / "0-1.json")
        calls = []
        def record_call(server, image, page, regions, settings):
            calls.append(page["number"])
            return successful_extract(server, image, page, regions, settings)
        code, resumed = extract(self.root, server_factory=FakeServer, extractor=record_call, model={"model": "fake"})
        manifest = read_json(resumed / "run.json")
        self.assertEqual(resumed, run)
        self.assertEqual(code, 0)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["run_errors"], [])
        self.assertEqual((manifest["completed_pages"], manifest["failed_pages"]), (2, 0))
        self.assertEqual(calls, [2])
        self.assertEqual(sha(run / "pages" / "0-1.json"), saved_sha)

    def test_settings_change_no_ocr_source_and_producer_invalidate(self):
        _, first = self.prepare()
        write_json(self.root / "抽出設定.json", {"item": "材質"})
        _, second = self.prepare()
        self.assertEqual(second["ocr_pages"], 0)
        (self.root / "input" / "a.pdf").write_bytes(b"different")
        _, third = self.prepare()
        self.assertEqual(third["ocr_pages"], 2)
        _, fourth = self.prepare({"v": 2})
        self.assertEqual(fourth["ocr_pages"], 4)

    def test_cache_corruption_requires_new_prepare(self):
        _, batch = self.prepare()
        folder = local_path(self.root, batch["documents"][0]["pages"][0]["cache"])
        (folder / "ocr.json").write_text("{}")
        self.assertFalse(cache_valid(folder))
        _, run = self.extract()
        self.assertEqual(read_json(run / "pages" / "0-1.json")["error"]["code"], "cache_corrupt")
        _, rebuilt = self.prepare()
        self.assertEqual(rebuilt["ocr_pages"], 1)
        self.assertTrue(cache_valid(folder))

    def test_wrong_json_shape_is_corrupt_cache(self):
        _, batch = self.prepare()
        folder = local_path(self.root, batch["documents"][0]["pages"][0]["cache"])
        write_json(folder / "complete.json", [])
        self.assertFalse(cache_valid(folder))
        _, rebuilt = self.prepare()
        self.assertEqual(rebuilt["ocr_pages"], 1)

    def test_resume_skips_saved_error_and_complete_pages(self):
        self.prepare()
        calls = []
        def stop(key, record):
            calls.append(key)
            if len(calls) == 2:
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.extract(on_saved=stop)
        folder = sorted((self.root / "data" / "runs").iterdir())[-1]
        self.assertFalse((folder / "抽出結果.xlsx").exists())
        self.assertEqual(read_json(folder / "run.json")["status"], "interrupted")
        self.assertEqual(read_json(folder / "run.json")["server_shutdown"],
            {"model_unloaded": True, "process_stopped": True, "errors": []})
        _, resumed = self.extract(on_saved=lambda key, record: calls.append(key))
        self.assertEqual(resumed, folder)
        self.assertEqual(calls, ["0:1", "0:2", "1:1", "1:2"])

    def test_changed_settings_starts_new_run(self):
        self.prepare()
        with self.assertRaises(KeyboardInterrupt):
            self.extract(on_saved=lambda key, record: (_ for _ in ()).throw(KeyboardInterrupt()))
        old = sorted((self.root / "data" / "runs").iterdir())[-1]
        write_json(self.root / "抽出設定.json", {"item": "材質"})
        _, new = self.extract()
        self.assertNotEqual(old, new)

    def test_page_exception_continues_other_pages(self):
        self.prepare()
        calls = []
        def fail_one(server, image, page, regions, settings):
            calls.append(page["number"])
            if len(calls) == 1:
                raise ValueError("one page failure")
            return fake_extract(server, image, page, regions, settings)
        code, run = extract(self.root, server_factory=FakeServer, extractor=fail_one, model={"model": "fake"})
        self.assertEqual(calls, [1, 2, 1, 2])
        self.assertEqual(read_json(run / "pages" / "0-1.json")["error"]["code"], "page")

    def test_modified_saved_input_starts_new_run(self):
        self.prepare()
        with self.assertRaises(KeyboardInterrupt):
            self.extract(on_saved=lambda key, record: (_ for _ in ()).throw(KeyboardInterrupt()))
        old = sorted((self.root / "data" / "runs").iterdir())[-1]
        (old / "inputs" / "0-1" / "payload.json").write_text("{}")
        _, new = self.extract()
        self.assertNotEqual(old, new)

    def test_interrupt_after_excel_publication_is_committed(self):
        self.prepare()
        from simple_drawing import workflow
        original = workflow.write_excel
        def publish_then_interrupt(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt()
        with patch.object(workflow, "write_excel", publish_then_interrupt):
            code, run = self.extract()
        self.assertTrue((run / "抽出結果.xlsx").exists())
        self.assertEqual(read_json(run / "run.json")["status"], "partial")

    def test_interrupt_before_excel_publication_resumes(self):
        self.prepare()
        from simple_drawing import workflow
        original = workflow.write_excel
        def interrupt_publish(path, rows, **kwargs):
            def before(temp):
                kwargs["before_publish"](temp)
                raise KeyboardInterrupt()
            return original(path, rows, before_publish=before)
        with patch.object(workflow, "write_excel", interrupt_publish):
            with self.assertRaises(KeyboardInterrupt): self.extract()
        old = sorted((self.root / "data" / "runs").iterdir())[-1]
        self.assertFalse((old / "抽出結果.xlsx").exists())
        _, resumed = self.extract()
        self.assertEqual(old, resumed)

    def test_total_page_cap(self):
        def inspect(path): return [{"number": n} for n in range(1, 502)]
        with self.assertRaisesRegex(ValueError, "上限"):
            prepare(self.root, inspector=inspect, renderer=fake_render, recognizer=fake_ocr, producer={"v": 1})

    def test_empty_and_relative_reference_rejection(self):
        for p in (self.root / "input").glob("*.pdf"): p.unlink()
        with self.assertRaisesRegex(ValueError, "PDF"):
            self.prepare()
        with self.assertRaises(ValueError): local_path(self.root, "../escape")

    def test_excel_literal_values(self):
        path = self.root / "strings.xlsx"
        values = ["0001", "=SUM(A1:A2)", "+1", "  02 mm  ", "https://example.com", "001", "001", ""]
        write_excel(path, [["file", "1", "item", v, ""] for v in values])
        with zipfile.ZipFile(path) as archive:
            strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            texts = ["".join(si.itertext()) for si in strings]
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            found = []
            for row in sheet.findall("m:sheetData/m:row", ns)[1:]:
                cell = next(c for c in row if c.attrib["r"].startswith("D"))
                found.append(texts[int(cell.find("m:v", ns).text)])
            self.assertEqual(found, values)
            self.assertIsNone(sheet.find(".//m:f", ns))
        with self.assertRaises(FileExistsError): write_excel(path, [])
        with self.assertRaises(ValueError): write_excel(self.root / "too-long.xlsx", [["x" * 32768]])

    def test_settings_strict_and_mm(self):
        self.assertEqual(ExtractionSettings.model_validate({"item": "番号", "region_hint_mm": {"x": 1, "y": 2, "width": 3, "height": 4}}).region_hint_mm.width, 3)
        for value in [{"item": " "}, {"item": "番号", "take": "many"}, {"item": "番号", "extra": 1}]:
            with self.assertRaises(ValueError): ExtractionSettings.model_validate(value)

    def test_saluki_backend_selects_provider_and_records_configuration(self):
        self.prepare()
        write_json(self.root / "実行設定.json", {"input_folder": "input", "llm_backend": "saluki", "saluki_thinking": True})
        with patch("simple_drawing.saluki.SalukiServer", FakeServer), \
             patch("simple_drawing.saluki.extract_page", fake_extract), \
             patch("simple_drawing.saluki.model_contract", return_value={"model": "Saluki", "adapter_sha256": "saluki-adapter"}):
            _, run = extract(self.root)
        contract = read_json(run / "run.json")["contract"]
        self.assertEqual(contract["model"]["model"], "Saluki")
        self.assertEqual(contract["provider_config"]["backend"], "saluki")
        self.assertTrue(contract["provider_config"]["thinking"])
        self.assertEqual(contract["adapter_sha256"], "saluki-adapter")

    def test_injected_provider_is_not_replaced_by_execution_settings(self):
        self.prepare()
        write_json(self.root / "実行設定.json", {"input_folder": "input", "llm_backend": "saluki"})
        with patch("simple_drawing.saluki.SalukiServer") as real_server:
            _, run = self.extract()
        real_server.assert_not_called()
        contract = read_json(run / "run.json")["contract"]
        self.assertEqual(contract["model"]["model"], "fake")
        self.assertEqual(contract["provider_config"]["backend"], "injected")

if __name__ == "__main__": unittest.main()

"""CPU fixtures exercise format routing, immutable caches and saved XDW OCR.

These inputs are synthetic bytes, not runtime/SDK acceptance evidence.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from simple_drawing.ocr_image_export import export_ocr_images
from simple_drawing.storage import local_path, read_json, relative, sha, write_json
from simple_drawing.workflow import extract, prepare


def page_metadata(number=1, *, xdw=False, rendered=False):
    page = {"number": number, "width_px": 320, "height_px": 160,
            "width_mm": 32.0, "height_mm": 16.0, "rotation": 0}
    if xdw:
        page.update(source_rotation=90, dimension_basis={
            "kind": "xdw_displayed_page", "coordinate_system": "top-left-x-right-y-down",
            "render_dpi": 300, "sdk_page_width_100mm": 3200,
            "sdk_page_height_100mm": 1600, "sdk_source_degree": 90,
            "sdk_page_type": 0, "orientation_applied_by": "sdk",
            "raster_size_basis": "actual_sdk_bmp" if rendered else "predicted_at_render_dpi"})
    return page


class FakeServer:
    def __init__(self, root):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown_report = {"model_unloaded": True, "process_stopped": True, "errors": []}


def successful_extract(server, image, page, regions, settings):
    return {"parsed": {"status": "found", "results": [
        {"value": regions[0]["text"], "source_ids": [regions[0]["id"]]}], "note": ""}}


class XdwWorkflowTests(unittest.TestCase):
    def setUp(self):
        artifacts = Path(__file__).resolve().parent.parent / "test-artifacts"
        artifacts.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=artifacts)
        self.root = Path(self.temp.name)
        (self.root / "input").mkdir()
        write_json(self.root / "実行設定.json", {"input_folder": "input"})
        write_json(self.root / "抽出設定.json", {"item": "部品番号"})
        source = self.root / "source" / "simple_drawing"
        source.mkdir(parents=True)
        (source / "xdw_input.py").write_text("synthetic adapter implementation", encoding="utf-8")
        (source / "llm.py").write_text("synthetic model implementation", encoding="utf-8")
        self.producer = {"recognition": "fixed CPU fixture", "dpi": 300}
        self.runtime_contract = {"core_version": "1.0.1", "runtime_version": "fixture-v1",
                                 "dll": {"name": "fixture.dll", "sha256": "a" * 64}, "render_dpi": 300}
        self.inspected = []
        self.rendered = []
        self.recognized = []
        self.xdw_entered = []
        self.xdw_closed = []
        self.xdw_error = None
        fixture = self

        class FakeXdwInput:
            def __init__(self, path):
                self.path = Path(path)

            def __enter__(self):
                fixture.xdw_entered.append(self.path)
                if fixture.xdw_error:
                    raise RuntimeError(fixture.xdw_error)
                if self.path.read_bytes() == b"broken":
                    raise ValueError("corrupt XDW fixture")
                return self

            def __exit__(self, *args):
                fixture.xdw_closed.append(self.path)

            def inspect(self):
                fixture.inspected.append(("xdw", self.path))
                return [page_metadata(xdw=True)]

            def renderer_contract(self):
                return deepcopy(fixture.runtime_contract)

            def render(self, number, output):
                fixture.rendered.append(("xdw", self.path, number))
                Image.new("RGB", (320, 160), "red").save(output)
                return page_metadata(number, xdw=True, rendered=True)

        self.fake_xdw = FakeXdwInput

    def tearDown(self):
        self.temp.cleanup()

    def add_source(self, name, content=b"same bytes"):
        path = self.root / "input" / name
        path.write_bytes(content)
        return path

    def inspect_pdf(self, path):
        self.inspected.append(("pdf", Path(path)))
        return [page_metadata()]

    def render_pdf(self, path, number, output):
        self.rendered.append(("pdf", Path(path), number))
        Image.new("RGB", (320, 160), "red").save(output)
        return page_metadata(number)

    def recognize(self, image, page, models, scratch):
        self.recognized.append(image)
        return [{"id": f"p{page['number']}:r000001", "text": "001-XDW", "confidence": 0.0,
                 "bbox_px": {"x": 10, "y": 10, "width": 150, "height": 30},
                 "bbox_mm": {"x": 1.0, "y": 1.0, "width": 15.0, "height": 3.0}}], {"synthetic": True}

    def prepare(self, *, inspector=None):
        # Patch the ordinary routing boundary: XDW must use its real branch,
        # rather than being disguised as a generic injected PDF inspector.
        with patch("simple_drawing.pdf_ocr.inspect_pdf", inspector or self.inspect_pdf), \
                patch("simple_drawing.pdf_ocr.render_page", self.render_pdf), \
                patch("simple_drawing.xdw_input.XdwInput", self.fake_xdw), \
                redirect_stdout(io.StringIO()):
            return prepare(self.root, recognizer=self.recognize, producer=self.producer)

    def document(self, batch, name):
        return next(document for document in batch["documents"] if document["name"] == name)

    def test_mixed_inputs_keep_extension_and_use_shared_extraction(self):
        self.add_source("a.pdf")
        self.add_source("日本語 図面.XDW")
        (self.root / "input" / "nested").mkdir()
        (self.root / "input" / "nested" / "ignored.xdw").write_bytes(b"nested")
        self.add_source("ignored.xbd")
        code, batch = self.prepare()
        self.assertEqual(code, 0)
        self.assertEqual(len(batch["documents"]), 2)
        self.assertEqual({doc["source_type"] for doc in batch["documents"]}, {"pdf", "xdw"})
        for doc in batch["documents"]:
            saved = local_path(self.root, doc["source"])
            self.assertEqual(saved.suffix, "." + doc["source_type"])
            self.assertEqual(sha(saved), doc["sha256"])
        self.assertEqual(len(self.recognized), 2)
        self.assertEqual({kind for kind, *_ in self.rendered}, {"pdf", "xdw"})
        self.assertTrue(all(path.parent == self.root / "data" / "sources" for path in self.xdw_entered))
        self.assertEqual(self.xdw_entered, self.xdw_closed)
        with redirect_stdout(io.StringIO()):
            extract_code, run = extract(self.root, server_factory=FakeServer,
                                       extractor=successful_extract, model={"model": "fixture"})
        self.assertEqual(extract_code, 0)
        self.assertTrue((run / "抽出結果.xlsx").is_file())
        records = [read_json(run / "pages" / filename) for filename in ("0-1.json", "1-1.json")]
        self.assertEqual({r["document_name"] for r in records}, {"a.pdf", "日本語 図面.XDW"})
        self.assertTrue(all(r["parsed"]["results"][0]["value"] == "001-XDW" for r in records))

    def test_same_bytes_pdf_and_xdw_have_separate_caches(self):
        self.add_source("a.pdf")
        self.add_source("b.xdw")
        _, batch = self.prepare()
        pdf, xdw = self.document(batch, "a.pdf"), self.document(batch, "b.xdw")
        self.assertEqual(pdf["sha256"], xdw["sha256"])
        self.assertNotEqual(pdf["producer_sha256"], xdw["producer_sha256"])
        self.assertNotEqual(pdf["pages"][0]["cache"], xdw["pages"][0]["cache"])
        self.assertEqual(batch["ocr_pages"], 2)
        for doc in (pdf, xdw):
            ocr = read_json(local_path(self.root, doc["pages"][0]["cache"]) / "ocr.json")
            self.assertEqual(ocr["producer_sha256"], doc["producer_sha256"])

    def test_adding_xdw_keeps_existing_pdf_cache(self):
        self.add_source("a.pdf")
        _, first = self.prepare()
        pdf_cache = first["documents"][0]["pages"][0]["cache"]
        self.add_source("b.xdw", b"xdw source")
        _, second = self.prepare()
        self.assertEqual(second["ocr_pages"], 1)
        self.assertEqual(second["reused_pages"], 1)
        self.assertEqual(self.document(second, "a.pdf")["pages"][0]["cache"], pdf_cache)
        self.assertEqual([kind for kind, *_ in self.rendered], ["pdf", "xdw"])

    def test_xdw_raster_provenance_does_not_prevent_reuse(self):
        self.add_source("a.xdw")
        _, first = self.prepare()
        code, second = self.prepare()
        self.assertEqual(code, 0)
        self.assertEqual((second["ocr_pages"], second["reused_pages"]), (0, 1))
        self.assertEqual(second["documents"][0]["pages"][0]["cache"],
                         first["documents"][0]["pages"][0]["cache"])
        self.assertEqual(len(self.recognized), 1)

    def test_runtime_contract_change_invalidates_only_xdw(self):
        self.add_source("a.pdf")
        self.add_source("b.xdw")
        _, first = self.prepare()
        self.runtime_contract["runtime_version"] = "fixture-v2"
        _, second = self.prepare()
        self.assertEqual((second["ocr_pages"], second["reused_pages"]), (1, 1))
        self.assertEqual(self.document(first, "a.pdf")["pages"][0]["cache"],
                         self.document(second, "a.pdf")["pages"][0]["cache"])
        self.assertNotEqual(self.document(first, "b.xdw")["pages"][0]["cache"],
                            self.document(second, "b.xdw")["pages"][0]["cache"])

    def test_pdf_runs_without_opening_xdw_or_core(self):
        self.add_source("a.pdf")
        self.xdw_error = "Core is not available"
        code, batch = self.prepare()
        self.assertEqual(code, 0)
        self.assertEqual(batch["completed_pages"], 1)
        self.assertEqual(self.xdw_entered, [])

    def test_missing_xdw_environment_does_not_stop_pdf(self):
        self.add_source("a.pdf")
        self.add_source("b.xdw")
        self.xdw_error = "Core is not available"
        code, batch = self.prepare()
        self.assertEqual(code, 2)
        self.assertEqual(batch["completed_pages"], 1)
        self.assertNotIn("error", self.document(batch, "a.pdf"))
        self.assertIn("Core", self.document(batch, "b.xdw")["error"])
        self.assertEqual([kind for kind, *_ in self.rendered], ["pdf"])

    def test_corrupt_xdw_does_not_stop_pdf(self):
        self.add_source("a.pdf")
        self.add_source("b.xdw", b"broken")
        code, batch = self.prepare()
        self.assertEqual(code, 2)
        self.assertEqual(batch["completed_pages"], 1)
        self.assertIn("corrupt XDW", self.document(batch, "b.xdw")["error"])
        self.assertEqual(len(self.recognized), 1)

    def test_original_or_saved_copy_changed_during_inspection_is_rejected(self):
        original = self.add_source("a.pdf")
        for target in ("original", "saved"):
            with self.subTest(target=target):
                original.write_bytes(b"before " + target.encode())
                def mutate(path):
                    (original if target == "original" else Path(path)).write_bytes(b"changed while inspecting")
                    return [page_metadata()]
                code, batch = self.prepare(inspector=mutate)
                self.assertEqual(code, 2)
                self.assertIn("変更", batch["documents"][0]["error"])
                self.assertEqual(batch["completed_pages"], 0)
                self.assertEqual(batch["ocr_pages"], 0)
        self.assertEqual(self.recognized, [])

    def test_xdw_original_or_copy_changed_during_inspection_is_rejected_and_closed(self):
        original = self.add_source("a.xdw")
        for target in ("original", "saved"):
            with self.subTest(target=target):
                original.write_bytes(b"before " + target.encode())
                def mutate(xdw):
                    (original if target == "original" else xdw.path).write_bytes(b"changed while inspecting")
                    return [page_metadata(xdw=True)]
                with patch.object(self.fake_xdw, "inspect", mutate):
                    code, batch = self.prepare()
                self.assertEqual(code, 2)
                self.assertIn("変更", batch["documents"][0]["error"])
                self.assertEqual(batch["completed_pages"], 0)
        self.assertEqual(self.recognized, [])
        self.assertEqual(self.xdw_entered, self.xdw_closed)

    def test_runtime_changes_between_inspection_and_render_are_not_cached(self):
        self.add_source("a.pdf")
        self.add_source("b.xdw")
        def changing_contract(xdw):
            contract = deepcopy(self.runtime_contract)
            if len(self.xdw_entered) > 1:
                contract["runtime_version"] = "changed after inspection"
            return contract
        with patch.object(self.fake_xdw, "renderer_contract", changing_contract):
            code, batch = self.prepare()
        self.assertEqual(code, 2)
        self.assertEqual((batch["completed_pages"], batch["failed_pages"]), (1, 1))
        xdw_page = self.document(batch, "b.xdw")["pages"][0]
        self.assertIn("環境が変更", xdw_page["error"])
        self.assertFalse((local_path(self.root, xdw_page["cache"]) / "complete.json").exists())
        self.assertEqual([kind for kind, *_ in self.rendered], ["pdf"])
        self.assertEqual(self.xdw_entered, self.xdw_closed)

    def test_saved_xdw_ocr_white_png_uses_document_producer_without_sdk(self):
        self.add_source("a.xdw")
        _, batch = self.prepare()
        doc = batch["documents"][0]
        self.assertNotEqual(doc["producer_sha256"], batch["producer_sha256"])
        immutable = {relative(self.root, path): sha(path) for path in (self.root / "data").rglob("*") if path.is_file()}
        with patch("simple_drawing.xdw_input.XdwInput", side_effect=AssertionError("SDK reopened")), \
                patch("simple_drawing.workflow.prepare", side_effect=AssertionError("OCR rerun")), \
                redirect_stdout(io.StringIO()):
            code, folder = export_ocr_images(self.root)
        self.assertEqual(code, 0)
        report = read_json(folder / "export.json")
        self.assertEqual(report["completed_pages"], 1)
        self.assertFalse(report["sdk_executed"])
        self.assertFalse(report["ocr_executed"])
        with Image.open(folder / report["records"][0]["image"]) as image:
            self.assertEqual(image.size, (320, 160))
            self.assertEqual(image.getpixel((310, 150)), (255, 255, 255))
            self.assertTrue(any(pixel == (0, 0, 0) for pixel in image.get_flattened_data()))
        for name, expected in immutable.items():
            self.assertEqual(sha(local_path(self.root, name)), expected)

    def test_legacy_pdf_batch_without_document_producer_still_exports_and_extracts(self):
        self.add_source("a.pdf")
        _, batch = self.prepare()
        for doc in batch["documents"]:
            for key in ("source_type", "producer", "producer_sha256"):
                doc.pop(key, None)
        pointer = read_json(self.root / "data" / "latest_batch.json")
        batch_path = local_path(self.root, pointer["path"])
        write_json(batch_path, batch)
        pointer["sha256"] = sha(batch_path)
        write_json(self.root / "data" / "latest_batch.json", pointer)
        with redirect_stdout(io.StringIO()):
            code, exported = export_ocr_images(self.root)
            extract_code, run = extract(self.root, server_factory=FakeServer,
                                       extractor=successful_extract, model={"model": "fixture"})
        self.assertEqual((code, extract_code), (0, 0))
        self.assertEqual(read_json(exported / "export.json")["completed_pages"], 1)
        self.assertTrue((run / "抽出結果.xlsx").is_file())


if __name__ == "__main__":
    unittest.main()

"""Export saved full OCR independently of extraction, with immutable input hashes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from simple_drawing.ocr_image_export import export_ocr_images
from simple_drawing.storage import digest, read_json, relative, sha, write_json
from simple_drawing.workflow import producer_contract


class OCRImageExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/"data/sources/source.pdf"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"synthetic source fixture")
        self.producer = digest({"test": 1})
        pages = []
        for number in (1, 2):
            page = {"number": number, "width_px": 320, "height_px": 160,
                    "width_mm": 32.0, "height_mm": 16.0, "rotation": 0}
            folder = self.root/f"data/cache/fixture/{number}"
            folder.mkdir(parents=True)
            Image.new("RGB", (320, 160), "red").save(folder/"page.png")
            regions = [] if number == 2 else [{"id": "p1:r000001", "text": " 0001 ± 材質 ",
                  "confidence": 0.0, "bbox_px": {"x": 10.0, "y": 10.0, "width": 250.0, "height": 30.0}}]
            write_json(folder/"ocr.json", {"version": 1, "page": page, "regions": regions,
                       "source_sha256": sha(self.source), "producer_sha256": self.producer})
            write_json(folder/"raw.json", {"synthetic": True})
            write_json(folder/"complete.json", {"version": 1, "files": {
                name: sha(folder/name) for name in ("page.png", "raw.json", "ocr.json")}})
            pages.append({**page, "cache": relative(self.root, folder),
                          "complete_sha256": sha(folder/"complete.json")})
        self.batch = {"version": 1, "status": "complete", "producer_sha256": self.producer,
                      "documents": [{"name": "../CON.pdf", "source": relative(self.root, self.source),
                                     "sha256": sha(self.source), "pages": pages}]}
        self.batch_path = self.root/"data/batches/fixture/batch.json"
        self.save_batch()

    def tearDown(self):
        self.temp.cleanup()

    def save_batch(self):
        write_json(self.batch_path, self.batch)
        write_json(self.root/"data/latest_batch.json", {"path": relative(self.root, self.batch_path),
                   "sha256": sha(self.batch_path)})

    def test_all_regions_and_empty_page_are_exported_without_ocr_or_model_calls(self):
        original_hashes = {relative(self.root, path): sha(path)
                           for path in (self.root/"data").rglob("*") if path.is_file()}
        received = []
        def renderer(ocr, output, **kwargs):
            received.append(ocr)
            Image.new("RGB", (ocr["page"]["width_px"], ocr["page"]["height_px"]), "white").save(output)
            return {"placed_regions": len(ocr["regions"]), "empty_regions": 0, "warnings": []}
        with patch("simple_drawing.workflow.prepare", side_effect=AssertionError("OCR")), \
                patch("simple_drawing.workflow.extract", side_effect=AssertionError("model")):
            code, folder = export_ocr_images(self.root, renderer=renderer)
        self.assertEqual(code, 0)
        self.assertEqual(received[0]["regions"][0]["text"], " 0001 ± 材質 ")
        self.assertEqual(received[0]["regions"][0]["confidence"], 0.0)
        self.assertEqual(received[1]["regions"], [])
        report = read_json(folder/"export.json")
        self.assertEqual(report["completed_pages"], 2)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["ocr_executed"])
        self.assertFalse(report["model_executed"])
        self.assertFalse(report["sdk_executed"])
        self.assertTrue(all(local_image.is_relative_to(folder) for local_image in folder.rglob("*.png")))
        for name, expected in original_hashes.items():
            self.assertEqual(sha(self.root/name), expected)

    def test_real_renderer_uses_white_page_and_keeps_pixel_and_mm_size(self):
        code, folder = export_ocr_images(self.root)
        self.assertEqual(code, 0)
        report = read_json(folder/"export.json")
        first, blank = [folder/row["image"] for row in report["records"]]
        with Image.open(first) as image:
            self.assertEqual(image.size, (320, 160))
            self.assertEqual(image.getpixel((310, 150)), (255, 255, 255))
            self.assertTrue(any(pixel == (0, 0, 0) for pixel in image.get_flattened_data()))
            self.assertAlmostEqual(image.info["dpi"][0], 254.0, places=1)
        with Image.open(blank) as image:
            self.assertEqual(image.getextrema(), ((255, 255), (255, 255), (255, 255)))

    def test_changed_cache_is_a_page_failure_and_other_pages_continue(self):
        folder = self.root/self.batch["documents"][0]["pages"][0]["cache"]
        (folder/"ocr.json").write_text("changed", encoding="utf-8")
        code, output = export_ocr_images(self.root)
        self.assertEqual(code, 2)
        report = read_json(output/"export.json")
        self.assertEqual((report["completed_pages"], report["failed_pages"]), (1, 1))
        self.assertEqual(report["records"][0]["status"], "failed")
        self.assertEqual(report["records"][1]["status"], "complete")

    def test_failed_preprocessing_and_document_remain_failures(self):
        self.batch["documents"][0]["pages"][0]["error"] = "synthetic OCR failure"
        self.batch["documents"].append({"name": "broken.pdf", "pages": [], "error": "bad PDF"})
        self.batch["status"] = "partial"
        self.save_batch()
        code, output = export_ocr_images(self.root)
        report = read_json(output/"export.json")
        self.assertEqual(code, 2)
        self.assertEqual(report["failed_pages"], 1)
        self.assertEqual(report["failed_documents"], 1)
        self.assertEqual(report["completed_pages"], 1)

    def test_new_export_does_not_overwrite_old_images(self):
        code, first = export_ocr_images(self.root)
        old_images = {relative(first, path): sha(path) for path in first.rglob("*.png")}
        code, second = export_ocr_images(self.root)
        self.assertEqual(code, 0)
        self.assertNotEqual(first, second)
        for path, expected in old_images.items():
            self.assertEqual(sha(first/path), expected)

    def test_interruption_is_recorded_and_completed_images_are_kept(self):
        def interrupted(ocr, output, **kwargs):
            if ocr["page"]["number"] == 2:
                raise KeyboardInterrupt()
            Image.new("RGB", (320, 160), "white").save(output)
            return {"placed_regions": 1}
        with self.assertRaises(KeyboardInterrupt):
            export_ocr_images(self.root, renderer=interrupted)
        folder = next((self.root/"data/ocr-images").iterdir())
        report = read_json(folder/"export.json")
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["completed_pages"], 1)
        self.assertEqual(len(list(folder.rglob("*.png"))), 1)

    def test_no_batch_gives_a_useful_message(self):
        (self.root/"data/latest_batch.json").unlink()
        with self.assertRaisesRegex(ValueError, "前処理.bat"):
            export_ocr_images(self.root)

    def test_export_code_changes_do_not_invalidate_recognition_contract(self):
        implementation = self.root/"source/simple_drawing"
        implementation.mkdir(parents=True)
        (implementation/"pdf_ocr.py").write_text("recognition fixture", encoding="utf-8")
        with patch("simple_drawing.workflow.importlib.metadata.version", return_value="fixture"):
            before = producer_contract(self.root)
            (implementation/"ocr_image.py").write_text("output only", encoding="utf-8")
            (implementation/"ocr_image_export.py").write_text("export only", encoding="utf-8")
            self.assertEqual(producer_contract(self.root), before)


if __name__ == "__main__":
    unittest.main()

"""CPU writer boundaries; fake SDKs never load DocuWorks or modify real data."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from simple_drawing.marker_excel import MarkerInputError, MarkerRow
from simple_drawing import xdw_marker as writer


class Mode(IntEnum):
    READONLY = 1
    UPDATE = 2


class Auth(IntEnum):
    NO_DIALOG = 1


class Kind(IntEnum):
    TEXT = 2
    MARKER = 3


class Colour(IntEnum):
    YELLOW = 123


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def rectangle(*, page=1, x=20.0, y=40.0, width=36.0, height=5.76, number=2):
    return MarkerRow(number, "drawing.xdw", page, x, y, width, height,
                     f"run:0:{page}:{number}", (f"p{page}:r000002",))


def fake_modules():
    package = ModuleType("docuworks_ctypes")
    package.__path__ = []
    package.XdwApi = SimpleNamespace(load=lambda: (_ for _ in ()).throw(
        AssertionError("The real SDK loader must not be called")))
    enums = ModuleType("docuworks_ctypes.enums")
    enums.OpenMode, enums.AuthMode = Mode, Auth
    enums.Color, enums.AnnotationType = Colour, Kind
    geometry = ModuleType("docuworks_ctypes.geometry")
    geometry.PointMM = Point
    return {"docuworks_ctypes": package, "docuworks_ctypes.enums": enums,
            "docuworks_ctypes.geometry": geometry}


class FakeAnnotation:
    def __init__(self, kind, *, points=(), width=0, color=Colour.YELLOW, transparent=True):
        self.annotation_type = kind
        self.values = {"%Points": tuple(points), "%BorderWidth": width,
                       "%BorderColor": color, "%BorderTransparent": transparent}

    def get_standard_attribute(self, name):
        return self.values[name]


class FakePage:
    def __init__(self, api, number):
        self.api, self.number = api, number

    def annotations(self, recursive=False):
        if recursive:
            raise AssertionError("This fake models top-level annotation counts")
        return iter(self.api.annotations[self.number])

    def add_marker(self, points, *, border_color, border_width, border_transparent):
        self.api.annotations[self.number].append(FakeAnnotation(
            Kind.MARKER, points=points, width=border_width, color=border_color,
            transparent=border_transparent))


class FakeDocument:
    page_count = 2

    def __init__(self, api, path, mode):
        self.api, self.path, self.mode = api, Path(path), mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.api.closed.append(self.mode)
        if self.api.failure == "close" and self.mode == Mode.UPDATE:
            raise RuntimeError("native close failed")
        return False

    def page(self, number):
        return FakePage(self.api, number)

    def save(self):
        if self.mode != Mode.UPDATE:
            raise AssertionError("Save must use update mode on the copy")
        if self.api.failure == "save":
            raise RuntimeError("native save failed")
        self.path.write_bytes(b"saved marked copy")
        self.api.saved = True


class FakeApi:
    runtime_info = SimpleNamespace(version_text="10.1.1")

    def __init__(self, target, excel, failure=None):
        self.target, self.excel, self.failure = target, excel, failure
        self.annotations = {1: [FakeAnnotation(Kind.TEXT)], 2: []}
        self.opened, self.closed, self.saved = [], [], False

    def open_document(self, path, *, mode, auth):
        if Path(path).resolve() == self.target.resolve():
            raise AssertionError("The original document must never be opened for SDK updates")
        if auth != Auth.NO_DIALOG:
            raise AssertionError("No authentication dialog is allowed")
        self.opened.append((Path(path), mode))
        if len(self.opened) == 3:
            if self.failure == "reopen":
                raise RuntimeError("native reopen failed")
            if self.failure == "verify":
                self.annotations[1][-1].values["%BorderColor"] = -1
            if self.failure == "target-change":
                self.target.write_bytes(b"concurrent external XDW edit")
            if self.failure == "excel-change":
                self.excel.write_bytes(b"concurrent external Excel edit")
            if self.failure == "target-delete":
                self.target.unlink()
            if self.failure == "annotations":
                self.annotations[2].append(FakeAnnotation(Kind.TEXT))
        return FakeDocument(self, path, mode)


class MarkerPlanningTests(unittest.TestCase):
    def test_tall_rectangle_is_covered_by_viewer_width_strokes_without_gaps(self):
        row = rectangle(height=30.0)
        strokes = writer.plan([row], {1: {"width_mm": 210.0, "height_mm": 297.0}})
        self.assertGreater(len(strokes), 1)
        spans = []
        for stroke in strokes:
            self.assertTrue(1 <= stroke.width_pt <= 13)
            self.assertEqual(stroke.points[0][1], stroke.points[1][1])
            self.assertLess(stroke.points[0][0], stroke.points[1][0])
            radius = stroke.width_pt * writer.MM_PER_PT / 2
            spans.append((stroke.points[0][1] - radius, stroke.points[0][1] + radius))
        self.assertLessEqual(spans[0][0], row.y_mm + 0.02)
        self.assertGreaterEqual(spans[-1][1], row.y_mm + row.height_mm - 0.02)
        self.assertTrue(all(a[1] >= b[0] - 0.02 for a, b in zip(spans, spans[1:])))

    def test_missing_page_and_out_of_page_rectangle_fail_before_sdk_write(self):
        dimensions = {1: {"width_mm": 100.0, "height_mm": 100.0}}
        with self.subTest("missing page"), self.assertRaises(MarkerInputError):
            writer.plan([rectangle(page=3)], dimensions)
        with self.subTest("outside page"), self.assertRaises(MarkerInputError):
            writer.plan([rectangle(x=90.0, width=20.0)], dimensions)
        with self.subTest("empty selection"), self.assertRaises(MarkerInputError):
            writer.plan([], dimensions)

    def test_marker_limit_rejects_large_plan(self):
        with patch.object(writer, "MAX_STROKES", 1), self.assertRaises(MarkerInputError):
            writer.plan([rectangle(height=20.0)], {1: {"width_mm": 210.0, "height_mm": 297.0}})


class MarkerWriteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="marker-writer-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.target = self.root / "drawing.xdw"
        self.excel = self.root / "results.xlsx"
        self.target.write_bytes(b"immutable original XDW")
        self.excel.write_bytes(b"immutable original Excel")
        self.target_hash, self.excel_hash = writer.sha(self.target), writer.sha(self.excel)
        self.patchers = [patch.dict("sys.modules", fake_modules()),
                         patch.object(writer, "version", return_value="1.0.1"),
                         patch.object(writer, "validate_saved_sources", return_value={
                             "saved_source_verified_rows": [2], "standalone_rows": []}),
                         patch.object(writer, "page_metadata", side_effect=lambda doc, n: {
                             "width_mm": 210.0, "height_mm": 296.0 if (
                                 doc.api.failure == "dimensions" and len(doc.api.opened) == 3) else 297.0,
                             "source_rotation": 90 if n == 1 else 0, "page_type": 1})]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def call(self, failure=None):
        api = FakeApi(self.target, self.excel, failure)
        result = writer.apply_markers(self.root, self.excel, self.target, [rectangle()],
                                      document="drawing.xdw", api_factory=lambda: api,
                                      progress=lambda message: None)
        return api, result

    def receipt(self):
        paths = list((self.root / "data" / "markers").glob("*/マーカー処理結果.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def assert_no_published_xdw(self):
        self.assertEqual(list((self.root / "data" / "markers").glob("*/*.xdw")), [])

    def assert_originals_unchanged(self):
        self.assertEqual(writer.sha(self.target), self.target_hash)
        self.assertEqual(writer.sha(self.excel), self.excel_hash)

    def test_success_verifies_saved_copy_and_preserves_originals_and_existing_annotation(self):
        api, (output, receipt) = self.call()
        self.assertTrue(api.saved)
        self.assertEqual([mode for _, mode in api.opened], [Mode.READONLY, Mode.UPDATE, Mode.READONLY])
        self.assertEqual(api.closed, [Mode.READONLY, Mode.UPDATE, Mode.READONLY])
        self.assertEqual(api.annotations[1][0].annotation_type, Kind.TEXT)
        self.assertEqual(output.read_bytes(), b"saved marked copy")
        report = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["pages"]["1"]["source_rotation"], 90)
        self.assertEqual(report["output_sha256"], writer.sha(output))
        self.assertEqual(len(report["verified_markers"]), report["markers"])
        self.assertTrue(report["original_preserved"] and report["excel_preserved"])
        self.assert_originals_unchanged()

    def test_save_close_reopen_and_saved_marker_verification_fail_without_publishing(self):
        for failure in ("save", "close", "reopen", "verify", "dimensions", "annotations"):
            with self.subTest(failure=failure):
                # Each failure gets a fresh task-owned output folder and originals.
                with tempfile.TemporaryDirectory(prefix="marker-failure-") as folder:
                    root = Path(folder)
                    target, excel = root / "drawing.xdw", root / "results.xlsx"
                    target.write_bytes(b"original XDW")
                    excel.write_bytes(b"original Excel")
                    hashes = writer.sha(target), writer.sha(excel)
                    api = FakeApi(target, excel, failure)
                    with self.assertRaises(RuntimeError):
                        writer.apply_markers(root, excel, target, [rectangle()], document="drawing.xdw",
                                             api_factory=lambda: api, progress=lambda message: None)
                    reports = list((root / "data" / "markers").glob("*/マーカー処理結果.json"))
                    self.assertEqual(len(reports), 1)
                    report = json.loads(reports[0].read_text(encoding="utf-8"))
                    self.assertEqual(report["status"], "failed")
                    self.assertTrue(report["original_preserved"] and report["excel_preserved"])
                    self.assertEqual((writer.sha(target), writer.sha(excel)), hashes)
                    self.assertEqual(list((root / "data" / "markers").glob("*/*.xdw")), [])

    def test_publish_failure_keeps_originals_and_records_failed_receipt(self):
        with patch.object(writer.os, "rename", side_effect=PermissionError("publish blocked")):
            with self.assertRaises(PermissionError):
                self.call()
        self.assert_no_published_xdw()
        report = self.receipt()
        self.assertEqual(report["status"], "failed")
        self.assertIn("publish blocked", report["error"])
        self.assert_originals_unchanged()

    def test_concurrent_original_edit_blocks_publish_and_reports_failed_integrity(self):
        with self.assertRaisesRegex(RuntimeError, "変更"):
            self.call("target-change")
        self.assert_no_published_xdw()
        report = self.receipt()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["original_preserved"])
        self.assertTrue(report["excel_preserved"])
        self.assertEqual(writer.sha(self.excel), self.excel_hash)

    def test_concurrent_excel_edit_blocks_publish_and_reports_failed_integrity(self):
        with self.assertRaisesRegex(RuntimeError, "変更"):
            self.call("excel-change")
        self.assert_no_published_xdw()
        report = self.receipt()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["original_preserved"])
        self.assertFalse(report["excel_preserved"])
        self.assertEqual(writer.sha(self.target), self.target_hash)

    def test_deleted_original_keeps_failed_receipt_and_does_not_mask_error(self):
        with self.assertRaises(FileNotFoundError):
            self.call("target-delete")
        self.assert_no_published_xdw()
        report = self.receipt()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["original_preserved"])
        self.assertTrue(report["excel_preserved"])

    def test_receipt_failure_after_publish_preserves_verified_output_and_originals(self):
        original_write = Path.write_text
        def write_receipt(path, *args, **kwargs):
            if path.name == "マーカー処理結果.json" and list(path.parent.glob("*.xdw")):
                raise OSError("receipt publish blocked")
            return original_write(path, *args, **kwargs)
        with patch.object(Path, "write_text", new=write_receipt):
            with self.assertRaisesRegex(OSError, "receipt publish blocked") as caught:
                self.call()
        self.assertTrue(any("JSON" in note for note in getattr(caught.exception, "__notes__", [])))
        outputs = list((self.root / "data" / "markers").glob("*/*.xdw"))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].read_bytes(), b"saved marked copy")
        self.assert_originals_unchanged()


class MarkerProvenanceTests(unittest.TestCase):
    """Frozen local run fixtures with small JSON files and mocked helper imports."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="marker-provenance-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "data/runs/known-run"
        self.run.mkdir(parents=True)
        self.source = self.root / "data/sources/original.xdw"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"saved original document")
        self.target_hash = writer.sha(self.source)
        self.dimensions = {1: {"width_mm": 210.0, "height_mm": 297.0}}
        self.row = MarkerRow(2, "drawing.xdw", 1, 20.0, 40.0, 36.0, 5.76,
                             "known-run:0:1:1", ("p1:r000002",))
        self.payload = {"ocr_regions": [{"id": "p1:r000002", "text": "A-101",
                         "bbox_mm": {"x": 20.0, "y": 40.0, "width": 36.0, "height": 5.76}}]}
        self.write(self.run / "payload.json", self.payload)
        self.record = {"parsed": {"results": [{"value": "A-101", "source_ids": ["p1:r000002"]}]},
                       "input_payload": "payload.json", "payload_sha256": writer.sha(self.run / "payload.json")}
        self.write(self.run / "record.json", self.record)
        self.batch_path = self.root / "data/batch.json"
        self.batch = {"documents": [{"name": "drawing.xdw", "source_type": "xdw",
                      "source": "data/sources/original.xdw", "sha256": self.target_hash,
                      "pages": [{"number": 1, "width_mm": 210.0, "height_mm": 297.0}]}]}
        self.write(self.batch_path, self.batch)
        self.manifest = {"status": "complete", "contract": {"batch_path": "data/batch.json",
                         "batch_sha256": writer.sha(self.batch_path)}, "records": {
                         "0:1": {"path": "record.json", "sha256": writer.sha(self.run / "record.json")}}}
        self.write(self.run / "run.json", self.manifest)

        storage = ModuleType("simple_drawing.storage")
        def local_path(root, reference):
            path = (Path(root) / reference).resolve()
            if not path.is_relative_to(Path(root).resolve()):
                raise ValueError("Path escaped fixture root")
            return path
        storage.local_path = local_path
        storage.read_json = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
        excel = ModuleType("simple_drawing.excel")
        def payload_regions(folder, record):
            path = local_path(folder, record["input_payload"])
            if writer.sha(path) != record["payload_sha256"]:
                raise ValueError("Frozen payload hash differs")
            return {region["id"]: region for region in storage.read_json(path)["ocr_regions"]}
        excel._payload_regions = payload_regions
        excel._box = lambda region: [region["bbox_mm"][key] for key in ("x", "y", "width", "height")]
        patcher = patch.dict("sys.modules", {"simple_drawing.storage": storage, "simple_drawing.excel": excel})
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def validate(self, row=None, *, target_hash=None):
        return writer.validate_saved_sources(self.root, [row or self.row], self.dimensions,
                                             target_hash or self.target_hash)

    def test_frozen_xdw_source_and_result_payload_are_verified(self):
        self.assertEqual(self.validate(), {"saved_source_verified_rows": [2], "standalone_rows": []})

    def test_different_selected_xdw_and_tampered_saved_source_are_rejected(self):
        with self.subTest("selected XDW"), self.assertRaises(MarkerInputError):
            self.validate(target_hash="0" * 64)
        self.source.write_bytes(b"tampered source")
        with self.subTest("saved XDW"), self.assertRaises(MarkerInputError):
            self.validate()

    def test_tampered_batch_record_and_payload_are_rejected(self):
        for name in ("batch", "record", "payload"):
            with self.subTest(name=name):
                path = self.batch_path if name == "batch" else self.run / (name + ".json")
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b" ")
                    with self.assertRaises(MarkerInputError):
                        self.validate()
                finally:
                    path.write_bytes(original)

    def test_changed_coordinates_or_evidence_ids_or_document_are_rejected(self):
        from dataclasses import replace
        for row in (replace(self.row, x_mm=21.0), replace(self.row, evidence_ids=("p1:wrong",)),
                    replace(self.row, document="different.xdw"), replace(self.row, result_id="known-run:0:2:1")):
            with self.subTest(row=row), self.assertRaises(MarkerInputError):
                self.validate(row)

    def test_absent_local_run_is_recorded_as_standalone_instead_of_claiming_provenance(self):
        from dataclasses import replace
        row = replace(self.row, result_id="external-run:0:1:1")
        self.assertEqual(self.validate(row), {"saved_source_verified_rows": [], "standalone_rows": [2]})


if __name__ == "__main__":
    unittest.main()

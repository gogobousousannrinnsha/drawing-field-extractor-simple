import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

from simple_drawing.excel import excel_rows, write_excel, reexport_excel, HEADERS, EVIDENCE_HEADERS
from simple_drawing.storage import read_json, write_json, sha, relative

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def cells(path, sheet=1):
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(si.itertext()) for si in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        xml = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml"))
        result = {}
        for cell in xml.findall(".//m:sheetData/m:row/m:c", NS):
            value = cell.find("m:v", NS)
            kind = cell.get("t", "n")
            result[cell.get("r")] = (None if value is None else strings[int(value.text)] if kind == "s" else float(value.text), kind)
        return result


class ExcelCoordinateTests(unittest.TestCase):
    def setUp(self):
        artifacts = Path(__file__).resolve().parent.parent / "test-artifacts"
        artifacts.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=artifacts)
        self.root = Path(self.temp.name)
        self.run = self.root / "data/runs/20261010-coordinate-test"
        self.run.mkdir(parents=True)
        self.settings = {"item": "部品番号"}
        self.batch = {"documents": [{"name": "日本語 図面.pdf", "pages": [{"number": 1}]}]}
        self.regions = [
            {"id": "p1:r1", "text": "0001", "bbox_mm": {"x": 0.0, "y": 2.123456789, "width": 3.5, "height": 4.25}},
            {"id": "p1:r2", "text": "別の根拠", "bbox_mm": {"x": 30, "y": 1, "width": 2, "height": 1}}]
        self.payload = self.run / "inputs/0-1/payload.json"
        write_json(self.payload, {"ocr_regions": self.regions})
        self.record = {"parsed": {"status": "found", "results": [{"value": "0001", "source_ids": ["p1:r1"]}], "note": ""},
                       "input_payload": relative(self.run, self.payload), "payload_sha256": sha(self.payload)}
    def tearDown(self):
        self.temp.cleanup()
    def rows(self):
        return list(excel_rows(self.batch, {"0:1": self.record}, self.settings, run_folder=self.run))
    def complete_run(self):
        batch_path = self.root / "data/batches/fixture/batch.json"
        write_json(batch_path, self.batch)
        page_path = self.run / "pages/0-1.json"
        write_json(page_path, self.record)
        manifest = {"status": "complete", "contract": {"batch_path": relative(self.root, batch_path),
            "batch_sha256": sha(batch_path), "settings": self.settings},
            "records": {"0:1": {"path": relative(self.run, page_path), "sha256": sha(page_path)}}}
        write_json(self.run / "run.json", manifest)
        (self.run / "抽出結果.xlsx").write_bytes(b"human corrected original")
        return manifest

    def test_single_numeric_coordinates_zero_and_precision(self):
        row = self.rows()[0]
        self.assertEqual(row.values[:5], ["日本語 図面.pdf", "1", "部品番号", "0001", ""])
        self.assertEqual(row.values[5:9], [0.0, 2.123456789, 3.5, 4.25])
        output = self.root / "coords.xlsx"
        write_excel(output, [row])
        main, detail = cells(output), cells(output, 2)
        self.assertEqual(main['D2'], ('0001', 's'))
        for column, expected in zip('FGHI', row.values[5:9]):
            self.assertEqual(main[column+'2'], (expected, 'n'))
            self.assertEqual(detail[column+'2'], (expected, 'n'))
        self.assertEqual(main['J2'][0], '["p1:r1"]')
        self.assertEqual(main['K2'][0], detail['A2'][0])
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(ET.fromstring(archive.read('xl/tables/table1.xml')).get('name'), 'Results')
            self.assertEqual(ET.fromstring(archive.read('xl/tables/table2.xml')).get('name'), 'EvidenceCoordinates')

    def test_multiple_boxes_and_evidence_order(self):
        self.record['parsed']['results'][0]['source_ids'] = ['p1:r2', 'p1:r1']
        row = self.rows()[0]
        self.assertEqual(row.values[5:8], [0.0, 1, 32.0])
        self.assertAlmostEqual(row.values[8], 5.373456789, places=12)
        self.assertEqual([r[3] for r in row.evidence], ['p1:r2', 'p1:r1'])
        self.assertEqual([r[4] for r in row.evidence], ['別の根拠', '0001'])

    def test_image_only_has_blank_coordinates_and_preserved_note(self):
        self.record['parsed']['results'][0]['source_ids'] = []
        self.record['parsed']['note'] = '画像から読み取った。'
        row = self.rows()[0]
        self.assertEqual(row.values[5:9], [None] * 4)
        self.assertEqual(row.values[4], '画像から読み取った。')
        self.assertEqual(row.evidence, [])
        output = self.root / 'image-only.xlsx'
        write_excel(output, [row])
        self.assertTrue(all(cells(output).get(c+'2', (None,))[0] is None for c in 'FGHI'))

    def test_empty_statuses_page_and_document_failures_are_blank(self):
        for status in ('not_found', 'unreadable', 'ambiguous'):
            self.record['parsed'] = {'status': status, 'results': [], 'note': '補足'}
            row = self.rows()[0]
            self.assertEqual(row.values[5:9], [None] * 4)
            self.assertEqual(row.evidence, [])
        self.record = {'error': {'message': 'capacity exceeded'}, 'parsed': None}
        self.assertIn('処理失敗', self.rows()[0].values[4])
        self.assertEqual(self.rows()[0].values[5:9], [None] * 4)
        self.batch['documents'][0]['error'] = 'broken PDF'
        self.assertEqual(self.rows()[0].values[5:9], [None] * 4)

    def test_duplicate_values_aliases_have_distinct_result_ids(self):
        self.record['parsed']['results'] *= 2
        self.batch['documents'].append({'name': '別名.pdf', 'pages': [{'number': 1}]})
        rows = list(excel_rows(self.batch, {'0:1': self.record, '1:1': self.record}, self.settings, run_folder=self.run))
        self.assertEqual([row.values[3] for row in rows], ['0001'] * 4)
        self.assertEqual(len({row.values[10] for row in rows}), 4)
        self.assertEqual(len(rows), 4)

    def test_payload_corruption_missing_id_and_nonfinite_are_rejected(self):
        self.payload.write_text('{}')
        with self.assertRaisesRegex(ValueError, '破損'): self.rows()
        write_json(self.payload, {'ocr_regions': self.regions})
        self.record['payload_sha256'] = sha(self.payload)
        self.record['parsed']['results'][0]['source_ids'] = ['other-page']
        with self.assertRaisesRegex(ValueError, '当該ページ'): self.rows()
        self.record['parsed']['results'][0]['source_ids'] = ['p1:r1']
        self.regions[0]['bbox_mm']['x'] = -1
        write_json(self.payload, {'ocr_regions': self.regions})
        self.record['payload_sha256'] = sha(self.payload)
        with self.assertRaisesRegex(ValueError, '座標'): self.rows()
        with self.assertRaisesRegex(ValueError, '有限'):
            write_excel(self.root / 'nan.xlsx', [['f', '1', 'i', 'v', '', float('nan')]])
        self.assertFalse((self.root / 'nan.xlsx').exists())

    def test_reexport_uses_snapshot_and_keeps_original_and_current_state(self):
        self.complete_run()
        write_json(self.root / '抽出設定.json', {'item': 'changed current setting'})
        write_json(self.root / 'data/latest_batch.json', {'path': 'missing'})
        before = {p: sha(p) for p in self.run.rglob('*') if p.is_file()}
        current = sha(self.root / '抽出設定.json')
        with patch('simple_drawing.llm.OllamaServer', side_effect=AssertionError('model must not run')), \
             patch('simple_drawing.pdf_ocr.OCRSession', side_effect=AssertionError('OCR must not run')):
            code, output = reexport_excel(self.root)
            _, second = reexport_excel(self.root)
        self.assertEqual(code, 0)
        self.assertNotEqual(output, second)
        self.assertEqual(cells(output)['C2'][0], '部品番号')
        self.assertEqual({p: sha(p) for p in before}, before)
        self.assertEqual(sha(self.root / '抽出設定.json'), current)
        report = read_json(output.parent / 'export.json')
        self.assertFalse(report['model_executed'])
        self.assertFalse(report['ocr_executed'])
        self.assertEqual(report['source_run'], relative(self.root, self.run))

    def test_reexport_skips_interrupted_and_rejects_explicit_interrupted(self):
        self.complete_run()
        newest = self.root / 'data/runs/999-interrupted'
        write_json(newest / 'run.json', {'status': 'interrupted'})
        _, output = reexport_excel(self.root)
        self.assertEqual(read_json(output.parent / 'export.json')['source_run'], relative(self.root, self.run))
        with self.assertRaisesRegex(ValueError, '未完了'):
            reexport_excel(self.root, relative(self.root, newest))

    def test_reexport_rejects_record_batch_and_payload_corruption(self):
        manifest = self.complete_run()
        page = self.run / 'pages/0-1.json'
        original = page.read_bytes()
        page.write_text('{}')
        with self.assertRaisesRegex(ValueError, '抽出結果'): reexport_excel(self.root)
        page.write_bytes(original)
        batch_path = self.root / manifest['contract']['batch_path']
        original = batch_path.read_bytes()
        batch_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, '前処理記録'): reexport_excel(self.root)
        batch_path.write_bytes(original)
        self.payload.write_text('{}')
        with self.assertRaisesRegex(ValueError, '抽出入力'): reexport_excel(self.root)
        self.assertFalse(any((self.root / 'data/exports').rglob('*.xlsx')))

    def test_no_completed_run_reports_reason_and_rejects_escape(self):
        with self.assertRaisesRegex(ValueError, '完了済み'): reexport_excel(self.root)
        with self.assertRaises(ValueError): reexport_excel(self.root, '../escape')

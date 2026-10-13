"""Independent XLSX XML fixtures; no Excel, SDK or third-party test modules."""
from dataclasses import dataclass, FrozenInstanceError
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
from xml.etree import ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simple_drawing import marker_excel as reader

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
MAIN = ["原本ファイル", "ページ", "項目", "値", "備考", "x_mm", "y_mm", "width_mm", "height_mm", "根拠ID", "結果ID"]
DETAIL = ["結果ID", "原本ファイル", "ページ", "根拠ID", "OCR原文", "x_mm", "y_mm", "width_mm", "height_mm"]


@dataclass
class Formula:
    cached: str = "1"


@dataclass
class Typed:
    kind: str
    value: str


def result(document="日本語 図面.xdw", page="1", box=(10.0, 20.0, 30.0, 40.0), ident="run:0:1:1", evidence=("p1:r1",)):
    return [document, page, "部品番号", "000123", "", *box, json.dumps(list(evidence)), ident]


def detail(document="日本語 図面.xdw", page="1", box=(10.0, 20.0, 30.0, 40.0), ident="run:0:1:1", evidence="p1:r1"):
    return [ident, document, page, evidence, "000123", *box]


def column_name(number):
    name = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xml_bytes(element):
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def fixture_parts(main_rows, detail_rows=None, *, headers=None, shared=False, hidden=()):
    strings, indices = [], {}

    def sheet(rows):
        root = ET.Element(f"{{{NS}}}worksheet")
        data = ET.SubElement(root, f"{{{NS}}}sheetData")
        for row_number, values in enumerate(rows, 1):
            attrs = {"r": str(row_number)}
            if row_number in hidden:
                attrs["hidden"] = "1"
            row = ET.SubElement(data, f"{{{NS}}}row", attrs)
            for number, value in enumerate(values, 1):
                if value is None:
                    continue
                cell = ET.SubElement(row, f"{{{NS}}}c", {"r": f"{column_name(number)}{row_number}"})
                if isinstance(value, Formula):
                    ET.SubElement(cell, f"{{{NS}}}f").text = "1+0"
                    ET.SubElement(cell, f"{{{NS}}}v").text = value.cached
                elif isinstance(value, Typed):
                    cell.set("t", value.kind)
                    ET.SubElement(cell, f"{{{NS}}}v").text = value.value
                elif isinstance(value, (float, int)):
                    ET.SubElement(cell, f"{{{NS}}}v").text = str(value)
                elif shared:
                    value = str(value)
                    if value not in indices:
                        indices[value] = len(strings)
                        strings.append(value)
                    cell.set("t", "s")
                    ET.SubElement(cell, f"{{{NS}}}v").text = str(indices[value])
                else:
                    cell.set("t", "inlineStr")
                    inline = ET.SubElement(cell, f"{{{NS}}}is")
                    ET.SubElement(inline, f"{{{NS}}}t").text = str(value)
        ET.SubElement(root, f"{{{NS}}}autoFilter", {"ref": "A1:K999"})
        return xml_bytes(root)

    parts = {"xl/worksheets/sheet1.xml": sheet([headers or MAIN, *main_rows])}
    names = ["未確認結果"]
    if detail_rows is not None:
        parts["xl/worksheets/sheet2.xml"] = sheet([DETAIL, *detail_rows])
        names.append("根拠座標")
    workbook = ET.Element(f"{{{NS}}}workbook")
    sheets = ET.SubElement(workbook, f"{{{NS}}}sheets")
    relationships = ET.Element(f"{{{PKG}}}Relationships")
    for number, name in enumerate(names, 1):
        ET.SubElement(sheets, f"{{{NS}}}sheet", {"name": name, "sheetId": str(number), f"{{{REL}}}id": f"rId{number}"})
        ET.SubElement(relationships, f"{{{PKG}}}Relationship", {"Id": f"rId{number}", "Type": REL + "/worksheet", "Target": f"worksheets/sheet{number}.xml"})
    content = ET.Element(f"{{{CT}}}Types")
    ET.SubElement(content, f"{{{CT}}}Override", {"PartName": "/xl/workbook.xml", "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"})
    parts.update({"[Content_Types].xml": xml_bytes(content), "xl/workbook.xml": xml_bytes(workbook), "xl/_rels/workbook.xml.rels": xml_bytes(relationships)})
    if shared:
        table = ET.Element(f"{{{NS}}}sst")
        for value in strings:
            item = ET.SubElement(table, f"{{{NS}}}si")
            ET.SubElement(item, f"{{{NS}}}t").text = value
        parts["xl/sharedStrings.xml"] = xml_bytes(table)
    return parts


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "抽出結果_座標付き.xlsx"

    def write(self, rows=None, detail_rows=None, *, parts=None, **options):
        parts = parts if parts is not None else fixture_parts([result()] if rows is None else rows, detail_rows, **options)
        with zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in parts.items():
                archive.writestr(name, value)
        return self.path

    def read(self, rows=None, detail_rows=None, **options):
        return reader.read_workbook(self.write(rows, detail_rows, **options))

    def test_main_only_exact_selection_and_immutable_rows(self):
        workbook = self.read([result("z.xdw"), result("a.pdf", ident="other"), result("日本語 図面.xdw", ident="jp")])
        self.assertEqual(workbook.documents, ("a.pdf", "z.xdw", "日本語 図面.xdw"))
        rows = workbook.select("日本語 図面.xdw")
        self.assertEqual(rows, [reader.MarkerRow(4, "日本語 図面.xdw", 1, 10, 20, 30, 40, "jp", ("p1:r1",))])
        with self.assertRaises(FrozenInstanceError):
            rows[0].page = 2
        rows.clear()
        self.assertEqual(len(workbook.select("日本語 図面.xdw")), 1)
        with self.assertRaises(reader.MarkerInputError):
            workbook.select("図面.xdw")
        with self.assertRaises(reader.MarkerInputError):
            workbook.skipped_no_coordinates("Z.xdw")

    def test_blank_coordinates_skipped_including_document_failure(self):
        blank = result(box=(None, None, None, None), ident="blank", evidence=())
        failed = result(page="", box=(None, None, None, None), ident="failed", evidence=())
        workbook = self.read([result(), blank, failed])
        self.assertEqual(len(workbook.select("日本語 図面.xdw")), 1)
        self.assertEqual(workbook.skipped_no_coordinates("日本語 図面.xdw"), 2)

    def test_all_hidden_filtered_rows_are_selected(self):
        workbook = self.read([result(), result(page="2", ident="second", evidence=("p2:r1",))], hidden=(2, 3))
        self.assertEqual([row.page for row in workbook.select("日本語 図面.xdw")], [1, 2])

    def test_shared_strings_numeric_pages_and_origin(self):
        workbook = self.read([result(page=1, box=(0, 0, 3, 4))], shared=True)
        row = workbook.select("日本語 図面.xdw")[0]
        self.assertEqual((row.page, row.x_mm, row.y_mm), (1, 0, 0))

    def test_reordered_headers_and_extra_column(self):
        order = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        headers = [MAIN[index] for index in order] + ["利用者メモ"]
        values = [result()[index] for index in order] + ["表示だけ"]
        row = self.read([values], headers=headers).select("日本語 図面.xdw")[0]
        self.assertEqual((row.x_mm, row.height_mm, row.result_id), (10, 40, "run:0:1:1"))

    def test_empty_workbook_and_document_without_coordinates(self):
        self.assertEqual(self.read([]).documents, ())
        workbook = self.read([result(box=(None,) * 4, evidence=())])
        self.assertEqual(workbook.select("日本語 図面.xdw"), [])
        self.assertEqual(workbook.skipped_no_coordinates("日本語 図面.xdw"), 1)

    def test_main_only_manual_coordinates_without_evidence_ids(self):
        self.assertEqual(len(self.read([result(evidence=())]).select("日本語 図面.xdw")), 1)

    def test_blank_and_partial_coordinate_cells(self):
        for box in ((1, None, 3, 4), (None, 2, None, None), (1, " ", 3, 4)):
            with self.subTest(box=box), self.assertRaises(reader.MarkerInputError):
                self.read([result(box=box)])

    def test_nonfinite_negative_zero_and_oversized_coordinates(self):
        cases = ((float("nan"), 0, 1, 1), (0, float("inf"), 1, 1), (-1, 0, 1, 1),
                 (0, -1, 1, 1), (0, 0, 0, 1), (0, 0, 1, -1), (0, 0, 1e20, 1), ("text", 0, 1, 1))
        for box in cases:
            with self.subTest(box=box), self.assertRaises(reader.MarkerInputError):
                self.read([result(box=box)])

    def test_invalid_pages(self):
        for page in ("", "0", "-1", "1.5", "1.0000000000000001", "NaN", "inf", "1000001", "one"):
            with self.subTest(page=page), self.assertRaises(reader.MarkerInputError):
                self.read([result(page=page)])

    def test_invalid_sources_and_empty_result_id(self):
        for document in ("", " ", "../sample.xdw", "C:" + chr(92) + "sample.xdw", "folder/sample.xdw", "invalid?.xdw", ".", ".."):
            with self.subTest(document=document), self.assertRaises(reader.MarkerInputError):
                self.read([result(document)])
        with self.assertRaises(reader.MarkerInputError):
            self.read([result(ident=" ")])

    def test_formulas_rejected_for_every_used_contract_column(self):
        for column in (0, 1, 5, 6, 7, 8, 9, 10):
            values = result()
            values[column] = Formula("1")
            with self.subTest(column=column), self.assertRaises(reader.MarkerInputError):
                self.read([values])

    def test_irrelevant_value_formula_does_not_control_plan(self):
        values = result()
        values[3] = Formula("999")
        row = self.read([values]).select("日本語 図面.xdw")[0]
        self.assertEqual(row.x_mm, 10)

    def test_boolean_error_and_date_used_cells_rejected(self):
        for kind, value in (("b", "1"), ("e", "#REF!"), ("d", "2026-10-13")):
            values = result()
            values[5] = Typed(kind, value)
            with self.subTest(kind=kind), self.assertRaises(reader.MarkerInputError):
                self.read([values])

    def test_duplicate_selected_result_ids_only(self):
        workbook = self.read([result(), result(ident="run:0:1:1")])
        with self.assertRaises(reader.MarkerInputError):
            workbook.select("日本語 図面.xdw")
        workbook = self.read([result("a.xdw"), result("b.xdw")])
        self.assertEqual(len(workbook.select("a.xdw")), 1)

    def test_invalid_evidence_json(self):
        for value in ("not-json", "{}", '[1]', '[""]', '["id","id"]', '[["id"]]'):
            values = result()
            values[9] = value
            with self.subTest(value=value), self.assertRaises(reader.MarkerInputError):
                self.read([values])

    def test_duplicate_missing_and_formula_headers(self):
        for headers in (MAIN + ["x_mm"], MAIN[:-1], [Formula(), *MAIN[1:]]):
            with self.subTest(headers=headers), self.assertRaises(reader.MarkerInputError):
                self.read(headers=headers)

    def test_evidence_union_validation(self):
        details = [detail(box=(10, 20, 10, 10), evidence="first"), detail(box=(20, 25, 20, 35), evidence="second")]
        workbook = self.read([result(evidence=("first", "second"))], details)
        self.assertEqual(len(workbook.select("日本語 図面.xdw")), 1)

    def test_evidence_union_tolerance(self):
        accepted = self.read([result(box=(10.05, 20, 30, 40))], [detail()])
        self.assertEqual(len(accepted.select("日本語 図面.xdw")), 1)
        rejected = self.read([result(box=(10.051, 20, 30, 40))], [detail()])
        with self.assertRaises(reader.MarkerInputError):
            rejected.select("日本語 図面.xdw")

    def test_evidence_wrong_document_page_missing_extra_and_duplicate(self):
        cases = ([detail(document="other.xdw")], [detail(page="2")], [],
                 [detail(evidence="wrong")], [detail(), detail(evidence="extra")], [detail(), detail()])
        for details in cases:
            with self.subTest(details=details):
                workbook = self.read(detail_rows=details)
                with self.assertRaises(reader.MarkerInputError):
                    workbook.select("日本語 図面.xdw")

    def test_evidence_sheet_without_main_ids_cannot_establish_box(self):
        workbook = self.read([result(evidence=())], [])
        with self.assertRaises(reader.MarkerInputError):
            workbook.select("日本語 図面.xdw")

    def test_unselected_evidence_consistency_does_not_block_selected_document(self):
        rows = [result("a.xdw", ident="a"), result("b.xdw", ident="b")]
        details = [detail("a.xdw", ident="a"), detail("b.xdw", box=(0, 0, 1, 1), ident="b")]
        workbook = self.read(rows, details)
        self.assertEqual(len(workbook.select("a.xdw")), 1)
        with self.assertRaises(reader.MarkerInputError):
            workbook.select("b.xdw")

    def test_evidence_partial_and_formula_coordinates_rejected(self):
        for values in (detail(box=(10, None, 30, 40)), detail(box=(Formula(), 20, 30, 40))):
            with self.subTest(values=values), self.assertRaises(reader.MarkerInputError):
                self.read(detail_rows=[values])

    def test_zip_invalid_duplicate_and_traversal_members(self):
        self.path.write_bytes(b"not a zip")
        with self.assertRaises(reader.MarkerInputError):
            reader.read_workbook(self.path)
        self.write()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "a") as archive:
                archive.writestr("xl/workbook.xml", b"duplicate")
        with self.assertRaises(reader.MarkerInputError):
            reader.read_workbook(self.path)
        for name in ("../outside", "/absolute", "xl\\workbook.xml", "xl/./sheet.xml", "xl//sheet.xml"):
            parts = fixture_parts([result()])
            parts[name] = b"unused"
            with self.subTest(name=name), self.assertRaises(reader.MarkerInputError):
                reader.read_workbook(self.write(parts=parts))

    def test_missing_members_wrong_type_and_malformed_xml(self):
        for name in ("[Content_Types].xml", "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "xl/worksheets/sheet1.xml"):
            parts = fixture_parts([result()])
            parts.pop(name)
            with self.subTest(missing=name), self.assertRaises(reader.MarkerInputError):
                reader.read_workbook(self.write(parts=parts))
        parts = fixture_parts([result()])
        parts["xl/worksheets/sheet1.xml"] = b"<broken"
        with self.assertRaises(reader.MarkerInputError):
            reader.read_workbook(self.write(parts=parts))
        parts = fixture_parts([result()])
        parts["[Content_Types].xml"] = parts["[Content_Types].xml"].replace(b"sheet.main+xml", b"macroEnabled.main+xml")
        with self.assertRaises(reader.MarkerInputError):
            reader.read_workbook(self.write(parts=parts))

    def test_dtd_and_bad_shared_reference(self):
        parts = fixture_parts([result()])
        parts["xl/worksheets/sheet1.xml"] = b'<!DOCTYPE worksheet [<!ENTITY value "10">]>' + parts["xl/worksheets/sheet1.xml"]
        with self.assertRaises(reader.MarkerInputError):
            reader.read_workbook(self.write(parts=parts))
        values = result()
        values[0] = Typed("s", "99999999")
        with self.assertRaises(reader.MarkerInputError):
            self.read([values], shared=True)

    def test_utf16_dtd_rejected(self):
        parts = fixture_parts([result()])
        source = parts["xl/workbook.xml"].decode("utf-8").split("?>", 1)[1]
        source = '<?xml version="1.0" encoding="utf-16"?><!DOCTYPE workbook [<!ENTITY unused "x">]>' + source
        parts["xl/workbook.xml"] = source.encode("utf-16")
        with self.assertRaises(reader.MarkerInputError):
            reader.read_workbook(self.write(parts=parts))

    def test_external_sheet_relationship_and_duplicate_sheet_name(self):
        parts = fixture_parts([result()])
        rel = ET.fromstring(parts["xl/_rels/workbook.xml.rels"])
        rel[0].set("TargetMode", "External")
        parts["xl/_rels/workbook.xml.rels"] = xml_bytes(rel)
        with self.assertRaises(reader.MarkerInputError):
            reader.read_workbook(self.write(parts=parts))
        parts = fixture_parts([result()], [])
        book = ET.fromstring(parts["xl/workbook.xml"])
        book.find(f"{{{NS}}}sheets")[1].set("name", "未確認結果")
        parts["xl/workbook.xml"] = xml_bytes(book)
        with self.assertRaises(reader.MarkerInputError):
            reader.read_workbook(self.write(parts=parts))

    def test_duplicate_cell_and_row_rejected(self):
        for duplicate in ("cell", "row"):
            parts = fixture_parts([result()])
            sheet = ET.fromstring(parts["xl/worksheets/sheet1.xml"])
            data = sheet.find(f"{{{NS}}}sheetData")
            if duplicate == "cell":
                data[1].append(ET.fromstring(ET.tostring(data[1][0])))
            else:
                data.append(ET.fromstring(ET.tostring(data[1])))
            parts["xl/worksheets/sheet1.xml"] = xml_bytes(sheet)
            with self.subTest(duplicate=duplicate), self.assertRaises(reader.MarkerInputError):
                reader.read_workbook(self.write(parts=parts))

    def test_resource_limits_are_rejected_without_large_fixtures(self):
        path = self.write()
        limits = (("MAX_ARCHIVE_BYTES", 1), ("MAX_ZIP_MEMBERS", 1), ("MAX_EXPANDED_BYTES", 1),
                  ("MAX_XML_BYTES", 1), ("MAX_ROWS", 0), ("MAX_CELLS", 1), ("MAX_TEXT_LENGTH", 1))
        for name, maximum in limits:
            with self.subTest(limit=name), patch.object(reader, name, maximum), self.assertRaises(reader.MarkerInputError):
                reader.read_workbook(path)

    def test_no_input_modification(self):
        path = self.write(detail_rows=[detail()], shared=True)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        workbook = reader.read_workbook(path)
        workbook.select("日本語 図面.xdw")
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_wrong_extension_and_missing_path(self):
        with self.assertRaises(reader.MarkerInputError):
            reader.read_workbook(Path(self.directory.name) / "missing.xlsx")
        with self.assertRaises(reader.MarkerInputError):
            reader.read_workbook(Path(self.directory.name) / "old.xls")


if __name__ == "__main__":
    unittest.main()

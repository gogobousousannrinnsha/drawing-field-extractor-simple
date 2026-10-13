"""Read exported coordinate workbooks without Excel or extra dependencies.

This module plans rectangles only. Document selection is an exact filename
match; the caller owns XDW geometry checks and all SDK operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path, PurePosixPath
import posixpath
import re
from types import MappingProxyType
from typing import Mapping
from xml.etree import ElementTree as ET
import zipfile
import zlib


MAIN_SHEET = "未確認結果"
EVIDENCE_SHEET = "根拠座標"
MAIN_HEADERS = ("原本ファイル", "ページ", "項目", "値", "備考",
                "x_mm", "y_mm", "width_mm", "height_mm", "根拠ID", "結果ID")
EVIDENCE_HEADERS = ("結果ID", "原本ファイル", "ページ", "根拠ID", "OCR原文",
                    "x_mm", "y_mm", "width_mm", "height_mm")
COORDINATE_HEADERS = ("x_mm", "y_mm", "width_mm", "height_mm")
MAIN_CONTRACT = ("原本ファイル", "ページ", *COORDINATE_HEADERS, "根拠ID", "結果ID")
EVIDENCE_CONTRACT = ("結果ID", "原本ファイル", "ページ", "根拠ID", *COORDINATE_HEADERS)
BOX_TOLERANCE_MM = 0.05

# These are resource limits, not DocuWorks page-size limits.
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_XML_BYTES = 32 * 1024 * 1024
MAX_ZIP_MEMBERS = 4096
MAX_ROWS = 100_000
MAX_CELLS = 1_200_000
MAX_SHARED_STRINGS = 300_000
MAX_TEXT_LENGTH = 32767
MAX_COORDINATE_MM = 1_000_000.0
MAX_PAGE_NUMBER = 1_000_000

_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_P = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_C = "{http://schemas.openxmlformats.org/package/2006/content-types}"
_WORKSHEET_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
_WORKBOOK_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
_CELL_REFERENCE = re.compile(r"([A-Z]{1,3})([1-9][0-9]{0,6})\Z")


class MarkerInputError(ValueError):
    """A workbook cannot provide an unambiguous, finite rectangle plan."""


@dataclass(frozen=True)
class MarkerRow:
    sheet_row: int
    document: str
    page: int
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    result_id: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class _EvidenceRow:
    sheet_row: int
    document: str
    page: int
    result_id: str
    evidence_id: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Workbook:
    documents: tuple[str, ...]
    _rows: tuple[MarkerRow, ...]
    _skipped: Mapping[str, int]
    _evidence: tuple[_EvidenceRow, ...] | None

    def _require_document(self, document: str) -> None:
        if document not in self.documents:
            raise MarkerInputError("Excel内に指定した原本ファイル名がありません: " + str(document))

    def skipped_no_coordinates(self, document: str) -> int:
        self._require_document(document)
        return self._skipped.get(document, 0)

    def select(self, document: str) -> list[MarkerRow]:
        self._require_document(document)
        selected = [row for row in self._rows if row.document == document]
        by_result = {}
        for row in selected:
            if row.result_id in by_result:
                raise MarkerInputError(f"{MAIN_SHEET}の結果IDが重複しています（行{row.sheet_row}）: {row.result_id}")
            by_result[row.result_id] = row
        if self._evidence is not None:
            evidence = {}
            for item in self._evidence:
                if item.result_id not in by_result:
                    continue
                row = by_result[item.result_id]
                if item.document != row.document or item.page != row.page:
                    raise MarkerInputError(f"{EVIDENCE_SHEET}行{item.sheet_row}の原本・ページが結果と一致しません。")
                group = evidence.setdefault(item.result_id, {})
                if item.evidence_id in group:
                    raise MarkerInputError(f"{EVIDENCE_SHEET}の結果ID・根拠IDが重複しています（行{item.sheet_row}）。")
                group[item.evidence_id] = item
            for row in selected:
                items = list(evidence.get(row.result_id, {}).values())
                if not row.evidence_ids or set(item.evidence_id for item in items) != set(row.evidence_ids):
                    raise MarkerInputError(f"{MAIN_SHEET}行{row.sheet_row}と{EVIDENCE_SHEET}の根拠IDが一致しません。")
                x = min(item.box[0] for item in items)
                y = min(item.box[1] for item in items)
                box = (x, y, max(item.box[0] + item.box[2] for item in items) - x,
                       max(item.box[1] + item.box[3] for item in items) - y)
                expected = (row.x_mm, row.y_mm, row.width_mm, row.height_mm)
                if any(abs(actual - wanted) > BOX_TOLERANCE_MM + 1e-9 for actual, wanted in zip(box, expected)):
                    raise MarkerInputError(f"{MAIN_SHEET}行{row.sheet_row}の座標が根拠の外接矩形と一致しません。")
        return selected


@dataclass(frozen=True)
class _Cell:
    value: str
    kind: str = "s"
    formula: bool = False


def _error(sheet: str, row: int, message: str) -> MarkerInputError:
    return MarkerInputError(f"{sheet} 行{row}: {message}")


def _xml(archive: zipfile.ZipFile, name: str) -> ET.Element:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise MarkerInputError("Excelの必要な内部ファイルがありません: " + name) from exc
    if info.file_size > MAX_XML_BYTES:
        raise MarkerInputError("ExcelのXMLが読取上限を超えています: " + name)
    data = archive.read(info)
    if len(data) > MAX_XML_BYTES:
        raise MarkerInputError("ExcelのXMLが読取上限を超えています: " + name)
    # Declaration keywords are ASCII in UTF-8, UTF-16 and UTF-32 XML. Remove
    # interleaved NULs for this guard, then parse the original bytes unchanged.
    declarations = data.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in declarations or b"<!ENTITY" in declarations:
        raise MarkerInputError("DTD・実体定義を含むExcel XMLは使用できません: " + name)
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise MarkerInputError("ExcelのXMLが壊れています: " + name) from exc


def _check_archive(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise MarkerInputError("Excelの内部ファイル数が上限を超えています。")
    names = set()
    expanded = 0
    for info in infos:
        name = info.filename
        parts = PurePosixPath(name).parts
        if (not name or name.startswith("/") or "\\" in name or ":" in name
                or any(part in ("", ".", "..") for part in name.split("/")[:-1])
                or ".." in parts or "\x00" in name):
            raise MarkerInputError("Excelの内部ファイル名が不正です。")
        if name in names:
            raise MarkerInputError("Excelの内部ファイル名が重複しています: " + name)
        names.add(name)
        if info.flag_bits & 1:
            raise MarkerInputError("暗号化されたExcelは使用できません。")
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise MarkerInputError("Excelの圧縮形式を読み取れません。")
        expanded += info.file_size
    if expanded > MAX_EXPANDED_BYTES:
        raise MarkerInputError("Excelの展開後サイズが読取上限を超えています。")


def _rich_text(element: ET.Element) -> str:
    # Phonetic guides (rPh) are not part of the cell's displayed string.
    return "".join(part.text or "" for part in element.findall(_S + "t")
                   + [part for run in element.findall(_S + "r") for part in run.findall(_S + "t")])


def _shared_strings(archive: zipfile.ZipFile) -> tuple[str, ...]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return ()
    root = _xml(archive, "xl/sharedStrings.xml")
    if root.tag != _S + "sst":
        raise MarkerInputError("Excelの共有文字列形式が不正です。")
    strings = []
    for item in root.findall(_S + "si"):
        text = _rich_text(item)
        if len(text) > MAX_TEXT_LENGTH:
            raise MarkerInputError("Excelの文字列が読取上限を超えています。")
        strings.append(text)
        if len(strings) > MAX_SHARED_STRINGS:
            raise MarkerInputError("Excelの共有文字列数が読取上限を超えています。")
    return tuple(strings)


def _sheet_paths(archive: zipfile.ZipFile) -> dict[str, str]:
    content = _xml(archive, "[Content_Types].xml")
    if content.tag != _C + "Types" or not any(
        part.get("PartName") == "/xl/workbook.xml" and part.get("ContentType") == _WORKBOOK_TYPE
        for part in content.findall(_C + "Override")
    ):
        raise MarkerInputError("通常の.xlsx形式のブックを指定してください。")
    workbook = _xml(archive, "xl/workbook.xml")
    relationships = _xml(archive, "xl/_rels/workbook.xml.rels")
    if workbook.tag != _S + "workbook" or relationships.tag != _P + "Relationships":
        raise MarkerInputError("Excelのブック定義が不正です。")
    rels = {}
    for rel in relationships.findall(_P + "Relationship"):
        ident = rel.get("Id")
        if not ident or ident in rels:
            raise MarkerInputError("Excelの内部参照IDが不正・重複しています。")
        rels[ident] = rel
    names, found = set(), {}
    sheets = workbook.find(_S + "sheets")
    if sheets is None:
        raise MarkerInputError("Excelにシートがありません。")
    for sheet in sheets.findall(_S + "sheet"):
        name = sheet.get("name", "")
        if not name or name.casefold() in names:
            raise MarkerInputError("Excelのシート名が不正・重複しています。")
        names.add(name.casefold())
        if name not in (MAIN_SHEET, EVIDENCE_SHEET):
            continue
        rel = rels.get(sheet.get(_R + "id"))
        if rel is None or rel.get("Type") != _WORKSHEET_REL or rel.get("TargetMode", "Internal") != "Internal":
            raise MarkerInputError("Excelの対象シート参照が不正です: " + name)
        target = rel.get("Target", "")
        if not target or any(char in target for char in ("\\", ":", "?", "#")) or ".." in PurePosixPath(target).parts:
            raise MarkerInputError("Excelの対象シートの参照先が不正です: " + name)
        path = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target))
        if not path.startswith("xl/") or path in found.values():
            raise MarkerInputError("Excelの対象シートの参照先が不正・重複しています: " + name)
        found[name] = path
    if MAIN_SHEET not in found:
        raise MarkerInputError("Excelに「未確認結果」シートがありません。")
    return found


def _cell_value(cell: ET.Element, strings: tuple[str, ...], sheet: str, row: int) -> _Cell:
    kind = cell.get("t", "n")
    value = cell.find(_S + "v")
    formula = cell.find(_S + "f") is not None
    raw = value.text or "" if value is not None else ""
    if kind == "inlineStr":
        inline = cell.find(_S + "is")
        text = _rich_text(inline) if inline is not None else ""
    elif kind == "s":
        if len(raw) > 8 or not re.fullmatch(r"[0-9]+", raw) or int(raw) >= len(strings):
            raise _error(sheet, row, "共有文字列の参照が不正です。")
        text = strings[int(raw)]
    elif kind in ("n", "str", "b", "e", "d"):
        text = raw
    else:
        raise _error(sheet, row, "未対応のセル形式です。")
    if len(text) > MAX_TEXT_LENGTH:
        raise _error(sheet, row, "セルの文字列が読取上限を超えています。")
    return _Cell(text, kind, formula)


def _sheet_rows(archive: zipfile.ZipFile, path: str, sheet: str, strings: tuple[str, ...]):
    root = _xml(archive, path)
    if root.tag != _S + "worksheet":
        raise MarkerInputError("Excelのシート形式が不正です: " + sheet)
    data = root.find(_S + "sheetData")
    if data is None:
        raise MarkerInputError("Excelにシートの行データがありません: " + sheet)
    previous_row, row_count, cell_count = 0, 0, 0
    for element in data.findall(_S + "row"):
        raw_row = element.get("r", str(previous_row + 1))
        if not re.fullmatch(r"[1-9][0-9]{0,6}", raw_row):
            raise MarkerInputError("Excelの行番号が不正です: " + sheet)
        row = int(raw_row)
        if row <= previous_row or row > 1048576:
            raise _error(sheet, row, "行番号が重複・範囲外です。")
        previous_row = row
        row_count += 1
        if row_count > MAX_ROWS + 1:
            raise MarkerInputError("Excelの行数が読取上限を超えています: " + sheet)
        cells = {}
        for cell in element.findall(_S + "c"):
            reference = cell.get("r", "")
            match = _CELL_REFERENCE.fullmatch(reference)
            if match is None or int(match.group(2)) != row:
                raise _error(sheet, row, "セルの番地が不正です。")
            column = 0
            for char in match.group(1):
                column = column * 26 + ord(char) - ord("A") + 1
            if column > 16384 or column in cells:
                raise _error(sheet, row, "セル番地が重複・範囲外です。")
            cell_count += 1
            if cell_count > MAX_CELLS:
                raise MarkerInputError("Excelのセル数が読取上限を超えています: " + sheet)
            cells[column] = _cell_value(cell, strings, sheet, row)
        yield row, cells


def _records(archive: zipfile.ZipFile, path: str, sheet: str, strings: tuple[str, ...], required: tuple[str, ...], contract: tuple[str, ...]):
    iterator = iter(_sheet_rows(archive, path, sheet, strings))
    first = next(iterator, None)
    if first is None or first[0] != 1:
        raise MarkerInputError("Excelの1行目に必要な見出しがありません: " + sheet)
    columns = {}
    for column, cell in first[1].items():
        if cell.formula or cell.kind in ("b", "e", "d"):
            raise _error(sheet, 1, "見出しには文字列を指定してください。")
        if not cell.value:
            continue
        if cell.value in columns:
            raise _error(sheet, 1, "見出しが重複しています: " + cell.value)
        columns[cell.value] = column
    missing = [name for name in required if name not in columns]
    if missing:
        raise MarkerInputError("Excelの見出しが不足しています（" + sheet + "）: " + "、".join(missing))
    for row, cells in iterator:
        values = {name: cells.get(column, _Cell("")) for name, column in columns.items()}
        if not any(cell.value or cell.formula for cell in values.values()):
            continue
        for name in contract:
            cell = values[name]
            if cell.formula:
                raise _error(sheet, row, "使用する列に数式があります: " + name)
            if cell.kind in ("b", "e", "d"):
                raise _error(sheet, row, "使用する列のセル形式が不正です: " + name)
        yield row, {name: cell.value for name, cell in values.items()}


def _document(value: str, sheet: str, row: int) -> str:
    if (not value.strip() or value in (".", "..") or any(char in value for char in '<>:"/\\|?*\x00')
            or any(ord(char) < 32 for char in value)):
        raise _error(sheet, row, "原本ファイルには有効なファイル名を指定してください。")
    return value


def _nonblank(value: str, field: str, sheet: str, row: int) -> str:
    if not value.strip():
        raise _error(sheet, row, field + "が空です。")
    return value


def _number(value: str, field: str, sheet: str, row: int) -> float:
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise _error(sheet, row, field + "は有限の数値で指定してください。") from exc
    if not math.isfinite(number):
        raise _error(sheet, row, field + "は有限の数値で指定してください。")
    return number


def _page(value: str, sheet: str, row: int) -> int:
    try:
        number = Decimal(value.strip())
    except InvalidOperation as exc:
        raise _error(sheet, row, "ページは正の整数で指定してください。") from exc
    if not number.is_finite() or not 1 <= number <= MAX_PAGE_NUMBER or number != number.to_integral_value():
        raise _error(sheet, row, "ページは正の整数で指定してください。")
    return int(number)


def _box(values: dict[str, str], sheet: str, row: int) -> tuple[float, float, float, float] | None:
    coordinates = [values[name].strip() for name in COORDINATE_HEADERS]
    present = [bool(value) for value in coordinates]
    if not any(present):
        return None
    if not all(present):
        raise _error(sheet, row, "座標4列の一部だけが記入されています。")
    numbers = tuple(_number(value, name, sheet, row) for name, value in zip(COORDINATE_HEADERS, coordinates))
    x, y, width, height = numbers
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise _error(sheet, row, "x/yは0以上、幅/高さは0より大きい座標が必要です。")
    if any(value > MAX_COORDINATE_MM for value in numbers):
        raise _error(sheet, row, "座標が読取上限を超えています。")
    return numbers


def _evidence_ids(value: str, sheet: str, row: int) -> tuple[str, ...]:
    try:
        ids = json.loads(value)
    except (ValueError, TypeError, RecursionError) as exc:
        raise _error(sheet, row, "根拠IDは文字列のJSON配列で指定してください。") from exc
    if (not isinstance(ids, list) or any(not isinstance(ident, str) or not ident.strip() for ident in ids)
            or len(ids) != len(set(ids))):
        raise _error(sheet, row, "根拠IDのJSON配列が不正・重複しています。")
    return tuple(ids)


def read_workbook(path) -> Workbook:
    """Read all physical worksheet rows, including hidden and filtered rows.

    Call ``select`` after choosing one exact source filename. Empty coordinate
    rows remain countable but never produce an annotation rectangle.
    """
    path = Path(path)
    if path.suffix.casefold() != ".xlsx":
        raise MarkerInputError("座標付き.xlsxを指定してください。")
    try:
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise MarkerInputError("Excelのファイルサイズが読取上限を超えています。")
        with zipfile.ZipFile(path) as archive:
            _check_archive(archive)
            paths = _sheet_paths(archive)
            strings = _shared_strings(archive)
            rows, skipped, documents = [], {}, set()
            for sheet_row, values in _records(archive, paths[MAIN_SHEET], MAIN_SHEET, strings, MAIN_HEADERS, MAIN_CONTRACT):
                document = _document(values["原本ファイル"], MAIN_SHEET, sheet_row)
                documents.add(document)
                box = _box(values, MAIN_SHEET, sheet_row)
                if box is None:
                    if values["ページ"].strip():
                        _page(values["ページ"], MAIN_SHEET, sheet_row)
                    if values["根拠ID"].strip():
                        _evidence_ids(values["根拠ID"], MAIN_SHEET, sheet_row)
                    skipped[document] = skipped.get(document, 0) + 1
                    continue
                rows.append(MarkerRow(sheet_row, document, _page(values["ページ"], MAIN_SHEET, sheet_row), *box,
                                      _nonblank(values["結果ID"], "結果ID", MAIN_SHEET, sheet_row),
                                      _evidence_ids(values["根拠ID"], MAIN_SHEET, sheet_row)))
            evidence = None
            if EVIDENCE_SHEET in paths:
                evidence = []
                for sheet_row, values in _records(archive, paths[EVIDENCE_SHEET], EVIDENCE_SHEET, strings, EVIDENCE_HEADERS, EVIDENCE_CONTRACT):
                    box = _box(values, EVIDENCE_SHEET, sheet_row)
                    if box is None:
                        raise _error(EVIDENCE_SHEET, sheet_row, "根拠の座標4列が空です。")
                    evidence.append(_EvidenceRow(sheet_row, _document(values["原本ファイル"], EVIDENCE_SHEET, sheet_row),
                        _page(values["ページ"], EVIDENCE_SHEET, sheet_row),
                        _nonblank(values["結果ID"], "結果ID", EVIDENCE_SHEET, sheet_row),
                        _nonblank(values["根拠ID"], "根拠ID", EVIDENCE_SHEET, sheet_row), box))
            return Workbook(tuple(sorted(documents, key=lambda name: (name.casefold(), name))), tuple(rows),
                            MappingProxyType(skipped), None if evidence is None else tuple(evidence))
    except MarkerInputError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError, OverflowError, EOFError, zlib.error) as exc:
        raise MarkerInputError("Excelを読み取れません: " + str(exc)) from exc

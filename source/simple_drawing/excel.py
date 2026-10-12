from dataclasses import dataclass
import json
import math
from pathlib import Path
import os
import uuid
import xlsxwriter

from .storage import local_path, new_id, now, read_json, relative, sha, write_json

REASONS = {"not_found": "記載なし", "unreadable": "判読できません", "ambiguous": "対象を一つに決められません"}
HEADERS = ["原本ファイル", "ページ", "項目", "値", "備考",
           "x_mm", "y_mm", "width_mm", "height_mm", "根拠ID", "結果ID"]
EVIDENCE_HEADERS = ["結果ID", "原本ファイル", "ページ", "根拠ID", "OCR原文",
                    "x_mm", "y_mm", "width_mm", "height_mm"]
COORDINATES = ("x", "y", "width", "height")
EXCEL_FORMAT_VERSION = 2


@dataclass
class ExcelRow:
    values: list
    evidence: list


def _box(region):
    result = []
    for name in COORDINATES:
        value = region["bbox_mm"][name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("保存された根拠OCRのmm座標が不正です。")
        result.append(value)
    return result


def _payload_regions(folder, record):
    payload = local_path(folder, record["input_payload"])
    if sha(payload) != record["payload_sha256"]:
        raise ValueError("保存された抽出入力が破損しています。座標は推測しません。")
    regions = read_json(payload)["ocr_regions"]
    by_id = {region["id"]: region for region in regions}
    if len(by_id) != len(regions):
        raise ValueError("保存されたOCRの根拠IDが重複しています。")
    return by_id


def excel_rows(batch, records, settings, *, run_folder):
    """Use the exact OCR snapshot sent for this run, without consulting current caches."""
    folder = Path(run_folder)
    for index, document in enumerate(batch["documents"]):
        if document.get("error"):
            values = [document["name"], "", settings["item"], "", "処理失敗: " + document["error"]]
            yield ExcelRow(values + [None] * 4 + ["[]", f"{folder.name}:{index}:document:0"], [])
            continue
        for page in document["pages"]:
            record = records[f"{index}:{page['number']}"]
            base = [document["name"], str(page["number"]), settings["item"]]
            prefix = f"{folder.name}:{index}:{page['number']}"
            if record.get("error"):
                error = record["error"]
                reason = error.get("message", str(error)) if isinstance(error, dict) else error
                yield ExcelRow(base + ["", "処理失敗: " + reason] + [None] * 4 + ["[]", prefix + ":0"], [])
                continue
            answer = record["parsed"]
            by_id = _payload_regions(folder, record)
            if not answer["results"]:
                reason = REASONS[answer["status"]]
                note = reason + (": " + answer["note"] if answer["note"] else "")
                yield ExcelRow(base + ["", note] + [None] * 4 + ["[]", prefix + ":0"], [])
                continue
            for ordinal, result in enumerate(answer["results"], 1):
                result_id = prefix + f":{ordinal}"
                evidence, boxes = [], []
                for source_id in result["source_ids"]:
                    if source_id not in by_id:
                        raise ValueError("回答の根拠IDが当該ページの保存OCRにありません。")
                    region = by_id[source_id]
                    box = _box(region)
                    boxes.append(box)
                    evidence.append([result_id, document["name"], str(page["number"]),
                                     source_id, region["text"], *box])
                rectangle = [None] * 4
                if len(boxes) == 1:
                    rectangle = boxes[0]
                elif boxes:
                    x, y = min(b[0] for b in boxes), min(b[1] for b in boxes)
                    rectangle = [x, y, max(b[0] + b[2] for b in boxes) - x,
                                 max(b[1] + b[3] for b in boxes) - y]
                source_ids = json.dumps(result["source_ids"], ensure_ascii=False, separators=(",", ":"))
                yield ExcelRow(base + [result["value"], answer["note"], *rectangle, source_ids, result_id], evidence)


def _write_row(sheet, number, values, numeric, number_format):
    if number >= 1048576:
        raise ValueError("Excelの行数上限を超えました。JSON結果は保存されています。")
    for col, value in enumerate(values):
        if value is None:
            sheet.write_blank(number, col, None, number_format if col in numeric else None)
        elif col in numeric:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Excelの座標には有限の数値が必要です。")
            sheet.write_number(number, col, value, number_format)
        else:
            if not isinstance(value, str) or len(value) > 32767:
                raise ValueError("Excelの文字数上限を超えました。JSON結果は保存されています。")
            if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
                raise ValueError("Excelで表現できない制御文字があります。JSON結果は保存されています。")
            sheet.write_string(number, col, value)


def write_excel(path, rows, *, before_publish=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError("既存のExcelは上書きしません。")
    temp = path.with_name(path.stem + "." + uuid.uuid4().hex + ".xlsx")
    try:
        with xlsxwriter.Workbook(temp, {"strings_to_numbers": False,
                "strings_to_formulas": False, "strings_to_urls": False}) as workbook:
            main = workbook.add_worksheet("未確認結果")
            detail = workbook.add_worksheet("根拠座標")
            header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7"})
            number_format = workbook.add_format({"num_format": "0.000", "valign": "vcenter"})
            ids_format = workbook.add_format({"text_wrap": True, "valign": "vcenter"})
            for sheet, headers in ((main, HEADERS), (detail, EVIDENCE_HEADERS)):
                sheet.freeze_panes(1, 0)
                for col, value in enumerate(headers):
                    sheet.write_string(0, col, value, header)
                sheet.set_column(5, 8, 15, number_format)
            main.set_column(0, 0, 36)
            main.set_column(1, 1, 8)
            main.set_column(2, 2, 20)
            main.set_column(3, 3, 40)
            main.set_column(4, 4, 64)
            main.set_column(9, 9, 30, ids_format)
            main.set_column(10, 10, 44)
            detail.set_column(0, 0, 44)
            detail.set_column(1, 1, 36)
            detail.set_column(2, 2, 8)
            detail.set_column(3, 3, 24)
            detail.set_column(4, 4, 50)
            count = detail_count = 0
            for count, row in enumerate(rows, 1):
                if isinstance(row, ExcelRow):
                    values, evidence = row.values, row.evidence
                    main.set_row(count, 15 * min(8, max(1, math.ceil(len(values[9]) / 28))))
                else:
                    values, evidence = row, []
                _write_row(main, count, values, range(5, 9), number_format)
                for values in evidence:
                    detail_count += 1
                    _write_row(detail, detail_count, values, range(5, 9), number_format)
            for sheet, count, headers, name in ((main, count, HEADERS, "Results"),
                    (detail, detail_count, EVIDENCE_HEADERS, "EvidenceCoordinates")):
                if count:
                    sheet.add_table(0, 0, count, len(headers) - 1, {"name": name, "style": "Table Style Light 9",
                        "columns": [{"header": value, "header_format": header} for value in headers]})
                else:
                    sheet.autofilter(0, 0, 0, len(headers) - 1)
        if path.exists():
            raise FileExistsError("既存のExcelは上書きしません。")
        if before_publish:
            before_publish(temp)
        os.rename(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def reexport_excel(root, run_reference=None):
    """Write a new Excel from a finished run; never start OCR or a model server."""
    root = Path(root).resolve()
    runs = root / "data" / "runs"
    if run_reference is not None:
        folder = local_path(root, run_reference)
        if not folder.is_relative_to(runs) or not folder.is_dir():
            raise ValueError("再出力の対象はdata/runs内の実行フォルダーで指定してください。")
    else:
        folder = None
        if runs.is_dir():
            for candidate in sorted((p for p in runs.iterdir() if p.is_dir()), reverse=True):
                if not (candidate / "run.json").is_file():
                    continue
                if read_json(candidate / "run.json")["status"] in ("complete", "partial"):
                    folder = candidate
                    break
        if folder is None:
            raise ValueError("完了済みの抽出記録がありません。先に抽出.batを実行してください。")
    manifest_path = folder / "run.json"
    manifest_sha = sha(manifest_path)
    manifest = read_json(manifest_path)
    if manifest["status"] not in ("complete", "partial"):
        raise ValueError("未完了の抽出からExcelを再出力できません。")
    contract = manifest["contract"]
    batch_path = local_path(root, contract["batch_path"])
    if sha(batch_path) != contract["batch_sha256"]:
        raise ValueError("当時の前処理記録が破損しています。")
    batch = read_json(batch_path)
    records = {}
    for key, reference in manifest["records"].items():
        path = local_path(folder, reference["path"])
        if sha(path) != reference["sha256"]:
            raise ValueError("保存された抽出結果が破損しています。")
        records[key] = read_json(path)
    output_folder = root / "data" / "exports" / new_id()
    output_folder.mkdir(parents=True)
    export = {"version": EXCEL_FORMAT_VERSION, "created_at": now(), "status": "running",
              "source_run": relative(root, folder), "source_run_sha256": manifest_sha,
              "batch_sha256": contract["batch_sha256"], "ocr_executed": False, "model_executed": False}
    write_json(output_folder / "export.json", export)
    output = output_folder / "抽出結果_座標付き.xlsx"
    try:
        write_excel(output, excel_rows(batch, records, contract["settings"], run_folder=folder))
        export.update(status="complete", excel=output.name, excel_sha256=sha(output), finished_at=now())
        write_json(output_folder / "export.json", export)
        print(f"Excel再出力完了: {output}", flush=True)
        return 0, output
    except BaseException as error:
        export.update(status="failed", error=str(error))
        write_json(output_folder / "export.json", export)
        raise

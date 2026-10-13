"""Create a new white-page image export from immutable saved full OCR."""
from __future__ import annotations

from pathlib import Path
import re

from .storage import local_path, new_id, now, read_json, relative, sha, write_json


def _document_folder(ordinal, name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:80] or "document"
    return f"{ordinal:04d}_{name}"


def _saved_batch(root):
    pointer_path = root/"data/latest_batch.json"
    if not pointer_path.is_file():
        raise ValueError("保存済みの全文OCRがありません。先に前処理.batを実行してください。")
    pointer = read_json(pointer_path)
    path = local_path(root, pointer["path"])
    if sha(path) != pointer["sha256"]:
        raise ValueError("前処理記録が破損しています。前処理を再実行してください。")
    batch = read_json(path)
    if (batch.get("status") not in ("complete", "partial") or batch.get("version") != 1
            or not isinstance(batch.get("documents"), list) or not batch["documents"]):
        raise ValueError("完了済みの前処理記録がありません。")
    return batch, pointer


def _page_ocr(root, batch, document, page):
    # Reuse the same integrity contract as extraction, without loading OCR models.
    from .workflow import cache_valid
    folder = local_path(root, page["cache"])
    if sha(folder/"complete.json") != page["complete_sha256"] or not cache_valid(folder):
        raise ValueError("保存されたページ画像・OCRが破損しています。前処理を再実行してください。")
    ocr = read_json(folder/"ocr.json")
    if (ocr["producer_sha256"] != document.get("producer_sha256", batch["producer_sha256"])
            or ocr["source_sha256"] != document["sha256"]):
        raise ValueError("保存OCRの原本・作成条件が一致しません。")
    for key in ("number", "width_px", "height_px", "width_mm", "height_mm", "rotation"):
        if ocr["page"][key] != page[key]:
            raise ValueError("保存OCRのページ番号・寸法・向きが前処理記録と一致しません。")
    return ocr, relative(root, folder/"ocr.json"), sha(folder/"ocr.json")


def export_ocr_images(root, *, renderer=None, font_path=None):
    """Called under operation.lock; never invokes OCR, a model, or DocuWorks."""
    root = Path(root).resolve(strict=True)
    batch, pointer = _saved_batch(root)
    if renderer is None:
        from .ocr_image import render_ocr_image
        renderer = render_ocr_image
    output_folder = root/"data/ocr-images"/new_id()
    output_folder.mkdir(parents=True)
    manifest = {
        "version": 1, "kind": "full-ocr-white-png", "created_at": now(), "status": "running",
        "source_batch": pointer["path"], "batch_sha256": pointer["sha256"],
        "coordinates": "saved-page-top-left-x-right-y-down", "records": [],
        "completed_pages": 0, "failed_pages": 0, "failed_documents": 0,
        "warning_pages": 0,
        "ocr_executed": False, "model_executed": False, "sdk_executed": False,
    }
    report_path = output_folder/"export.json"
    write_json(report_path, manifest)
    try:
        for ordinal, document in enumerate(batch["documents"], 1):
            name = document["name"]
            if document.get("error"):
                manifest["failed_documents"] += 1
                manifest["records"].append({"document": name, "status": "failed",
                                            "error": document["error"], "stage": "preprocessing"})
                print(f"{name} / 前処理失敗: {document['error']}", flush=True)
                write_json(report_path, manifest)
                continue
            try:
                if sha(local_path(root, document["source"])) != document["sha256"]:
                    raise ValueError("保存原本のSHAが一致しません。前処理を再実行してください。")
            except (OSError, ValueError, KeyError) as error:
                manifest["failed_documents"] += 1
                manifest["records"].append({"document": name, "status": "failed", "error": str(error)})
                print(f"{name} / 原本検査失敗: {error}", flush=True)
                write_json(report_path, manifest)
                continue
            document_folder = output_folder/_document_folder(ordinal, name)
            for page in document["pages"]:
                record = {"document": name, "source_sha256": document["sha256"], "page": page["number"]}
                print(f"{name} / ページ {page['number']} / 全文OCR画像出力 / "
                      f"完了 {manifest['completed_pages']} / 失敗 {manifest['failed_pages']}", flush=True)
                try:
                    if page.get("error"):
                        raise ValueError("前処理失敗: " + page["error"])
                    ocr, source_ocr, source_ocr_sha = _page_ocr(root, batch, document, page)
                    document_folder.mkdir(exist_ok=True)
                    output = document_folder/f"page-{page['number']:04d}.png"
                    rendered = renderer(ocr, output, font_path=font_path)
                    if rendered.get("warnings"):
                        manifest["warning_pages"] += 1
                        codes = ", ".join(sorted({warning["code"] for warning in rendered["warnings"]}))
                        print(f"{name} / ページ {page['number']} / 表示上の注意: {codes}（export.json参照）", flush=True)
                    record.update(status="complete", image=relative(output_folder, output),
                                  image_sha256=sha(output), source_ocr=source_ocr,
                                  source_ocr_sha256=source_ocr_sha, page_info=ocr["page"], rendering=rendered)
                    manifest["completed_pages"] += 1
                except Exception as error:
                    record.update(status="failed", error=str(error))
                    manifest["failed_pages"] += 1
                    print(f"{name} / ページ {page['number']} / 出力失敗: {error}", flush=True)
                manifest["records"].append(record)
                write_json(report_path, manifest)
        partial = bool(manifest["failed_pages"] or manifest["failed_documents"])
        manifest.update(status="partial" if partial else "complete", finished_at=now())
        write_json(report_path, manifest)
        warning = "（表示上の注意あり）" if manifest["warning_pages"] else ""
        print(f"全文OCR画像出力{'（一部失敗）' if partial else '完了'}{warning}: {output_folder}", flush=True)
        return (2 if partial else 0), output_folder
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(report_path, manifest)
        raise

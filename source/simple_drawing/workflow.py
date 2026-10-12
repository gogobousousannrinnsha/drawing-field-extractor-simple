from __future__ import annotations
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import time

from .storage import digest, local_path, new_id, now, read_json, relative, sha, write_json
from .settings import ExecutionSettings, ExtractionSettings
from .excel import excel_rows, write_excel

MAX_BATCH_PAGES = 1000

def progress(name, number, stage, completed, failed):
    print(f"{name} / ページ {number or '-'} / {stage} / 完了 {completed} / 失敗 {failed}", flush=True)

def producer_contract(root):
    packages = {name: importlib.metadata.version(name) for name in
        ["pypdfium2", "pypdf", "paddleocr", "paddlex", "paddlepaddle-gpu", "Pillow", "numpy"]}
    files = {relative(root, path): sha(path) for path in sorted((root / "source" / "simple_drawing").glob("*ocr*.py"))}
    vendor = root / "source" / "simple_drawing" / "vendor"
    if vendor.exists():
        files.update({relative(root, path): sha(path) for path in sorted(vendor.rglob("*.py"))})
    model_root = root / "models" / "ocr"
    models = {relative(model_root, path): sha(path) for path in sorted(model_root.rglob("*")) if path.is_file()}
    return {"version": 1, "python": sys.version.split()[0], "packages": packages,
        "implementation": files, "models": models, "dpi": 300, "tile_size": 2048,
        "tile_overlap": 128, "orientations": [0], "recognizer_contrast": 2.0,
        "recognition_batch_size": 1, "coordinates": "displayed_png_xywh_mm_px_v1"}

def cache_valid(folder):
    try:
        complete = read_json(folder / "complete.json")
        if not isinstance(complete, dict) or not isinstance(complete.get("files"), dict):
            return False
        if complete.get("version") != 1 or set(complete["files"]) != {"page.png", "raw.json", "ocr.json"}:
            return False
        for name, expected in complete["files"].items():
            if sha(folder / name) != expected:
                return False
        ocr = read_json(folder / "ocr.json")
        if not isinstance(ocr, dict):
            return False
        return ocr["version"] == 1 and isinstance(ocr["regions"], list) and bool(ocr["page"])
    except (OSError, ValueError, KeyError, TypeError):
        return False

def save_source(root, source):
    source_hash = sha(source)
    destination = root / "data" / "sources" / (source_hash + ".pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha(destination) == source_hash:
        return source_hash, relative(root, destination)
    temporary = destination.with_name(destination.name + "." + new_id() + ".tmp")
    shutil.copyfile(source, temporary)
    if sha(temporary) != source_hash or sha(source) != source_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError("コピー中に原本が変更されました。")
    os.replace(temporary, destination)
    return source_hash, relative(root, destination)

def prepare(root, *, inspector=None, renderer=None, recognizer=None, producer=None):
    """Worker owns the OCR engine; process exit releases all GPU allocations."""
    root = Path(root).resolve()
    from .pdf_ocr import inspect_pdf, render_page, recognize_page, OCRSession
    inspector = inspector or inspect_pdf
    renderer = renderer or render_page
    recognizer_override = recognizer
    recognizer = recognizer or recognize_page
    settings = ExecutionSettings.model_validate(read_json(root / "実行設定.json"))
    input_folder = Path(settings.input_folder)
    if not input_folder.is_absolute():
        input_folder = root / input_folder
    input_folder = input_folder.resolve(strict=True)
    if not input_folder.is_dir():
        raise ValueError("入力先はフォルダーで指定してください。")
    sources = sorted((p for p in input_folder.iterdir() if p.is_file() and p.suffix.casefold() == ".pdf"),
        key=lambda p: (p.name.casefold(), p.name))
    if not sources:
        raise ValueError("入力フォルダー直下にPDFがありません。")
    producer = producer or producer_contract(root)
    producer_sha = digest(producer)
    batch_id = new_id()
    batch_path = root / "data" / "batches" / batch_id / "batch.json"
    batch = {"version": 1, "id": batch_id, "created_at": now(), "status": "running",
        "producer": producer, "producer_sha256": producer_sha, "documents": [],
        "completed_pages": 0, "failed_pages": 0, "reused_pages": 0, "ocr_pages": 0}
    write_json(batch_path, batch)
    write_json(root / "data" / "active_prepare.json", {"path": relative(root, batch_path)})
    session = None
    try:
        total = 0
        for source in sources:
            document = {"name": source.name, "pages": []}
            batch["documents"].append(document)
            progress(source.name, 0, "原本保存・PDF検査", 0, 0)
            try:
                source_hash, source_ref = save_source(root, source)
                document.update(sha256=source_hash, source=source_ref)
                document["pages"] = inspector(local_path(root, source_ref))
                total += len(document["pages"])
            except Exception as error:
                document["error"] = str(error)
            write_json(batch_path, batch)
        batch["total_pages"] = total
        if total > MAX_BATCH_PAGES:
            raise ValueError(f"一括対象が{total}ページあります。合計上限は{MAX_BATCH_PAGES}ページです。")
        for document in batch["documents"]:
            if document.get("error"):
                continue
            for page in document["pages"]:
                folder = root / "data" / "cache" / producer_sha / document["sha256"] / str(page["number"])
                page["cache"] = relative(root, folder)
                try:
                    if cache_valid(folder):
                        saved = read_json(folder / "ocr.json")
                        if saved["producer_sha256"] != producer_sha or saved["source_sha256"] != document["sha256"]:
                            raise ValueError("OCRキャッシュの原本・作成条件が一致しません。")
                        batch["reused_pages"] += 1
                        progress(document["name"], page["number"], "保存済みOCRを再利用", batch["completed_pages"], batch["failed_pages"])
                    else:
                        folder.parent.mkdir(parents=True, exist_ok=True)
                        stage = folder.with_name(folder.name + ".pending-" + new_id())
                        stage.mkdir()
                        started = time.perf_counter()
                        progress(document["name"], page["number"], "300dpi画像化", batch["completed_pages"], batch["failed_pages"])
                        rendered = renderer(local_path(root, document["source"]), page["number"], stage / "page.png")
                        page.update(rendered)
                        progress(document["name"], page["number"], "全文OCR", batch["completed_pages"], batch["failed_pages"])
                        if recognizer_override is None:
                            if session is None:
                                session = OCRSession(root / "models" / "ocr")
                            regions, raw = recognizer(stage / "page.png", rendered, root / "models" / "ocr", stage / "scratch", session=session)
                        else:
                            regions, raw = recognizer(stage / "page.png", rendered, root / "models" / "ocr", stage / "scratch")
                        write_json(stage / "raw.json", raw)
                        write_json(stage / "ocr.json", {"version": 1, "page": rendered, "regions": regions,
                            "source_sha256": document["sha256"], "producer_sha256": producer_sha,
                            "seconds": time.perf_counter() - started})
                        write_json(stage / "complete.json", {"version": 1, "finished_at": now(),
                            "files": {name: sha(stage / name) for name in ["page.png", "raw.json", "ocr.json"]}})
                        if folder.exists():
                            # Keep damaged cache for diagnosis instead of accepting it.
                            damaged = folder.with_name(folder.name + ".damaged-" + new_id())
                            os.rename(folder, damaged)
                        os.rename(stage, folder)
                        batch["ocr_pages"] += 1
                    page["complete_sha256"] = sha(folder / "complete.json")
                    batch["completed_pages"] += 1
                except Exception as error:
                    page["error"] = str(error)
                    batch["failed_pages"] += 1
                    progress(document["name"], page["number"], "処理失敗: " + str(error), batch["completed_pages"], batch["failed_pages"])
                write_json(batch_path, batch)
        batch["status"] = "partial" if batch["failed_pages"] or any(d.get("error") for d in batch["documents"]) else "complete"
        batch["finished_at"] = now()
        write_json(batch_path, batch)
        write_json(root / "data" / "latest_batch.json", {"path": relative(root, batch_path), "sha256": sha(batch_path)})
        return (2 if batch["status"] == "partial" else 0), batch
    except BaseException as error:
        batch["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        batch["error"] = str(error)
        write_json(batch_path, batch)
        raise
    finally:
        if session is not None:
            session.close()

def load_latest_batch(root):
    pointer = read_json(root / "data" / "latest_batch.json")
    path = local_path(root, pointer["path"])
    if sha(path) != pointer["sha256"]:
        raise ValueError("前処理記録が破損しています。前処理を再実行してください。")
    batch = read_json(path)
    if batch["status"] not in ("complete", "partial"):
        raise ValueError("前処理が完了していません。")
    for document in batch["documents"]:
        if not document.get("error") and sha(local_path(root, document["source"])) != document["sha256"]:
            raise ValueError("保存原本が破損しています。前処理を再実行してください。")
    return batch, pointer

def extraction_contract(root, batch, pointer, settings, model, provider_config=None, adapter_hash=None):
    return {"version": 1, "batch_path": pointer["path"], "batch_sha256": pointer["sha256"],
        "settings": settings, "model": model, "workflow_sha256": sha(Path(__file__)),
        "validation_sha256": sha(root / "source" / "simple_drawing" / "llm.py"),
        "provider_config": provider_config or {"backend": "gemma"}, "adapter_sha256": adapter_hash,
        "prepared_pages": [[d["name"], p["number"], p.get("complete_sha256"), p.get("error")]
            for d in batch["documents"] for p in d["pages"]]}

def resume_run(root, contract):
    runs = root / "data" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    folders = sorted((p for p in runs.iterdir() if p.is_dir()), reverse=True)
    if folders:
        latest = folders[0]
        try:
            manifest = read_json(latest / "run.json")
            if manifest["status"] in ("running", "interrupted") and manifest["contract"] == contract:
                records = {}
                for key, record in manifest["records"].items():
                    path = local_path(latest, record["path"])
                    if sha(path) != record["sha256"]:
                        raise ValueError("中断runの保存結果が破損しています。")
                    saved = read_json(path)
                    for reference, hash_key in [("input_image", "input_image_sha256"), ("input_payload", "payload_sha256")]:
                        if reference in saved and sha(local_path(latest, saved[reference])) != saved[hash_key]:
                            raise ValueError("中断runの保存入力が破損しています。")
                    records[key] = saved
                manifest["status"] = "running"
                return latest, manifest, records
        except (OSError, ValueError, KeyError, TypeError):
            pass
    folder = runs / new_id()
    folder.mkdir()
    manifest = {"version": 1, "created_at": now(), "status": "running", "contract": contract, "records": {}}
    write_json(folder / "run.json", manifest)
    return folder, manifest, {}

def shutdown_errors(report):
    # Older injected providers may not report cleanup. Only reported failures
    # affect their outcome; actual inference servers provide all three fields.
    reasons = []
    if not isinstance(report, dict):
        reasons.append("終了処理の報告が不正です。")
    else:
        if "model_unloaded" in report and report["model_unloaded"] is not True:
            reasons.append("モデルの解放を確認できませんでした。")
        if "process_stopped" in report and report["process_stopped"] is not True:
            reasons.append("専用サーバーの終了を確認できませんでした。")
        if report.get("errors"):
            errors = report["errors"]
            details = " / ".join(map(str, errors)) if isinstance(errors, list) else str(errors)
            reasons.append("終了処理のエラー: " + details)
    return [{"code": "server_shutdown", "message": " ".join(reasons)}] if reasons else []

def extract(root, *, server_factory=None, extractor=None, model=None, on_saved=None):
    root = Path(root).resolve()
    from .llm import OllamaServer, build_payload, extract_page, model_contract
    execution = ExecutionSettings.model_validate(read_json(root / "実行設定.json"))
    provider_config = {"backend": "gemma"}
    adapter_hash = None
    if server_factory is None and extractor is None and execution.llm_backend == "saluki":
        from .saluki import SalukiServer, extract_page, model_contract
        options = {"thinking": execution.saluki_thinking, "context_size": execution.saluki_context_size,
            "max_tokens": execution.saluki_max_tokens,
            "image_max_tokens": execution.saluki_image_max_tokens, "batch_size": execution.saluki_batch_size,
            "micro_batch_size": execution.saluki_micro_batch_size, "kv_cache_type": execution.saluki_kv_cache_type}
        server_factory = lambda root: SalukiServer(root, **options)
        model = model or model_contract(root, **options)
        provider_config = {"backend": "saluki", **options}
        adapter_hash = model.get("adapter_sha256")
    elif server_factory is not None or extractor is not None:
        provider_config = {"backend": "injected"}
    else:
        from .profiles import get_profile
        from .model_manager import ensure_usable
        selected = get_profile(execution.ollama_profile)
        # Installation/functional verification belongs to model management.
        # Injected test providers and the existing Saluki branch bypass it.
        ensure_usable(root, selected)
        server_factory = lambda root: OllamaServer(root, profile=selected)
        model = model or model_contract(root, selected)
        provider_config = {"backend": "gemma", "profile": selected.snapshot()}
        adapter_hash = sha(Path(__file__).with_name("profiles.py"))
    server_factory = server_factory or OllamaServer
    extractor = extractor or extract_page
    batch, pointer = load_latest_batch(root)
    settings = ExtractionSettings.model_validate(read_json(root / "抽出設定.json")).model_dump()
    contract = extraction_contract(root, batch, pointer, settings, model or model_contract(root), provider_config, adapter_hash)
    folder, manifest, records = resume_run(root, contract)
    write_json(root / "data" / "active_extract.json", {"path": relative(root, folder / "run.json")})
    write_json(folder / "run.json", manifest)
    failed = sum(bool(r.get("error")) for r in records.values())
    def show_run_errors():
        for error in manifest.get("run_errors", []):
            print("終了処理失敗: " + error["message"], flush=True)
    def commit(key, record):
        nonlocal failed
        path = folder / "pages" / (key.replace(":", "-") + ".json")
        write_json(path, record)
        records[key] = record
        manifest["records"][key] = {"path": relative(folder, path), "sha256": sha(path)}
        write_json(folder / "run.json", manifest)
        if record.get("error"):
            failed += 1
        if on_saved:
            on_saved(key, record)
    try:
        pending = [(i, d, p) for i, d in enumerate(batch["documents"]) for p in d["pages"]
            if f"{i}:{p['number']}" not in records]
        # Preprocessing failures never become a model 'not_found' answer.
        viable = []
        for index, document, page in pending:
            key = f"{index}:{page['number']}"
            if page.get("error"):
                commit(key, {"error": {"code": "preprocessing", "message": page["error"]}, "parsed": None})
            else:
                cache = local_path(root, page["cache"])
                if not cache_valid(cache) or sha(cache / "complete.json") != page["complete_sha256"]:
                    commit(key, {"error": {"code": "cache_corrupt", "message": "保存OCRが破損しています。前処理を再実行してください。"}, "parsed": None})
                else:
                    viable.append((index, document, page))
        if viable:
            server_instance = None
            try:
                server_instance = server_factory(root)
                with server_instance as server:
                    if provider_config["backend"] == "gemma" and server.contract != contract["model"]:
                        raise ValueError("抽出開始時に選択モデルの資産が変更されました。モデル管理で確認してください。")
                    for index, document, page in viable:
                        key = f"{index}:{page['number']}"
                        progress(document["name"], page["number"], str(contract["model"].get("model", "local model")) + "抽出", len(records) - failed, failed)
                        try:
                            cache = local_path(root, page["cache"])
                            if not cache_valid(cache) or sha(cache / "complete.json") != page["complete_sha256"]:
                                raise ValueError("抽出開始後に保存OCRが変更されました。前処理を再実行してください。")
                            complete = read_json(cache / "complete.json")
                            ocr = read_json(cache / "ocr.json")
                            input_folder = folder / "inputs" / key.replace(":", "-")
                            input_folder.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(cache / "page.png", input_folder / "page.png")
                            if sha(input_folder / "page.png") != complete["files"]["page.png"]:
                                raise ValueError("送信画像のコピー中に保存画像が変更されました。")
                            write_json(input_folder / "payload.json", build_payload(ocr["page"], ocr["regions"], settings))
                            record = extractor(server, input_folder / "page.png", ocr["page"], ocr["regions"], settings)
                            record.update(document_name=document["name"], page_number=page["number"],
                                input_image=relative(folder, input_folder / "page.png"), input_image_sha256=sha(input_folder / "page.png"),
                                input_payload=relative(folder, input_folder / "payload.json"), payload_sha256=sha(input_folder / "payload.json"))
                        except Exception as error:
                            record = {"error": {"code": "page", "message": str(error)}, "parsed": None,
                                "document_name": document["name"], "page_number": page["number"]}
                        commit(key, record)
            except Exception as error:
                for index, document, page in viable:
                    key = f"{index}:{page['number']}"
                    if key not in records:
                        commit(key, {"error": {"code": "server", "message": str(error)}, "parsed": None})
            finally:
                if server_instance is not None:
                    manifest["server_shutdown"] = getattr(server_instance, "shutdown_report", {})
                    manifest["server_runtime"] = getattr(server_instance, "contract", {})
                    # Replace the latest outcome so a successful resumed cleanup
                    # cannot retain a stale failure from the interrupted attempt.
                    manifest["run_errors"] = shutdown_errors(manifest["server_shutdown"])
                    write_json(folder / "run.json", manifest)
        progress("Excel一覧", 0, "保存", len(records) - failed, failed)
        output = folder / "抽出結果.xlsx"
        if output.exists():
            if sha(output) != manifest.get("pending_excel_sha256"):
                raise ValueError("保存済みExcelが変更されています。上書きしません。新runで再実行してください。")
        else:
            def before_publish(temporary):
                manifest["pending_excel_sha256"] = sha(temporary)
                write_json(folder / "run.json", manifest)
            write_excel(output, excel_rows(batch, records, settings, run_folder=folder), before_publish=before_publish)
        failed_documents = sum(bool(d.get("error")) for d in batch["documents"])
        manifest.update(status="partial" if failed or failed_documents or manifest.get("run_errors") else "complete", finished_at=now(),
            completed_pages=len(records) - failed, failed_pages=failed, failed_documents=failed_documents,
            excel="抽出結果.xlsx", excel_sha256=sha(folder / "抽出結果.xlsx"))
        write_json(folder / "run.json", manifest)
        show_run_errors()
        print(f"{'一部失敗' if manifest['status'] == 'partial' else '完了'}: {folder / '抽出結果.xlsx'}", flush=True)
        return (2 if manifest["status"] == "partial" else 0), folder
    except BaseException as error:
        output = folder / "抽出結果.xlsx"
        if output.is_file() and sha(output) == manifest.get("pending_excel_sha256"):
            # Publication is committed. An interrupt after rename cannot undo it.
            failed_documents = sum(bool(d.get("error")) for d in batch["documents"])
            manifest.update(status="partial" if failed or failed_documents or manifest.get("run_errors") else "complete", finished_at=now(),
                completed_pages=len(records) - failed, failed_pages=failed, failed_documents=failed_documents,
                excel="抽出結果.xlsx", excel_sha256=sha(output))
            write_json(folder / "run.json", manifest)
            show_run_errors()
            print(f"出力保存は完了しています: {output}", flush=True)
            return (2 if manifest["status"] == "partial" else 0), folder
        manifest.update(status="interrupted", interrupted_at=now(), interruption_reason=str(error))
        write_json(folder / "run.json", manifest)
        show_run_errors()
        raise

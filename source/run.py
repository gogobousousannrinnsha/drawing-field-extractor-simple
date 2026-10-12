"""Portable entry point. No global Python/Ollama settings are changed."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
from simple_drawing.storage import local_path, read_json, write_json, now

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "extract", "export", "internal-prepare", "models-gui"])
    parser.add_argument("--root", type=Path, default=SOURCE.parent)
    parser.add_argument("--run", help="Excel再出力対象の実行フォルダー（候補内の相対パス）。省略時は最新の完了済み抽出。")
    args = parser.parse_args()
    if args.run and args.command != "export":
        parser.error("--runはExcel再出力だけに指定できます。")
    root = args.root.resolve(strict=True)
    for name in ("data", "logs", "temp"):
        (root / name).mkdir(exist_ok=True)
    # Redirect dependency caches into this portable instance before imports.
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(root / "temp" / "paddlex")
    os.environ["PADDLE_HOME"] = str(root / "temp" / "paddle")
    os.environ["HF_HOME"] = str(root / "temp" / "huggingface")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["TMP"] = os.environ["TEMP"] = str(root / "temp")
    if args.command == "models-gui":
        from simple_drawing.model_gui import launch
        return launch(root)
    if args.command == "internal-prepare":
        from simple_drawing.workflow import prepare
        return prepare(root)[0]
    from filelock import FileLock, Timeout
    try:
        with FileLock(root / "data" / "operation.lock", timeout=0):
            if args.command == "export":
                from simple_drawing.excel import reexport_excel
                return reexport_excel(root, args.run)[0]
            if args.command == "prepare":
                log = root / "logs" / ("ocr-" + now().replace(":", "-") + ".log")
                with log.open("w", encoding="utf-8") as errors:
                    worker = subprocess.Popen([sys.executable, "-I", "-B", "-X", "utf8", str(SOURCE / "run.py"),
                        "internal-prepare", "--root", str(root)], cwd=root, stderr=errors,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    import psutil
                    owned_worker = psutil.Process(worker.pid)
                    worker_created = owned_worker.create_time()
                    try:
                        return worker.wait()
                    except KeyboardInterrupt:
                        if worker.poll() is None:
                            if owned_worker.create_time() != worker_created or Path(owned_worker.exe()).resolve() != Path(sys.executable).resolve():
                                raise RuntimeError("OCRワーカーの識別が変わったため停止できません。")
                            children = owned_worker.children(recursive=True)
                            for process in [*children, owned_worker]:
                                try:
                                    process.terminate()
                                except psutil.NoSuchProcess:
                                    pass
                            _, alive = psutil.wait_procs([owned_worker, *children], timeout=5)
                            for process in alive:
                                process.kill()
                            worker.wait(timeout=30)
                        try:
                            pointer = read_json(root / "data" / "active_prepare.json")
                            path = local_path(root, pointer["path"])
                            batch = read_json(path)
                            if batch["status"] == "running":
                                batch.update(status="interrupted", interrupted_at=now())
                                write_json(path, batch)
                        except (OSError, ValueError, KeyError):
                            pass
                        raise
            from simple_drawing.workflow import extract
            return extract(root)[0]
    except Timeout:
        print("別の処理が実行中です。終了後に再実行してください。", flush=True)
        return 1

if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        print("中断しました。完成ページは保存されています。Excelは作成していません。", flush=True)
        code = 130
    except Exception as error:
        print("処理できません: " + str(error), flush=True)
        code = 1
    raise SystemExit(code)

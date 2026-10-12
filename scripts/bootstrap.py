"""Build the app locally from pinned upstream assets; never download the user's documents."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

ALLOWED_HOSTS = {"files.pythonhosted.org", "paddle-whl.cdn.bcebos.com", "paddle-model-ecology.bj.bcebos.com",
                 "github.com", "huggingface.co", "raw.githubusercontent.com"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def operation_lock(root):
    """Use the same Windows byte lock as FileLock, before FileLock is installed."""
    import msvcrt
    path = Path(root)/"data/operation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("他の初期設定・BAT・モデル操作が実行中です") from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def validate_asset(item):
    from urllib.parse import urlparse
    url = urlparse(item["url"])
    if url.scheme != "https" or url.hostname not in ALLOWED_HOSTS or url.username or url.password:
        raise ValueError("取得先が公式資産一覧の許可範囲外です")
    digest = item["sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("SHA-256が不正です")
    filename = item["filename"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*", filename) or filename in (".", ".."):
        raise ValueError("取得ファイル名が不正です")


def download(item, folder, offline=False):
    validate_asset(item)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / item["filename"]
    expected = item["sha256"]
    if target.is_file() and sha(target) == expected:
        print(f"[再利用] {target.name}", flush=True)
        return target
    if target.exists():
        target.rename(target.with_name(target.name+f".bad-sha-{time.time_ns()}"))
    if offline:
        raise RuntimeError(f"オフライン: 検査済み資産がありません: {target.name}")
    partial = target.with_name(target.name+".part")
    print(f"[取得] {target.name}", flush=True)
    command = [str(Path(os.environ.get("SystemRoot", "C:/Windows"))/"System32/curl.exe"),
               "--fail", "--location", "--silent", "--show-error", "--retry", "3",
               "--connect-timeout", "30", "--proto", "=https", "--proto-redir", "=https",
               "--continue-at", "-", "--output", str(partial), item["url"]]
    result = subprocess.run(command, check=False)
    if result.returncode in (33, 36) and partial.exists():
        partial.rename(partial.with_name(partial.name+f".resume-unavailable-{time.time_ns()}"))
        result = subprocess.run(command, check=False)
    if result.returncode == 23:
        raise RuntimeError(f"保存できませんでした。空き容量・書込み権限を確認して再実行してください: {target.name}")
    if result.returncode:
        raise RuntimeError(f"取得に失敗しました({result.returncode})。再実行すると途中から再開します: {target.name}")
    if sha(partial) != expected:
        partial.rename(partial.with_name(partial.name+f".bad-sha-{time.time_ns()}"))
        raise RuntimeError(f"取得内容が固定SHAと一致しません: {target.name}")
    partial.replace(target)
    return target


def safe_member(name, root):
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in ("..", "") or ":" in part for part in path.parts):
        raise ValueError(f"アーカイブの保存先が不正です: {name}")
    destination = Path(root).joinpath(*path.parts)
    destination.resolve().relative_to(Path(root).resolve())
    return destination


def extract(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = safe_member(member.filename, destination)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("リンクを含むアーカイブは展開しません")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
    else:
        with tarfile.open(archive) as bundle:
            for member in bundle:
                target = safe_member(member.name, destination)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                else:
                    raise ValueError("リンク・特殊ファイルを含むアーカイブは展開しません")


def installed_wheels_ok(root, packages):
    from importlib.metadata import distributions
    import base64
    import csv
    import io
    site = root/"runtime/Lib/site-packages"
    installed = {dist.metadata["Name"].lower().replace("_", "-"): dist
                 for dist in distributions(path=[str(site)])}
    for item in packages:
        dist = installed.get(item["name"].lower().replace("_", "-"))
        if dist is None or dist.version != item["version"]:
            return False
        record = dist.read_text("RECORD")
        if not record:
            return False
        for row in csv.reader(io.StringIO(record)):
            if len(row) != 3:
                return False
            name, file_hash, size = row
            # Bytecode is generated locally and may be absent with -B/--no-compile.
            if name.endswith(".pyc"):
                continue
            path = Path(dist.locate_file(name))
            try:
                path.resolve().relative_to((root/"runtime").resolve())
            except ValueError:
                return False
            # pip generates launchers locally; two distributions can intentionally
            # own the same console command (modelscope/modelscope-hub). The app
            # uses python.exe and imports, never these generated entry points.
            if path.parent.resolve() == (root/"runtime/Scripts").resolve() and path.suffix == ".exe":
                continue
            if not path.is_file():
                return False
            if size and path.stat().st_size != int(size):
                return False
            if file_hash:
                if not file_hash.startswith("sha256="):
                    return False
                actual = base64.urlsafe_b64encode(bytes.fromhex(sha(path))).rstrip(b"=").decode("ascii")
                if actual != file_hash.partition("=")[2]:
                    return False
    return True


def check_capacity(root):
    marker = root/"data/setup-started.json"
    manifest_sha = sha(root/"assets/python-packages.json")
    started = marker.exists() and read(marker).get("python_packages_sha256") == manifest_sha
    free = shutil.disk_usage(root).free
    if not started and free < 25*1024**3:
        raise RuntimeError("初回構築には25GiB以上の空き容量が必要です")
    if free < 1024**3:
        raise RuntimeError("作業用の空き容量が1GiB未満です。容量を確保して再実行してください")
    if not started:
        write(marker, {"status": "started", "python_packages_sha256": manifest_sha,
                       "initial_free_bytes": free})


def pip_environment(root):
    # Python -I does not isolate pip's own PIP_* variables or pip.ini files.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PIP_")}
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_CACHE_DIR"] = str(root/"temp/pip")
    env["TEMP"] = env["TMP"] = str(root/"temp")
    return env


def setup_dependencies(root, offline):
    packages = read(root/"assets/python-packages.json")["packages"]
    wheel_folder = root/"downloads/wheels"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: download(item, wheel_folder, offline), packages))
    lock = root/"data/setup-requirements.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("\n".join(f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}" for item in packages)+"\n", encoding="utf-8")
    python = root/"runtime/python.exe"
    marker = root/"data/setup-dependencies.json"
    signature = sha(root/"assets/python-packages.json")
    if not (marker.exists() and read(marker).get("manifest_sha256") == signature and installed_wheels_ok(root, packages)):
        subprocess.run([str(python), "-I", "-B", "-X", "utf8", "-m", "pip", "--isolated", "--no-cache-dir", "install", "--prefix", str(root/"runtime"), "--no-user", "--force-reinstall", "--no-index", "--find-links", str(wheel_folder),
                        "--only-binary", ":all:", "--require-hashes", "--no-compile", "--no-warn-script-location",
                        "--disable-pip-version-check", "-r", str(lock)], env=pip_environment(root), check=True)
    subprocess.run([str(python), "-I", "-B", "-X", "utf8", "-m", "pip", "--isolated", "--no-cache-dir", "check"], env=pip_environment(root), check=True)
    if not installed_wheels_ok(root, packages):
        raise RuntimeError("依存パッケージの版が固定一覧と一致しません")
    write(marker, {"status": "complete", "manifest_sha256": signature, "packages": len(packages)})


def bundled_core_manifest(root):
    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from bundled_core import validate_core_bundle
    return validate_core_bundle(root)


def installed_core_ok(root, item):
    """Compare installed code with the pinned wheel, not a mutable installed RECORD."""
    site = root/"runtime/Lib/site-packages"
    with zipfile.ZipFile(root/item["path"]) as bundle:
        expected_sources = set()
        for member in bundle.infolist():
            if member.filename.endswith("/RECORD"):
                # pip adds its own INSTALLER/direct_url metadata to installed RECORD.
                continue
            path = site.joinpath(*PurePosixPath(member.filename).parts)
            try:
                path.resolve().relative_to((root/"runtime").resolve())
            except ValueError:
                return False
            if path.is_symlink() or not path.is_file() or path.read_bytes() != bundle.read(member):
                return False
            if member.filename.startswith("docuworks_ctypes/"):
                expected_sources.add(member.filename)
        actual_sources = {path.relative_to(site).as_posix()
                          for path in (site/"docuworks_ctypes").rglob("*")
                          if path.is_file() and "__pycache__" not in path.parts}
        return actual_sources == expected_sources


def setup_bundled_core(root):
    """Install the user's MIT Core only from the validated ZIP member, without network."""
    marker = root/"data/setup-core.json"
    write(marker, {"status": "in_progress", "name": "docuworks-ctypes", "version": "1.0.1"})
    try:
        _install_bundled_core(root)
    except (Exception, KeyboardInterrupt) as error:
        write(marker, {"status": "failed", "name": "docuworks-ctypes", "version": "1.0.1",
                       "reason": str(error) or type(error).__name__})
        raise


def _install_bundled_core(root):
    item = bundled_core_manifest(root)
    lock = root/"data/setup-core.lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    (root/"temp").mkdir(exist_ok=True)
    lock.write_text(f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}\n",
                    encoding="utf-8", newline="\n")
    python = root/"runtime/python.exe"
    if not installed_core_ok(root, item):
        subprocess.run([str(python), "-I", "-B", "-X", "utf8", "-m", "pip", "--isolated",
                        "--no-cache-dir", "install", "--prefix", str(root/"runtime"), "--no-user",
                        "--force-reinstall", "--no-index", "--no-deps", "--find-links",
                        str((root/item["path"]).parent), "--only-binary", ":all:", "--require-hashes",
                        "--no-compile", "--no-warn-script-location", "--disable-pip-version-check",
                        "-r", str(lock)], env=pip_environment(root), check=True)
    if not installed_core_ok(root, item):
        raise RuntimeError("同封Coreの導入結果が固定配布物と一致しません")
    subprocess.run([str(python), "-I", "-B", "-X", "utf8", "-m", "pip", "--isolated",
                    "--no-cache-dir", "check"], env=pip_environment(root), check=True)
    write(root/"data/setup-core.json", {"status": "complete", "name": item["name"],
          "version": item["version"], "wheel_sha256": item["sha256"],
          "manifest_sha256": sha(root/"assets/bundled-core.json"), "source": "bundled-local-wheel"})


def setup_ocr(root, offline):
    original = root/"assets/ocr-models.json"
    models = read(original)
    model_root = root/"models/ocr"
    for item in models:
        item_asset = {"filename": item["source"].rsplit("/",1)[-1], "url": item["source"], "sha256": item["archive_sha256"]}
        archive = download(item_asset, root/"downloads/ocr", offline)
        valid = all((model_root / name).is_file() and sha(model_root/name) == expected for name, expected in item["files"].items())
        if not valid:
            stage = root/"temp"/("setup-ocr-"+item["name"]+"-"+str(time.time_ns()))
            extract(archive, stage)
            # Upstream tar directories may carry _infer in their top-level name.
            for name, expected in item["files"].items():
                filename = Path(name).name
                candidates = [path for path in stage.rglob(filename) if path.is_file() and sha(path) == expected]
                if len(candidates) != 1:
                    raise RuntimeError(f"OCR展開結果が固定一覧と一致しません: {name}")
                destination = model_root/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidates[0], destination)
    model_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, model_root/"model-manifest.json")
    for name in ("LICENSE-APACHE-2.0.txt", "NOTICE.txt"):
        notice = root/"licenses/ocr"/name
        if not notice.is_file():
            raise RuntimeError(f"OCRの利用条件文書がありません: {name}")
        shutil.copy2(notice, model_root/name)
    write(root/"data/setup-ocr.json", {"status": "complete", "manifest_sha256": sha(original)})


def setup_ollama(root, offline):
    item = read(root/"assets/ollama-runtime.json")
    archive = download(item, root/"downloads/ollama", offline)
    target = root/"ollama"
    marker = root/"data/setup-ollama.json"
    valid = marker.exists() and read(marker).get("archive_sha256") == item["sha256"]
    if valid:
        valid = all((target / file["path"]).is_file() and sha(target/file["path"]) == file["sha256"]
                    for file in read(marker)["files"])
    if not valid:
        stage = root/"temp"/("setup-ollama-"+str(time.time_ns()))
        extract(archive, stage)
        exe = stage/"ollama.exe"
        if not exe.is_file() or sha(exe) != item["executable_sha256"]:
            raise RuntimeError("Ollama実行ファイルが固定版と一致しません")
        shutil.copytree(stage, target, dirs_exist_ok=True)
        files = [{"path": file.relative_to(stage).as_posix(), "sha256": sha(file)} for file in sorted(stage.rglob("*")) if file.is_file()]
        write(marker, {"status": "complete", "archive_sha256": item["sha256"], "files": files})


def check_runtime(root):
    env = os.environ.copy()
    for key, value in {"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True", "PADDLE_HOME": str(root/"temp/paddle"),
                       "PADDLE_PDX_CACHE_HOME": str(root/"temp/paddlex"), "HF_HOME": str(root/"temp/huggingface"),
                       "PYTHONIOENCODING": "utf-8", "TMP": str(root/"temp"), "TEMP": str(root/"temp")}.items():
        env[key] = value
    code = "import json,sys,tkinter;import paddle; print(json.dumps({'python':sys.version,'cuda':paddle.device.is_compiled_with_cuda(),'gpu_count':paddle.device.cuda.device_count(),'tk':tkinter.Tcl().eval('info patchlevel')}));assert paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count()>0"
    subprocess.run([str(root/"runtime/python.exe"), "-I", "-B", "-X", "utf8", "-c", code], env=env, check=True)
    subprocess.run([str(root/"runtime/python.exe"), "-I", "-B", "-X", "utf8", str(root/"scripts/verify_setup.py"),
                    "--root", str(root), "--ocr-only", "--lock-held"], env=env, check=True)


def setup_model(root, offline):
    sys.path.insert(0, str(root/"source"))
    from simple_drawing.model_manager import ModelManager
    manager = ModelManager(root)
    last_print = [0.0, ""]
    def progress(item):
        status = item.get("status", "モデルを処理しています")
        stage = item.get("stage", "")
        if stage == "download" and time.monotonic()-last_print[0] < 1 and status == last_print[1]:
            return
        last_print[:] = [time.monotonic(), status]
        total, completed = item.get("total"), item.get("completed")
        transfer = f" {completed/1024**3:.2f}/{total/1024**3:.2f}GiB" if total and completed is not None else ""
        print(f"Gemma4: {status}{transfer}", flush=True)
    manifest = root/"models/ollama/manifests/registry.ollama.ai/library/gemma4/e4b-it-q8_0"
    if not offline and not manifest.is_file():
        manager.install("gemma4-q8", progress=progress)
    else:
        print("Gemma4: 保存済みモデルを検査して動作確認します。", flush=True)
    manager.check("gemma4-q8", progress=progress)
    # Keep an existing user's selection. Fresh published settings already select Gemma4.
    manager.ensure_usable("gemma4-q8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-model", action="store_true", help="構成部品だけを取得する。初期設定完了にはしない")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if sys.version_info[:2] != (3,13) or os.name != "nt":
        raise RuntimeError("Windows x64 Python 3.13用の初期設定です")
    for name in ("temp", "data", "logs", "input"):
        (root/name).mkdir(exist_ok=True)
    sys.path.insert(0, str(root/"source"))
    with operation_lock(root):
        write(root/"data/setup-complete.json", {"status": "in_progress", "app_version": "0.1.4"})
        bundled_core_manifest(root)
        check_capacity(root)
        print("[1/6] Pythonライブラリ", flush=True)
        setup_dependencies(root, args.offline)
        print("[2/6] 同封Core 1.0.1（ローカル導入）", flush=True)
        setup_bundled_core(root)
        print("[3/6] OCRモデル", flush=True)
        setup_ocr(root, args.offline)
        print("[4/6] Ollama", flush=True)
        setup_ollama(root, args.offline)
        print("[5/6] GUI・GPU環境確認", flush=True)
        check_runtime(root)
    # ModelManager uses the same lock; do not hold a second copy while invoking it.
    if args.skip_model:
        print("構成部品の準備が完了しました。Gemma4の導入・確認は未完了です。", flush=True)
        return 0
    print("[6/6] Gemma4の導入と動作確認", flush=True)
    setup_model(root, args.offline)
    model_record = read(root/"data/model-registry.json")["models"]["gemma4-q8"]
    write(root/"data/setup-complete.json", {"status": "complete", "app_version": "0.1.4",
                                          "bundled_core": read(root/"data/setup-core.json"),
                                          "python_packages_sha256": sha(root/"assets/python-packages.json"),
                                          "ollama_archive_sha256": read(root/"assets/ollama-runtime.json")["sha256"],
                                          "model": "gemma4:e4b-it-q8_0",
                                          "model_digest": model_record["digest"],
                                          "model_check_sha256": model_record["check"]["report_sha256"],
                                          "real_ocr_check_sha256": sha(root/"data/setup-real-ocr.json"),
                                          "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("初期設定が完了しました。前処理.bat・抽出.bat・モデル管理.batを使用できます。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("中断しました。初期設定.batを再実行すると検査済み資産を再利用します。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"初期設定失敗: {error}", file=sys.stderr)
        raise SystemExit(1)

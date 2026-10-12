"""Prepare the optional, pinned Saluki backend without changing model selection."""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

# Isolated Python does not add the script directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import download, extract, operation_lock, read, safe_member, sha, validate_asset, write


EXPECTED_EXECUTABLE_SHA256 = "7e236243c9b670c6a49410c03186fd575c57a0efe3fd8a21d8727503d571d5b6"


def adapter_constants(root):
    """Read the existing contract without importing or running the GPU adapter."""
    tree = ast.parse((Path(root) / "source/simple_drawing/saluki.py").read_text(encoding="utf-8-sig"))
    values = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in {"MODEL_PATH", "PROJECTOR_PATH", "RUNTIME_VERSION", "ASSETS"}:
                    if target.id == "ASSETS":
                        values[target.id] = {values[key.id]: ast.literal_eval(value)
                                             for key, value in zip(statement.value.keys, statement.value.values)}
                    else:
                        values[target.id] = ast.literal_eval(statement.value)
    if set(values) != {"MODEL_PATH", "PROJECTOR_PATH", "RUNTIME_VERSION", "ASSETS"}:
        raise ValueError("Salukiアダプターの固定資産定義がありません")
    return values


def validate_manifest(root, manifest):
    if manifest.get("schema_version") != 1 or manifest.get("backend") != "saluki":
        raise ValueError("Saluki資産一覧の版・種類が不正です")
    adapter = adapter_constants(root)
    models = manifest["models"]
    expected = adapter["ASSETS"]
    if len(models) != len(expected) or {item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]}
                                         for item in models} != expected:
        raise ValueError("Salukiモデルの資産一覧がアダプターと一致しません")
    runtime = manifest["runtime"]
    if runtime["version"] != adapter["RUNTIME_VERSION"] or runtime["executable_sha256"] != EXPECTED_EXECUTABLE_SHA256:
        raise ValueError("llama.cppの固定版・実行ファイルが一致しません")
    files = runtime["files"]
    if len({item["path"] for item in files}) != len(files):
        raise ValueError("llama.cpp資産一覧の保存先が重複しています")
    executable = [item for item in files if item["path"] == "llama-server.exe"]
    if len(executable) != 1 or executable[0]["sha256"] != EXPECTED_EXECUTABLE_SHA256:
        raise ValueError("llama.cpp資産一覧に固定実行ファイルがありません")
    for item in [*models, *manifest["notices"], *runtime["archives"]]:
        validate_asset(item)
        if type(item.get("bytes")) is not int or item["bytes"] <= 0:
            raise ValueError("取得資産のサイズが不正です")
    for item in [*models, *manifest["notices"]]:
        safe_member(item["path"], root)
        if not item["path"].startswith(("models/saluki/", "runtime/llama-cpp/")):
            raise ValueError("Saluki資産の保存先が専用フォルダー外です")
    for item in files:
        safe_member(item["path"], Path(root) / "runtime/llama-cpp")
        if type(item.get("bytes")) is not int or item["bytes"] <= 0:
            raise ValueError("llama.cppファイルサイズが不正です")
        if len(item["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in item["sha256"]):
            raise ValueError("llama.cppファイルのSHA-256が不正です")


def valid_file(path, item):
    path = Path(path)
    return path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]


def existing_complete(root, manifest, signature):
    marker = Path(root) / "data/setup-saluki.json"
    try:
        record = read(marker)
        if record.get("status") != "complete" or record.get("manifest_sha256") != signature:
            return False
        files = record["installed_files"]
        # A receipt cannot bypass the pinned models, runtime and notices.
        required = {item["path"]: item for item in [*manifest["models"], *manifest["notices"]]}
        required.update({"runtime/llama-cpp/" + item["path"]: item for item in manifest["runtime"]["files"]})
        recorded = {item["path"]: item for item in files}
        if len(recorded) != len(files) or any(recorded.get(name, {}).get("sha256") != item["sha256"]
                                                or recorded.get(name, {}).get("bytes") != item["bytes"]
                                                for name, item in required.items()):
            return False
        if not all(valid_file(safe_member(item["path"], root), item) for item in files):
            return False
        runtime_path = Path(root) / "runtime/llama-cpp/asset-manifest.json"
        if sha(runtime_path) != record["runtime_manifest_sha256"]:
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_assets(root, manifest, offline):
    """Stage and check every asset before publishing a new completion receipt."""
    root = Path(root)
    staged = []
    for item in [*manifest["models"], *manifest["notices"]]:
        target = safe_member(item["path"], root)
        # Reuse installed model files too, avoiding another 8 GB download/cache.
        source = target if valid_file(target, item) else download(item, root / "downloads/saluki", offline)
        if not valid_file(source, item):
            raise RuntimeError("Saluki取得資産のSHA・サイズが一致しません: " + item["filename"])
        staged.append((source, target, item))
    runtime_root = root / "runtime/llama-cpp"
    runtime = manifest["runtime"]
    runtime_ok = all(valid_file(safe_member(item["path"], runtime_root), item) for item in runtime["files"])
    extras = []
    with tempfile.TemporaryDirectory(prefix="setup-saluki-", dir=root / "temp") as temporary:
        directory = Path(temporary)
        if not runtime_ok:
            for number, item in enumerate(runtime["archives"]):
                archive = download(item, root / "downloads/saluki", offline)
                if not valid_file(archive, item):
                    raise RuntimeError("llama.cppアーカイブのSHA・サイズが一致しません: " + item["filename"])
                package = directory / str(number)
                extract(archive, package)
                # Retain additional upstream notices even when not in the baseline list.
                for source in sorted(package.rglob("*")):
                    if source.is_file() and any(token in source.name.upper() for token in ("LICENSE", "NOTICE", "COPYING")):
                        relative = source.relative_to(package).as_posix()
                        target = runtime_root / "legal" / item["filename"] / relative
                        extras.append((source, target, {"bytes": source.stat().st_size, "sha256": sha(source)}))
            for item in runtime["files"]:
                candidates = [source for source in directory.rglob(Path(item["path"]).name) if valid_file(source, item)]
                if len(candidates) != 1:
                    raise RuntimeError("llama.cpp展開結果が一意に固定一覧と一致しません: " + item["path"])
                staged.append((candidates[0], safe_member(item["path"], runtime_root), item))
        else:
            staged.extend((runtime_root / item["path"], runtime_root / item["path"], item) for item in runtime["files"])
        installed = []
        for source, target, item in [*staged, *extras]:
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve():
                building = target.with_name(target.name + ".building")
                shutil.copy2(source, building)
                if not valid_file(building, item):
                    raise RuntimeError("Saluki配置後のSHA・サイズが一致しません: " + target.name)
                building.replace(target)
            installed.append({"path": target.relative_to(root).as_posix(), "bytes": item["bytes"], "sha256": item["sha256"]})
    # Already retained upstream notices remain part of the checked runtime contract.
    legal = runtime_root / "legal"
    known = {item["path"] for item in installed}
    for file in sorted(legal.rglob("*")) if legal.is_dir() else []:
        name = file.relative_to(root).as_posix()
        if file.is_file() and name not in known:
            installed.append({"path": name, "bytes": file.stat().st_size, "sha256": sha(file)})
    runtime_files = [{"path": item["path"].removeprefix("runtime/llama-cpp/"), "bytes": item["bytes"], "sha256": item["sha256"]}
                     for item in installed if item["path"].startswith("runtime/llama-cpp/")]
    runtime_manifest = runtime_root / "asset-manifest.json"
    write(runtime_manifest, {"version": runtime["version"], "tag": runtime["version"],
                             "source_commit": runtime["source_commit"], "executable_sha256": runtime["executable_sha256"],
                             "files": sorted(runtime_files, key=lambda item: item["path"])})
    if not all(valid_file(root / item["path"], item) for item in installed):
        raise RuntimeError("Saluki配置後の最終資産検査に失敗しました")
    return installed, sha(runtime_manifest)


def setup(root, *, offline=False):
    root = Path(root).resolve(strict=True)
    manifest_path = root / "assets/saluki-assets.json"
    manifest = read(manifest_path)
    validate_manifest(root, manifest)
    signature = sha(manifest_path)
    marker = root / "data/setup-saluki.json"
    for name in ("data", "temp"):
        (root / name).mkdir(exist_ok=True)
    with operation_lock(root):
        if existing_complete(root, manifest, signature):
            print("Saluki資産は検査済みです。選択設定は保持しています。", flush=True)
            return read(marker)
        # Downloads and their installed copies coexist until explicitly removed.
        if not all(valid_file(root / item["path"], item) for item in manifest["models"]):
            if shutil.disk_usage(root).free < manifest["minimum_initial_free_bytes"]:
                raise RuntimeError("Salukiの初回追加には22.5GiB以上の空き容量が必要です")
        state = {"status": "in_progress", "backend": "saluki", "manifest_sha256": signature,
                 "model_source_revision": manifest["source_revision"], "runtime_version": manifest["runtime"]["version"],
                 "inference_verified": False, "selection_changed": False}
        write(marker, state)
        try:
            print("Salukiの固定モデル・実行資産・許諾文書を準備します。", flush=True)
            installed, runtime_sha = prepare_assets(root, manifest, offline)
            state.update(status="complete", installed_files=installed, runtime_manifest_sha256=runtime_sha,
                         completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            write(marker, state)
            print("Saluki資産の準備が完了しました。推論動作は未確認です。現在のモデル選択は保持しています。", flush=True)
            return state
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            write(marker, state)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="検査済み取得キャッシュ・配置済み資産だけを使う")
    args = parser.parse_args()
    if os.name != "nt" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Windows x64 Python 3.13用の追加設定です")
    setup(args.root, offline=args.offline)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("中断しました。再実行すると検査済み資産と取得途中データを再利用します。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Saluki追加設定失敗: {error}", file=sys.stderr)
        raise SystemExit(1)

"""Own the portable model store and keep selection separate from downloads."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading

from filelock import FileLock, Timeout as LockTimeout
import httpx

from . import llm, model_download
from .model_probe import PROBE_VERSION, create_probe
from .model_terms import ModelTerms, ModelTermsError, require_model_terms
from .profiles import DEFAULT_PROFILE_ID, list_profiles as catalog, resolve_profile
from .storage import digest, local_path, new_id, now, read_json, relative, sha, write_json


class ModelManagerError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ModelOperationCancelled(ModelManagerError):
    def __init__(self):
        super().__init__("CANCELLED", "操作を中止しました。使用モデルは変更していません。")


def _code_fingerprints() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    return {name: sha(source / name) for name in
            ("llm.py", "model_manager.py", "model_download.py", "model_probe.py", "profiles.py", "model_terms.py")}


def check_fingerprint(profile, model_digest: str) -> str:
    profile = resolve_profile(profile)
    return digest({"profile": profile.snapshot(), "model_digest": model_digest,
                   "runtime_version": llm.RUNTIME_VERSION, "executable_sha256": llm.EXPECTED_EXE,
                   "system_prompt_sha256": hashlib.sha256(llm.SYSTEM.encode()).hexdigest(),
                   "schema_sha256": digest(llm.SCHEMA), "probe_version": PROBE_VERSION,
                   "source_sha256": _code_fingerprints()})


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise ModelOperationCancelled()


def _emit(progress, stage, status, **details):
    if progress is not None:
        progress({"stage": stage, "status": status, **details})


def _profile(value=None):
    try:
        selected = resolve_profile(value)
        known = {profile.id: profile for profile in catalog()}
        if selected.id not in known or selected.model != known[selected.id].model:
            raise ValueError("unregistered model identity")
        return selected
    except (ValueError, TypeError) as exc:
        raise ModelManagerError("UNKNOWN_PROFILE", "このモデルの設定は登録されていません。") from exc


@contextmanager
def _watch_cancel(server, cancel, client=None):
    """Close the request and its own daemon if a blocked operation is cancelled."""
    finished = threading.Event()
    watcher = None
    def watch():
        while not finished.wait(.05):
            if cancel.is_set():
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
                # A blocked non-streaming check needs the owned server socket closed.
                server.stop()
                return
    if cancel is not None:
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
    try:
        yield
    except Exception:
        _cancelled(cancel)
        raise
    finally:
        finished.set()
        if watcher is not None:
            watcher.join()


def _pull_model(server, profile, cancel=None, progress=None):
    with httpx.Client(trust_env=False, timeout=httpx.Timeout(120, connect=10)) as client, _watch_cancel(server, cancel, client):
        with client.stream("POST", f"http://127.0.0.1:{server.port}/api/pull",
                           json={"model": profile.model, "stream": True}) as response:
            if response.status_code != 200:
                raise ModelManagerError("DOWNLOAD_FAILED", f"モデル導入に失敗しました: HTTP {response.status_code}")
            success = False
            for line in response.iter_lines():
                _cancelled(cancel)
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except ValueError as exc:
                    raise ModelManagerError("DOWNLOAD_FAILED", "導入の進捗を読み取れません。") from exc
                if not isinstance(item, dict):
                    raise ModelManagerError("DOWNLOAD_FAILED", "進捗が不正です。")
                if item.get("error"):
                    raise ModelManagerError("DOWNLOAD_FAILED", str(item["error"]))
                _emit(progress, "download", item.get("status", "導入中"),
                      **{key: item[key] for key in ("digest", "completed", "total") if key in item})
                success = item.get("status") == "success"
            if not success:
                raise ModelManagerError("DOWNLOAD_INCOMPLETE", "導入の正常完了を確認できません。")
    _cancelled(cancel)


class ModelManager:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.registry_path = self.root / "data/model-registry.json"

    @contextmanager
    def _operation(self):
        self.root.joinpath("data").mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(self.root / "data/operation.lock"), timeout=0):
                yield
        except LockTimeout as exc:
            raise ModelManagerError("BUSY", "別の処理が実行中です。終了後にやり直してください。") from exc

    def _registry(self):
        if not self.registry_path.exists():
            return {"version": 1, "models": {}}
        try:
            saved = read_json(self.registry_path)
            if (not isinstance(saved, dict) or saved.get("version") != 1 or
                    not isinstance(saved.get("models"), dict)):
                raise ValueError("invalid registry")
            for key, record in saved["models"].items():
                if key not in {p.id for p in catalog()} or not isinstance(record, dict):
                    raise ValueError("invalid registered profile")
                if record.get("profile_id") != key:
                    raise ValueError("profile identity mismatch")
                if record.get("model_tag") != _profile(key).model:
                    raise ValueError("model tag mismatch")
                if not isinstance(record.get("digest"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["digest"]):
                    raise ValueError("invalid model digest")
                if type(record.get("full_hash_verified")) is not bool:
                    raise ValueError("invalid hash verification state")
                check = record.get("check")
                if check is not None and (not isinstance(check, dict) or
                        check.get("status") not in ("passed", "failed") or
                        not isinstance(check.get("fingerprint"), str)):
                    raise ValueError("invalid check record")
            return saved
        except (OSError, ValueError, TypeError) as exc:
            raise ModelManagerError("INVALID_REGISTRY", "モデルの記録を読めません。自動で上書きしません。") from exc

    def terms_status(self, profile_id):
        profile = _profile(profile_id)
        try:
            return ModelTerms(self.root).status(profile.id)
        except ModelTermsError as exc:
            raise ModelManagerError(exc.code, str(exc)) from exc

    def accept_terms(self, profile_id, document_hashes, *, accepted):
        profile = _profile(profile_id)
        with self._operation():
            try:
                return ModelTerms(self.root).accept(profile.id, document_hashes, accepted=accepted)
            except ModelTermsError as exc:
                raise ModelManagerError(exc.code, str(exc)) from exc

    def _require_terms(self, profile):
        try:
            return require_model_terms(self.root, profile.id)
        except ModelTermsError as exc:
            raise ModelManagerError(exc.code, str(exc)) from exc

    def _asset_identity(self, profile, *, full_hash=False, cancel=None, progress=None):
        profile = _profile(profile)
        # Catalog tags are fixed, never user-controlled paths.
        name, tag = profile.model.split(":", 1)
        path = local_path(self.root, f"models/ollama/manifests/registry.ollama.ai/library/{name}/{tag}")
        if not path.is_file():
            raise ModelManagerError("NOT_INSTALLED", "このモデルは導入されていません。")
        try:
            data = read_json(path)
            refs = [data["config"], *data["layers"]]
            model_digest = sha(path)
            if profile.expected_digest is not None and model_digest not in (profile.expected_digest, profile.fallback_digest):
                raise ModelManagerError("MODEL_CHANGED", "既存の検証済みモデル定義と一致しません。")
            blobs = []
            for ref in refs:
                _cancelled(cancel)
                blob_digest, size = ref["digest"], ref["size"]
                if (not isinstance(blob_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", blob_digest)
                        or type(size) is not int or size < 0):
                    raise ValueError("invalid blob reference")
                blob = local_path(self.root, "models/ollama/blobs/" + blob_digest.replace(":", "-"))
                if not blob.is_file() or blob.stat().st_size != size:
                    raise ModelManagerError("CORRUPT_MODEL", "モデルの参照ファイルが不足または破損しています。")
                if full_hash:
                    _emit(progress, "verify", "モデルファイルのSHA-256を検査しています。", digest=blob_digest)
                    hash_value = hashlib.sha256()
                    with blob.open("rb") as stream:
                        for block in iter(lambda: stream.read(4*1024*1024), b""):
                            _cancelled(cancel)
                            hash_value.update(block)
                    if hash_value.hexdigest() != blob_digest.removeprefix("sha256:"):
                        raise ModelManagerError("CORRUPT_MODEL", "モデルファイルのSHA-256が一致しません。")
                blobs.append({"digest": blob_digest, "bytes": size})
            return {"profile_id": profile.id, "model_tag": profile.model, "digest": model_digest,
                    "asset_variant": "huggingface" if model_digest == profile.fallback_digest else "ollama",
                    "manifest_sha256": model_digest, "manifest_path": relative(self.root, path),
                    "blobs": blobs, "size_bytes": sum(b["bytes"] for b in blobs)}
        except ModelManagerError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelManagerError("CORRUPT_MODEL", "モデル定義を正しく読めません。") from exc

    def _capabilities(self, server, profile):
        shown = server.api("/api/show", {"model": profile.model}, timeout=30)
        caps = shown.get("capabilities", [])
        if shown.get("remote_host") or shown.get("remote_model"):
            raise ModelManagerError("REMOTE_MODEL", "クラウドモデルはこのアプリで使用できません。")
        if not isinstance(caps, list) or not {"completion", "vision"}.issubset(caps):
            raise ModelManagerError("UNSUPPORTED_MODEL", "文字と画像の抽出に対応するモデルではありません。")
        if profile.think is True and "thinking" not in caps:
            raise ModelManagerError("UNSUPPORTED_THINKING", "このモデルは設定されたThinkingに対応しません。")
        requires = shown.get("requires")
        if requires:
            match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", requires)
            current = tuple(map(int, llm.RUNTIME_VERSION.split(".")))
            if match is None or tuple(map(int, match.groups())) > current:
                raise ModelManagerError("UNSUPPORTED_RUNTIME", "モデルが必要とするOllamaの版に対応していません。")
        return shown

    def _record_identity(self, identity, registry, **extra):
        previous = registry["models"].get(identity["profile_id"], {})
        check = previous.get("check")
        fingerprint = check_fingerprint(identity["profile_id"], identity["digest"])
        if not isinstance(check, dict) or check.get("fingerprint") != fingerprint:
            check = None
        acquisition = previous.get("acquisition")
        return {**identity, **({"acquisition": deepcopy(acquisition)} if isinstance(acquisition, dict) else {}),
                "installed_at": previous.get("installed_at", now()),
                "full_hash_verified": True, "hash_verified_at": now(), "check": check, **extra}

    def list_profiles(self):
        with self._operation():
            registry = self._registry()
            execution = read_json(self.root / "実行設定.json")
            selected = execution.get("ollama_profile", DEFAULT_PROFILE_ID) if execution.get("llm_backend", "gemma") == "gemma" else None
            with llm.OllamaServer(self.root, management=True) as server:
                tags = server.api("/api/tags", timeout=30).get("models", [])
                entries = {item.get("name", item.get("model")): item for item in tags if isinstance(item, dict)}
                cards = []
                for profile in catalog():
                    card = {"profile_id": profile.id, "label": profile.label, "model_tag": profile.model,
                            "size_bytes": 0, "installed": False, "checked": False,
                            "check_status": "未導入", "selected": selected == profile.id}
                    if profile.model in entries:
                        card["installed"] = True
                        card["size_bytes"] = entries[profile.model].get("size", 0)
                        try:
                            identity = self._asset_identity(profile)
                            card["asset_variant"] = identity["asset_variant"]
                            if identity["asset_variant"] == "huggingface":
                                card["label"] += "（Hugging Face版）"
                            if entries[profile.model].get("digest") != identity["digest"]:
                                raise ModelManagerError("MODEL_CHANGED", "APIとモデル定義が一致しません。")
                            self._capabilities(server, profile)
                            record = registry["models"].get(profile.id, {})
                            check = record.get("check") or {}
                            card["checked"] = bool(record.get("full_hash_verified") and
                                record.get("digest") == identity["digest"] and check.get("status") == "passed" and
                                check.get("fingerprint") == check_fingerprint(profile, identity["digest"]))
                            if card["checked"]:
                                self.ensure_usable(profile, full_hash=False)
                            card["check_status"] = "確認済み" if card["checked"] else ("再確認が必要" if check else "未確認")
                            card["digest"] = identity["digest"]
                        except ModelManagerError as exc:
                            card.update(checked=False,
                                        check_status="再確認が必要" if exc.code == "UNCHECKED_MODEL" else "利用不可",
                                        error_code=exc.code, message=str(exc))
                    cards.append(card)
            if not server.shutdown_report.get("process_stopped") or server.shutdown_report.get("errors"):
                raise ModelManagerError("CLEANUP_FAILED", "専用サーバーの終了確認に失敗しました。")
            return cards

    def install(self, profile_id, *, cancel=None, progress=None):
        profile = _profile(profile_id)
        with self._operation():
            registry = self._registry()
            _cancelled(cancel)
            self._require_terms(profile)
            with llm.OllamaServer(self.root, management=True) as server:
                tags = server.api("/api/tags", timeout=30).get("models", [])
                existing = any(item.get("name", item.get("model")) == profile.model for item in tags)
                acquisition = {"source": "existing" if existing else "ollama"}
                if existing:
                    _emit(progress, "verify", "導入済みです。再ダウンロードせず検査します。")
                else:
                    estimate = profile.download_size_bytes
                    if shutil.disk_usage(self.root).free < estimate + 1024**3:
                        raise ModelManagerError("DISK_FULL", "モデル導入先の空き容量が不足しています。")
                    try:
                        _pull_model(server, profile, cancel, progress)
                    except ModelOperationCancelled:
                        raise
                    except (httpx.HTTPError, ModelManagerError) as primary_error:
                        _cancelled(cancel)
                        if isinstance(primary_error, ModelManagerError) and primary_error.code not in (
                                "DOWNLOAD_FAILED", "DOWNLOAD_INCOMPLETE"):
                            raise
                        _emit(progress, "fallback", "Ollamaから取得できませんでした。Hugging Face直接取得を試します。")
                        try:
                            acquisition = model_download.install_from_huggingface(
                                self.root, profile, cancel=cancel, progress=progress)
                        except ModelOperationCancelled:
                            raise
                        except (model_download.HuggingFaceDownloadError, OSError) as fallback_error:
                            _cancelled(cancel)
                            raise ModelManagerError("MODEL_DOWNLOAD_FAILED",
                                f"Ollamaから取得できませんでした: {primary_error}\n"
                                f"Hugging Faceからも取得できませんでした: {fallback_error}\n"
                                "モデル導入を停止します。ネット接続・プロキシ・保存先を確認してください。") from fallback_error
                    _cancelled(cancel)
                identity = self._asset_identity(profile, full_hash=True, cancel=cancel, progress=progress)
                tag = next((item for item in server.api("/api/tags").get("models", [])
                            if item.get("name", item.get("model")) == profile.model), None)
                if tag is None or tag.get("digest") != identity["digest"]:
                    raise ModelManagerError("MODEL_CHANGED", "導入後のAPIとモデル定義が一致しません。")
                shown = self._capabilities(server, profile)
                _cancelled(cancel)
                record = self._record_identity(identity, registry, capabilities=shown["capabilities"],
                                               acquisition=acquisition)
            if not server.shutdown_report.get("process_stopped") or server.shutdown_report.get("errors"):
                raise ModelManagerError("CLEANUP_FAILED", "専用サーバーの終了確認に失敗しました。導入記録を確定していません。")
            registry["models"][profile.id] = record
            write_json(self.registry_path, registry)
            _emit(progress, "complete", "導入・ファイル検査が完了しました。動作確認を行ってください。")
            return record

    def check(self, profile_id, *, cancel=None, progress=None):
        profile = _profile(profile_id)
        with self._operation():
            registry = self._registry()
            _cancelled(cancel)
            self._require_terms(profile)
            identity = self._asset_identity(profile, full_hash=True, cancel=cancel, progress=progress)
            fingerprint = check_fingerprint(profile, identity["digest"])
            report_folder = self.root / "data/model-checks" / new_id()
            report = {"version": 1, "profile": profile.snapshot(), "digest": identity["digest"],
                      "fingerprint": fingerprint, "probe_version": PROBE_VERSION,
                      "started_at": now(), "status": "running", "cases": []}
            write_json(report_folder / "check.json", report)
            server = llm.OllamaServer(self.root, profile=profile, management=True)
            try:
                with server, _watch_cancel(server, cancel):
                    shown = self._capabilities(server, profile)
                    server.contract = llm.model_contract(self.root, profile)
                    # A new unregistered tag has just been hash checked. Use that identity
                    # without weakening runtime executable verification.
                    image, page, regions, cases = create_probe(report_folder / "input")
                    write_json(report_folder / "input/ocr.json", {"page": page, "regions": regions})
                    report["input_files"] = [{"path": "input/page.png", "sha256": sha(image)},
                                             {"path": "input/ocr.json", "sha256": sha(report_folder / "input/ocr.json")}]
                    for case in cases:
                        _cancelled(cancel)
                        _emit(progress, "check", "画像とOCRで動作確認しています。", case=case["name"])
                        record = llm.extract_page(server, image, page, regions, case["settings"])
                        write_json(report_folder / f"{case['name']}.json", record)
                        _cancelled(cancel)
                        actual = record.get("parsed")
                        passed = bool(actual and actual.get("status") == case["expected"]["status"] and
                                      actual.get("results") == case["expected"]["results"])
                        report["cases"].append({"name": case["name"], "expected": case["expected"],
                                                "passed": passed, "record_path": f"{case['name']}.json",
                                                "record_sha256": sha(report_folder / f"{case['name']}.json")})
                        write_json(report_folder / "check.json", report)
                    if not all(case["passed"] for case in report["cases"]):
                        raise ModelManagerError("CHECK_FAILED", "動作確認の期待値または根拠IDと一致しません。結果は保存しました。")
                    # Detect replacement while the model was being checked.
                    if self._asset_identity(profile)["digest"] != identity["digest"]:
                        raise ModelManagerError("MODEL_CHANGED", "動作確認中にモデル定義が変更されました。")
                    if check_fingerprint(profile, identity["digest"]) != fingerprint:
                        raise ModelManagerError("PROFILE_CHANGED", "動作確認中に設定またはプログラムが変更されました。")
                if (not server.shutdown_report.get("process_stopped") or
                        not server.shutdown_report.get("model_unloaded") or server.shutdown_report.get("errors")):
                    raise ModelManagerError("CLEANUP_FAILED", "モデルの解放または専用サーバーの終了確認に失敗しました。")
                _cancelled(cancel)
                report.update(status="passed", finished_at=now(), shutdown=deepcopy(server.shutdown_report))
                write_json(report_folder / "check.json", report)
                saved = self._record_identity(identity, registry, capabilities=shown["capabilities"],
                    check={"status": "passed", "fingerprint": fingerprint,
                           "checked_at": now(), "report_path": relative(self.root, report_folder / "check.json"),
                           "report_sha256": sha(report_folder / "check.json")})
                registry["models"][profile.id] = saved
                write_json(self.registry_path, registry)
            except BaseException as exc:
                report["status"] = "cancelled" if isinstance(exc, ModelOperationCancelled) else "failed"
                report["error"] = {"code": getattr(exc, "code", "CHECK_ERROR"), "message": str(exc)}
                if getattr(exc, "code", None) == "CHECK_FAILED":
                    failed = self._record_identity(identity, registry, check={"status": "failed",
                        "fingerprint": check_fingerprint(profile, identity["digest"]),
                        "checked_at": now(), "report_path": relative(self.root, report_folder / "check.json")})
                    registry["models"][profile.id] = failed
                    write_json(self.registry_path, registry)
                raise
            finally:
                report.setdefault("finished_at", now())
                report["shutdown"] = deepcopy(server.shutdown_report)
                write_json(report_folder / "check.json", report)
            _emit(progress, "complete", "動作確認に合格しました。図面の実務精度とは別の確認です。")
            return saved

    def ensure_usable(self, profile=None, *, full_hash=True):
        """Validate under the caller's operation lock, once before a run/selection.

        Only the read-only model list skips full blob hashing. A passed historical
        hash does not prove that today's same-size files still match their digest.
        """
        profile = _profile(profile)
        self._require_terms(profile)
        registry = self._registry()
        identity = self._asset_identity(profile, full_hash=full_hash)
        record = registry["models"].get(profile.id, {})
        if record.get("digest") and record["digest"] != identity["digest"]:
            raise ModelManagerError("MODEL_CHANGED", "動作確認後にモデルが変わりました。再確認してください。")
        check = record.get("check") or {}
        if (not record.get("full_hash_verified") or check.get("status") != "passed" or
                check.get("fingerprint") != check_fingerprint(profile, identity["digest"])):
            raise ModelManagerError("UNCHECKED_MODEL", "モデル管理画面で動作確認を行ってください。")
        try:
            report_path = local_path(self.root, check["report_path"])
            if check.get("kind") == "historical-regression":
                valid_report = report_path.is_file() and sha(report_path) == check.get("report_sha256")
            else:
                report = read_json(report_path)
                valid_report = (report.get("status") == "passed" and report.get("digest") == identity["digest"]
                                and report.get("profile") == profile.snapshot() and
                                report.get("fingerprint") == check.get("fingerprint") and
                                sha(report_path) == check.get("report_sha256"))
                if valid_report:
                    artifacts = [*report["input_files"], *({"path": case["record_path"],
                        "sha256": case["record_sha256"]} for case in report["cases"])]
                    valid_report = all(sha(local_path(report_path.parent, artifact["path"])) == artifact["sha256"]
                                       for artifact in artifacts)
            if not valid_report:
                raise ValueError("invalid check evidence")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ModelManagerError("UNCHECKED_MODEL", "動作確認の保存記録が不足または破損しています。再確認してください。") from exc
        if not llm.model_contract(self.root, profile).get("ready"):
            raise ModelManagerError("CORRUPT_MODEL", "モデルまたは実行資産の検査に失敗しました。")
        return deepcopy(record)

    def select(self, profile_id):
        profile = _profile(profile_id)
        with self._operation():
            record = self.ensure_usable(profile)
            path = self.root / "実行設定.json"
            execution = read_json(path)
            if not isinstance(execution, dict):
                raise ModelManagerError("INVALID_SETTINGS", "実行設定がオブジェクト形式ではありません。")
            # Keep input and Saluki settings byte-for-value; only selection changes.
            execution.update(llm_backend="gemma", ollama_profile=profile.id)
            from .settings import ExecutionSettings
            ExecutionSettings.model_validate(execution)
            write_json(path, execution)
            return {"profile_id": profile.id, "model_tag": profile.model,
                    "digest": record["digest"], "selected": True}

    def register_historical_check(self, profile_id=DEFAULT_PROFILE_ID, *, report_path, summary=None):
        """Called only after the owner has verified a recorded regression result."""
        profile = _profile(profile_id)
        if profile.id != DEFAULT_PROFILE_ID:
            raise ModelManagerError("INVALID_HISTORICAL_CHECK", "既存Gemma4以外は新しい動作確認が必要です。")
        with self._operation():
            report = local_path(self.root, report_path)
            if not report.is_file():
                raise ModelManagerError("MISSING_CHECK_REPORT", "検証記録がありません。")
            registry = self._registry()
            identity = self._asset_identity(profile, full_hash=True)
            if identity["digest"] != profile.expected_digest:
                raise ModelManagerError("INVALID_HISTORICAL_CHECK", "Hugging Face版は新しい動作確認が必要です。")
            record = self._record_identity(identity, registry, check={"status": "passed",
                "fingerprint": check_fingerprint(profile, identity["digest"]), "checked_at": now(),
                "report_path": relative(self.root, report), "report_sha256": sha(report),
                "kind": "historical-regression", "summary": summary or {}})
            registry["models"][profile.id] = record
            write_json(self.registry_path, registry)
            return record


def ensure_usable(root: Path, profile=None, *, full_hash=True):
    return ModelManager(root).ensure_usable(profile, full_hash=full_hash)


def register_historical_check(root: Path, profile_id=DEFAULT_PROFILE_ID, *, report_path, summary=None):
    return ModelManager(root).register_historical_check(profile_id, report_path=report_path, summary=summary)

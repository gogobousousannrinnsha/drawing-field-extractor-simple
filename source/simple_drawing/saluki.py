"""Saluki vision extraction through a pinned, application-owned llama.cpp server."""
from __future__ import annotations

import base64
from copy import deepcopy
import json
import mimetypes
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from typing import Any

import httpx
import psutil

from .llm import SCHEMA, SYSTEM, OutputError, _gpu_sample, _sha_bytes, _sha_file, build_payload, validate_output

MODEL = "Underdog-Saluki-27B-1.0-IQ2-mix"
PORT = 11450
RUNTIME_VERSION = "b11429"
MODEL_PATH = "models/saluki/Underdog-Saluki-27B-1.0-IQ2-mix.gguf"
PROJECTOR_PATH = "models/saluki/mmproj-Underdog-Saluki-27B-1.0-F16.gguf"
ASSETS = {
    MODEL_PATH: {"bytes": 7898369152, "sha256": "4a673518b11b1c4445f9b9a9d3c40356f5ba39f6a6260dcb25f3fbfd475d9efb"},
    PROJECTOR_PATH: {"bytes": 927606912, "sha256": "f3aa57813a513420c2ef765503b057c1946bf4c8cdf3679d266a6644fc6cfd50"},
}


class SalukiAPIError(RuntimeError):
    code = "API_ERROR"

    def __init__(self, message: str, response: dict[str, Any]):
        self.response = response
        # The server rejects a full prompt when context shifting is disabled.
        if re.search(r"exceed.*context|context.*(overflow|exceed|too (?:long|large))|exceed_context_size", message, re.I):
            self.code = "CONTEXT_OVERFLOW"
        super().__init__(message)


def runtime_config(*, thinking: bool = False, context_size: int = 32768,
                   max_tokens: int | None = None, image_max_tokens: int | None = None,
                   batch_size: int = 512, micro_batch_size: int = 256,
                   kv_cache_type: str = "f16") -> dict[str, Any]:
    if type(thinking) is not bool:
        raise ValueError("Saluki thinkingはtrueまたはfalseです。")
    if type(context_size) is not int or not 8192 <= context_size <= 131072:
        raise ValueError("Saluki context_sizeは8192から131072の整数です。")
    if max_tokens is None:
        max_tokens = 16384 if thinking else 8192
    if type(max_tokens) is not int or not 1 <= max_tokens <= min(32768, context_size):
        raise ValueError("Saluki max_tokensはcontext_size以下の1から32768の整数です。")
    if image_max_tokens is not None and (type(image_max_tokens) is not int or not 256 <= image_max_tokens <= 16384):
        raise ValueError("Saluki image_max_tokensは256から16384の整数です。")
    if (type(batch_size) is not int or type(micro_batch_size) is not int
            or not 32 <= micro_batch_size <= batch_size <= 4096):
        raise ValueError("Salukiのbatch設定は32から4096でmicro_batch_size以下にならないbatch_sizeです。")
    if kv_cache_type not in ("f16", "q8_0"):
        raise ValueError("Saluki kv_cache_typeはf16またはq8_0です。")
    return {"thinking": thinking, "context_size": context_size, "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1, "top_k": 64, "repeat_penalty": 1,
            "presence_penalty": 0, "seed": 42, "flash_attention": True,
            "gpu_layers": 99, "parallel": 1, "batch_size": batch_size, "ubatch_size": micro_batch_size,
            "cache_type_k": kv_cache_type, "cache_type_v": kv_cache_type, "mmproj_gpu": True,
            "context_shift": False, "image_tokens": image_max_tokens or "runtime_default"}


def model_contract(root: Path, *, thinking: bool = False, context_size: int = 32768,
                   max_tokens: int | None = None, image_max_tokens: int | None = None,
                   batch_size: int = 512, micro_batch_size: int = 256,
                   kv_cache_type: str = "f16") -> dict[str, Any]:
    root = Path(root).resolve()
    config = runtime_config(thinking=thinking, context_size=context_size, max_tokens=max_tokens,
        image_max_tokens=image_max_tokens, batch_size=batch_size, micro_batch_size=micro_batch_size,
        kv_cache_type=kv_cache_type)
    runtime = root / "runtime/llama-cpp"
    executable = runtime / "llama-server.exe"
    manifest_path = runtime / "asset-manifest.json"
    issues: list[str] = []
    runtime_files: list[dict[str, Any]] = []
    manifest_hash = None
    exe_hash = None
    try:
        manifest_hash = _sha_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != RUNTIME_VERSION or manifest.get("tag") != RUNTIME_VERSION:
            raise ValueError("runtime manifest version mismatch")
        entries = manifest["files"]
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ValueError("runtime manifest file list missing")
        files = {entry["path"]: entry for entry in entries}
        if len(files) != len(entries) or "llama-server.exe" not in files:
            raise ValueError("runtime manifest executable missing")
        for name, entry in sorted(files.items()):
            expected = entry["sha256"]
            path = (runtime / name).resolve()
            if not path.is_relative_to(runtime.resolve()) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("invalid runtime manifest path/hash")
            actual = _sha_file(path) if path.is_file() else None
            valid = actual == expected and path.stat().st_size == entry["bytes"]
            runtime_files.append({"path": path.relative_to(root).as_posix(), "sha256": actual,
                                  "expected_sha256": expected, "bytes": entry["bytes"], "valid": valid})
            if not valid:
                issues.append("llama.cppのファイルが不足または一致しません: " + name)
            if name == "llama-server.exe":
                exe_hash = actual
        if exe_hash != manifest.get("executable_sha256"):
            issues.append("llama.cpp実行ファイルのSHA-256が定義と一致しません。")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        issues.append("検証済みのllama.cpp実行ファイル定義がありません、または一致しません: " + str(exc))
    assets = []
    for name, expected in ASSETS.items():
        path = root / name
        valid = path.is_file() and path.stat().st_size == expected["bytes"]
        assets.append({"path": name, **expected, "size_valid": valid})
        if not valid:
            issues.append("Salukiのモデルファイルが不足またはサイズが一致しません: " + name)
    return {"backend": "saluki", "model": MODEL, "runtime_version": RUNTIME_VERSION,
            "executable": executable.relative_to(root).as_posix(), "executable_sha256": exe_hash,
            "manifest": manifest_path.relative_to(root).as_posix(), "manifest_sha256": manifest_hash,
            "runtime_files": runtime_files, "assets": assets, "model_hashes_checked_at_startup": False,
            "adapter_sha256": _sha_file(Path(__file__)), "config": config,
            "system_prompt": SYSTEM, "system_prompt_sha256": _sha_bytes(SYSTEM.encode()),
            "schema": deepcopy(SCHEMA), "stream": False, "truncate": False,
            "backend_dll": "runtime/llama-cpp/ggml-cuda.dll", "log_verbosity": 4,
            "ready": not issues, "issues": issues}


def gpu_offload_evidence(log: str) -> dict[str, Any]:
    """Require positive full-layer offload and vision/projector CUDA evidence."""
    layers = re.findall(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", log, re.I)
    full = bool(layers) and all(int(done) == int(total) and int(total) > 0 for done, total in layers)
    projector_lines = [line for line in log.splitlines() if
        re.search(r"(?:clip|mtmd|mmproj|projector).*CUDA|CUDA.*(?:clip|mtmd|mmproj|projector)", line, re.I)]
    return {"full_gpu_offload": full, "layers": [list(map(int, pair)) for pair in layers],
            "projector_gpu": bool(projector_lines), "projector_log_lines": projector_lines}


class SalukiServer:
    """Own a dedicated local server, refusing shared ports and CPU fallback."""
    def __init__(self, root: Path, port: int = PORT, *, thinking: bool = False,
                 context_size: int = 32768, max_tokens: int | None = None,
                 image_max_tokens: int | None = None, batch_size: int = 512,
                 micro_batch_size: int = 256, kv_cache_type: str = "f16"):
        if port != PORT:
            raise ValueError("Saluki専用ポートは11450です。")
        self.root = Path(root).resolve()
        self.port = port
        self.config = runtime_config(thinking=thinking, context_size=context_size, max_tokens=max_tokens,
            image_max_tokens=image_max_tokens, batch_size=batch_size, micro_batch_size=micro_batch_size,
            kv_cache_type=kv_cache_type)
        self.contract: dict[str, Any] = {}
        self.process: subprocess.Popen | None = None
        self._process_created: float | None = None
        self._log_files: list[Any] = []
        self._log_paths: list[Path] = []
        self.shutdown_report: dict[str, Any] = {}

    def api(self, path: str, body: dict | None = None, timeout: float = 900) -> dict:
        if path not in ("/health", "/props", "/v1/chat/completions", "/v1/models"):
            raise ValueError("Invalid local API path")
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            response = client.request("GET" if body is None else "POST",
                f"http://127.0.0.1:{self.port}{path}", content=data,
                headers={"Content-Type": "application/json"})
            diagnostic = {"http_status": response.status_code,
                          "body": response.content.decode("utf-8", errors="replace")}
            if response.status_code != 200:
                raise SalukiAPIError(f"HTTP {response.status_code}: {diagnostic['body']}", diagnostic)
            try:
                parsed = json.loads(response.content)
            except (ValueError, TypeError) as exc:
                raise SalukiAPIError("API response is not valid JSON", diagnostic) from exc
            if not isinstance(parsed, dict):
                raise SalukiAPIError("API response must be a JSON object", diagnostic)
            if "error" in parsed:
                raise SalukiAPIError(str(parsed["error"]), {**diagnostic, "json": parsed})
            return parsed

    def command(self) -> list[str]:
        command = [str(self.root / self.contract["executable"]), "--model", str(self.root / MODEL_PATH),
                "--mmproj", str(self.root / PROJECTOR_PATH), "--alias", MODEL,
                "--host", "127.0.0.1", "--port", str(self.port), "--jinja", "-ngl", "99",
                "-lv", "4", "-fa", "on", "-c", str(self.config["context_size"]), "-np", "1",
                "--no-context-shift", "-b", str(self.config["batch_size"]), "-ub", str(self.config["ubatch_size"]),
                "-ctk", self.config["cache_type_k"], "-ctv", self.config["cache_type_v"]]
        if isinstance(self.config["image_tokens"], int):
            command += ["--image-max-tokens", str(self.config["image_tokens"])]
        return command

    def start(self) -> "SalukiServer":
        if self.process is not None:
            raise RuntimeError("Dedicated server has already been started")
        self.contract = model_contract(self.root, thinking=self.config["thinking"],
            context_size=self.config["context_size"], max_tokens=self.config["max_tokens"],
            image_max_tokens=self.config["image_tokens"] if isinstance(self.config["image_tokens"], int) else None,
            batch_size=self.config["batch_size"], micro_batch_size=self.config["ubatch_size"],
            kv_cache_type=self.config["cache_type_k"])
        if not self.contract["ready"]:
            raise RuntimeError("\n".join(self.contract["issues"]))
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError("Saluki専用ポート11450は使用中です。既存サーバーは使用・停止しません。")
        logs = self.root / "logs"
        temp = self.root / "runtime/temp"
        for directory in (logs, temp):
            directory.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        runtime = (self.root / self.contract["executable"]).parent
        # CUDA and its dependent DLLs belong to this Portable runtime. Point
        # llama.cpp at the exact backend instead of depending on caller cwd.
        env.update({"TEMP": str(temp), "TMP": str(temp),
                    "GGML_BACKEND_PATH": str(runtime / "ggml-cuda.dll"),
                    "PATH": str(runtime) + os.pathsep + env.get("PATH", "")})
        stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + str(time.time_ns())
        self._log_paths = [logs / f"saluki-{stamp}.{name}.log" for name in ("stdout", "stderr")]
        self._log_files = [path.open("ab") for path in self._log_paths]
        self.contract["server_logs"] = [path.relative_to(self.root).as_posix() for path in self._log_paths]
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(self.command(), cwd=runtime, env=env,
                stdin=subprocess.DEVNULL, stdout=self._log_files[0], stderr=self._log_files[1],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._process_created = psutil.Process(self.process.pid).create_time()
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("Salukiサーバーが起動中に終了しました。logsを確認してください。")
                try:
                    health = self.api("/health", timeout=1)
                    if health.get("status") == "ok":
                        log = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in self._log_paths)
                        evidence = gpu_offload_evidence(log)
                        self.contract["gpu_offload"] = evidence
                        if not evidence["full_gpu_offload"] or not evidence["projector_gpu"]:
                            raise RuntimeError("Salukiの全層と画像projectorのGPU配置を確認できません。CPUへの自動退避を許可しません。logsを確認してください。")
                        self.contract["server_props"] = self.api("/props", timeout=3)
                        self.contract["startup_wall_seconds"] = time.perf_counter() - started
                        return self
                except (OSError, httpx.HTTPError):
                    pass
                except SalukiAPIError as exc:
                    if exc.response.get("http_status") != 503:
                        raise
                time.sleep(.2)
            raise TimeoutError("Salukiサーバーの起動を180秒以内に確認できませんでした。")
        except BaseException:
            self.stop()
            raise

    def _owned_process(self) -> psutil.Process | None:
        if self.process is None:
            return None
        try:
            process = psutil.Process(self.process.pid)
            if (process.create_time() != self._process_created or
                    Path(process.exe()).resolve() != (self.root / self.contract["executable"]).resolve()):
                raise RuntimeError("Process identity changed; refusing to stop it")
            return process
        except psutil.NoSuchProcess:
            return None

    def stop(self) -> dict[str, Any]:
        report: dict[str, Any] = {"model_unloaded": False, "process_stopped": False, "errors": []}
        try:
            process = self._owned_process()
            if process is not None:
                children = process.children(recursive=True)
                # Confirm descendant identities before any later forced kill.
                identities = {child.pid: child.create_time() for child in children}
                identities[process.pid] = self._process_created
                for owned in [*children, process]:
                    try:
                        if owned.create_time() == identities[owned.pid]:
                            owned.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs([process, *children], timeout=5)
                for owned in alive:
                    try:
                        if owned.create_time() == identities[owned.pid]:
                            owned.kill()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(alive, timeout=5)
                report["process_stopped"] = not alive
            else:
                report["process_stopped"] = True
            # llama.cpp owns model allocations inside this process.
            report["model_unloaded"] = report["process_stopped"]
        except Exception as exc:
            report["errors"].append(str(exc))
        finally:
            for file in self._log_files:
                file.close()
            self._log_files = []
            self.shutdown_report = report
        return report

    def __enter__(self) -> "SalukiServer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


def extract_page(server: SalukiServer, image_path: Path, page: dict,
                 regions: list[dict], settings: dict) -> dict[str, Any]:
    record: dict[str, Any] = {"response": None, "parsed": None, "timings": {}, "request_metadata": {}}
    samples: list[dict] = []
    finished = threading.Event()
    monitor: threading.Thread | None = None
    start = time.perf_counter()
    try:
        if settings.get("take") not in ("one", "all"):
            raise OutputError("INVALID_INPUT", "takeはoneまたはallです。")
        identifiers = [r["id"] for r in regions]
        if len(identifiers) != len(set(identifiers)):
            raise OutputError("INVALID_INPUT", "OCR領域IDが重複しています。")
        image_bytes = Path(image_path).read_bytes()
        payload = build_payload(page, regions, settings)
        payload_text = json.dumps(payload, ensure_ascii=False)
        config = server.config
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            raise OutputError("INVALID_INPUT", "対応するページ画像形式ではありません。")
        request = {"model": MODEL, "stream": False, "temperature": config["temperature"],
            "top_p": config["top_p"], "top_k": config["top_k"], "repeat_penalty": config["repeat_penalty"],
            "presence_penalty": config["presence_penalty"], "seed": config["seed"], "max_tokens": config["max_tokens"],
            "chat_template_kwargs": {"enable_thinking": config["thinking"]},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "drawing_extraction", "strict": True, "schema": deepcopy(SCHEMA)}},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
                {"type": "text", "text": "次の設定で画像の項目を抽出してください。\n" + payload_text},
                {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(image_bytes).decode("ascii")}}]}]}
        image_reference = Path(image_path).resolve()
        root = getattr(server, "root", None)
        image_reference = image_reference.relative_to(root).as_posix() if root and image_reference.is_relative_to(root) else str(image_reference)
        record["request_metadata"] = {"backend": "saluki", "model": MODEL, "config": deepcopy(config),
            "think": config["thinking"], "stream": False, "truncate": False, "shift": False,
            "image_path": image_reference, "image_sha256": _sha_bytes(image_bytes),
            "system_prompt_sha256": _sha_bytes(SYSTEM.encode()), "input_payload_sha256": _sha_bytes(payload_text.encode()),
            "input_payload": payload, "request_schema": deepcopy(SCHEMA), "ocr_region_count": len(regions),
            "manifest_sha256": server.contract.get("manifest_sha256"),
            "executable_sha256": server.contract.get("executable_sha256"),
            "adapter_sha256": server.contract.get("adapter_sha256"), "runtime_version": server.contract.get("runtime_version")}
        def sample_resources():
            while not finished.is_set():
                sample = _gpu_sample()
                if sample:
                    samples.append(sample)
                finished.wait(.5)
        monitor = threading.Thread(target=sample_resources, daemon=True)
        monitor.start()
        response = server.api("/v1/chat/completions", request, timeout=900)
        record["response"] = response
        native = response.get("timings", {})
        record["provider_timings"] = deepcopy(native)
        if isinstance(native, dict):
            for native_key, key in (("prompt_ms", "prompt_eval_seconds"), ("predicted_ms", "eval_seconds")):
                if isinstance(native.get(native_key), (int, float)):
                    record["timings"][key] = native[native_key] / 1000
            record["timings"].update(prompt_eval_count=native.get("prompt_n"), eval_count=native.get("predicted_n"),
                                     prompt_eval_cached_count=response.get("tokens_cached", native.get("cache_n")))
            if "prompt_eval_seconds" in record["timings"] and "eval_seconds" in record["timings"]:
                record["timings"]["total_seconds"] = record["timings"]["prompt_eval_seconds"] + record["timings"]["eval_seconds"]
        record["usage"] = deepcopy(response.get("usage", {}))
        choices = response.get("choices", [])
        if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
                or choices[0].get("finish_reason") != "stop"):
            raise OutputError("INCOMPLETE_RESPONSE", "モデルの回答が正常完了していません。")
        if response.get("truncated") is True or (isinstance(native, dict) and native.get("truncated") is True):
            raise OutputError("CONTEXT_OVERFLOW", "モデルサーバーが入力を切り詰めました。回答を採用しません。")
        record["parsed"] = validate_output(choices[0].get("message", {}).get("content", ""), settings, regions)
        record["loaded_models"] = {"backend": "saluki", "model": MODEL, "gpu_offload": server.contract.get("gpu_offload")}
    except Exception as exc:
        record["error"] = {"code": getattr(exc, "code", "API_ERROR"), "message": str(exc)}
        if isinstance(exc, SalukiAPIError):
            record["api_error_response"] = exc.response
    finally:
        finished.set()
        if monitor:
            monitor.join(timeout=4)
        record["timings"]["wall_seconds"] = time.perf_counter() - start
        record["resource_samples"] = samples
        record["peak_system_gpu_memory_mib"] = max((s["system_gpu_memory_mib"] for s in samples), default=None)
    return record

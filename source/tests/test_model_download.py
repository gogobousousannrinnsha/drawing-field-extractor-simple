"""Tiny offline fixtures for pinned downloads, interruption, and publication safety."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest

from simple_drawing import model_download as download
from simple_drawing.model_manager import ModelOperationCancelled


REVISION = "a" * 40
REPOSITORY = "fixture/model-GGUF"
BINARY = b"GGUF\x00\x01small offline fixture\xff"


def sha(content):
    return hashlib.sha256(content).hexdigest()


@dataclass
class Fixture:
    root: Path
    profile: object
    catalog: dict
    manifest: dict
    encoded: bytes
    payloads: dict
    binary_digest: str

    def save(self):
        path = self.root / "assets/huggingface-ollama.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.catalog, ensure_ascii=False), encoding="utf-8")

    @property
    def definition(self):
        return self.catalog["profiles"][self.profile.id]

    @property
    def active(self):
        return self.root / "models/ollama/manifests/registry.ollama.ai/library/fixture-model/exact"

    def blob(self, digest):
        return self.root / "models/ollama/blobs" / ("sha256-" + digest[7:])

    def partial(self, digest):
        return self.root / "downloads/huggingface" / self.profile.id / ("sha256-" + digest[7:] + ".part")

    def preserve_active(self):
        self.active.parent.mkdir(parents=True, exist_ok=True)
        previous = b'{"previous":"must survive failure"}\n'
        self.active.write_bytes(previous)
        return previous


@pytest.fixture
def model(tmp_path):
    payloads = {}
    blobs = {}
    refs = []
    for content, inline in (("設定\n".encode(), True), (b"TEMPLATE\n", True), (BINARY, False)):
        digest = "sha256:" + sha(content)
        payloads[digest] = content
        ref = {"digest": digest, "size": len(content), "mediaType": "fixture/model"}
        asset = {"sha256": digest[7:], "bytes": len(content)}
        if inline:
            asset["text"] = content.decode()
        else:
            asset.update(repository=REPOSITORY, revision=REVISION,
                         url=f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.gguf")
        refs.append(ref)
        blobs[digest] = asset
    manifest = {"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "config": refs[0], "layers": refs[1:]}
    encoded = (json.dumps(manifest, indent=3) + "\n\n").encode()
    profile = SimpleNamespace(id="fixture-model", model="fixture-model:exact", expected_digest=sha(encoded))
    definition = {"model": profile.model, "manifest_text": encoded.decode(),
                  "manifest_sha256": sha(encoded), "blobs": blobs}
    result = Fixture(tmp_path, profile, {"schema_version": 1, "profiles": {profile.id: definition}},
                     manifest, encoded, payloads, refs[-1]["digest"])
    result.save()
    return result


class CurlProcess:
    """Only writes tiny specified bytes; no executable or network is invoked."""
    def __init__(self, command, kwargs, outcome):
        self.command, self.kwargs = command, kwargs
        self.initial_partial = Path(command[command.index("--output") + 1])
        self.before = self.initial_partial.read_bytes() if self.initial_partial.exists() else None
        self.returncode = outcome.get("code", 0)
        self.running = outcome.get("running", False)
        self.timeout_on_terminate = outcome.get("timeout_on_terminate", False)
        self.terminated = False
        self.killed = False
        self.waits = []
        if "content" in outcome:
            self.initial_partial.write_bytes(outcome["content"])
        if "diagnostic" in outcome:
            kwargs["stderr"].write(outcome["diagnostic"])

    def poll(self):
        return None if self.running else self.returncode

    def terminate(self):
        self.terminated = True
        if not self.timeout_on_terminate:
            self.running = False

    def kill(self):
        self.killed = True
        self.running = False

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.running:
            raise subprocess.TimeoutExpired(self.command, timeout)
        return self.returncode


def fake_curl(monkeypatch, *outcomes):
    instances = []
    remaining = iter(outcomes)

    def spawn(command, **kwargs):
        process = CurlProcess(command, kwargs, next(remaining))
        instances.append(process)
        return process

    monkeypatch.setattr(download.subprocess, "Popen", spawn)
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    return instances


def cache_all(model):
    for digest, content in model.payloads.items():
        target = model.blob(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def no_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("This fixture must not start a downloader")
    monkeypatch.setattr(download.subprocess, "Popen", fail)


def test_success_verifies_inline_and_binary_and_publishes_exact_manifest(model, monkeypatch):
    processes = fake_curl(monkeypatch, {"content": BINARY})
    events = []
    result = download.install_from_huggingface(model.root, model.profile, progress=events.append)
    assert len(processes) == 1
    assert model.active.read_bytes() == model.encoded
    assert result == {"source": "huggingface", "model": model.profile.model,
                      "asset_variant": "huggingface", "repository": None, "revision": None,
                      "manifest_sha256": sha(model.encoded),
                      "blobs": [{"digest": digest, "bytes": len(content)}
                                for digest, content in model.payloads.items()]}
    for digest, content in model.payloads.items():
        assert model.blob(digest).read_bytes() == content
    assert all(event["source"] == "huggingface" for event in events)
    assert not list(model.root.rglob("*.part"))
    assert not list(model.active.parent.glob("*.tmp"))
    assert not (model.root / "data/model-registry.json").exists()


@pytest.mark.parametrize("location", ["blobs", "partials"])
def test_verified_cache_reused_without_network(model, monkeypatch, location):
    no_network(monkeypatch)
    for digest, content in model.payloads.items():
        target = model.blob(digest) if location == "blobs" else model.partial(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    download.install_from_huggingface(model.root, model.profile)
    assert model.active.read_bytes() == model.encoded
    assert all(model.blob(digest).read_bytes() == value for digest, value in model.payloads.items())


@pytest.mark.parametrize("pinned", [True, False])
def test_primary_unpinned_profile_still_requires_explicit_hf_identity(model, monkeypatch, pinned):
    model.profile.expected_digest = None
    model.profile.fallback_digest = sha(model.encoded) if pinned else "0" * 64
    cache_all(model)
    no_network(monkeypatch)
    if pinned:
        download.install_from_huggingface(model.root, model.profile)
        assert model.active.read_bytes() == model.encoded
    else:
        with pytest.raises(download.HuggingFaceDownloadError):
            download.install_from_huggingface(model.root, model.profile)
        assert not model.active.exists()


@pytest.mark.parametrize("change", [
    "schema", "model", "manifest-sha", "expected-sha", "manifest-text",
    "missing-blob", "extra-blob", "ref-sha", "ref-size-negative", "ref-size-bool",
    "ref-size-huge", "asset-sha", "asset-size", "inline-sha", "inline-size", "oversized-catalog",
])
def test_invalid_descriptor_refused_before_any_artifact_changes(model, monkeypatch, change):
    previous = model.preserve_active()
    original_files = {path.relative_to(model.root).as_posix(): path.read_bytes()
                      for path in model.root.rglob("*") if path.is_file() and path.name != "huggingface-ollama.json"}
    definition = model.definition
    binary = definition["blobs"][model.binary_digest]
    inline_digest = model.manifest["config"]["digest"]
    if change == "schema":
        model.catalog["schema_version"] = 2
    elif change == "model":
        definition["model"] = "different:tag"
    elif change == "manifest-sha":
        definition["manifest_sha256"] = "0" * 64
    elif change == "expected-sha":
        model.profile.expected_digest = "0" * 64
    elif change == "manifest-text":
        definition["manifest_text"] += " "
    elif change == "missing-blob":
        del definition["blobs"][model.binary_digest]
    elif change == "extra-blob":
        definition["blobs"]["sha256:" + "0" * 64] = deepcopy(binary)
    elif change.startswith("ref-"):
        ref = model.manifest["layers"][-1]
        if change == "ref-sha":
            ref["digest"] = "sha256:" + "A" * 64
        else:
            ref["size"] = {"ref-size-negative": -1, "ref-size-bool": True,
                           "ref-size-huge": 20 * 1024**3 + 1}[change]
        definition["manifest_text"] = json.dumps(model.manifest)
        definition["manifest_sha256"] = sha(definition["manifest_text"].encode())
        model.profile.expected_digest = definition["manifest_sha256"]
    elif change == "asset-sha":
        binary["sha256"] = "0" * 64
    elif change == "asset-size":
        binary["bytes"] += 1
    elif change == "inline-sha":
        definition["blobs"][inline_digest]["text"] = "壊れ\n"
    elif change == "inline-size":
        definition["blobs"][inline_digest]["text"] += "x"
    model.save()
    if change == "oversized-catalog":
        (model.root / "assets/huggingface-ollama.json").write_bytes(b" " * (256 * 1024 + 1))
    no_network(monkeypatch)
    with pytest.raises(download.HuggingFaceDownloadError):
        download.install_from_huggingface(model.root, model.profile)
    assert model.active.read_bytes() == previous
    observed = {path.relative_to(model.root).as_posix(): path.read_bytes()
                for path in model.root.rglob("*") if path.is_file() and path.name != "huggingface-ollama.json"}
    assert observed == original_files


@pytest.mark.parametrize("field,value", [
    ("url", f"http://huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.gguf"),
    ("url", f"https://evil.invalid/{REPOSITORY}/resolve/{REVISION}/model.gguf"),
    ("url", f"https://huggingface.co.evil.invalid/{REPOSITORY}/resolve/{REVISION}/model.gguf"),
    ("url", f"https://user:password@huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.gguf"),
    ("url", f"https://huggingface.co:8443/{REPOSITORY}/resolve/{REVISION}/model.gguf"),
    ("url", f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.gguf?download=true"),
    ("url", f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.gguf#part"),
    ("url", f"https://huggingface.co/{REPOSITORY}/resolve/main/model.gguf"),
    ("url", f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/../model.gguf"),
    ("url", f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/folder/model.gguf"),
    ("url", f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.bin"),
    ("revision", "a" * 39), ("revision", "A" * 40), ("revision", "main"),
    ("repository", "fixture/other"), ("repository", "../fixture"),
])
def test_unpinned_or_untrusted_urls_refused(model, monkeypatch, field, value):
    model.definition["blobs"][model.binary_digest][field] = value
    model.save()
    no_network(monkeypatch)
    with pytest.raises(download.HuggingFaceDownloadError):
        download.install_from_huggingface(model.root, model.profile)
    assert not model.active.exists()
    assert not (model.root / "models").exists()


@pytest.mark.parametrize("code", [5, 7, 22, 28, 60])
def test_source_failure_retains_active_manifest_and_partial(model, monkeypatch, code):
    previous = model.preserve_active()
    partial = model.partial(model.binary_digest)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(BINARY[:3])
    processes = fake_curl(monkeypatch, {"code": code, "diagnostic": b"fixture transfer failure"})
    with pytest.raises(download.HuggingFaceDownloadError, match=f"{code}"):
        download.install_from_huggingface(model.root, model.profile)
    assert len(processes) == 1
    assert processes[0].before == BINARY[:3]
    assert partial.read_bytes() == BINARY[:3]
    assert model.active.read_bytes() == previous
    assert not model.blob(model.binary_digest).exists()


@pytest.mark.parametrize("bad", [BINARY[:-1], b"x" * len(BINARY), BINARY + b"extra"])
def test_wrong_size_or_sha_stops_without_manifest_activation(model, monkeypatch, bad):
    previous = model.preserve_active()
    processes = fake_curl(monkeypatch, {"content": bad})
    with pytest.raises(download.HuggingFaceDownloadError, match="SHA-256"):
        download.install_from_huggingface(model.root, model.profile)
    assert len(processes) == 1
    assert model.partial(model.binary_digest).read_bytes() == bad
    assert model.active.read_bytes() == previous
    assert not model.blob(model.binary_digest).exists()


def test_corrupt_full_partial_quarantined_then_single_fresh_download(model, monkeypatch):
    bad = b"x" * len(BINARY)
    partial = model.partial(model.binary_digest)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(bad)
    processes = fake_curl(monkeypatch, {"content": BINARY})
    download.install_from_huggingface(model.root, model.profile)
    assert len(processes) == 1
    assert processes[0].before is None
    retained = list(partial.parent.glob(partial.name + ".bad-sha-*"))
    assert len(retained) == 1
    assert retained[0].read_bytes() == bad
    assert model.blob(model.binary_digest).read_bytes() == BINARY


def test_corrupt_target_retained_only_after_replacement_verified(model, monkeypatch):
    bad = b"x" * len(BINARY)
    target = model.blob(model.binary_digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bad)
    processes = fake_curl(monkeypatch, {"content": BINARY})
    download.install_from_huggingface(model.root, model.profile)
    assert len(processes) == 1
    assert target.read_bytes() == BINARY
    assert [path.read_bytes() for path in target.parent.glob(target.name + ".bad-sha-*")] == [bad]


@pytest.mark.parametrize("code", [33, 36])
def test_unsupported_resume_preserves_partial_and_retries_once(model, monkeypatch, code):
    partial = model.partial(model.binary_digest)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(BINARY[:3])
    processes = fake_curl(monkeypatch, {"code": code}, {"content": BINARY})
    download.install_from_huggingface(model.root, model.profile)
    assert len(processes) == 2
    assert processes[0].before == BINARY[:3]
    assert processes[1].before is None
    assert [path.read_bytes() for path in partial.parent.glob(partial.name + ".resume-unavailable-*")] == [BINARY[:3]]
    assert model.active.read_bytes() == model.encoded


def test_fresh_retry_rechecks_capacity_and_stops_before_second_process(model, monkeypatch):
    previous = model.preserve_active()
    partial = model.partial(model.binary_digest)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(BINARY[:3])
    free = 32 * 1024**2 + len(BINARY) - 3
    calls = []
    def disk_usage(path):
        calls.append(path)
        return SimpleNamespace(free=free)
    monkeypatch.setattr(download.shutil, "disk_usage", disk_usage)
    processes = fake_curl(monkeypatch, {"code": 33})
    with pytest.raises(download.HuggingFaceDownloadError, match="空き容量"):
        download.install_from_huggingface(model.root, model.profile)
    assert len(processes) == 1
    assert len(calls) == 2
    assert model.active.read_bytes() == previous
    assert [path.read_bytes() for path in partial.parent.glob(partial.name + ".resume-unavailable-*")] == [BINARY[:3]]


def test_repeated_resume_failure_is_bounded_and_never_published(model, monkeypatch):
    previous = model.preserve_active()
    partial = model.partial(model.binary_digest)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(BINARY[:3])
    processes = fake_curl(monkeypatch, {"code": 33}, {"code": 33, "content": BINARY[:2]})
    with pytest.raises(download.HuggingFaceDownloadError, match="33"):
        download.install_from_huggingface(model.root, model.profile)
    assert len(processes) == 2
    assert model.active.read_bytes() == previous
    assert partial.read_bytes() == BINARY[:2]


@pytest.mark.parametrize("kill_required", [False, True])
def test_cancellation_stops_and_waits_curl_without_publishing(model, monkeypatch, kill_required):
    previous = model.preserve_active()
    cancel = threading.Event()
    processes = fake_curl(monkeypatch, {"running": True, "content": BINARY[:3],
                                       "timeout_on_terminate": kill_required})
    monkeypatch.setattr(download.time, "monotonic", lambda: 10.0)
    events = []

    def progress(event):
        events.append(event)
        if event["stage"] == "download":
            cancel.set()

    with pytest.raises(ModelOperationCancelled):
        download.install_from_huggingface(model.root, model.profile, cancel=cancel, progress=progress)
    assert len(processes) == 1
    process = processes[0]
    assert process.terminated and process.killed == kill_required
    assert process.waits == ([5, 5] if kill_required else [5])
    assert not process.running
    assert model.active.read_bytes() == previous
    assert model.partial(model.binary_digest).read_bytes() == BINARY[:3]
    assert any(event.get("completed") == 3 and event.get("total") == len(BINARY) for event in events)


def test_precancel_has_no_download_or_mutation(model, monkeypatch):
    no_network(monkeypatch)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ModelOperationCancelled):
        download.install_from_huggingface(model.root, model.profile, cancel=cancel)
    assert not (model.root / "models").exists()
    assert not (model.root / "downloads").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_cancel_immediately_after_manifest_publish_rolls_back(model, monkeypatch, existing):
    cache_all(model)
    no_network(monkeypatch)
    previous = model.preserve_active() if existing else None
    cancel = threading.Event()
    replace = Path.replace

    def replacement(path, target):
        result = replace(path, target)
        if Path(target) == model.active and Path(target).read_bytes() == model.encoded:
            cancel.set()
        return result

    monkeypatch.setattr(Path, "replace", replacement)
    with pytest.raises(ModelOperationCancelled):
        download.install_from_huggingface(model.root, model.profile, cancel=cancel)
    if existing:
        assert model.active.read_bytes() == previous
    else:
        assert not model.active.exists()
    assert not list(model.active.parent.glob("*.tmp"))


def test_curl_keeps_proxy_environment_tls_restrictions_and_timeouts(model, monkeypatch):
    proxy_vars = {"HTTP_PROXY": "http://fixture-proxy.invalid:8080", "HTTPS_PROXY": "http://fixture-proxy.invalid:8080",
                  "ALL_PROXY": "http://fixture-proxy.invalid:8080", "NO_PROXY": "localhost,127.0.0.1"}
    for key, value in proxy_vars.items():
        monkeypatch.setenv(key, value)
    processes = fake_curl(monkeypatch, {"content": BINARY})
    download.install_from_huggingface(model.root, model.profile)
    command, kwargs = processes[0].command, processes[0].kwargs
    assert Path(command[0]).name == "curl.exe"
    assert command[1] == "--disable"
    for flag, value in (("--proto", "=https"), ("--proto-redir", "=https"), ("--connect-timeout", "30"),
                        ("--speed-limit", "1024"), ("--speed-time", "60"), ("--max-time", "3600"),
                        ("--continue-at", "-")):
        assert command[command.index(flag) + 1] == value
    assert "--fail" in command and "--location" in command
    assert not {"--insecure", "-k", "--proxy-insecure", "--noproxy", "--proxy"}.intersection(command)
    assert "env" not in kwargs and "shell" not in kwargs
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["creationflags"] == (subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert {key: os.environ[key] for key in proxy_vars} == proxy_vars


def test_curl_launch_failure_stops_and_preserves_manifest(model, monkeypatch):
    previous = model.preserve_active()

    def fail(*args, **kwargs):
        raise FileNotFoundError("fixture missing executable")

    monkeypatch.setattr(download.subprocess, "Popen", fail)
    with pytest.raises(download.HuggingFaceDownloadError, match="curl.exe"):
        download.install_from_huggingface(model.root, model.profile)
    assert model.active.read_bytes() == previous


@pytest.mark.parametrize("kind", ["is_symlink", "is_junction"])
@pytest.mark.parametrize("blocked", ["models/ollama/blobs", "downloads/huggingface", "models/ollama/manifests"])
def test_link_or_junction_ancestor_refused_before_write(model, monkeypatch, kind, blocked):
    if kind == "is_junction" and not hasattr(Path, kind):
        pytest.skip("This Python lacks junction detection")
    previous = model.preserve_active()
    cache_all(model) if blocked == "models/ollama/manifests" else None
    path = model.root / blocked
    path.mkdir(parents=True, exist_ok=True)
    original = getattr(Path, kind)
    monkeypatch.setattr(Path, kind, lambda self: self == path or original(self))
    no_network(monkeypatch)
    with pytest.raises(download.HuggingFaceDownloadError, match="リンク"):
        download.install_from_huggingface(model.root, model.profile)
    assert model.active.read_bytes() == previous


def test_real_windows_junction_to_external_directory_is_refused(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("Windows junction fixture")
    root, external = tmp_path / "portable", tmp_path / "external"
    root.mkdir()
    external.mkdir()
    link = root / "models"
    result = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(external)],
                            capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode != 0:
        pytest.skip("The environment cannot create the optional real junction fixture")
    assert link.is_junction()
    try:
        with pytest.raises(download.HuggingFaceDownloadError, match="リンク"):
            download._regular_path(root, "models/ollama/blobs/fixture")
        assert not list(external.iterdir())
    finally:
        link.rmdir()


def test_relative_path_escape_refused(tmp_path):
    root = tmp_path / "portable"
    root.mkdir()
    with pytest.raises(ValueError, match="外"):
        download._regular_path(root, "../external/model")
    assert not (tmp_path / "external").exists()

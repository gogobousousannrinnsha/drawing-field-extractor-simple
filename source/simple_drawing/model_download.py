"""Acquire pinned Hugging Face assets with their own verified identity."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlparse
import uuid

from .storage import local_path, read_json


class HuggingFaceDownloadError(RuntimeError):
    pass


def _repository_ok(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', value) and all(
        part not in ('.', '..') for part in value.split('/'))


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        # Imported only when needed; ModelManager owns the public cancellation type.
        from .model_manager import ModelOperationCancelled
        raise ModelOperationCancelled()


def _emit(progress, stage, status, **details):
    if progress is not None:
        progress({'stage': stage, 'status': status, 'source': 'huggingface', **details})


def _hash(path, cancel=None):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            _cancelled(cancel)
            value.update(block)
    return value.hexdigest()


def _matches(path, size, digest, cancel=None):
    return path.is_file() and path.stat().st_size == size and _hash(path, cancel) == digest


def _regular_path(root, name):
    """Check unresolved ancestors too, so links cannot redirect acquisition."""
    path = Path(root) / name
    for current in (path, *path.parents):
        if current == Path(root):
            break
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise HuggingFaceDownloadError('モデル取得先にリンクを使用できません。')
    return local_path(root, name)


def _catalog(root, profile):
    try:
        path = Path(root) / 'assets/huggingface-ollama.json'
        if path.stat().st_size > 256 * 1024:
            raise ValueError('oversized catalog')
        catalog = read_json(path)
        if catalog['schema_version'] != 1:
            raise ValueError('unknown catalog format')
        definition = catalog['profiles'][profile.id]
        if definition['model'] != profile.model:
            raise ValueError('different model tag')
        encoded = definition['manifest_text'].encode('utf-8')
        digest = hashlib.sha256(encoded).hexdigest()
        allowed = {value for value in (profile.expected_digest, getattr(profile, 'fallback_digest', None)) if value}
        if digest != definition['manifest_sha256'] or not allowed or digest not in allowed:
            raise ValueError('different model manifest')
        manifest = json.loads(encoded)
        refs = [manifest['config'], *manifest['layers']]
        assets = definition['blobs']
        if not isinstance(assets, dict) or set(assets) != {item['digest'] for item in refs}:
            raise ValueError('incomplete blob catalog')
        if len(refs) > 20 or manifest['schemaVersion'] != 2:
            raise ValueError('unexpected manifest')
        for ref in refs:
            digest, size = ref['digest'], ref['size']
            if not isinstance(digest, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
                raise ValueError('invalid blob identity')
            if type(size) is not int or size < 0 or size > 20 * 1024**3:
                raise ValueError('invalid blob size')
            asset = assets[digest]
            if asset['bytes'] != size or asset['sha256'] != digest[7:]:
                raise ValueError('different blob reference')
            if 'text' in asset:
                content = asset['text'].encode('utf-8')
                if size > 64 * 1024 or len(content) != size or hashlib.sha256(content).hexdigest() != digest[7:]:
                    raise ValueError('invalid inline blob')
            else:
                url = urlparse(asset['url'])
                revision = asset['revision']
                repository = asset['repository']
                if (url.scheme != 'https' or url.hostname != 'huggingface.co' or url.username or url.password
                        or url.query or url.fragment or url.port not in (None, 443)
                        or not _repository_ok(repository)
                        or not re.fullmatch(r'[0-9a-f]{40}', revision)
                        or not re.fullmatch('/' + re.escape(repository) + '/resolve/' + revision +
                                            r'/[A-Za-z0-9_.-]+\.gguf', url.path)):
                    raise ValueError('untrusted or unpinned Hugging Face URL')
        return definition, encoded, refs
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise HuggingFaceDownloadError('このモデルの検証済みHugging Face取得情報が不足または破損しています。') from error


def _curl_download(asset, partial, cancel=None, progress=None):
    """Honor the user's proxy, keep TLS verification, and bound stalled transfers."""
    command = [str(Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/curl.exe'),
               '--disable', '--fail', '--location', '--silent', '--show-error',
               '--proto', '=https', '--proto-redir', '=https', '--connect-timeout', '30',
               '--speed-limit', '1024', '--speed-time', '60', '--max-time', '3600',
               '--continue-at', '-', '--output', str(partial), asset['url']]
    for attempt in range(2):
        _cancelled(cancel)
        remaining = max(0, asset['bytes'] - (partial.stat().st_size if partial.exists() else 0))
        if shutil.disk_usage(partial.parent).free < remaining + 32 * 1024**2:
            raise HuggingFaceDownloadError('Hugging Face取得先の空き容量が不足しています。容量を確保して再実行してください。')
        with tempfile.TemporaryFile() as diagnostic:
            try:
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=diagnostic,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except OSError as error:
                raise HuggingFaceDownloadError('Hugging Face取得用のcurl.exeを起動できません。') from error
            try:
                last_emit = 0.0
                while process.poll() is None:
                    _cancelled(cancel)
                    current = time.monotonic()
                    if current - last_emit >= 1:
                        completed = partial.stat().st_size if partial.exists() else 0
                        _emit(progress, 'download', 'Hugging Faceから直接取得しています。',
                              completed=completed, total=asset['bytes'], digest='sha256:' + asset['sha256'])
                        last_emit = current
                    time.sleep(.1)
                _cancelled(cancel)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            code = process.returncode
            diagnostic.seek(0)
            detail = diagnostic.read(2048).decode('utf-8', 'replace').strip()
        if code == 0:
            return
        if code in (33, 36) and attempt == 0 and partial.exists():
            partial.replace(partial.with_name(partial.name + '.resume-unavailable-' + uuid.uuid4().hex))
            continue
        raise HuggingFaceDownloadError(f'Hugging Faceから取得できませんでした（curl終了コード{code}）。{detail}')


def install_from_huggingface(root, profile, *, cancel=None, progress=None):
    """Publish a manifest only after every exact blob has been verified."""
    root = Path(root).resolve()
    _cancelled(cancel)
    definition, encoded_manifest, refs = _catalog(root, profile)
    _emit(progress, 'fallback', 'Ollamaの取得に失敗したため、Hugging Faceからの直接取得を試します。')
    verified = []
    for ref in refs:
        _cancelled(cancel)
        digest = ref['digest'][7:]
        asset = definition['blobs'][ref['digest']]
        target = _regular_path(root, 'models/ollama/blobs/sha256-' + digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        _emit(progress, 'verify', '保存済みモデル部品を検査しています。', digest=ref['digest'])
        if not _matches(target, asset['bytes'], digest, cancel):
            partial = _regular_path(root, 'downloads/huggingface/' + profile.id + '/sha256-' + digest + '.part')
            partial.parent.mkdir(parents=True, exist_ok=True)
            if not _matches(partial, asset['bytes'], digest, cancel):
                if 'text' in asset:
                    partial.write_bytes(asset['text'].encode('utf-8'))
                else:
                    # A completed wrong download cannot be safely resumed from its full size.
                    if partial.exists() and partial.stat().st_size >= asset['bytes']:
                        partial.replace(partial.with_name(partial.name + '.bad-sha-' + uuid.uuid4().hex))
                    _curl_download(asset, partial, cancel, progress)
            _cancelled(cancel)
            _emit(progress, 'verify', '取得した部品のSHA-256を検査しています。', digest=ref['digest'])
            if not _matches(partial, asset['bytes'], digest, cancel):
                raise HuggingFaceDownloadError('Hugging Face取得ファイルのサイズまたはSHA-256が一致しません。初期設定を停止します。')
            if target.exists():
                target.replace(target.with_name(target.name + '.bad-sha-' + uuid.uuid4().hex))
            partial.replace(target)
        verified.append({'digest': ref['digest'], 'bytes': asset['bytes']})
    _cancelled(cancel)
    model, tag = profile.model.split(':', 1)
    target = _regular_path(root, f'models/ollama/manifests/registry.ollama.ai/library/{model}/{tag}')
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = target.read_bytes() if target.exists() else None
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(encoded_manifest)
        _cancelled(cancel)
        temporary.replace(target)
        _cancelled(cancel)
    except BaseException:
        # Cancellation after the atomic rename must not leave a new manifest active.
        if target.exists() and target.read_bytes() == encoded_manifest:
            if previous is None:
                target.unlink()
            else:
                temporary.write_bytes(previous)
                temporary.replace(target)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    result = {'source': 'huggingface', 'model': profile.model,
              'asset_variant': 'huggingface', 'repository': definition.get('repository'),
              'revision': definition.get('revision'),
              'manifest_sha256': definition['manifest_sha256'], 'blobs': verified}
    _emit(progress, 'verify', 'Hugging Face直接取得のファイル検査が完了しました。Ollamaで互換性を確認します。')
    return result

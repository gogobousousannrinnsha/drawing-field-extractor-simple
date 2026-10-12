"""Validate the one reviewed, user-owned Core wheel bundled with this release.

This narrow exception permits a pure Python wheel, never arbitrary runtime
archives or proprietary DocuWorks binaries. Validation performs no extraction,
import, download, DLL loading, or DocuWorks operation.
"""
from __future__ import annotations

import ast
import base64
import csv
from email.parser import Parser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


CORE_NAME = "docuworks-ctypes"
CORE_VERSION = "1.0.1"
CORE_FILENAME = "docuworks_ctypes-1.0.1-py3-none-any.whl"
CORE_PATH = "bundled/core/" + CORE_FILENAME
CORE_SIZE = 48173
CORE_SHA256 = "7ea3a6d1623238180ccb400237a753b616dccf72761ba83d97fee1b55f7f5f58"
CORE_URL = (
    "https://github.com/gogobousousannrinnsha/dw-ocr/releases/download/v0.8.1/"
    + CORE_FILENAME
)
DIST_INFO = "docuworks_ctypes-1.0.1.dist-info/"
LICENSE_MEMBER = DIST_INFO + "licenses/LICENSE"
RECORD_MEMBER = DIST_INFO + "RECORD"
CORE_MEMBERS = frozenset({
    "docuworks_ctypes/__init__.py", "docuworks_ctypes/api.py",
    "docuworks_ctypes/attributes.py", "docuworks_ctypes/capabilities.py",
    "docuworks_ctypes/document.py", "docuworks_ctypes/encoding.py",
    "docuworks_ctypes/enums.py", "docuworks_ctypes/errors.py",
    "docuworks_ctypes/evidence.py", "docuworks_ctypes/geometry.py",
    "docuworks_ctypes/_raw/__init__.py", "docuworks_ctypes/_raw/api.py",
    "docuworks_ctypes/_raw/constants.py", "docuworks_ctypes/_raw/functions.py",
    "docuworks_ctypes/_raw/types.py", "docuworks_ctypes/runtime/__init__.py",
    "docuworks_ctypes/runtime/loader.py", "docuworks_ctypes/runtime/models.py",
    "docuworks_ctypes/runtime/pe.py", "docuworks_ctypes/runtime/probe_worker.py",
    "docuworks_ctypes/runtime/resolver.py", "docuworks_ctypes/simple/__init__.py",
    "docuworks_ctypes/simple/facade.py", LICENSE_MEMBER,
    DIST_INFO + "METADATA", DIST_INFO + "WHEEL", DIST_INFO + "top_level.txt",
    RECORD_MEMBER,
})
PRIVATE_KEY = re.compile(r"(?m)^-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----")
ACCESS_TOKEN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b")
LOCAL_PATH = re.compile(r"\b[A-Za-z]:[\\/][^\s\"'`]+")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _normal_newlines(text: str) -> str:
    # Normalize header line endings for the email metadata parser.
    return text.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")


def _safe_member(name: str) -> None:
    parts = PurePosixPath(name).parts
    if (not name or "\\" in name or ":" in name or "\x00" in name
            or name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/"))
            or any(part.rstrip(" .") != part for part in parts)):
        raise ValueError(f"Unsafe Core wheel member: {name}")


def _scan_text(name: str, text: str) -> None:
    if PRIVATE_KEY.search(text) or ACCESS_TOKEN.search(text):
        raise ValueError(f"Secret-like material in Core wheel: {name}")
    for match in LOCAL_PATH.finditer(text):
        normalized = match.group(0).replace("/", "\\").casefold()
        # Upstream public documentation uses fictional installation examples;
        # resolver uses the standard Windows directory as an OS fallback.
        roots = {"c:" + "\\" + leaf for leaf in ("windows", "sdk", "test", "path")}
        if not any(normalized == root or normalized.startswith(root + "\\") for root in roots):
            raise ValueError(f"Private-looking path in Core wheel: {name}")


def audit_core_wheel(path: Path) -> dict:
    """Audit structural content and require the exact reviewed public bytes."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Bundled Core wheel must be a regular file")
    content = path.read_bytes()
    if len(content) > 1024 * 1024:
        raise ValueError("Bundled Core wheel exceeds its review size limit")
    texts: dict[str, str] = {}
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        infos = archive.infolist()
        names = [entry.filename for entry in infos]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate Core wheel member")
        total = 0
        for entry in infos:
            _safe_member(entry.filename)
            mode = entry.external_attr >> 16
            if entry.is_dir() or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in {0, stat.S_IFREG}:
                raise ValueError(f"Non-regular Core wheel member: {entry.filename}")
            if entry.flag_bits & 1:
                raise ValueError("Encrypted Core wheel member")
            if entry.filename not in CORE_MEMBERS:
                raise ValueError(f"Unreviewed Core wheel member: {entry.filename}")
            total += entry.file_size
            if entry.file_size > 128 * 1024 or total > 512 * 1024:
                raise ValueError("Expanded Core wheel exceeds its review size limit")
            data = archive.read(entry)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Non-UTF-8 Core wheel member: {entry.filename}") from exc
            if "\x00" in text:
                raise ValueError(f"Binary content in Core wheel: {entry.filename}")
            _scan_text(entry.filename, text)
            if entry.filename.endswith(".py"):
                ast.parse(text, filename=entry.filename)
            members[entry.filename] = data
            texts[entry.filename] = text
        if set(members) != CORE_MEMBERS:
            raise ValueError("Required reviewed Core wheel member is missing")

    metadata = Parser().parsestr(_normal_newlines(texts[DIST_INFO + "METADATA"]))
    if (metadata.get("Name") != CORE_NAME or metadata.get("Version") != CORE_VERSION
            or metadata.get("License-Expression") != "MIT"
            or metadata.get_all("License-File") != ["LICENSE"]
            or metadata.get_all("Requires-Dist")):
        raise ValueError("Unexpected Core package metadata or dependencies")
    wheel = Parser().parsestr(_normal_newlines(texts[DIST_INFO + "WHEEL"]))
    if (wheel.get("Wheel-Version") != "1.0" or wheel.get("Root-Is-Purelib") != "true"
            or wheel.get_all("Tag") != ["py3-none-any"]):
        raise ValueError("Core wheel must remain pure Python py3-none-any")
    license_text = _normal_newlines(texts[LICENSE_MEMBER])
    for clause in (
        "MIT License", "Copyright (c) 2026 gogobousousannrinnsha",
        "Permission is hereby granted, free of charge",
        "The above copyright notice and this permission notice shall be included",
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND',
    ):
        if clause not in license_text:
            raise ValueError("Required Core MIT license notice is missing")
    if texts[DIST_INFO + "top_level.txt"].strip() != "docuworks_ctypes":
        raise ValueError("Unexpected Core top-level package")
    rows = list(csv.reader(io.StringIO(texts[RECORD_MEMBER], newline="")))
    seen = set()
    for row in rows:
        if len(row) != 3 or row[0] in seen or row[0] not in members:
            raise ValueError("Invalid Core RECORD entry")
        name, digest, size = row
        seen.add(name)
        if name == RECORD_MEMBER:
            if digest or size:
                raise ValueError("Core RECORD must not hash itself")
            continue
        expected = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(members[name]).digest()
        ).rstrip(b"=").decode("ascii")
        if digest != expected or size != str(len(members[name])):
            raise ValueError(f"Core RECORD mismatch: {name}")
    if seen != set(members):
        raise ValueError("Core RECORD does not cover every member")
    if len(content) != CORE_SIZE or _sha(content) != CORE_SHA256:
        raise ValueError("Bundled Core wheel differs from the pinned public artifact")
    return {
        "schema": 1, "name": CORE_NAME, "version": CORE_VERSION,
        "size_bytes": len(content), "sha256": _sha(content), "license": "MIT",
        "pure_python": True, "member_count": len(members),
        "python_source_count": sum(name.endswith(".py") for name in members),
        "record_verified": True, "proprietary_binaries_present": False,
        "members": [{"path": name, "size_bytes": len(data), "sha256": _sha(data)}
                    for name, data in sorted(members.items())],
    }


def validate_core_bundle(root: Path) -> dict:
    """Validate local manifest, exact wheel, and the copied unchanged license."""
    root = Path(root).resolve()
    manifest_path = root / "assets/bundled-core.json"
    if manifest_path.is_symlink() or manifest_path.resolve() != root / "assets/bundled-core.json":
        raise ValueError("Core manifest must stay inside the application")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema": 1, "name": CORE_NAME, "version": CORE_VERSION,
        "filename": CORE_FILENAME, "path": CORE_PATH, "size_bytes": CORE_SIZE,
        "sha256": CORE_SHA256, "url": CORE_URL, "license": "MIT",
    }
    if (type(manifest) is not dict or manifest != expected
            or type(manifest["schema"]) is not int or type(manifest["size_bytes"]) is not int):
        raise ValueError("Bundled Core manifest differs from the reviewed artifact")
    wheel_path = root / CORE_PATH
    if wheel_path.resolve() != root / CORE_PATH:
        raise ValueError("Core wheel must stay inside the application")
    audit_core_wheel(wheel_path)
    license_path = root / "licenses/core/LICENSE.txt"
    if license_path.is_symlink() or license_path.resolve() != root / "licenses/core/LICENSE.txt":
        raise ValueError("Core license must stay inside the application")
    with zipfile.ZipFile(wheel_path) as archive:
        if license_path.read_bytes() != archive.read(LICENSE_MEMBER):
            raise ValueError("Copied Core license differs from the unchanged wheel notice")
    return manifest

"""Build a reproducible source archive with one audited user-owned Core wheel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib
import zipfile


ROOT_FILES = {
    "LICENSE", "README.md", "THIRD_PARTY_NOTICES.md",
    "実行設定.json", "抽出設定.json",
    "初期設定.bat", "前処理.bat", "抽出.bat", "Excel再出力.bat", "モデル管理.bat",
    "追加資産取得.bat", "Saluki導入.bat", "Saluki追加設定.bat", "全文OCR画像出力.bat",
    "黄色マーカー追加.bat", "黄色マーカー_使い方.txt", "docs/VERIFICATION_JA.md",
}
SOURCE_FILES = {"source/run.py", "source/marker_run.py", "source/pyproject.toml", "source/LICENSE"}
PUBLIC_SCRIPTS = {
    "bootstrap.py", "build_source_zip.py", "verify_setup.py",
    "fetch_optional_assets.py", "setup_saluki.py", "bundled_core.py",
}
PUBLIC_ASSETS = {
    "python-runtime.json", "python-packages.json", "requirements-runtime.lock.txt",
    "ollama-runtime.json", "ocr-models.json", "saluki-runtime.json", "saluki-models.json",
    "optional-assets.json", "saluki-assets.json",
    "bundled-core.json",
    "huggingface-ollama.json",
}
BUNDLED_CORE_PATH = "bundled/core/docuworks_ctypes-1.0.1-py3-none-any.whl"
EXCLUDED_PARTS = {"__pycache__", ".git", ".ssh", ".venv", "node_modules"}
PRIVATE_PATH = re.compile(r"\b[A-Za-z]:[\\/][^\s\"']+")
PRIVATE_KEY = re.compile(r"(?m)^-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----")
ACCESS_TOKEN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b")


def permitted(relative: Path) -> bool:
    if any(part in EXCLUDED_PARTS or part.startswith(".") for part in relative.parts):
        return False
    name = relative.as_posix()
    if name in ROOT_FILES or name in SOURCE_FILES:
        return True
    if name == BUNDLED_CORE_PATH:
        return True
    if len(relative.parts) >= 3 and relative.parts[:2] in (
        ("source", "simple_drawing"), ("source", "tests")
    ):
        return relative.suffix == ".py"
    if len(relative.parts) == 2 and relative.parts[0] == "scripts":
        return relative.name in PUBLIC_SCRIPTS or (
            relative.name.startswith("setup") and relative.suffix == ".ps1"
        )
    if len(relative.parts) == 2 and relative.parts[0] == "assets":
        return relative.name in PUBLIC_ASSETS
    if relative.parts[0] == "licenses":
        return relative.suffix.lower() in {".json", ".txt", ".md"}
    return False


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_files(root: Path) -> dict[str, bytes]:
    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from bundled_core import validate_core_bundle
    core = validate_core_bundle(root)
    if core["path"] != BUNDLED_CORE_PATH:
        raise ValueError("The bundled Core path differs from the publication allowlist")
    selected = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if not permitted(relative):
            continue
        if path.is_symlink() or path.resolve() != path.absolute():
            raise ValueError(f"Public input must not be a link: {relative.as_posix()}")
        content = path.read_bytes()
        if relative.as_posix() == BUNDLED_CORE_PATH:
            # This exact public MIT wheel has its hash, members, licence and RECORD audited.
            if sha256(content) != core["sha256"]:
                raise ValueError("Bundled Core changed during publication inspection")
            selected[relative.as_posix()] = content
            continue
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Public input is not UTF-8 text: {relative.as_posix()}") from exc
        if path.suffix.lower() == ".ps1" and any(ord(char) > 127 for char in text):
            if not content.startswith(b"\xef\xbb\xbf"):
                raise ValueError(f"Non-ASCII PowerShell requires UTF-8 BOM: {relative.as_posix()}")
        for label, pattern in (("absolute local path", PRIVATE_PATH), ("private key", PRIVATE_KEY),
                               ("access token", ACCESS_TOKEN)):
            matches = list(pattern.finditer(text))
            if label == "absolute local path":
                # The SystemRoot fallback is an OS location, not a development/user path.
                safe_literals = {"c:/windows"}
                if relative.parts[:2] == ("source", "tests"):
                    # A synthetic archive-traversal rejection fixture, never a user's file.
                    safe_literals.add("c:" + "/outside")
                matches = [match for match in matches
                           if match.group(0).replace("\\", "/").lower() not in safe_literals]
            if matches:
                raise ValueError(f"Refusing {label} in {relative.as_posix()}")
        selected[relative.as_posix()] = content
    required = {
        "LICENSE", "README.md", "THIRD_PARTY_NOTICES.md", "初期設定.bat",
        "前処理.bat", "抽出.bat", "Excel再出力.bat", "モデル管理.bat",
        "実行設定.json", "抽出設定.json", "source/pyproject.toml",
        "source/LICENSE", "source/simple_drawing/__init__.py", "source/run.py",
        "scripts/bootstrap.py", "scripts/setup.ps1", "scripts/setup_python.ps1", "scripts/verify_setup.py",
        "assets/python-runtime.json", "assets/python-packages.json",
        "assets/requirements-runtime.lock.txt", "assets/ollama-runtime.json", "assets/ocr-models.json",
        "licenses/models/gemma3-terms.txt", "licenses/models/gemma3-prohibited-use-policy.txt",
        "licenses/ocr/LICENSE-APACHE-2.0.txt", "licenses/ocr/NOTICE.txt",
        BUNDLED_CORE_PATH, "assets/bundled-core.json", "scripts/bundled_core.py",
        "licenses/core/LICENSE.txt", "licenses/core/NOTICE.md",
        "全文OCR画像出力.bat", "source/simple_drawing/ocr_image.py",
        "source/simple_drawing/ocr_image_export.py",
        "source/simple_drawing/model_download.py", "assets/huggingface-ollama.json",
        "黄色マーカー追加.bat", "黄色マーカー_使い方.txt", "source/marker_run.py",
        "source/tests/test_marker_excel.py", "source/tests/test_marker_writer.py",
        "licenses/黄色マーカー-MIT-LICENSE.txt", "docs/VERIFICATION_JA.md",
    }
    required.update("source/simple_drawing/" + name + ".py" for name in (
        "excel", "llm", "model_gui", "model_manager", "model_probe", "model_terms",
        "pdf_ocr", "profiles", "saluki", "settings", "storage", "workflow", "xdw_input",
        "marker_excel", "xdw_marker",
    ))
    missing = sorted(required - selected.keys())
    if missing:
        raise ValueError("Required public inputs are missing: " + ", ".join(missing))
    return selected


def project_version(files: dict[str, bytes]) -> str:
    project = tomllib.loads(files["source/pyproject.toml"].decode("utf-8"))["project"]
    version = project["version"]
    init = files["source/simple_drawing/__init__.py"].decode("utf-8")
    match = re.search(r'^__version__\s*=\s*[\"\']([^\"\']+)[\"\']', init, re.MULTILINE)
    if not match or match.group(1) != version:
        raise ValueError("Package and module versions differ")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.]+)?", version):
        raise ValueError("Version cannot be used as an archive filename")
    return version


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 0
    info.external_attr = 0
    return info


def build(root: Path, output: Path | None = None, *, check_only: bool = False) -> dict:
    root = root.resolve(strict=True)
    files = source_files(root)
    version = project_version(files)
    manifest = {
        "schema_version": 1,
        "package": "drawing-field-extractor-simple",
        "version": version,
        "distribution_kind": "source-with-bundled-core-online-setup",
        "bundled_core": {"name": "docuworks-ctypes", "version": "1.0.1",
                         "path": BUNDLED_CORE_PATH, "sha256": sha256(files[BUNDLED_CORE_PATH])},
        "files": [{"path": name, "bytes": len(content), "sha256": sha256(content)}
                  for name, content in sorted(files.items())],
    }
    summary = {"version": version, "source_files": len(files),
               "uncompressed_bytes": sum(map(len, files.values())), "check_only": check_only}
    if check_only:
        return summary
    output = output or root / "dist" / f"drawing-field-extractor-simple-v{version}-source.zip"
    output = output.resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("Output must have the .zip extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    encoded_manifest = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(zip_info(name), content, compresslevel=9)
        archive.writestr(zip_info("SOURCE_MANIFEST.json"), encoded_manifest, compresslevel=9)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        if set(archive.namelist()) != set(files) | {"SOURCE_MANIFEST.json"}:
            raise ValueError("Archive file inventory differs")
        for name, content in files.items():
            if archive.read(name) != content:
                raise ValueError("Archive content differs: " + name)
        if archive.read("SOURCE_MANIFEST.json") != encoded_manifest:
            raise ValueError("Archive manifest differs")
    temporary.replace(output)
    digest = sha256(output.read_bytes())
    output.with_name(output.name + ".sha256").write_text(
        digest + "  " + output.name + "\n", encoding="utf-8", newline="\n")
    output.with_name(output.name + ".manifest.json").write_text(
        json.dumps({**manifest, "archive": {"name": output.name,
            "bytes": output.stat().st_size, "sha256": digest}}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return {**summary, "archive": output.name, "archive_bytes": output.stat().st_size,
            "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output, check_only=args.check_only),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

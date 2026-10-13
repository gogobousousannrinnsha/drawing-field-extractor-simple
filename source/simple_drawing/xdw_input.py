"""Read-only XDW input, normalized to the saved visual-page PNG contract.

Core is imported only when an XDW input is opened. The SDK applies page
orientation during rasterization; this adapter never rotates that raster again.
"""
from __future__ import annotations

import ctypes
from importlib.metadata import version
import math
import os
from pathlib import Path
import tempfile

from .pdf_ocr import MAX_IMAGE_PIXELS, MAX_PAGES, RENDER_DPI
from .storage import sha

ADAPTER_VERSION = 1
SDK_PATH_LIMIT = 255


def _load_api(api_factory=None):
    if api_factory is not None:
        return api_factory()
    try:
        from docuworks_ctypes import XdwApi
    except ImportError as exc:
        raise RuntimeError("XDW入力には同封するDocuWorks Coreが必要です。初期設定を確認してください。") from exc
    try:
        return XdwApi.load()
    except Exception as exc:
        raise RuntimeError("XDW入力用のDocuWorks環境を読み込めません: " + str(exc)) from exc


def _renderer_contract(api):
    runtime = api.runtime_info
    dll_path = runtime.dll_path
    if dll_path is None:
        raise ValueError("XDW画像化のDLLを特定できません。")
    dll_path = Path(dll_path)
    dll_hash = sha(dll_path)
    recorded_hash = getattr(runtime, "dll_sha256", None)
    if recorded_hash and dll_hash != recorded_hash.casefold():
        raise ValueError("XDW画像化のDLLが読み込み後に変更されました。")
    return {"adapter": "simple-xdw-visual-page", "adapter_version": ADAPTER_VERSION,
            "core_version": version("docuworks-ctypes"), "runtime_version": runtime.version_text,
            "dll": {"name": dll_path.name, "sha256": dll_hash, "size_bytes": dll_path.stat().st_size},
            "render_dpi": RENDER_DPI, "color": "rgb", "orientation": "sdk-displayed-page"}


def renderer_contract(*, api_factory=None):
    """Identify the selected runtime without persisting its machine-specific path."""
    return _renderer_contract(_load_api(api_factory))


def _commit_new_file(temporary, output):
    # Windows rename is atomic and rejects an existing destination.
    if os.name == "nt":
        os.rename(temporary, output)
    else:
        os.link(temporary, output)
        temporary.unlink()


class XdwInput:
    """Own exactly one read-only Core document handle; optional factory aids tests."""

    def __init__(self, path, *, api_factory=None):
        self.path = Path(path).resolve()
        self.api_factory = api_factory
        self.api = None
        self.doc = None
        self.page_count = None

    def __enter__(self):
        if self.doc is not None:
            raise RuntimeError("XDW入力は既に開かれています。")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if self.path.suffix.casefold() != ".xdw":
            raise ValueError("XDW入力には.xdwファイルを指定してください。")
        self.api = _load_api(self.api_factory)
        from docuworks_ctypes.enums import AuthMode, OpenMode
        self.doc = self.api.open_document(self.path, mode=OpenMode.READONLY, auth=AuthMode.NO_DIALOG)
        try:
            self.page_count = self.doc.page_count
            if type(self.page_count) is not int or not 1 <= self.page_count <= MAX_PAGES:
                raise ValueError(f"XDW must contain 1 to {MAX_PAGES} pages")
            return self
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def __exit__(self, exc_type, exc, traceback):
        document, self.doc = self.doc, None
        self.page_count = None
        if document is not None:
            try:
                document.close()
            except BaseException as close_error:
                if exc is None:
                    raise
                exc.add_note("XDW document close also failed: " + str(close_error))
        return False

    def renderer_contract(self):
        self._ensure_open()
        return _renderer_contract(self.api)

    def _ensure_open(self):
        if self.doc is None:
            raise RuntimeError("XDW入力が開かれていません。")

    def _metadata(self, number):
        self._ensure_open()
        if type(number) is not int or not 1 <= number <= self.page_count:
            raise ValueError("Page number is outside the XDW")
        from docuworks_ctypes._raw import types as T
        from docuworks_ctypes.errors import check_result
        # GetPageInformation accepts the extended layout by its nSize. The
        # declared Core signature uses the compatible base-structure pointer.
        info = T.XDW_PAGE_INFO_EX()
        info.nSize = ctypes.sizeof(info)
        pointer = ctypes.cast(ctypes.byref(info), ctypes.POINTER(T.XDW_PAGE_INFO))
        check_result(self.doc.raw.XDW_GetPageInformation(self.doc.handle, number, pointer),
                     "XDW_GetPageInformation")
        width, height = info.nWidth / 100, info.nHeight / 100
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError("XDW page dimensions must be positive and finite")
        if info.nDegree % 90:
            raise ValueError("XDW page rotation must be a multiple of 90 degrees")
        pixels = (math.ceil(width * RENDER_DPI / 25.4), math.ceil(height * RENDER_DPI / 25.4))
        if min(pixels) <= 0 or pixels[0] * pixels[1] > MAX_IMAGE_PIXELS:
            raise ValueError("XDW page exceeds supported raster dimensions")
        return {"number": number, "width_mm": width, "height_mm": height,
                "width_px": pixels[0], "height_px": pixels[1], "rotation": 0,
                "source_rotation": info.nDegree % 360,
                "dimension_basis": {"kind": "xdw_displayed_page",
                    "coordinate_system": "top-left-x-right-y-down", "render_dpi": RENDER_DPI,
                    "sdk_page_width_100mm": info.nWidth, "sdk_page_height_100mm": info.nHeight,
                    "sdk_source_degree": info.nDegree, "sdk_page_type": info.nPageType,
                    "raster_size_basis": "predicted_at_render_dpi", "orientation_applied_by": "sdk"}}

    def inspect(self):
        self._ensure_open()
        return [self._metadata(n) for n in range(1, self.page_count + 1)]

    def render(self, number, image_path):
        from docuworks_ctypes._raw import constants as C, types as T
        from docuworks_ctypes.encoding import wchar_buffer
        from docuworks_ctypes.errors import check_result
        from PIL import Image
        metadata = self._metadata(number)
        output = Path(image_path).absolute()
        if output.resolve() == self.path:
            raise ValueError("The page image must not replace the source XDW")
        if output.suffix.casefold() != ".png":
            raise ValueError("XDW page image output must be PNG")
        if os.path.lexists(output):
            raise FileExistsError("XDW page image output already exists: " + str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        # SDK output paths are limited to 255 UTF-16 units. Cache directories
        # contain two SHA hashes, so use process TEMP for the SDK BMP. PNG
        # staging stays beside the destination for an atomic same-volume move.
        with tempfile.TemporaryDirectory(prefix="xdw-") as sdk_temporary, \
                tempfile.TemporaryDirectory(prefix="xdw-", dir=output.parent) as png_temporary:
            bmp, png = Path(sdk_temporary) / "p.bmp", Path(png_temporary) / "p.png"
            if len(str(bmp).encode("utf-16-le")) // 2 > SDK_PATH_LIMIT:
                raise ValueError("SDK image path exceeds 255 UTF-16 code units; use a shorter portable folder path")
            options = T.XDW_IMAGE_OPTION()
            options.nSize = ctypes.sizeof(options)
            options.nDpi, options.nColor = RENDER_DPI, C.XDW_IMAGE_COLOR
            check_result(self.doc.raw.XDW_ConvertPageToImageFileW(
                self.doc.handle, number, wchar_buffer(str(bmp)), ctypes.byref(options)),
                "XDW_ConvertPageToImageFileW")
            previous_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
            try:
                # Own the binary stream explicitly: some Pillow BMP modes keep
                # an internal file reference after conversion on Windows.
                with bmp.open("rb") as stream, Image.open(stream) as image:
                    width, height = image.size
                    if min(width, height) <= 0 or width * height > MAX_IMAGE_PIXELS:
                        raise ValueError("XDW image exceeds supported raster dimensions")
                    # SDK pixel rounding may differ by one pixel from ceil.
                    if any(abs(actual - expected) > 1 for actual, expected in
                           zip((width, height), (metadata["width_px"], metadata["height_px"]))):
                        raise ValueError("Rendered image dimensions disagree with XDW page geometry")
                    with image.convert("RGB") as rgb:
                        rgb.save(png, format="PNG", dpi=(RENDER_DPI, RENDER_DPI))
            finally:
                Image.MAX_IMAGE_PIXELS = previous_limit
            metadata.update(width_px=width, height_px=height)
            metadata["dimension_basis"]["raster_size_basis"] = "actual_sdk_bmp"
            _commit_new_file(png, output)
        return metadata


def inspect_xdw(path, *, api_factory=None):
    with XdwInput(path, api_factory=api_factory) as document:
        return document.inspect()


def render_page(path, number, image_path, *, api_factory=None):
    with XdwInput(path, api_factory=api_factory) as document:
        return document.render(number, image_path)

"""Core-backed input contract checks with fake SDK calls; no installed DLL loads."""
import ctypes
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image
from docuworks_ctypes._raw import constants as C, types as T
from docuworks_ctypes.encoding import decode_wchar_buffer
from docuworks_ctypes.enums import AuthMode, OpenMode

from simple_drawing import xdw_input
from simple_drawing.storage import sha


class FakeRaw:
    def __init__(self, pages):
        self.pages = pages
        self.info_result = self.convert_result = 0
        self.rendered = []
        self.size_override = None
        self.mode = "L"

    def XDW_GetPageInformation(self, handle, number, pointer):
        assert handle == 77
        info = ctypes.cast(pointer, ctypes.POINTER(T.XDW_PAGE_INFO_EX)).contents
        assert info.nSize == ctypes.sizeof(T.XDW_PAGE_INFO_EX)
        for key, value in self.pages[number - 1].items():
            setattr(info, key, value)
        return self.info_result

    def XDW_ConvertPageToImageFileW(self, handle, number, path, option):
        assert handle == 77
        option = ctypes.cast(option, ctypes.POINTER(T.XDW_IMAGE_OPTION)).contents
        assert option.nSize == ctypes.sizeof(T.XDW_IMAGE_OPTION)
        assert option.nDpi == 300 and option.nColor == C.XDW_IMAGE_COLOR
        output = Path(decode_wchar_buffer(path))
        self.rendered.append(output)
        page = self.pages[number - 1]
        size = self.size_override or (round(page["nWidth"] / 100 * 300 / 25.4),
                                      round(page["nHeight"] / 100 * 300 / 25.4))
        with Image.new(self.mode, size, "white") as image:
            image.putpixel((1, 1), 0 if self.mode == "L" else (0, 0, 0))
            image.save(output, format="BMP")
        return self.convert_result


class FakeDocument:
    def __init__(self, raw):
        self.raw, self.handle = raw, 77
        self.count = len(raw.pages)
        self.count_error = self.close_error = None
        self.closes = 0

    @property
    def page_count(self):
        if self.count_error:
            raise self.count_error
        return self.count

    def close(self):
        self.closes += 1
        if self.close_error:
            raise self.close_error


class FakeApi:
    def __init__(self, root, pages=None):
        self.doc = FakeDocument(FakeRaw(pages or [
            {"nWidth": 2540, "nHeight": 1270, "nDegree": 0, "nPageType": 1}]))
        dll = root / "dll location" / "xdwapi.dll"
        dll.parent.mkdir(parents=True, exist_ok=True)
        dll.write_bytes(b"fake-verified-dll")
        self.runtime_info = SimpleNamespace(dll_path=dll, version_text="9.1.7",
                                            dll_sha256=sha(dll))
        self.opened = []
        self.open_error = None

    def open_document(self, path, *, mode, auth):
        self.opened.append((path, mode, auth))
        if self.open_error:
            raise self.open_error
        return self.doc


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "日本語 original.XDW"
    source.write_bytes(b"immutable-source")
    return source, FakeApi(tmp_path)


def test_module_import_does_not_load_core_paddle_or_models():
    script = """
import importlib.abc, sys
class Blocker(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path, target=None):
  if fullname.split('.')[0] in {'docuworks_ctypes', 'paddle', 'paddleocr', 'paddlex'}:
   raise AssertionError('Unexpected dependency load: ' + fullname)
sys.meta_path.insert(0, Blocker())
import simple_drawing.xdw_input
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, "-B", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("degree", [0, 90, 180, 270, -90])
def test_normalized_visual_geometry_without_second_rotation(fixture, tmp_path, degree):
    source, api = fixture
    api.doc.raw.pages[0]["nDegree"] = degree
    original = source.read_bytes()
    output = tmp_path / "日本語 保存" / "page.png"
    with xdw_input.XdwInput(source, api_factory=lambda: api) as document:
        metadata = document.inspect()[0]
        assert metadata["width_mm"] == 25.4 and metadata["height_mm"] == 12.7
        assert metadata["rotation"] == 0 and metadata["source_rotation"] == degree % 360
        assert metadata["dimension_basis"]["sdk_source_degree"] == degree
        rendered = document.render(1, output)
    assert source.read_bytes() == original
    assert api.opened == [(source.resolve(), OpenMode.READONLY, AuthMode.NO_DIALOG)]
    assert api.doc.closes == 1
    assert rendered["dimension_basis"]["raster_size_basis"] == "actual_sdk_bmp"
    with Image.open(output) as image:
        assert image.mode == "RGB" and image.size == (300, 150)
        assert image.getpixel((1, 1)) == (0, 0, 0)
    assert not list(output.parent.glob("xdw-*"))


def test_contract_identifies_runtime_without_absolute_path(fixture):
    source, api = fixture
    with xdw_input.XdwInput(source, api_factory=lambda: api) as document:
        contract = document.renderer_contract()
    assert contract["core_version"] == version("docuworks-ctypes")
    assert contract["runtime_version"] == "9.1.7"
    assert contract["dll"] == {"name": "xdwapi.dll", "sha256": sha(api.runtime_info.dll_path),
                               "size_bytes": 17}
    assert str(api.runtime_info.dll_path.parent) not in json.dumps(contract)
    assert xdw_input.renderer_contract(api_factory=lambda: api) == contract


def test_changed_runtime_is_rejected(fixture):
    source, api = fixture
    api.runtime_info.dll_path.write_bytes(b"modified")
    with xdw_input.XdwInput(source, api_factory=lambda: api) as document:
        with pytest.raises(ValueError, match="DLL"):
            document.renderer_contract()
    assert api.doc.closes == 1


def test_uppercase_sdk_hash_identifies_same_dll(fixture):
    source, api = fixture
    api.runtime_info.dll_sha256 = api.runtime_info.dll_sha256.upper()
    with xdw_input.XdwInput(source, api_factory=lambda: api) as document:
        assert document.renderer_contract()["dll"]["sha256"] == sha(api.runtime_info.dll_path)


@pytest.mark.parametrize("count", [0, 1001, True])
def test_page_count_rejected_and_handle_closed(fixture, count):
    source, api = fixture
    api.doc.count = count
    with pytest.raises(ValueError, match="pages"):
        with xdw_input.XdwInput(source, api_factory=lambda: api):
            pytest.fail("must reject")
    assert api.doc.closes == 1


def test_count_failure_also_closes(fixture):
    source, api = fixture
    api.doc.count_error = RuntimeError("page count failure")
    with pytest.raises(RuntimeError, match="page count"):
        xdw_input.XdwInput(source, api_factory=lambda: api).__enter__()
    assert api.doc.closes == 1


def test_close_failure_preserves_primary_error(fixture):
    source, api = fixture
    api.doc.close_error = RuntimeError("close failure")
    with pytest.raises(ValueError, match="primary") as caught:
        with xdw_input.XdwInput(source, api_factory=lambda: api):
            raise ValueError("primary")
    assert api.doc.closes == 1
    assert "close failure" in caught.value.__notes__[0]


def test_close_failure_without_primary_is_reported(fixture):
    source, api = fixture
    api.doc.close_error = RuntimeError("close failure")
    with pytest.raises(RuntimeError, match="close failure"):
        with xdw_input.XdwInput(source, api_factory=lambda: api):
            pass
    assert api.doc.closes == 1


@pytest.mark.parametrize("key,value", [("nWidth", 0), ("nHeight", -1),
                                     ("nWidth", 2147483647), ("nDegree", 45)])
def test_invalid_geometry_rejected(fixture, key, value):
    source, api = fixture
    api.doc.raw.pages[0][key] = value
    with pytest.raises(ValueError):
        xdw_input.inspect_xdw(source, api_factory=lambda: api)
    assert api.doc.closes == 1


@pytest.mark.parametrize("number", [0, -1, 2, 1.0, True])
def test_invalid_page_number_rejected(fixture, tmp_path, number):
    source, api = fixture
    with pytest.raises(ValueError, match="outside"):
        xdw_input.render_page(source, number, tmp_path / "page.png", api_factory=lambda: api)
    assert not api.doc.raw.rendered
    assert api.doc.closes == 1


def test_existing_output_and_source_are_preserved(fixture, tmp_path):
    source, api = fixture
    output = tmp_path / "existing.png"
    output.write_bytes(b"keep-existing")
    with xdw_input.XdwInput(source, api_factory=lambda: api) as document:
        with pytest.raises(FileExistsError):
            document.render(1, output)
        with pytest.raises(ValueError, match="source"):
            document.render(1, source)
    assert output.read_bytes() == b"keep-existing"
    assert source.read_bytes() == b"immutable-source"
    assert not api.doc.raw.rendered


def test_racing_output_is_preserved(fixture, tmp_path, monkeypatch):
    source, api = fixture
    output = tmp_path / "page.png"
    commit = xdw_input._commit_new_file
    def race(temporary, destination):
        destination.write_bytes(b"racing-output")
        commit(temporary, destination)
    monkeypatch.setattr(xdw_input, "_commit_new_file", race)
    with pytest.raises(FileExistsError):
        xdw_input.render_page(source, 1, output, api_factory=lambda: api)
    assert output.read_bytes() == b"racing-output"
    assert not list(output.parent.glob("xdw-*"))


@pytest.mark.parametrize("failure", ["sdk", "geometry", "png"])
def test_failed_render_cleans_owned_temporaries(fixture, tmp_path, monkeypatch, failure):
    source, api = fixture
    output = tmp_path / "page.png"
    unrelated = tmp_path / "keep.bmp"
    unrelated.write_bytes(b"keep")
    if failure == "sdk":
        api.doc.raw.convert_result = -1
    elif failure == "geometry":
        api.doc.raw.size_override = (150, 300)
    else:
        original_save = Image.Image.save
        def fail_png(image, destination, *args, **kwargs):
            if kwargs.get("format") == "PNG":
                raise OSError("PNG write failed")
            return original_save(image, destination, *args, **kwargs)
        monkeypatch.setattr(Image.Image, "save", fail_png)
    with pytest.raises(Exception):
        xdw_input.render_page(source, 1, output, api_factory=lambda: api)
    assert not output.exists() and not list(tmp_path.glob("xdw-*"))
    assert unrelated.read_bytes() == b"keep" and source.read_bytes() == b"immutable-source"
    assert api.doc.closes == 1


def test_sdk_path_limit_counts_utf16_code_units(fixture, tmp_path, monkeypatch):
    source, api = fixture
    # A supplementary character occupies two UTF-16 units. Reject an unusually
    # deep process TEMP, rather than exposing the deep cache path to the SDK.
    parent = tmp_path / ("日\U0001f600" * 35) / ("日\U0001f600" * 35)
    if len(str(parent).encode("utf-16-le")) // 2 < 235:
        parent /= "日" * 40
    parent.mkdir(parents=True)
    monkeypatch.setattr(xdw_input.tempfile, "tempdir", str(parent))
    output = tmp_path / "page.png"
    with pytest.raises(ValueError, match="255 UTF-16"):
        xdw_input.render_page(source, 1, output, api_factory=lambda: api)
    assert not api.doc.raw.rendered and not output.exists()
    assert not list(parent.glob("xdw-*"))


def test_deep_cache_output_uses_short_sdk_temp_and_same_volume_commit(fixture, tmp_path, monkeypatch):
    source, api = fixture
    short_temp = tmp_path / "temp"
    short_temp.mkdir()
    monkeypatch.setattr(xdw_input.tempfile, "tempdir", str(short_temp))
    output = (tmp_path / "data" / "cache" / ("a" * 64) / ("b" * 64) /
              "1.pending-20261013-123456-abcdef" / "日本語 空白" / "page.png")
    assert len(str(output).encode("utf-16-le")) // 2 > 255
    commit = xdw_input._commit_new_file
    commits = []
    def record_commit(temporary, destination):
        commits.append((temporary, destination))
        assert temporary.parent.parent == destination.parent
        commit(temporary, destination)
    monkeypatch.setattr(xdw_input, "_commit_new_file", record_commit)
    rendered = xdw_input.render_page(source, 1, output, api_factory=lambda: api)
    assert rendered["width_px"] == 300 and output.exists()
    assert len(commits) == 1
    assert api.doc.raw.rendered[0].is_relative_to(short_temp)
    assert len(str(api.doc.raw.rendered[0]).encode("utf-16-le")) // 2 <= 255
    assert not list(short_temp.iterdir()) and not list(output.parent.glob("xdw-*"))


def test_actual_pixel_rounding_is_preserved(fixture, tmp_path):
    source, api = fixture
    api.doc.raw.size_override = (299, 149)
    rendered = xdw_input.render_page(source, 1, tmp_path / "page.png", api_factory=lambda: api)
    assert (rendered["width_px"], rendered["height_px"]) == (299, 149)


def test_missing_source_does_not_load_api(tmp_path):
    called = []
    with pytest.raises(FileNotFoundError):
        xdw_input.inspect_xdw(tmp_path / "missing.xdw", api_factory=lambda: called.append(True))
    assert not called


def test_closed_input_cannot_inspect_or_render(fixture, tmp_path):
    source, api = fixture
    document = xdw_input.XdwInput(source, api_factory=lambda: api)
    with document:
        pass
    with pytest.raises(RuntimeError, match="開かれて"):
        document.inspect()
    with pytest.raises(RuntimeError, match="開かれて"):
        document.render(1, tmp_path / "page.png")

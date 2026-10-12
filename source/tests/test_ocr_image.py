"""Saved OCR image rendering checks; no model, SDK or original page is used."""
from copy import deepcopy
import math
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageFont

from simple_drawing.ocr_image import (
    FALLBACK_FONT_NAMES, MAX_IMAGE_PIXELS, MAX_PATCH_PIXELS, SYSTEM_FONT_NAMES,
    _font_path, _missing_characters, _select_font, _system_font_paths, render_ocr_image,
)


def saved_ocr(text="部品番号 0001", *, box=None, polygon=None):
    box = box or {"x": 40, "y": 50, "width": 220, "height": 40}
    region = {"id": "p1:r000001", "text": text, "confidence": 0,
              "bbox_px": box}
    if polygon is not None:
        region["polygon_px"] = polygon
    return {"version": 1, "page": {"number": 1, "width_px": 320, "height_px": 240,
            "width_mm": 320 * 25.4 / 300, "height_mm": 240 * 25.4 / 300,
            "rotation": 0, "source_rotation": 90}, "regions": [region]}


@pytest.fixture
def japanese_font():
    try:
        return _font_path(None)
    except FileNotFoundError:
        pytest.skip("A Japanese Windows system font is required for raster tests")


def ink_bounds(image):
    with ImageChops.invert(image.convert("L")) as ink:
        return ink.getbbox()


def test_white_original_size_black_text_exact_position_and_actual_mm_dpi(tmp_path, japanese_font):
    output = tmp_path / "認識文字 白紙.png"
    ocr = saved_ocr()
    preserved = deepcopy(ocr)
    report = render_ocr_image(ocr, output, japanese_font)
    assert ocr == preserved
    assert report["placed_regions"] == 1
    assert report["empty_regions"] == 0
    assert report["font"]["name"]
    assert report["dpi"] == pytest.approx([300, 300])
    assert report["warnings"] == []
    with Image.open(output) as image:
        assert image.mode == "RGB"
        assert image.size == (320, 240)
        assert image.info["dpi"] == pytest.approx([300, 300], abs=.02)
        assert image.getpixel((0, 0)) == (255, 255, 255)
        bounds = ink_bounds(image)
        assert bounds is not None
        assert 40 <= bounds[0] < bounds[2] <= 260
        assert 50 <= bounds[1] < bounds[3] <= 90
        assert all(r == g == b for r, g, b in image.get_flattened_data())


def test_all_original_strings_are_passed_unchanged_without_confidence_filter(tmp_path, japanese_font):
    text = "  00000001 部品番号\n材質 SUS304  "
    ocr = saved_ocr(text)
    ocr["regions"].append({"id": "p1:r000002", "text": "0002", "confidence": -999,
                           "bbox_px": {"x": 5, "y": 150, "width": 50, "height": 25}})
    seen = []
    actual_draw = ImageDraw.ImageDraw.multiline_text
    def record(self, position, value, *args, **kwargs):
        seen.append(value)
        return actual_draw(self, position, value, *args, **kwargs)
    with patch.object(ImageDraw.ImageDraw, "multiline_text", record):
        report = render_ocr_image(ocr, tmp_path / "raw.png", japanese_font)
    assert seen == [text, "0002"]
    assert report["placed_regions"] == 2


def test_clockwise_90_degree_quad_supplies_text_direction(tmp_path, japanese_font):
    horizontal = saved_ocr("0001 ABC", box={"x": 20, "y": 20, "width": 180, "height": 30},
                           polygon=[[20, 20], [200, 20], [200, 50], [20, 50]])
    vertical = saved_ocr("0001 ABC", box={"x": 40, "y": 20, "width": 30, "height": 180},
                         polygon=[[70, 20], [70, 200], [40, 200], [40, 20]])
    render_ocr_image(horizontal, tmp_path / "horizontal.png", japanese_font)
    render_ocr_image(vertical, tmp_path / "vertical.png", japanese_font)
    with Image.open(tmp_path / "horizontal.png") as a, Image.open(tmp_path / "vertical.png") as b:
        horizontal_bounds, vertical_bounds = ink_bounds(a), ink_bounds(b)
        assert horizontal_bounds[2] - horizontal_bounds[0] > (horizontal_bounds[3] - horizontal_bounds[1]) * 2
        assert vertical_bounds[3] - vertical_bounds[1] > (vertical_bounds[2] - vertical_bounds[0]) * 2
        assert 40 <= vertical_bounds[0] < vertical_bounds[2] <= 70
        assert 20 <= vertical_bounds[1] < vertical_bounds[3] <= 200
        with a.crop((20, 20, 200, 50)) as source, b.crop((40, 20, 70, 200)) as rotated:
            with source.transpose(Image.Transpose.ROTATE_270) as expected:
                assert expected.tobytes() == rotated.tobytes()


def test_slanted_quad_keeps_ink_inside_polygon(tmp_path, japanese_font):
    ocr = saved_ocr("傾いた認識文字", box={"x": 30, "y": 30, "width": 250, "height": 75},
                    polygon=[[30, 30], [280, 65], [270, 105], [35, 70]])
    render_ocr_image(ocr, tmp_path / "slanted.png", japanese_font)
    with Image.open(tmp_path / "slanted.png") as image:
        assert ink_bounds(image) is not None
        for point in [(30, 104), (279, 31), (100, 29), (281, 80)]:
            assert image.getpixel(point) == (255, 255, 255)


def test_long_value_is_never_shortened_and_warns_about_tiny_text(tmp_path, japanese_font):
    text = "000000001-" * 100
    ocr = saved_ocr(text, box={"x": 5, "y": 10, "width": 10, "height": 8})
    seen = []
    actual_draw = ImageDraw.ImageDraw.multiline_text
    def record(self, position, value, *args, **kwargs):
        seen.append(value)
        return actual_draw(self, position, value, *args, **kwargs)
    with patch.object(ImageDraw.ImageDraw, "multiline_text", record):
        report = render_ocr_image(ocr, tmp_path / "condensed.png", japanese_font)
    assert seen == [text]
    assert any(row["code"] == "small_text" for row in report["warnings"])
    with Image.open(tmp_path / "condensed.png") as image:
        assert ink_bounds(image) is not None


def test_zero_regions_and_blank_strings_produce_blank_page_success(tmp_path, japanese_font):
    for name, regions, expected_empty in (("zero", [], 0), ("blank", [saved_ocr(" \n ")["regions"][0]], 1)):
        ocr = saved_ocr()
        ocr["regions"] = regions
        report = render_ocr_image(ocr, tmp_path / (name + ".png"), japanese_font)
        assert report["placed_regions"] == 0
        assert report["empty_regions"] == expected_empty
        with Image.open(tmp_path / (name + ".png")) as image:
            assert ink_bounds(image) is None


@pytest.mark.parametrize("change", [
    lambda o: o["page"].update(width_px=0),
    lambda o: o["page"].update(width_px=True),
    lambda o: o["page"].update(width_px=MAX_IMAGE_PIXELS + 1, height_px=1),
    lambda o: o["page"].update(width_mm=math.inf),
    lambda o: o["page"].update(height_mm=math.nan),
    lambda o: o["page"].update(rotation=90),
    lambda o: o["regions"][0]["bbox_px"].update(x=-1),
    lambda o: o["regions"][0]["bbox_px"].update(y=math.nan),
    lambda o: o["regions"][0]["bbox_px"].update(width=0),
    lambda o: o["regions"][0]["bbox_px"].update(width=999),
    lambda o: o["regions"][0].update(text=None),
    lambda o: o["regions"][0].update(polygon_px=None),
    lambda o: o["regions"][0].update(polygon_px=[[40, 50], [260, 90], [260, 50], [40, 90]]),
    lambda o: o["regions"][0].update(polygon_px=[[40, 50], [260, 50], [260, 90], [40, 91]]),
    lambda o: o["regions"].append(deepcopy(o["regions"][0])),
])
def test_invalid_saved_geometry_fails_before_output_is_created(tmp_path, change):
    ocr = saved_ocr()
    change(ocr)
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError):
        render_ocr_image(ocr, output, tmp_path / "font-does-not-exist.ttf")
    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp-*"))


def test_patch_limit_failure_does_not_allocate_full_page(tmp_path):
    ocr = saved_ocr()
    side = math.ceil(math.sqrt(MAX_PATCH_PIXELS)) + 1
    ocr["page"].update(width_px=side, height_px=side)
    ocr["regions"][0]["bbox_px"] = {"x": 0, "y": 0, "width": side, "height": side}
    with patch("simple_drawing.ocr_image.Image.new", side_effect=AssertionError("no allocation")):
        with pytest.raises(ValueError, match="patch pixel limit"):
            render_ocr_image(ocr, tmp_path / "large.png")


def test_missing_font_and_missing_glyph_report_clear_reasons(tmp_path, japanese_font):
    with pytest.raises(FileNotFoundError, match="フォント"):
        render_ocr_image(saved_ocr(), tmp_path / "missing.png", tmp_path / "missing.ttf")
    report = render_ocr_image(saved_ocr("部品\U0010ffff"), tmp_path / "tofu.png", japanese_font)
    warning = next(row for row in report["warnings"] if row["code"] == "font_missing_characters")
    assert "U+10FFFF" in warning["characters"]


def test_default_installed_fallback_covers_chinese_japanese_and_digits(tmp_path):
    text = "查 部品番号 材質 あいう アイウ SUS304 0001"
    candidates = _system_font_paths()
    if not any(not _missing_characters(path, [text]) for path in candidates):
        pytest.skip("An installed system font covering this sample is required")
    report = render_ocr_image(saved_ocr(text), tmp_path / "fallback.png")
    assert not any(row["code"] == "font_missing_characters" for row in report["warnings"])
    assert not _missing_characters(Path(report["font"]["path"]), [text])
    assert Path(report["font"]["path"]) in candidates


def test_default_fallback_order_stops_at_first_full_coverage_font(tmp_path):
    folder = tmp_path / "Fonts"
    folder.mkdir()
    names = SYSTEM_FONT_NAMES + FALLBACK_FONT_NAMES
    for name in names:
        (folder / name).write_bytes(b"mock system font")
    scanned = []
    def missing(path, texts):
        assert texts == ("查 部品番号 0001",)
        scanned.append(path.name)
        return [] if path.name == "msyh.ttc" else ["U+67E5"]
    with patch.dict(os.environ, {"SystemRoot": str(tmp_path)}), \
            patch("simple_drawing.ocr_image._missing_characters", missing):
        selected, absent = _select_font(None, ["查 部品番号 0001"])
    assert selected.name == "msyh.ttc"
    assert absent == []
    assert scanned == list(SYSTEM_FONT_NAMES) + ["NotoSansJP-VF.ttf", "msyh.ttc"]


def test_complete_primary_coverage_only_scans_primary_and_explicit_choice_is_honored(tmp_path):
    primary = tmp_path / "primary.ttf"
    primary.write_bytes(b"mock font")
    with patch("simple_drawing.ocr_image._font_path", return_value=primary), \
            patch("simple_drawing.ocr_image._missing_characters", return_value=[]) as coverage, \
            patch("simple_drawing.ocr_image._system_font_paths", side_effect=AssertionError("no fallbacks")):
        assert _select_font(None, ["部品番号"])[0] == primary
    assert coverage.call_count == 1
    with patch("simple_drawing.ocr_image._missing_characters", return_value=["U+67E5"]) as coverage, \
            patch("simple_drawing.ocr_image._system_font_paths", side_effect=AssertionError("honor explicit choice")):
        assert _select_font(primary, ["查"])[1] == ["U+67E5"]
    assert coverage.call_count == 1


def test_no_full_coverage_chooses_fewest_missing_with_stable_preference(tmp_path):
    folder = tmp_path / "Fonts"
    folder.mkdir()
    for name in SYSTEM_FONT_NAMES + FALLBACK_FONT_NAMES:
        (folder / name).write_bytes(b"mock font")
    def missing(path, texts):
        return ["U+AAAA"] if path.name in {"NotoSansJP-VF.ttf", "msyh.ttc"} else ["U+AAAA", "U+BBBB"]
    with patch.dict(os.environ, {"SystemRoot": str(tmp_path)}), \
            patch("simple_drawing.ocr_image._missing_characters", missing):
        selected, absent = _select_font(None, ["a"])
    assert selected.name == "NotoSansJP-VF.ttf"
    assert absent == ["U+AAAA"]


def test_existing_output_and_racing_writer_are_never_overwritten(tmp_path, japanese_font):
    output = tmp_path / "protected.png"
    output.write_bytes(b"human output")
    with pytest.raises(FileExistsError):
        render_ocr_image(saved_ocr(), output, japanese_font)
    assert output.read_bytes() == b"human output"
    output.unlink()
    actual_save = Image.Image.save
    def race(self, path, *args, **kwargs):
        result = actual_save(self, path, *args, **kwargs)
        output.write_bytes(b"other writer")
        return result
    with patch.object(Image.Image, "save", race):
        with pytest.raises(FileExistsError):
            render_ocr_image(saved_ocr(), output, japanese_font)
    assert output.read_bytes() == b"other writer"
    assert not list(tmp_path.glob("*.tmp-*"))


def test_save_failure_leaves_no_partial_png(tmp_path, japanese_font):
    output = tmp_path / "failed.png"
    def fail(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")
    with patch.object(Image.Image, "save", fail):
        with pytest.raises(OSError, match="disk full"):
            render_ocr_image(saved_ocr(), output, japanese_font)
    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp-*"))

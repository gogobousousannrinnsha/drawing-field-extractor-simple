"""Geometry, tile coverage and evidence tests; no Paddle/GPU imports."""
import math
from pathlib import Path

import pytest
from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, FloatObject, NameObject, NumberObject, RectangleObject

from simple_drawing.pdf_ocr import inspect_pdf, render_page, recognize_page, tile_boxes


def make_pdf(path, *, rotation=0, unit=1, crop=True, inherited=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    page[NameObject("/UserUnit")] = FloatObject(unit)
    page[NameObject("/Rotate")] = NumberObject(rotation)
    if crop:
        page.cropbox = RectangleObject((100, 200, 500, 700))
    stream = DecodedStreamObject()
    # Red evidence inside the crop; blue evidence outside it.
    stream.set_data(b"1 0 0 rg 150 250 30 20 re f\n0 0 1 rg 10 10 20 20 re f\n")
    page[NameObject("/Contents")] = writer._add_object(stream)
    if inherited:
        parent = page["/Parent"].get_object()
        for name in ("/MediaBox", "/CropBox", "/Rotate"):
            if name in page:
                parent[NameObject(name)] = page[name]
                del page[name]
    writer.write(path)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("unit", [1, 2])
@pytest.mark.parametrize("crop", [False, True])
def test_visible_page_size_userunit_and_rotation(tmp_path, rotation, unit, crop):
    source = tmp_path / "日本語 original.pdf"
    image_path = tmp_path / "保存画像" / "page.png"
    make_pdf(source, rotation=rotation, unit=unit, crop=crop)
    original = source.read_bytes()
    metadata = inspect_pdf(source)[0]
    assert render_page(source, 1, image_path) == metadata
    assert source.read_bytes() == original
    w, h = (400, 500) if crop else (600, 800)
    if rotation in (90, 270):
        w, h = h, w
    assert metadata["number"] == 1
    assert metadata["source_rotation"] == rotation
    assert metadata["rotation"] == 0
    assert metadata["width_mm"] == pytest.approx(w * unit * 25.4 / 72)
    assert metadata["height_mm"] == pytest.approx(h * unit * 25.4 / 72)
    assert metadata["dimension_basis"]["user_unit"] == unit
    with Image.open(image_path) as image:
        assert image.mode == "RGB"
        assert image.size == (metadata["width_px"], metadata["height_px"])
        # Find the red mark in the rotated raster and project back to mm.
        r, g, b = image.split()
        with ImageChops.subtract(r, g) as difference:
            red = difference.point(lambda v: 255 if v > 100 else 0)
        red_box = red.getbbox()
        left, bottom, right, top = (100, 200, 500, 700) if crop else (0, 0, 600, 800)
        corners = [(150, 250), (180, 250), (180, 270), (150, 270)]
        def visual(x, y):
            if rotation == 0:
                return x - left, top - y
            if rotation == 90:
                return y - bottom, x - left
            if rotation == 180:
                return right - x, y - bottom
            return top - y, right - x
        expected = [visual(x, y) for x, y in corners]
        factor = unit * 25.4 / 72
        expected_mm = [min(x for x, y in expected) * factor, min(y for x, y in expected) * factor,
                       max(x for x, y in expected) * factor, max(y for x, y in expected) * factor]
        actual_mm = [red_box[0] * metadata["width_mm"] / image.width,
                     red_box[1] * metadata["height_mm"] / image.height,
                     red_box[2] * metadata["width_mm"] / image.width,
                     red_box[3] * metadata["height_mm"] / image.height]
        assert actual_mm == pytest.approx(expected_mm, abs=.2)
        for channel in (r, g, b, red):
            channel.close()
    assert not list(image_path.parent.glob("*.tmp-*"))


def test_inherited_boxes_and_rotation(tmp_path):
    source = tmp_path / "inherited.pdf"
    make_pdf(source, unit=2, rotation=90, inherited=True)
    info = inspect_pdf(source)[0]
    assert info["dimension_basis"]["effective_box_canvas"] == [100, 200, 500, 700]
    assert info["width_mm"] == pytest.approx(500 * 2 * 25.4 / 72)
    assert info["source_rotation"] == 90


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("crop,effective", [
    ((150, 200, 550, 700), (150, 200, 550, 700)),
    ((-50, 50, 450, 600), (50, 100, 450, 600)),
    ((250, 500, 750, 1000), (250, 500, 650, 900)),
], ids=["inside", "outside-left-bottom", "outside-right-top"])
def test_nonzero_media_origin_and_cropped_visible_intersection(tmp_path, rotation, crop, effective):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    media = (50, 100, 650, 900)
    page.mediabox = RectangleObject(media)
    page.cropbox = RectangleObject(crop)
    page[NameObject("/Rotate")] = NumberObject(rotation)
    page[NameObject("/UserUnit")] = FloatObject(2)
    left, bottom, right, top = effective
    mark_x, mark_y = left + 50, bottom + 50
    stream = DecodedStreamObject()
    stream.set_data(f"1 0 0 rg {mark_x} {mark_y} 30 20 re f\n".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    source, output = tmp_path / "nonzero.pdf", tmp_path / "page.png"
    writer.write(source)
    metadata = inspect_pdf(source)[0]
    assert render_page(source, 1, output) == metadata
    assert metadata["dimension_basis"]["media_box_canvas"] == list(media)
    assert metadata["dimension_basis"]["crop_box_canvas"] == list(crop)
    assert metadata["dimension_basis"]["effective_box_canvas"] == list(effective)
    width, height = right - left, top - bottom
    if rotation in (90, 270):
        width, height = height, width
    factor = 2 * 25.4 / 72
    assert metadata["width_mm"] == pytest.approx(width * factor)
    assert metadata["height_mm"] == pytest.approx(height * factor)
    corners = [(mark_x, mark_y), (mark_x + 30, mark_y),
               (mark_x + 30, mark_y + 20), (mark_x, mark_y + 20)]
    def visual(x, y):
        if rotation == 0:
            return x - left, top - y
        if rotation == 90:
            return y - bottom, x - left
        if rotation == 180:
            return right - x, y - bottom
        return top - y, right - x
    displayed = [visual(x, y) for x, y in corners]
    expected_mm = [min(x for x, y in displayed) * factor,
                   min(y for x, y in displayed) * factor,
                   max(x for x, y in displayed) * factor,
                   max(y for x, y in displayed) * factor]
    with Image.open(output) as image:
        assert image.size == (metadata["width_px"], metadata["height_px"])
        r, g, b = image.split()
        with ImageChops.subtract(r, g) as difference:
            red = difference.point(lambda v: 255 if v > 100 else 0)
        bounds = red.getbbox()
        assert bounds is not None
        actual_mm = [bounds[0] * metadata["width_mm"] / image.width,
                     bounds[1] * metadata["height_mm"] / image.height,
                     bounds[2] * metadata["width_mm"] / image.width,
                     bounds[3] * metadata["height_mm"] / image.height]
        assert actual_mm == pytest.approx(expected_mm, abs=.2)
        for channel in (r, g, b, red):
            channel.close()


@pytest.mark.parametrize("crop", [(700, 100, 800, 200), (650, 100, 750, 200)],
                         ids=["disjoint", "boundary-only"])
def test_empty_media_crop_intersection_is_rejected(tmp_path, crop):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    page.mediabox = RectangleObject((50, 100, 650, 900))
    page.cropbox = RectangleObject(crop)
    source, output = tmp_path / "empty-visible.pdf", tmp_path / "page.png"
    writer.write(source)
    with pytest.raises(ValueError, match="intersection"):
        inspect_pdf(source)
    with pytest.raises(ValueError, match="intersection"):
        render_page(source, 1, output)
    assert not output.exists()


def test_parent_userunit_is_not_independently_inherited(tmp_path):
    writer = PdfWriter()
    page = writer.add_blank_page(width=144, height=216)
    parent = page["/Parent"].get_object()
    parent[NameObject("/UserUnit")] = FloatObject(3)
    assert "/UserUnit" not in page
    source, output = tmp_path / "parent-userunit.pdf", tmp_path / "page.png"
    writer.write(source)
    # The PageObject property is the authority; UserUnit is not one of the
    # inheritable page-tree entries resolved by pypdf.
    with PdfReader(source) as reader:
        assert reader.pages[0].user_unit == 1
        assert reader.pages[0]["/Parent"].get_object()["/UserUnit"] == 3
    metadata = render_page(source, 1, output)
    assert metadata["dimension_basis"]["user_unit"] == 1
    assert metadata["width_mm"] == pytest.approx(50.8)
    assert metadata["height_mm"] == pytest.approx(76.2)
    assert (metadata["width_px"], metadata["height_px"]) == (
        math.ceil(144 * (300 / 72)), math.ceil(216 * (300 / 72)))


def test_encrypted_pdf_is_rejected_even_with_empty_password(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("")
    path = tmp_path / "protected.pdf"
    writer.write(path)
    with pytest.raises(ValueError, match="暗号化"):
        inspect_pdf(path)


def test_invalid_userunit_and_page_number_are_rejected(tmp_path):
    path = tmp_path / "invalid.pdf"
    make_pdf(path, unit=0)
    with pytest.raises(ValueError, match="UserUnit"):
        inspect_pdf(path)
    make_pdf(path)
    with pytest.raises(ValueError, match="Page number"):
        render_page(path, 2, tmp_path / "page.png")
    with pytest.raises(ValueError, match="source PDF"):
        render_page(path, 1, path)


def test_fractional_canvas_and_render_rounding(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=792, height=576)
    path = tmp_path / "letter.pdf"
    writer.write(path)
    metadata = render_page(path, 1, tmp_path / "page.png")
    assert metadata["width_px"] == math.ceil(792 * (300 / 72))
    assert metadata["height_px"] == math.ceil(576 * (300 / 72))
    assert metadata["width_mm"] == pytest.approx(279.4)


def test_tile_boxes_cover_large_page_edges_and_overlap():
    boxes = tile_boxes(5001, 4001)
    assert boxes[0] == (0, 0, 2048, 2048)
    assert boxes[-1] == (2953, 1953, 5001, 4001)
    xs, ys = sorted({b[0] for b in boxes}), sorted({b[1] for b in boxes})
    assert all(b - a <= 2048 - 128 for a, b in zip(xs, xs[1:]))
    assert all(b - a <= 2048 - 128 for a, b in zip(ys, ys[1:]))
    assert all(0 <= l < r <= 5001 and 0 <= t < b <= 4001 for l, t, r, b in boxes)


class FakeSession:
    signature = "test-only"
    def __init__(self, duplicate=False, malformed=False):
        self.calls = 0
        self.duplicate = duplicate
        self.malformed = malformed

    def recognize(self, path, cache):
        assert Path(path).is_file()
        x = 100 if self.calls == 0 else (48 if self.duplicate else 100)
        self.calls += 1
        points = [[x, 2], [x + 20, 2], [x + 20, 12], [x, 12]]
        if self.malformed:
            points[1][0] = 9999
        return {"res": {"rec_texts": [" 0007 ", "   "], "rec_scores": [.9, .5],
                        "rec_polys": [points, [[1, 1], [2, 1], [2, 2], [1, 2]]]}}


def metadata(number=1):
    return {"number": number, "width_px": 2100, "height_px": 20,
            "width_mm": 210, "height_mm": 4, "rotation": 0}


def test_full_page_offsets_mm_original_text_and_deterministic_ids(tmp_path):
    image = tmp_path / "page.png"
    with Image.new("RGB", (2100, 20), "white") as pixels:
        pixels.save(image)
    session = FakeSession()
    regions, raw = recognize_page(image, metadata(3), tmp_path / "unused-models", tmp_path / "scratch", session=session)
    assert session.calls == 2
    assert [r["id"] for r in regions] == ["p3:r000001", "p3:r000002"]
    assert [r["text"] for r in regions] == [" 0007 ", " 0007 "]
    assert [r["bbox_px"]["x"] for r in regions] == [100, 152]
    assert regions[0]["bbox_mm"] == {"x": 10, "y": .4, "width": 2, "height": 2}
    assert regions[1]["polygon_mm"][0] == pytest.approx([15.2, .4])
    assert len(raw["tiles"]) == 2
    assert not list((tmp_path / "scratch").glob("tiles-*"))


def test_repeated_tile_observation_is_removed_but_raw_is_kept(tmp_path):
    image = tmp_path / "page.png"
    with Image.new("RGB", (2100, 20), "white") as pixels:
        pixels.save(image)
    regions, raw = recognize_page(image, metadata(), tmp_path / "unused-models", tmp_path / "scratch", session=FakeSession(duplicate=True))
    assert len(regions) == 1
    assert len(raw["observations_px"]) == 2
    assert len(raw["tiles"]) == 2


def test_out_of_tile_polygon_is_rejected_and_scratch_is_cleaned(tmp_path):
    image = tmp_path / "page.png"
    with Image.new("RGB", (2100, 20), "white") as pixels:
        pixels.save(image)
    with pytest.raises(ValueError, match="source tile"):
        recognize_page(image, metadata(), tmp_path / "unused-models", tmp_path / "scratch", session=FakeSession(malformed=True))
    assert not list((tmp_path / "scratch").glob("tiles-*"))

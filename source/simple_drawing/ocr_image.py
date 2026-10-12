"""Render saved OCR strings on a white visual page without running inference.

Only the saved visual-page geometry and original strings are used. A quad's
first edge supplies the text direction; no font or pixels are copied from the
original document. The selected Windows font remains a system dependency.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import uuid

from PIL import Image, ImageDraw, ImageFont

RENDERER_VERSION = 1
MAX_IMAGE_PIXELS = 600_000_000
MAX_PATCH_PIXELS = 64_000_000
MAX_REGION_CHARACTERS = 100_000
MAX_FONT_SIZE = 4096
SYSTEM_FONT_NAMES = ("YuGothR.ttc", "meiryo.ttc", "msgothic.ttc")
FALLBACK_FONT_NAMES = ("NotoSansJP-VF.ttf", "msyh.ttc", "msjh.ttc", "simsun.ttc", "malgun.ttf")


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return float(value)


def _font_path(font_path):
    if font_path is not None:
        path = Path(font_path)
        if not path.is_file():
            raise FileNotFoundError(f"OCR画像用フォントが見つかりません: {path}")
        return path.resolve()
    available = _system_font_paths()
    if available:
        return available[0]
    raise FileNotFoundError(
        "OCR画像用のWindowsフォントが見つかりません。游ゴシック、メイリオ、"
        "MS ゴシック、Noto Sans JPなどの対応フォントが必要です。"
    )


def _system_font_paths():
    system_root = os.environ.get("SystemRoot")
    if not system_root:
        return []
    folder = Path(system_root) / "Fonts"
    return [(folder / name).resolve() for name in SYSTEM_FONT_NAMES + FALLBACK_FONT_NAMES
            if (folder / name).is_file()]


def _select_font(font_path, texts):
    """Prefer Japanese fonts; use another installed font if it covers more text.

    An explicit font choice is honored. The ordinary fully-covered Japanese
    page performs one coverage scan, rather than scanning every fallback.
    """
    texts = tuple(texts)
    selected = _font_path(font_path)
    missing = _missing_characters(selected, texts)
    if font_path is not None or not missing:
        return selected, missing
    for alternative in _system_font_paths():
        if alternative == selected:
            continue
        alternative_missing = _missing_characters(alternative, texts)
        if len(alternative_missing) < len(missing):
            selected, missing = alternative, alternative_missing
            if not missing:
                break
    return selected, missing


def _page_geometry(ocr):
    if not isinstance(ocr, dict) or ocr.get("version") != 1:
        raise ValueError("Unsupported saved OCR version")
    page = ocr.get("page")
    if not isinstance(page, dict) or page.get("rotation") != 0:
        raise ValueError("OCR image requires a canonical visual page with rotation zero")
    if type(page.get("number")) is not int or page["number"] < 1:
        raise ValueError("OCR page number must be a positive integer")
    for key in ("width_px", "height_px"):
        if type(page.get(key)) is not int or page[key] <= 0:
            raise ValueError(f"OCR page {key} must be a positive integer")
    width, height = page["width_px"], page["height_px"]
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"OCR page exceeds supported {MAX_IMAGE_PIXELS:,} pixel limit")
    # Pillow accepts signed 32-bit dimensions even for extremely thin pages.
    if max(width, height) > 2_147_483_647:
        raise ValueError("OCR page dimensions exceed the image library limit")
    width_mm = _finite(page.get("width_mm"), "Page width_mm", positive=True)
    height_mm = _finite(page.get("height_mm"), "Page height_mm", positive=True)
    regions = ocr.get("regions")
    if not isinstance(regions, list):
        raise ValueError("Saved OCR regions must be a list")
    return width, height, (width * 25.4 / width_mm, height * 25.4 / height_mm), regions


def _region_geometry(region, width, height):
    if not isinstance(region, dict):
        raise ValueError("OCR region must be an object")
    identifier = region.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("OCR region requires a nonempty ID")
    text = region.get("text")
    if not isinstance(text, str):
        raise ValueError(f"OCR region {identifier} requires an original string")
    if len(text) > MAX_REGION_CHARACTERS:
        raise ValueError(f"OCR region {identifier} exceeds supported character limit")
    box = region.get("bbox_px")
    if not isinstance(box, dict):
        raise ValueError(f"OCR region {identifier} requires bbox_px")
    x, y = (_finite(box.get(key), f"{identifier} bbox {key}") for key in ("x", "y"))
    w, h = (_finite(box.get(key), f"{identifier} bbox {key}", positive=True)
            for key in ("width", "height"))
    if x < 0 or y < 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
        raise ValueError(f"OCR region {identifier} bbox is outside the page")
    polygon = region.get("polygon_px")
    if "polygon_px" not in region:
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    else:
        if not isinstance(polygon, (list, tuple)) or len(polygon) != 4:
            raise ValueError(f"OCR region {identifier} requires a four-corner polygon")
        points = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"OCR region {identifier} polygon requires xy pairs")
            px, py = (_finite(value, f"{identifier} polygon point") for value in point)
            if not (0 <= px <= width + 1e-6 and 0 <= py <= height + 1e-6):
                raise ValueError(f"OCR region {identifier} polygon is outside the page")
            points.append((px, py))
        extremes = (min(p[0] for p in points), min(p[1] for p in points),
                    max(p[0] for p in points), max(p[1] for p in points))
        if any(abs(a - b) > 1e-5 for a, b in zip(extremes, (x, y, x + w, y + h))):
            raise ValueError(f"OCR region {identifier} polygon and bbox disagree")
    crosses = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) -
                       (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(v > 1e-9 for v in crosses) or all(v < -1e-9 for v in crosses)):
        raise ValueError(f"OCR region {identifier} polygon is degenerate or self-intersecting")
    left, top = math.floor(x), math.floor(y)
    right, bottom = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
    if (right - left) * (bottom - top) > MAX_PATCH_PIXELS:
        raise ValueError(f"OCR region {identifier} exceeds supported patch pixel limit")
    edge_width = (math.dist(points[0], points[1]) + math.dist(points[3], points[2])) / 2
    edge_height = (math.dist(points[0], points[3]) + math.dist(points[1], points[2])) / 2
    local_width, local_height = max(1, math.ceil(edge_width)), max(1, math.ceil(edge_height))
    if local_width * local_height > MAX_PATCH_PIXELS:
        raise ValueError(f"OCR region {identifier} exceeds supported text patch pixel limit")
    return identifier, text, points, (left, top, right, bottom), (local_width, local_height)


def _perspective_coefficients(points, width, height):
    """Solve target-to-source homography for Pillow without NumPy/SciPy."""
    equations = []
    for (x, y), (u, v) in zip(points, ((0, 0), (width, 0), (width, height), (0, height))):
        equations.append([x, y, 1, 0, 0, 0, -u * x, -u * y, u])
        equations.append([0, 0, 0, x, y, 1, -v * x, -v * y, v])
    for column in range(8):
        pivot = max(range(column, 8), key=lambda row: abs(equations[row][column]))
        if abs(equations[pivot][column]) <= 1e-12:
            raise ValueError("OCR polygon cannot be projected")
        equations[column], equations[pivot] = equations[pivot], equations[column]
        scale = equations[column][column]
        equations[column] = [value / scale for value in equations[column]]
        for row in range(8):
            if row != column:
                scale = equations[row][column]
                equations[row] = [a - scale * b for a, b in zip(equations[row], equations[column])]
    coefficients = tuple(equations[row][-1] for row in range(8))
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("OCR polygon projection is nonfinite")
    return coefficients


def _text_mask(text, size, font_path, measure, fonts):
    width, height = size
    def geometry(point_size):
        if point_size not in fonts:
            # Keep long batches from retaining every size of the same font.
            if len(fonts) >= 64:
                fonts.clear()
            fonts[point_size] = ImageFont.truetype(str(font_path), point_size)
        font = fonts[point_size]
        spacing = round(point_size * .1)
        bounds = measure.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
        return font, spacing, bounds
    lower, upper = 1, min(MAX_FONT_SIZE, max(1, height * 2))
    best = None
    while lower <= upper:
        middle = (lower + upper) // 2
        candidate = geometry(middle)
        bounds = candidate[2]
        if bounds[2] - bounds[0] <= width and bounds[3] - bounds[1] <= height:
            best = candidate
            lower = middle + 1
        else:
            upper = middle - 1
    compressed = best is None
    font, spacing, bounds = best or geometry(1)
    text_width, text_height = max(1, math.ceil(bounds[2] - bounds[0])), max(1, math.ceil(bounds[3] - bounds[1]))
    if text_width * text_height > MAX_PATCH_PIXELS:
        raise ValueError("OCR original string exceeds supported text-rendering pixel limit")
    with Image.new("L", (text_width, text_height), 0) as ink:
        draw = ImageDraw.Draw(ink)
        # The complete original string is passed unchanged. No ellipsis,
        # stripping, character replacement, confidence filter or line wrapping.
        draw.multiline_text((-bounds[0], -bounds[1]), text, fill=255, font=font, spacing=spacing)
        if compressed:
            result = ink.resize((min(text_width, width), min(text_height, height)), Image.Resampling.LANCZOS)
        else:
            result = ink.copy()
    mask = Image.new("L", size, 0)
    try:
        mask.paste(result, (0, 0))
    finally:
        result.close()
    return mask, compressed, font.size


def _missing_characters(font_path, texts):
    """Report likely unsupported characters rather than substituting them."""
    font = ImageFont.truetype(str(font_path), 24)
    fallback = font.getmask("\U0010ffff")
    missing_mask = (fallback.size, bytes(fallback))
    missing = []
    for character in sorted({character for text in texts for character in text if not character.isspace()}):
        mask = font.getmask(character)
        if (mask.size, bytes(mask)) == missing_mask:
            missing.append(f"U+{ord(character):04X}")
    return missing


def _commit_new_file(temporary, output):
    # Windows rename is an atomic move that rejects an existing destination.
    # POSIX rename would overwrite, so use an exclusive hard-link there.
    if os.name == "nt":
        os.rename(temporary, output)
    else:
        os.link(temporary, output)
        temporary.unlink()


def render_ocr_image(ocr: dict, output_path: Path, font_path=None) -> dict:
    """Save a new white RGB PNG with all saved OCR text at its recorded quad.

    Regions with empty/whitespace-only strings have no visible ink and are
    counted separately. Long strings are reduced to fit; extreme inputs fail
    explicitly. Existing files are never replaced, including a racing writer.
    """
    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("OCR image output must be a PNG file")
    if os.path.lexists(output):
        raise FileExistsError(f"OCR画像の出力先は既に存在します: {output}")
    width, height, dpi, regions = _page_geometry(ocr)
    prepared = [_region_geometry(region, width, height) for region in regions]
    identifiers = [region[0] for region in prepared]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Saved OCR region IDs are duplicated")
    selected_font, missing = _select_font(font_path, (row[1] for row in prepared))
    probe_font = ImageFont.truetype(str(selected_font), 24)
    name, style = probe_font.getname()
    warnings = []
    if missing:
        warnings.append({"code": "font_missing_characters", "characters": missing,
                         "message": "選択フォントにない文字は欠字記号で表示される可能性があります。"})
    placed, empty = 0, 0
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    fonts = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.new("RGB", (width, height), "white") as canvas, Image.new("L", (1, 1)) as sample:
            measure = ImageDraw.Draw(sample)
            for identifier, text, points, bounds, local_size in prepared:
                if not text or text.isspace():
                    empty += 1
                    continue
                mask, compressed, font_size = _text_mask(text, local_size, selected_font, measure, fonts)
                try:
                    left, top, right, bottom = bounds
                    local_points = [(x - left, y - top) for x, y in points]
                    coefficients = _perspective_coefficients(local_points, *local_size)
                    with mask.transform((right - left, bottom - top), Image.Transform.PERSPECTIVE,
                                        coefficients, resample=Image.Resampling.BICUBIC, fillcolor=0) as projected:
                        canvas.paste((0, 0, 0), bounds, projected)
                finally:
                    mask.close()
                if compressed or font_size < 4:
                    warnings.append({"code": "small_text", "region_id": identifier,
                                     "message": "全文を領域内に収めましたが、小さく判読しにくい可能性があります。"})
                placed += 1
            canvas.save(temporary, format="PNG", dpi=dpi)
        _commit_new_file(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"renderer_version": RENDERER_VERSION, "placed_regions": placed, "empty_regions": empty,
            "font": {"path": str(selected_font), "name": name, "style": style},
            "width_px": width, "height_px": height, "dpi": list(dpi), "warnings": warnings}

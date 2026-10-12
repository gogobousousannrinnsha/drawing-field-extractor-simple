"""Small authored connection checks; these do not certify drawing accuracy."""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROBE_VERSION = "authored-image-ocr-v1"


def create_probe(folder: Path) -> tuple[Path, dict, list[dict], list[dict]]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=36)
    lines = [(80, 70, "MODEL CONNECTION CHECK"),
             (80, 220, "PART NUMBER"), (520, 220, "001-020"),
             (80, 370, "MATERIAL"), (520, 370, "STEEL")]
    regions = []
    for index, (x, y, value) in enumerate(lines, 1):
        draw.text((x, y), value, font=font, fill="black")
        left, top, right, bottom = draw.textbbox((x, y), value, font=font)
        regions.append({"id": f"p1:r{index:06d}", "text": value,
                        "bbox_mm": {"x": left / 5, "y": top / 5,
                                    "width": (right-left) / 5,
                                    "height": (bottom-top) / 5}})
    path = folder / "page.png"
    image.save(path)
    page = {"number": 1, "width_mm": 200.0, "height_mm": 140.0,
            "width_px": 1000, "height_px": 700,
            "dimension_basis": "authored model connection check"}
    cases = [
        {"name": "value-and-evidence", "settings": {
            "item": "Part number", "region_hint_mm": None, "take": "one", "instruction": ""},
         "expected": {"status": "found", "results": [
             {"value": "001-020", "source_ids": ["p1:r000003"]}]}},
        {"name": "missing-item", "settings": {
            "item": "Mass", "region_hint_mm": None, "take": "one",
            "instruction": "Mass/weight is not the part number or material. Return not_found if absent."},
         "expected": {"status": "not_found", "results": []}},
    ]
    return path, page, regions, cases

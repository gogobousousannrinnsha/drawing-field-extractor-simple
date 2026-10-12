"""Run an authored real-OCR check without changing the user's input or settings."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time


def check_ocr(root):
    for key, value in {"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True", "PADDLE_HOME": str(root/"temp/paddle"),
                       "PADDLE_PDX_CACHE_HOME": str(root/"temp/paddlex"), "HF_HOME": str(root/"temp/huggingface"),
                       "TMP": str(root/"temp"), "TEMP": str(root/"temp")}.items():
        os.environ[key] = value
    sys.path.insert(0, str(root/"source"))
    from PIL import Image, ImageDraw, ImageFont
    from simple_drawing.pdf_ocr import inspect_pdf, render_page, recognize_page
    from simple_drawing.storage import write_json
    folder = root/"data/setup-checks"/("real-ocr-"+str(time.time_ns()))
    folder.mkdir(parents=True)
    original = Image.new("RGB", (1240,1754), "white")
    draw = ImageDraw.Draw(original)
    font = ImageFont.load_default(size=48)
    for xy, value in [((90,120),"ONLINE SETUP CHECK"),((90,400),"PART NUMBER"),((650,400),"000777-C"),
                      ((90,650),"MATERIAL"),((650,650),"STEEL")]:
        draw.text(xy, value, fill="black", font=font)
    pdf = folder/"authored-check.pdf"
    original.save(pdf, "PDF", resolution=150)
    original.close()
    pages = inspect_pdf(pdf)
    if len(pages) != 1:
        raise RuntimeError("試料のページ数が一致しません")
    page = render_page(pdf, 1, folder/"page.png")
    regions, raw = recognize_page(folder/"page.png", page, root/"models/ocr", folder/"scratch")
    write_json(folder/"raw.json", raw)
    write_json(folder/"ocr.json", {"version":1, "page":page, "regions":regions})
    texts = [region["text"] for region in regions]
    if not any("000777-C" in text for text in texts):
        raise RuntimeError("実OCR試料の部品番号を確認できません。記録を確認してください")
    report = {"status":"passed", "kind":"authored-real-ocr", "regions":len(regions),
              "expected_value":"000777-C", "record":folder.relative_to(root).as_posix()}
    write_json(root/"data/setup-real-ocr.json", report)
    print("実OCR確認に合格しました（自作試料1ページ）。", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ocr-only", action="store_true", required=True)
    parser.add_argument("--lock-held", action="store_true", help="初期設定の子ワーカー専用")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.lock_held:
        return check_ocr(root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bootstrap import operation_lock
    with operation_lock(root):
        return check_ocr(root)


if __name__ == "__main__":
    raise SystemExit(main())

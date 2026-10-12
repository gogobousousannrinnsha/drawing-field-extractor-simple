"""PDF visual-page geometry and a small, process-local tiled OCR adapter.

Coordinates always refer to the saved, displayed page PNG. PDF /Rotate is
applied by PDFium once; /UserUnit is read separately with pypdf. Paddle is
loaded lazily, and importing this module or inspecting PDFs never loads it.
The contrast crop hook and tile seam policy originate from the verified v0.2
adapter; no existing app, installed dependency, or model file is modified.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import tempfile
import uuid

RENDER_DPI = 300
MAX_PAGES = 1000
MAX_IMAGE_PIXELS = 600_000_000
OCR_PROFILE = {
    "adapter_version": 1, "tile_size": 2048, "overlap": 128,
    "orientations": [0], "dedup_version": 1,
    "models": ["PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"],
    "paddleocr_version": "3.7.0", "paddlex_version": "3.7.2",
    "paddlepaddle_gpu_version": "3.2.2", "pillow_version": "12.3.0",
    "recognizer_preprocessing": "quad_contrast2_v1", "contrast_factor": 2.0,
    "text_recognition_batch_size": 1, "text_rec_score_thresh": 0.0,
    "use_doc_orientation_classify": False, "use_doc_unwarping": False,
    "use_textline_orientation": False, "device": "gpu:0",
}


def _finite(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return value


def _box(value, name):
    if len(value) != 4:
        raise ValueError(f"{name} must have four coordinates")
    result = [_finite(v, name) for v in value]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{name} has no positive area")
    return result


@contextmanager
def _pdf(path):
    import pypdfium2 as pdfium
    from pypdf import PdfReader
    path = Path(path)
    reader = PdfReader(path)
    try:
        if reader.is_encrypted:
            raise ValueError("暗号化・パスワード保護されたPDFには対応していません。")
        if not 1 <= len(reader.pages) <= MAX_PAGES:
            raise ValueError(f"PDF must contain 1 to {MAX_PAGES} pages")
        with pdfium.PdfDocument(path) as document:
            if len(document) != len(reader.pages):
                raise ValueError("PDF parsers disagree on the page count")
            yield reader, document
    finally:
        reader.close()


def _metadata(reader, document, number):
    import pypdfium2 as pdfium
    source = reader.pages[number - 1]
    media = _box(source.mediabox, "MediaBox")
    crop = _box(source.cropbox, "CropBox")
    effective = [max(media[0], crop[0]), max(media[1], crop[1]),
                 min(media[2], crop[2]), min(media[3], crop[3])]
    _box(effective, "MediaBox/CropBox intersection")
    unit = _finite(source.user_unit, "UserUnit", positive=True)
    raw_rotation = _finite(source.get("/Rotate", 0), "Rotate")
    if raw_rotation != int(raw_rotation) or int(raw_rotation) % 90:
        raise ValueError("PDF Rotate must be a multiple of 90 degrees")
    rotation = int(raw_rotation) % 360
    width, height = effective[2] - effective[0], effective[3] - effective[1]
    if rotation in (90, 270):
        width, height = height, width
    page = document[number - 1]
    try:
        canvas_width, canvas_height = page.get_size()
        if page.get_rotation() != rotation:
            raise ValueError("PDF parsers disagree on page rotation")
        if any(abs(a - b) > 0.001 for a, b in zip(page.get_bbox(), effective)):
            raise ValueError("PDF parsers disagree on the visible page box")
        if any(abs(a - b) > 0.001 for a, b in zip((canvas_width, canvas_height), (width, height))):
            raise ValueError("PDF parsers disagree on the visible page size")
    finally:
        page.close()
    scale = RENDER_DPI * unit / 72
    width_px, height_px = math.ceil(canvas_width * scale), math.ceil(canvas_height * scale)
    if min(width_px, height_px) <= 0 or width_px * height_px > MAX_IMAGE_PIXELS:
        raise ValueError("PDF page exceeds supported raster dimensions")
    return {
        "number": number, "width_mm": width * unit * 25.4 / 72,
        "height_mm": height * unit * 25.4 / 72,
        "width_px": width_px, "height_px": height_px,
        "source_rotation": rotation, "rotation": 0,
        "dimension_basis": {
            "kind": "pdf_visible_page", "coordinate_system": "top-left-x-right-y-down",
            "media_box_canvas": media, "crop_box_canvas": crop,
            "effective_box_canvas": effective, "user_unit": unit,
            "source_rotation": rotation, "render_dpi": RENDER_DPI,
            "render_scale": scale, "pypdf_version": version("pypdf"),
            "pypdfium2_version": version("pypdfium2"),
            "pdfium_version": str(pdfium.PDFIUM_INFO),
        },
    }


def inspect_pdf(path) -> list[dict]:
    """Read PDF visual-page geometry without rasterizing or loading OCR."""
    with _pdf(path) as (reader, document):
        return [_metadata(reader, document, i) for i in range(1, len(document) + 1)]


def render_page(path, number, image_path) -> dict:
    """Atomically save one uncommitted page PNG, including intrinsic rotation."""
    if type(number) is not int or number < 1:
        raise ValueError("Page number must be a positive integer")
    image_path = Path(image_path)
    if image_path.resolve() == Path(path).resolve():
        raise ValueError("The page image must not replace the source PDF")
    with _pdf(path) as (reader, document):
        if number > len(document):
            raise ValueError("Page number is outside the PDF")
        metadata = _metadata(reader, document, number)
        page = document[number - 1]
        bitmap = None
        temporary = image_path.with_name(image_path.name + ".tmp-" + uuid.uuid4().hex)
        try:
            bitmap = page.render(scale=metadata["dimension_basis"]["render_scale"],
                                 rotation=0, draw_annots=True, rev_byteorder=True)
            with bitmap.to_pil() as view, view.convert("RGB") as image:
                if image.size != (metadata["width_px"], metadata["height_px"]):
                    raise ValueError("Rendered image dimensions disagree with PDF geometry")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(temporary, format="PNG")
            os.replace(temporary, image_path)
        finally:
            temporary.unlink(missing_ok=True)
            if bitmap is not None:
                bitmap.close()
            page.close()
    return metadata


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_signature(model_root):
    """Verify the two fixed model assets and return a producer signature."""
    root = Path(model_root).resolve()
    manifest = root / "model-manifest.json"
    models = json.loads(manifest.read_text(encoding="utf-8"))
    if {model["name"] for model in models} != set(OCR_PROFILE["models"]):
        raise ValueError("The fixed OCR model manifest has unexpected models")
    for model in models:
        for name, expected in model["files"].items():
            path = (root / name).resolve()
            path.relative_to(root)
            if file_sha256(path) != expected:
                raise ValueError(f"OCR model file SHA-256 mismatch: {name}")
    payload = {"manifest_sha256": file_sha256(manifest), "profile": OCR_PROFILE}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def _native_model_paths(inference, root):
    """Adapt only the two local model filenames to Windows' narrow native API."""
    if os.name != "nt" or str(root).isascii():
        yield
        return
    allowed = {root / name / "inference.json": root / name / "inference.pdiparams"
               for name in OCR_PROFILE["models"]}
    original = inference.Config
    def config(*args, **kwargs):
        if len(args) == 2 and not kwargs:
            model, params = (Path(arg).resolve() for arg in args)
            if allowed.get(model) == params:
                return original(str(model).encode("mbcs"), str(params).encode("mbcs"))
        return original(*args, **kwargs)
    inference.Config = config
    try:
        yield
    finally:
        inference.Config = original


class OCRSession:
    """One GPU engine shared by pages in one worker; never a global singleton."""
    def __init__(self, model_root):
        self.model_root = Path(model_root).resolve()
        self.signature = model_signature(self.model_root)
        self.engine = None
        self._crop_trace = []

    def _initialize(self):
        for package, expected in (("paddleocr", "3.7.0"), ("paddlex", "3.7.2"),
                                  ("Pillow", "12.3.0"), ("paddlepaddle-gpu", "3.2.2")):
            if version(package) != expected:
                raise RuntimeError(f"Fixed {package} {expected} is required")
        import paddle
        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
            raise RuntimeError("CUDA GPU is required; CPU fallback is disabled")
        paddle.set_device("gpu:0")
        from paddleocr import PaddleOCR
        with _native_model_paths(paddle.inference, self.model_root):
            engine = PaddleOCR(
                text_detection_model_name=OCR_PROFILE["models"][0],
                text_recognition_model_name=OCR_PROFILE["models"][1],
                text_detection_model_dir=str(self.model_root / OCR_PROFILE["models"][0]),
                text_recognition_model_dir=str(self.model_root / OCR_PROFILE["models"][1]),
                device="gpu:0", use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False,
                text_rec_score_thresh=0.0, text_recognition_batch_size=1,
                enable_mkldnn=False,
            )
        pipeline = engine.paddlex_pipeline._pipeline
        if pipeline.text_rec_model.batch_sampler.batch_size != 1:
            raise ValueError("Contrast crop preprocessing requires recognizer batch size one")
        cropper = pipeline._crop_by_polys
        trace = self._crop_trace
        class ContrastCropper:
            def __call__(self, pixels, polygons):
                import numpy as np
                from PIL import Image, ImageEnhance
                for ordinal, crop in enumerate(cropper(pixels, polygons)):
                    processed = crop
                    if crop.size:
                        with Image.fromarray(crop[:, :, ::-1]) as image:
                            with ImageEnhance.Contrast(image).enhance(2.0) as enhanced:
                                processed = np.array(enhanced)[:, :, ::-1].copy()
                    trace.append({"ordinal": ordinal, "shape": list(crop.shape),
                                  "processing_applied": bool(crop.size),
                                  "original_bgr_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                                  "processed_bgr_sha256": hashlib.sha256(processed.tobytes()).hexdigest()})
                    yield processed
        pipeline._crop_by_polys = ContrastCropper()
        # Publish the engine only after every required hook is installed. A
        # failed initialization cannot leave a reusable, unprocessed engine.
        self.engine = engine

    def recognize(self, image_path, cache_root):
        from PIL import Image
        import numpy as np
        cache_root = Path(cache_root).resolve()
        profile = cache_root / "profile"
        profile.mkdir(parents=True, exist_ok=True)
        scoped = {"USERPROFILE": str(profile), "PADDLE_PDX_CACHE_HOME": str(cache_root / "paddlex"),
                  "PADDLE_HOME": str(cache_root / "paddle"), "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"}
        previous = {key: os.environ.get(key) for key in scoped}
        try:
            os.environ.update(scoped)
            with redirect_stdout(sys.stderr):
                if self.engine is None:
                    self._initialize()
                with Image.open(image_path) as image:
                    pixels = np.array(image.convert("RGB"))[:, :, ::-1].copy()
                self._crop_trace.clear()
                results = list(self.engine.predict(pixels))
                if len(results) != 1:
                    raise ValueError("Expected one OCR tile result")
                payload = results[0].json
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return {**payload, "recognizer_preprocessing": {
                    "producer_signature": self.signature, "config": dict(OCR_PROFILE),
                    "crop_inputs": list(self._crop_trace)}}
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def close(self):
        if self.engine is not None:
            closer = getattr(self.engine, "close", None)
            if closer:
                closer()
            self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def tile_boxes(width, height):
    size, overlap = OCR_PROFILE["tile_size"], OCR_PROFILE["overlap"]
    def starts(length):
        if length <= size:
            return [0]
        values = list(range(0, length - size + 1, size - overlap))
        if values[-1] != length - size:
            values.append(length - size)
        return values
    return [(x, y, min(x + size, width), min(y + size, height))
            for y in starts(height) for x in starts(width)]


def _bbox(points):
    xs, ys = zip(*points)
    return {"x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys)}


def _intersection(a, b):
    w = max(0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    h = max(0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    return w * h


def _deduplicate(rows):
    selected = []
    for row in sorted(rows, key=lambda r: (-r["confidence"], r["bbox_px"]["y"], r["bbox_px"]["x"], r["text"])):
        box = row["bbox_px"]
        area = box["width"] * box["height"]
        if row["edge_clipped"] and any(not other["edge_clipped"] and
                _intersection(box, other["bbox_px"]) / area >= .8 for other in rows):
            continue
        duplicate = False
        for old in selected:
            other = old["bbox_px"]
            overlap = _intersection(box, other)
            union = area + other["width"] * other["height"] - overlap
            # The seam comparison follows v0.2; the winning original text is
            # retained unchanged, and every observation also remains in Raw.
            if row["text"].strip() == old["text"].strip() and union and overlap / union >= .5:
                duplicate = True
                break
        if not duplicate:
            selected.append(row)
    return sorted(selected, key=lambda r: (r["bbox_px"]["y"], r["bbox_px"]["x"], r["text"]))


def _tile_rows(payload, box, page_width, page_height):
    data = payload.get("res", payload)
    texts, scores, polygons = (data[key] for key in ("rec_texts", "rec_scores", "rec_polys"))
    if not len(texts) == len(scores) == len(polygons):
        raise ValueError("OCR text/score/polygon lengths differ")
    rows = []
    for text, score, polygon in zip(texts, scores, polygons):
        if not isinstance(text, str) or len(polygon) != 4:
            raise ValueError("OCR requires text and a four-corner polygon")
        confidence = _finite(score, "OCR confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("OCR confidence must be between zero and one")
        points = []
        for point in polygon:
            if len(point) != 2:
                raise ValueError("OCR polygon requires xy pairs")
            x, y = (_finite(v, "OCR point") for v in point)
            if not (-1e-6 <= x <= box[2] - box[0] + 1e-6 and -1e-6 <= y <= box[3] - box[1] + 1e-6):
                raise ValueError("OCR polygon is outside its source tile")
            points.append([min(box[2], max(box[0], x + box[0])), min(box[3], max(box[1], y + box[1]))])
        crosses = []
        for i in range(4):
            a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
            crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
            raise ValueError("OCR polygon is degenerate or self-intersecting")
        if not text.strip():
            continue
        bounds = _bbox(points)
        clipped = ((box[0] > 0 and bounds["x"] - box[0] <= 3)
                   or (box[1] > 0 and bounds["y"] - box[1] <= 3)
                   or (box[2] < page_width and bounds["x"] + bounds["width"] >= box[2] - 3)
                   or (box[3] < page_height and bounds["y"] + bounds["height"] >= box[3] - 3))
        rows.append({"text": text, "confidence": confidence, "polygon_px": points,
                     "bbox_px": bounds, "tile_px": list(box), "edge_clipped": bool(clipped)})
    return rows


def recognize_page(image_path, page, model_root, scratch, *, session=None):
    """Return full-page OCR regions with exact original strings and px/mm evidence."""
    from PIL import Image
    if type(page.get("number")) is not int or page["number"] < 1 or page.get("rotation") != 0:
        raise ValueError("OCR requires a canonical visual page")
    if any(type(page.get(key)) is not int or page[key] <= 0 for key in ("width_px", "height_px")):
        raise ValueError("OCR requires positive integer page pixel dimensions")
    sx = _finite(page["width_mm"], "Page width", positive=True) / page["width_px"]
    sy = _finite(page["height_mm"], "Page height", positive=True) / page["height_px"]
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    own_session = session is None
    if own_session:
        session = OCRSession(model_root)
    elif hasattr(session, "model_root") and Path(model_root).resolve() != session.model_root:
        raise ValueError("OCR session belongs to another model directory")
    raw_tiles, rows = [], []
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(image_path) as opened:
            if opened.size != (page["width_px"], page["height_px"]):
                raise ValueError("OCR image dimensions disagree with page metadata")
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError("OCR image exceeds supported raster dimensions")
            image = opened if opened.mode == "RGB" else opened.convert("RGB")
            try:
                with tempfile.TemporaryDirectory(prefix="tiles-", dir=scratch) as temporary:
                    for ordinal, box in enumerate(tile_boxes(*image.size)):
                        path = Path(temporary) / f"tile-{ordinal:04d}.png"
                        with image.crop(box) as tile:
                            tile.save(path)
                        payload = session.recognize(path, scratch / "ocr-cache")
                        raw_tiles.append({"box_px": list(box), "result": payload})
                        rows.extend(_tile_rows(payload, box, *image.size))
                        path.unlink()
            finally:
                if image is not opened:
                    image.close()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
        if own_session:
            session.close()
    regions = []
    for ordinal, row in enumerate(_deduplicate(rows), 1):
        bounds = row["bbox_px"]
        regions.append({"id": f"p{page['number']}:r{ordinal:06d}", "text": row["text"],
                        "confidence": row["confidence"], "bbox_px": bounds,
                        "bbox_mm": {key: value * (sx if key in ("x", "width") else sy)
                                    for key, value in bounds.items()},
                        "polygon_px": row["polygon_px"],
                        "polygon_mm": [[x * sx, y * sy] for x, y in row["polygon_px"]]})
    return regions, {"schema": "simple-tiled-ocr-v1", "profile": dict(OCR_PROFILE),
                     "producer_signature": getattr(session, "signature", "injected-test-session"),
                     "tiles": raw_tiles, "observations_px": rows}

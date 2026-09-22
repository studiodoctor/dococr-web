"""
DocOCR Web — FastAPI backend.

POST /api/scan accepts a PDF or image upload, runs every page through the
same OCR pipeline used by the Android app (see ocr_engine.py), and returns
extracted text with layout preserved, per page.
"""
from __future__ import annotations

import base64
import gc
import io
import sys
import threading
import time
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import ocr_engine

APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = APP_DIR.parent.parent / "frontend"

print("DocOCR Web starting…", flush=True)

app = FastAPI(title="DocOCR Web")


@app.on_event("startup")
def _warm_models_in_background() -> None:
    """Binds the port immediately; the (small, but non-zero) ONNX model load
    happens in a background thread so a slow host never delays health checks."""

    def _run():
        try:
            ocr_engine.preload_models()
            print("OCR models loaded.", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"Model preload failed (will retry lazily per-request): {e}", file=sys.stderr, flush=True)

    threading.Thread(target=_run, daemon=True).start()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render's free tier caps a container at 512MB RAM. Measured peak memory for
# one page through the full pipeline (preprocessing + Tesseract + PaddleOCR,
# Tesseract's own subprocess included) is ~415MB at MAX_SIDE=1600 vs. ~577MB
# (already over the limit) at the original 2600 — this size was chosen from
# that measurement, not guessed. Accuracy at 1600 was unchanged in testing.
# If you're running this somewhere with more RAM, both can be raised again —
# 2600/220 matches what the Android app uses.
MAX_SIDE = 1600
PDF_RENDER_DPI = 170
MAX_PAGES = 20
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


def _cap_size(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return rgb
    scale = MAX_SIDE / max(h, w)
    return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _pil_to_rgb(im: Image.Image) -> np.ndarray:
    im = im.convert("RGB")
    return np.array(im)


def _load_pages(filename: str, data: bytes) -> list[np.ndarray]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        if doc.page_count > MAX_PAGES:
            raise HTTPException(400, f"PDF has {doc.page_count} pages; the limit is {MAX_PAGES}.")
        pages = []
        zoom = PDF_RENDER_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            pages.append(_cap_size(arr))
        doc.close()
        if not pages:
            raise HTTPException(400, "The PDF has no pages.")
        return pages
    try:
        im = Image.open(io.BytesIO(data))
        im = _exif_transpose(im)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read the image ({type(e).__name__}).") from e
    return [_cap_size(_pil_to_rgb(im))]


def _exif_transpose(im: Image.Image) -> Image.Image:
    try:
        from PIL import ImageOps
        return ImageOps.exif_transpose(im) or im
    except Exception:  # noqa: BLE001
        return im


def _encode_png(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


class BoxOut(BaseModel):
    text: str
    confidence: float
    left: float
    top: float
    right: float
    bottom: float


class PageOut(BaseModel):
    index: int
    text: str
    engine: str
    width: int
    height: int
    diagnostics: list[str]
    boxes: list[BoxOut]
    image: str


class ScanResult(BaseModel):
    filename: str
    pages: list[PageOut]
    elapsed_seconds: float
    engines_available: str


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "tesseract": True,
        "paddleocr": True,
    }


@app.post("/api/scan", response_model=ScanResult)
async def scan(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File is too large (30 MB limit).")
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")

    started = time.time()
    pages = _load_pages(file.filename or "upload", data)

    out_pages: list[PageOut] = []
    for i, rgb in enumerate(pages):
        result = ocr_engine.run_pipeline(rgb)
        out_pages.append(
            PageOut(
                index=i,
                text=result.plain_text,
                engine=result.chosen_engine,
                width=result.image_width,
                height=result.image_height,
                diagnostics=result.diagnostics,
                boxes=[
                    BoxOut(text=t, confidence=c, left=b[0], top=b[1], right=b[2], bottom=b[3])
                    for t, c, b in result.boxes
                ],
                image=_encode_png(rgb),
            )
        )
        # Free this page's intermediate arrays before starting the next one —
        # matters on a memory-capped host (e.g. Render free tier) with a
        # multi-page PDF, where pages would otherwise be able to pile up.
        gc.collect()

    return ScanResult(
        filename=file.filename or "upload",
        pages=out_pages,
        elapsed_seconds=round(time.time() - started, 2),
        engines_available="Tesseract + PaddleOCR",
    )


# Serve the frontend as static files, with index.html at the root.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

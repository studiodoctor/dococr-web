"""
DocOCR Web — FastAPI backend.

Large documents (many pages, or just slow on a constrained host) are handled
as a background JOB rather than one long synchronous request: POST /api/scan
returns a job_id immediately, pages are OCR'd one at a time in a background
thread, and the frontend polls GET /api/scan/{job_id} for progress and
results as each page finishes. This matters specifically on a memory-and-CPU
capped free host (Render's free tier: 512MB RAM, throttled CPU) — a
synchronous multi-page request there can run long enough to hit the
platform's own proxy timeout, which looks identical to a memory crash (a
502) from the outside. Keeping every HTTP request short avoids that
regardless of how long the whole document takes to finish.
"""
from __future__ import annotations

import base64
import gc
import io
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import fitz  # PyMuPDF
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

# The Android app (and this app's own full-quality default) uses 2600/220,
# but a 512MB-RAM host — Render's free tier — can't fit a page that large
# through the full pipeline (measured peak ~577MB, over the limit). These are
# lowered to 1600/170, measured safe at ~415-450MB with no accuracy loss on
# clean documents (dense small print can still be imperfect regardless of
# this setting — that's a separate OCR-quality issue, not memory). If you
# ever move to a host with more RAM, raise both back to 2600/220.
MAX_SIDE = 1600
PDF_RENDER_DPI = 170
MAX_PAGES = 40
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

# How long a finished/failed job's result stays in memory before being
# swept — the frontend polls until done, so this is just a safety net
# against a growing job dict if a browser tab is abandoned mid-scan.
JOB_TTL_SECONDS = 30 * 60


def _cap_size(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return rgb
    scale = MAX_SIDE / max(h, w)
    return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _pil_to_rgb(im: Image.Image) -> np.ndarray:
    im = im.convert("RGB")
    return np.array(im)


def _exif_transpose(im: Image.Image) -> Image.Image:
    try:
        from PIL import ImageOps
        return ImageOps.exif_transpose(im) or im
    except Exception:  # noqa: BLE001
        return im


def _load_pages(filename: str, data: bytes) -> list[np.ndarray]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        if doc.page_count > MAX_PAGES:
            doc.close()
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
    table: list[list[str]]


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "done", "error"]
    filename: str
    total_pages: int
    completed_pages: int
    pages: list[PageOut]
    elapsed_seconds: float
    engines_available: str
    error: str | None = None


@dataclass
class _Job:
    job_id: str
    filename: str
    status: Literal["queued", "processing", "done", "error"] = "queued"
    total_pages: int = 0
    pages: list[PageOut] = field(default_factory=list)
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def to_status(self) -> JobStatus:
        with self.lock:
            elapsed = (self.finished_at or time.time()) - self.started
            return JobStatus(
                job_id=self.job_id,
                status=self.status,
                filename=self.filename,
                total_pages=self.total_pages,
                completed_pages=len(self.pages),
                pages=list(self.pages),
                elapsed_seconds=round(elapsed, 2),
                engines_available="Tesseract + PaddleOCR",
                error=self.error,
            )


_jobs: dict[str, _Job] = {}
_jobs_lock = threading.Lock()


def _sweep_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with _jobs_lock:
        stale = [
            jid for jid, j in _jobs.items()
            if j.finished_at is not None and j.finished_at < cutoff
        ]
        for jid in stale:
            del _jobs[jid]


def _run_job(job: _Job, filename: str, data: bytes) -> None:
    try:
        pages = _load_pages(filename, data)
        with job.lock:
            job.total_pages = len(pages)
            job.status = "processing"

        for i, rgb in enumerate(pages):
            result = ocr_engine.run_pipeline(rgb)
            page_out = PageOut(
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
                table=result.table,
            )
            with job.lock:
                job.pages.append(page_out)
            # Free this page's intermediate arrays before starting the next
            # one — matters on a memory-capped host with a multi-page PDF,
            # where pages would otherwise be able to pile up.
            del result, rgb
            gc.collect()

        with job.lock:
            job.status = "done"
            job.finished_at = time.time()
    except HTTPException as e:
        with job.lock:
            job.status = "error"
            job.error = str(e.detail)
            job.finished_at = time.time()
    except Exception as e:  # noqa: BLE001
        print(f"Job {job.job_id} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        with job.lock:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.finished_at = time.time()
    finally:
        _sweep_old_jobs()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "tesseract": True,
        "paddleocr": True,
    }


@app.post("/api/scan", response_model=JobStatus)
async def scan(file: UploadFile = File(...)):
    """Starts a scan job and returns immediately (status: queued/processing) —
    poll GET /api/scan/{job_id} for progress and the finished result. This
    never blocks on OCR itself, so large documents can't time out a single
    HTTP request even on a slow host."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File is too large (30 MB limit).")
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")

    filename = file.filename or "upload"
    job = _Job(job_id=uuid.uuid4().hex, filename=filename)
    with _jobs_lock:
        _jobs[job.job_id] = job

    threading.Thread(target=_run_job, args=(job, filename, data), daemon=True).start()
    return job.to_status()


@app.get("/api/scan/{job_id}", response_model=JobStatus)
def scan_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    return job.to_status()


def _get_finished_job(job_id: str) -> "_Job":
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    if job.status != "done":
        raise HTTPException(409, "The scan isn't finished yet.")
    return job


def _download_stem(filename: str) -> str:
    return Path(filename or "scan").stem or "scan"


@app.get("/api/scan/{job_id}/csv")
def scan_csv(job_id: str):
    """Downloads every page's reconstructed table as one CSV file, with a
    'Page N' marker row between pages when there's more than one."""
    job = _get_finished_job(job_id)
    status = job.to_status()
    lines: list[str] = []
    multi_page = len(status.pages) > 1
    for page in status.pages:
        if multi_page:
            if lines:
                lines.append("")
            lines.append(f"Page {page.index + 1}")
        if page.table:
            lines.append(ocr_engine.table_to_csv(page.table).rstrip("\r\n"))
        else:
            lines.append(page.text.replace(",", " "))
    csv_bytes = ("\n".join(lines) + "\n").encode("utf-8-sig")
    stem = _download_stem(status.filename)
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'},
    )


@app.get("/api/scan/{job_id}/xlsx")
def scan_xlsx(job_id: str):
    """Downloads every page's reconstructed table as one Excel workbook, one
    sheet per page."""
    job = _get_finished_job(job_id)
    status = job.to_status()

    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    for page in status.pages:
        title = f"Page {page.index + 1}"[:31] or "Sheet"
        ws = wb.create_sheet(title=title)
        rows = page.table or [[line] for line in page.text.splitlines()]
        widths: dict[int, int] = {}
        for r, row in enumerate(rows, start=1):
            for c, value in enumerate(row, start=1):
                ws.cell(row=r, column=c, value=value)
                widths[c] = max(widths.get(c, 8), min(60, len(str(value)) + 2))
        for c, w in widths.items():
            ws.column_dimensions[get_column_letter(c)].width = w
        ws.freeze_panes = "A2"
    if not wb.sheetnames:
        wb.create_sheet(title="Sheet1")

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    stem = _download_stem(status.filename)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{stem}.xlsx"'},
    )


# Serve the frontend as static files, with index.html at the root.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

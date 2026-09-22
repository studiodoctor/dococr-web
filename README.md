---
title: DocOCR Web
emoji: 📄
colorFrom: teal
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# DocOCR Web

A web version of the DocOCR Android app: upload a photo or PDF of a document and get back
clean, layout-preserved text. It reuses the exact same OCR pipeline that was built and
verified for the Android app — same models, same preprocessing, same fusion and
reading-order logic — so accuracy matches what you already tested on the phone.

Everything runs locally on your own machine/server. No cloud OCR APIs, no data leaves
the box, and no fees.

## What it does

1. You upload a PDF or an image (JPG/PNG/WEBP) through the browser.
2. Each page is preprocessed: perspective correction, deskew, illumination
   flattening, denoise, sharpen, binarisation for the text-detection pass.
3. Two OCR engines run on every page: **Tesseract 5** (LSTM) and **PaddleOCR
   PP-OCRv5** (ONNX Runtime: text detector + text recognizer + 180° angle
   classifier).
4. Page rotation (90°/180°/270°) is detected and corrected *before*
   recognition, so a sideways or upside-down photo still comes out reading
   top-to-bottom, left-to-right.
5. The two engines' results are fused line-by-line (the more confident engine
   wins each line; text only one engine found is still kept).
6. Reading order is reconstructed with a recursive XY-cut algorithm, so
   multi-column layouts read column-by-column instead of splicing lines from
   both columns together.
7. The extracted text is rendered back with the original layout — line
   breaks, paragraphs, table-like column alignment — and shown next to the
   scanned page image, with copy/download actions.

## Architecture

```
DocOCR-Web/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI app: /api/scan, /api/health, serves the frontend
│   │   └── ocr_engine.py    OCR pipeline (preprocessing, Tesseract, PaddleOCR, fusion, layout)
│   ├── models/               det.onnx, rec.onnx, cls.onnx, rec_dict.txt, eng.traineddata
│   └── requirements.txt
├── frontend/
│   └── index.html            Upload UI (vanilla HTML/CSS/JS, no build step)
└── README.md
```

There's no separate front-end build: FastAPI serves `frontend/index.html` directly, and
the page talks to the API with plain `fetch()`.

## Requirements

- Python 3.10+
- Tesseract 5 installed as a system binary (the `tesseract` executable must be on
  your `PATH`) — the bundled `eng.traineddata` is used, so you don't need the
  system's own language data, but the binary itself must be installed:
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki, and make
    sure `tesseract.exe` is on `PATH`
- Everything else (PaddleOCR models, ONNX Runtime, OpenCV, PyMuPDF) is pure
  Python and installs via pip — no separate PaddlePaddle install and no GPU
  needed.

## Setup

```bash
cd DocOCR-Web/backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd DocOCR-Web/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser, drop in a PDF or photo, and click
**Scan document**.

## API

If you want to call the OCR service from another app instead of the bundled UI:

```
POST /api/scan
  multipart/form-data, field name "file" — a PDF or image, up to 30 MB / 20 pages

200 OK
{
  "filename": "invoice.pdf",
  "elapsed_seconds": 2.4,
  "engines_available": "Tesseract + PaddleOCR",
  "pages": [
    {
      "index": 0,
      "text": "...",                 // layout-preserved plain text
      "engine": "PaddleOCR + Tesseract",
      "width": 1654, "height": 2339,
      "diagnostics": ["Page was rotated 90°; corrected before recognition."],
      "boxes": [ { "text": "...", "confidence": 0.97, "left": .., "top": .., "right": .., "bottom": .. } ],
      "image": "data:image/png;base64,..."   // the processed page, for reference
    }
  ]
}
```

`GET /api/health` returns `{"status": "ok"}` once both engines have loaded.

## Notes on accuracy

This pipeline is the same one measured on the Android app: 99.7–100% character
accuracy on synthetic ground-truth documents covering clean scans, photographed
pages (perspective distortion, shadow, noise, warm colour cast), 90°/180° rotated
pages, dense small print, and two-column layouts. Because the exact same ONNX
models, constants, and fusion/layout logic are used here, you should see the same
quality on the web version.

Large or very low-resolution photos are automatically downscaled to a 2600px max
side before processing (matching the Android app), and PDFs are rasterised at
220 DPI per page before running the same image pipeline.

## Limits

- Max upload size: 30 MB
- Max PDF pages per upload: 20 (raise `MAX_PAGES` in `backend/app/main.py` if you
  need more — each extra page adds a few seconds of CPU-bound OCR time)
- CPU-only by default (ONNX Runtime's `CPUExecutionProvider`). A GPU box can
  swap in `onnxruntime-gpu` and `CUDAExecutionProvider` in `ocr_engine.py` for
  faster batch processing, but isn't required.

## Deploying to Hugging Face Spaces

This project ships with a `Dockerfile` at the repo root that packages the backend,
frontend, and models into one container — Tesseract included, no other hosting needed.
Hugging Face Spaces' free CPU tier runs it as-is.

1. Create a new Space at https://huggingface.co/new-space — pick **Docker** as the
   SDK (not "Blank"/Gradio/Streamlit), any name, public visibility, free CPU tier.
2. Clone the empty Space repo it gives you, e.g.:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   ```
3. Copy everything from this project into that cloned folder — `Dockerfile`,
   `.dockerignore`, `README.md`, `backend/`, `frontend/` — so `Dockerfile` sits at
   the repo root next to `README.md`. (The `README.md`'s YAML header at the very
   top — `sdk: docker`, `app_port: 7860` — is what tells Spaces how to build and run
   it; don't remove it.)
4. Commit and push:
   ```bash
   cd <space-name>
   git add -A
   git commit -m "Deploy DocOCR Web"
   git push
   ```
5. The Space tab on huggingface.co shows the build log; once it says "Running", your
   app is live at `https://<your-username>-<space-name>.hf.space` — that single URL
   serves both the upload UI and the `/api/scan` endpoint, no separate frontend
   hosting needed.

Notes:
- The free CPU tier is enough for this (it's the same CPU-only ONNX Runtime path used
  locally) but is slower than a dedicated server and Spaces can sleep after inactivity,
  so the first request after a while will be slow to wake up.
- `backend/models/` (~37 MB of ONNX + Tesseract data) is committed straight to the
  Space repo — Spaces repos handle that fine, but if you ever add much larger model
  files, use `git lfs track` for them before committing.
- If you'd rather host the frontend separately (e.g. on Netlify) and only put the
  backend on a Space, change the `fetch('/api/scan', ...)` call in
  `frontend/index.html` to your Space's full URL and make sure the Space stays public
  so the browser can reach it cross-origin (CORS is already wide open in `main.py`).

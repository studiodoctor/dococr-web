# DocOCR Web — single container: FastAPI backend (Tesseract + PaddleOCR via
# ONNX Runtime) plus the static frontend, served on one port. Built for
# Hugging Face Spaces (Docker SDK, port 7860) but runs anywhere Docker does.

FROM python:3.11-slim

# Tesseract is a system binary (not a pip package) — the OCR engine needs it.
# libgl1/libglib2.0-0 are OpenCV's usual runtime dependencies even in
# "headless" builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY frontend frontend

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

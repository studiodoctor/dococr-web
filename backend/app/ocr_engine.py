"""
Core OCR engine for the web service.

This is a direct, faithful port of the verified Android pipeline (DocOCR):
  - PaddleOCR PP-OCRv5 (det.onnx + rec.onnx + cls.onnx) via ONNX Runtime, run
    exactly as in com.dococr.ocr.PaddleEngine — same constants, same CTC
    decoding, same page-level rotation correction (90/180/270) applied BEFORE
    recognition so reading order stays correct.
  - Tesseract 5 (LSTM) via the same eng.traineddata bundled with the app,
    used as the second engine.
  - The same XY-cut reading-order reconstruction and length-weighted engine
    fusion used on Android (ResultFuser / LayoutBuilder), so a page that read
    correctly on the phone reads correctly here too.

Every accuracy-relevant constant mirrors app/src/main/java/com/dococr/ocr/*.kt
and app/src/main/java/com/dococr/image/Preprocessor.kt exactly.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytesseract

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ---------------------------------------------------------------------------
# PaddleOCR (PP-OCRv5) — mirrors PaddleEngine.kt
# ---------------------------------------------------------------------------

DET_MAX_SIDE = 1280
DET_THRESH = 0.3
DET_BOX_THRESH = 0.5
DET_UNCLIP = 1.6

REC_HEIGHT = 48
REC_MAX_WIDTH = 1600

CLS_HEIGHT = 48
CLS_WIDTH = 192
CLS_THRESH = 0.9
CLS_SAMPLE = 12

ORIENT_MAX_SIDE = 800

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_providers = ["CPUExecutionProvider"]
_DICT = [ln.rstrip("\n") for ln in open(MODELS_DIR / "rec_dict.txt", encoding="utf-8")]

TESSDATA_DIR = str(MODELS_DIR)

# Sessions are created lazily (on first use, not at import) and pinned to a
# single thread. This keeps the process' startup fast and its memory/CPU
# footprint predictable on small/free hosting tiers (e.g. Render's free
# instance) — the port binds and /api/health responds immediately, instead of
# the whole request potentially blocking on model loads before the server is
# even listening.
_sessions: dict[str, ort.InferenceSession] = {}


def _session(name: str) -> ort.InferenceSession:
    sess = _sessions.get(name)
    if sess is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess = ort.InferenceSession(str(MODELS_DIR / name), sess_options=opts, providers=_providers)
        _sessions[name] = sess
    return sess


def _det_sess() -> ort.InferenceSession:
    return _session("det.onnx")


def _rec_sess() -> ort.InferenceSession:
    return _session("rec.onnx")


def _cls_sess() -> ort.InferenceSession:
    return _session("cls.onnx")


def preload_models() -> None:
    """Warms up all three sessions. Called once from a background thread on
    app startup so the first real request isn't the one paying for it, while
    still letting uvicorn bind the port and answer /api/health immediately."""
    _det_sess()
    _rec_sess()
    _cls_sess()


def _order_clockwise(pts):
    pts = list(pts)
    by_sum = sorted(pts, key=lambda q: q[0] + q[1])
    by_diff = sorted(pts, key=lambda q: q[1] - q[0])
    return [by_sum[0], by_diff[0], by_sum[-1], by_diff[-1]]


def _detect(rgb: np.ndarray):
    h0, w0 = rgb.shape[:2]
    ratio = min(1.0, DET_MAX_SIDE / max(w0, h0))
    w = max(32, int(math.ceil(w0 * ratio / 32) * 32))
    h = max(32, int(math.ceil(h0 * ratio / 32) * 32))
    resized = cv2.resize(rgb, (w, h))
    sx, sy = w0 / w, h0 / h

    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]

    det_sess = _det_sess()
    prob = det_sess.run(None, {det_sess.get_inputs()[0].name: x})[0][0][0]
    _, binmap = cv2.threshold(prob, DET_THRESH, 255.0, cv2.THRESH_BINARY)
    binmap = binmap.astype(np.uint8)
    contours, _ = cv2.findContours(binmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in contours:
        if len(c) < 4:
            continue
        rect = cv2.minAreaRect(c)
        if min(rect[1]) < 3:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [c], 255)
        score = cv2.mean(prob, mask)[0]
        if score < DET_BOX_THRESH:
            continue
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        dist = area * DET_UNCLIP / peri
        (cx, cy), (bw, bh), ang = cv2.minAreaRect(c)
        expanded = ((cx, cy), (bw + 2 * dist, bh + 2 * dist), ang)
        pts = cv2.boxPoints(expanded)
        pts = [(p[0] * sx, p[1] * sy) for p in pts]
        out.append(_order_clockwise(pts))
    out.sort(key=lambda poly: (min(p[1] for p in poly), min(p[0] for p in poly)))
    return out


def _crop_rotated(rgb: np.ndarray, poly):
    p = np.float32(poly)
    w = int(round(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3]))))
    h = int(round(max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1]))))
    if w < 4 or h < 4:
        return None
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    out = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(p, dst), (w, h), flags=cv2.INTER_CUBIC)
    if out.shape[0] > out.shape[1] * 1.5:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return out


def _char_for(idx: int) -> str:
    return _DICT[idx - 1] if 0 <= idx - 1 < len(_DICT) else " "


def _ctc_decode(logits):
    sb, last, conf, n = [], -1, 0.0, 0
    for step in logits:
        best = int(np.argmax(step))
        if best != 0 and best != last:
            sb.append(_char_for(best))
            conf += float(step[best])
            n += 1
        last = best
    return "".join(sb), (conf / n if n else 0.0)


def _recognise_crop(crop: np.ndarray):
    ratio = crop.shape[1] / crop.shape[0]
    w = int(np.clip(round(REC_HEIGHT * ratio), REC_HEIGHT, REC_MAX_WIDTH))
    resized = cv2.resize(crop, (w, REC_HEIGHT))
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = x.transpose(2, 0, 1)[None]
    rec_sess = _rec_sess()
    logits = rec_sess.run(None, {rec_sess.get_inputs()[0].name: x})[0][0]
    return _ctc_decode(logits)


def _classify_180(crops):
    out = []
    for img in crops:
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            out.append(False)
            continue
        rw = min(CLS_WIDTH, int(math.ceil(CLS_HEIGHT * w / h)))
        rw = max(1, rw)
        r = cv2.resize(img, (rw, CLS_HEIGHT)).astype(np.float32).transpose(2, 0, 1) / 255.0
        r = (r - 0.5) / 0.5
        pad = np.zeros((3, CLS_HEIGHT, CLS_WIDTH), np.float32)
        pad[:, :, :rw] = r
        cls_sess = _cls_sess()
        p = cls_sess.run(None, {cls_sess.get_inputs()[0].name: pad[None]})[0][0]
        out.append(bool(p[1] > CLS_THRESH))
    return out


def _quad_wh(poly):
    p = np.float32(poly)
    w = max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3]))
    h = max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1]))
    return w, h


def _page_is_sideways(polys) -> bool:
    if len(polys) < 3:
        return False
    tall = sum(1 for p in polys if _quad_wh(p)[1] > _quad_wh(p)[0] * 1.2)
    return tall > len(polys) * 0.6


def detect_rotation(rgb: np.ndarray) -> int:
    """Returns the clockwise rotation (0/90/180/270) the page needs before recognition."""
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, ORIENT_MAX_SIDE / max(w0, h0))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    polys = _detect(small)
    rotation = 90 if _page_is_sideways(polys) else 0

    working = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE) if rotation == 90 else small
    polys = _detect(working) if rotation == 90 else polys
    if not polys:
        return rotation

    by_width = sorted(polys, key=lambda p: -_quad_wh(p)[0])[:CLS_SAMPLE]
    crops = [c for c in (_crop_rotated(working, p) for p in by_width) if c is not None]
    if not crops:
        return rotation
    flips = _classify_180(crops)
    if flips and sum(flips) > len(flips) * 0.5:
        rotation = (rotation + 180) % 360
    return rotation


def paddle_recognise(rgb: np.ndarray):
    """Returns list of (text, conf, (l,t,r,b)) in the given image's own coordinate space."""
    polys = _detect(rgb)
    crops, keep = [], []
    for poly in polys:
        c = _crop_rotated(rgb, poly)
        if c is not None:
            crops.append(c)
            keep.append(poly)
    if crops:
        flips = _classify_180(crops)
        for i, flip in enumerate(flips):
            if flip:
                crops[i] = cv2.rotate(crops[i], cv2.ROTATE_180)
    out = []
    for poly, crop in zip(keep, crops):
        text, conf = _recognise_crop(crop)
        if text.strip():
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            out.append((text.strip(), conf, (min(xs), min(ys), max(xs), max(ys))))
    return out


def tesseract_recognise(rgb: np.ndarray):
    """Returns list of (text, conf, (l,t,r,b)) via Tesseract word-level boxes."""
    import os

    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR
    data = pytesseract.image_to_data(
        rgb, lang="eng", config="--psm 3", output_type=pytesseract.Output.DICT
    )
    out = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        conf_raw = data["conf"][i]
        try:
            conf = max(0.0, float(conf_raw)) / 100.0
        except (TypeError, ValueError):
            conf = 0.0
        l, t, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append((text, conf, (float(l), float(t), float(l + w), float(t + h))))
    return out


# ---------------------------------------------------------------------------
# Preprocessing — mirrors Preprocessor.kt
# ---------------------------------------------------------------------------

def perspective_correct(rgb: np.ndarray, skip: bool = False):
    if skip:
        return rgb, False
    scale = 600.0 / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = small.shape[0] * small.shape[1]
    best, best_area = None, 0
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = abs(cv2.contourArea(approx))
        if len(approx) == 4 and area > image_area * 0.25 and area > best_area and cv2.isContourConvex(approx):
            best, best_area = approx, area
    if best is None or best_area > image_area * 0.97:
        return rgb, False
    pts = _order_clockwise([(p[0][0] / scale, p[0][1] / scale) for p in best])
    p = np.float32(pts)
    w = int(round(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3]))))
    h = int(round(max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1]))))
    if w < 200 or h < 200:
        return rgb, False
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(p, dst), (w, h), flags=cv2.INTER_CUBIC), True


def estimate_skew(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = 1000.0 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    lines = cv2.HoughLinesP(b, 1, np.pi / 720, 80, minLineLength=gray.shape[1] / 4, maxLineGap=20)
    if lines is None:
        return 0.0
    angs = []
    for l in lines[:, 0]:
        a = np.degrees(np.arctan2(l[3] - l[1], l[2] - l[0]))
        if abs(a) < 20:
            angs.append(a)
    if len(angs) < 3:
        return 0.0
    return float(np.median(angs))


def enhance(rgb: np.ndarray) -> np.ndarray:
    out = cv2.bilateralFilter(rgb, 7, 50, 50)
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    bg = cv2.GaussianBlur(bg, (0, 0), 15).astype(np.float32) / 255.0
    bg = np.maximum(bg, 0.05)
    f = out.astype(np.float32) / 255.0
    f = np.minimum(f / bg[..., None], 1.0)
    out = (f * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(out, (0, 0), 2)
    out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)
    return out


@dataclass
class Prepared:
    color: np.ndarray
    binary: np.ndarray
    corrected: bool
    skew: float


def preprocess(rgb: np.ndarray, skip_perspective: bool = False) -> Prepared:
    warped, corrected = perspective_correct(rgb, skip_perspective)
    ang = estimate_skew(warped)
    if 0.3 < abs(ang) < 20:
        c = (warped.shape[1] / 2, warped.shape[0] / 2)
        warped = cv2.warpAffine(
            warped, cv2.getRotationMatrix2D(c, ang, 1.0), (warped.shape[1], warped.shape[0]),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )
    clean = enhance(warped)
    gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
    # Kept single-channel (not expanded to RGB) — it's a pure black/white image
    # so the two extra channels were pure waste, and Tesseract (the only
    # consumer) accepts grayscale directly. Cuts a meaningful chunk of peak
    # memory on memory-capped hosts for free.
    return Prepared(clean, binary, corrected, ang)


def rotate_prepared(p: Prepared, degrees: int) -> Prepared:
    def rot(img):
        if degrees == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if degrees == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        if degrees == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    return Prepared(rot(p.color), rot(p.binary), p.corrected, p.skew)


# ---------------------------------------------------------------------------
# Engine fusion — mirrors ResultFuser.kt
# ---------------------------------------------------------------------------

REPLACE_MARGIN = 0.12
SAME_LINE_IOU = 0.35


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _mean_conf(boxes):
    if not boxes:
        return 0.0
    total_len = sum(max(1, len(t)) for t, c, b in boxes)
    if total_len == 0:
        return 0.0
    return sum(c * max(1, len(t)) for t, c, b in boxes) / total_len


def _line_bounds(line):
    return (
        min(b[2][0] for b in line), min(b[2][1] for b in line),
        max(b[2][2] for b in line), max(b[2][3] for b in line),
    )


def _line_text(line):
    return " ".join(b[0] for b in line)


def _line_conf(line):
    total_len = sum(max(1, len(t)) for t, c, b in line)
    if total_len == 0:
        return 0.0
    return sum(c * max(1, len(t)) for t, c, b in line) / total_len


def fuse(tess_boxes, paddle_boxes):
    """Returns (fused_boxes, chosen_engine_label). Mirrors ResultFuser.kt exactly:
    fusion compares whole LINES (not individual word boxes), which is what keeps a
    word-level engine (Tesseract) from producing duplicate text next to a
    line-level engine (PaddleOCR)."""
    tess_boxes = tess_boxes or []
    paddle_boxes = paddle_boxes or []
    if not tess_boxes and not paddle_boxes:
        return [], "none"
    if not tess_boxes:
        return paddle_boxes, "PaddleOCR"
    if not paddle_boxes:
        return tess_boxes, "Tesseract"

    t_conf = _mean_conf(tess_boxes)
    p_conf = _mean_conf(paddle_boxes)
    # Paddle wins ties: its detector is far better on photographed, low-contrast documents.
    paddle_primary = p_conf + 0.03 >= t_conf
    primary = group_lines(paddle_boxes if paddle_primary else tess_boxes)
    secondary = group_lines(tess_boxes if paddle_primary else paddle_boxes)
    label = "PaddleOCR + Tesseract" if paddle_primary else "Tesseract + PaddleOCR"

    out = []
    used_secondary = set()
    for p in primary:
        pb = _line_bounds(p)
        best_i, best_iou = -1, 0.0
        for i, s in enumerate(secondary):
            if i in used_secondary:
                continue
            iou = _iou(pb, _line_bounds(s))
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i < 0 or best_iou < SAME_LINE_IOU:
            out.extend(p)
            continue
        used_secondary.add(best_i)
        s = secondary[best_i]
        p_conf_line, s_conf_line = _line_conf(p), _line_conf(s)
        p_text, s_text = _line_text(p), _line_text(s)
        better = (s_conf_line - p_conf_line > REPLACE_MARGIN) or (
            p_conf_line < 0.5 and len(s_text) > len(p_text) * 1.3
        )
        out.extend(s if better else p)

    for i, s in enumerate(secondary):
        if i in used_secondary:
            continue
        sb = _line_bounds(s)
        if _line_conf(s) >= 0.6 and all(_iou(_line_bounds(p), sb) <= 0.1 for p in primary):
            out.extend(s)

    return out, label


# ---------------------------------------------------------------------------
# Reading order + layout — mirrors LayoutBuilder.kt
# ---------------------------------------------------------------------------

def _holes(intervals):
    merged, gaps = [], []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    for a, b in zip(merged, merged[1:]):
        gaps.append((b[0] - a[1], a[1], b[0]))
    return gaps


def _v_split(boxes, min_gap_ratio):
    if len(boxes) < 4:
        return None
    xs0 = min(b[2][0] for b in boxes)
    xs1 = max(b[2][2] for b in boxes)
    page_w = max(1.0, xs1 - xs0)
    g = _holes([(b[2][0], b[2][2]) for b in boxes])
    if not g:
        return None
    w, ga, gb = max(g)
    if w <= page_w * min_gap_ratio:
        return None
    cut = (ga + gb) / 2
    left = [b for b in boxes if b[2][2] <= cut]
    right = [b for b in boxes if b[2][0] >= cut]
    if left and right and len(left) + len(right) == len(boxes):
        return left, right
    return None


def xy_regions(boxes, min_gap_ratio: float = 0.10, depth: int = 0):
    if len(boxes) <= 1 or depth > 10:
        return [list(boxes)]
    v = _v_split(boxes, min_gap_ratio)
    if v:
        return xy_regions(v[0], min_gap_ratio, depth + 1) + xy_regions(v[1], min_gap_ratio, depth + 1)

    heights = sorted(b[2][3] - b[2][1] for b in boxes)
    median_h = heights[len(heights) // 2]
    g = _holes([(b[2][1], b[2][3]) for b in boxes])
    if g:
        h, ga, gb = max(g)
        if h > median_h * 0.8:
            cut = (ga + gb) / 2
            top = [b for b in boxes if b[2][3] <= cut]
            bot = [b for b in boxes if b[2][1] >= cut]
            if top and bot and len(top) + len(bot) == len(boxes):
                if _v_split(top, min_gap_ratio) or _v_split(bot, min_gap_ratio):
                    return xy_regions(top, min_gap_ratio, depth + 1) + xy_regions(bot, min_gap_ratio, depth + 1)
    return [list(boxes)]


def group_lines(boxes):
    items = sorted(boxes, key=lambda b: (b[2][1] + b[2][3]) / 2)
    lines = []
    for b in items:
        cy = (b[2][1] + b[2][3]) / 2
        h = b[2][3] - b[2][1]
        placed = None
        for ln in reversed(lines):
            ref = statistics.fmean([(x[2][1] + x[2][3]) / 2 for x in ln])
            hh = statistics.fmean([x[2][3] - x[2][1] for x in ln])
            if abs(cy - ref) < max(hh, h) * 0.5:
                placed = ln
                break
        if placed is not None:
            placed.append(b)
        else:
            lines.append([b])
    out = [sorted(ln, key=lambda x: x[2][0]) for ln in lines]
    out.sort(key=lambda ln: min(x[2][1] for x in ln))
    return out


def _bounds(ln):
    return (
        min(x[2][0] for x in ln), min(x[2][1] for x in ln),
        max(x[2][2] for x in ln), max(x[2][3] for x in ln),
    )


def group_blocks(lines):
    blocks = []
    for ln in lines:
        b = _bounds(ln)
        if blocks:
            pb = _bounds(blocks[-1][-1])
            if (b[1] - pb[3]) < (pb[3] - pb[1]) * 0.9 and b[0] < pb[2] and pb[0] < b[2]:
                blocks[-1].append(ln)
                continue
        blocks.append([ln])
    return blocks


def build_regions(boxes):
    regions = xy_regions(boxes)
    return [group_blocks(group_lines(r)) for r in regions]


def _region_plain_text(blocks):
    all_lines = [ln for blk in blocks for ln in blk]
    if not all_lines:
        return ""
    heights = sorted(_bounds(ln)[3] - _bounds(ln)[1] for ln in all_lines)
    median_h = heights[len(heights) // 2]
    char_w = max(4.0, median_h * 0.55)
    out, last = [], None
    for blk in blocks:
        bb = (
            min(_bounds(l)[0] for l in blk), min(_bounds(l)[1] for l in blk),
            max(_bounds(l)[2] for l in blk), max(_bounds(l)[3] for l in blk),
        )
        if last is not None:
            gap = bb[1] - last[3]
            out.extend([""] * min(3, max(1, round(gap / (median_h * 1.4)))))
        for ln in blk:
            row = ""
            for t, c, bx in ln:
                col = round((bx[0] - _region_left(blk)) / char_w)
                if len(row) < col:
                    row += " " * (col - len(row))
                if row and not row.endswith(" "):
                    row += " "
                row += t
            out.append(row.rstrip())
        last = bb
    return "\n".join(out).strip()


def _region_left(blk):
    return min(x[2][0] for ln in blk for x in ln)


def to_plain_text(regions):
    parts = [_region_plain_text(r) for r in regions]
    parts = [p for p in parts if p]
    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    plain_text: str
    boxes: list = field(default_factory=list)  # (text, conf, (l,t,r,b))
    chosen_engine: str = "none"
    image_width: int = 0
    image_height: int = 0
    diagnostics: list = field(default_factory=list)
    region_count: int = 1


def run_pipeline(rgb: np.ndarray, skip_perspective: bool = False) -> PageResult:
    notes: list[str] = []

    try:
        prepared = preprocess(rgb, skip_perspective)
    except Exception as e:  # noqa: BLE001
        notes.append(f"Preprocessing failed ({type(e).__name__}: {e}); using the original image.")
        prepared = Prepared(rgb, rgb, False, 0.0)

    try:
        rotation = detect_rotation(prepared.color)
    except Exception:  # noqa: BLE001
        rotation = 0
    if rotation:
        notes.append(f"Page was rotated {rotation}°; corrected before recognition.")
        prepared = rotate_prepared(prepared, rotation)

    def recognise_both(colour, binary):
        try:
            t = tesseract_recognise(binary)
        except Exception as e:  # noqa: BLE001
            notes.append(f"Tesseract failed — {type(e).__name__}: {e}")
            t = []
        try:
            p = paddle_recognise(colour)
        except Exception as e:  # noqa: BLE001
            notes.append(f"PaddleOCR failed — {type(e).__name__}: {e}")
            p = []
        return t, p

    tess_boxes, paddle_boxes = recognise_both(prepared.color, prepared.binary)

    if not tess_boxes and not paddle_boxes:
        notes.append("No text in the cleaned image — retried the original.")
        rt, rp = recognise_both(rgb, rgb)
        if rt or rp:
            tess_boxes, paddle_boxes = rt, rp
            prepared = Prepared(rgb, rgb, False, 0.0)

    boxes, chosen = fuse(tess_boxes, paddle_boxes)
    regions = build_regions(boxes) if boxes else []
    if len(regions) > 1:
        notes.append(f"Detected {len(regions)} text regions (columns/bands).")

    return PageResult(
        plain_text=to_plain_text(regions) if regions else "",
        boxes=boxes,
        chosen_engine=chosen,
        image_width=prepared.color.shape[1],
        image_height=prepared.color.shape[0],
        diagnostics=notes,
        region_count=max(1, len(regions)),
    )

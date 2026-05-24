from __future__ import annotations

import base64
import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

OUT_SIZE = int(os.getenv('OUT_SIZE', '384'))
PAD_PCT = float(os.getenv('PAD_PCT', '0.08'))
YOLO_CONF = float(os.getenv('YOLO_CONF', '0.10'))
PERSON_CLS = 0

app = FastAPI(title='preprocessing')

_yolo = None


def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO('yolov8n.pt')
    return _yolo


CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))


def enhance(pil_img: Image.Image) -> Image.Image:
    arr = np.array(pil_img.convert('L'))
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    arr = CLAHE.apply(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))


def contour_bbox(pil_img: Image.Image):
    g = np.array(pil_img.convert('L'))
    H, W = g.shape
    mh, mw = int(H * 0.03), int(W * 0.03)
    inner = g[mh:H - mh, mw:W - mw]
    Hi, Wi = inner.shape
    img_area = Hi * Wi

    bw = cv2.adaptiveThreshold(inner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 10)
    bw = cv2.medianBlur(bw, 3)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n_lbl <= 1:
        return None

    keep = []
    for i in range(1, n_lbl):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        ba = w * h
        if ba < 0.0005 * img_area:
            continue
        if w > 0.92 * Wi and h > 0.92 * Hi:
            continue
        if max(w, h) / max(min(w, h), 1) > 12:
            continue
        if a < 0.005 * ba:
            continue
        keep.append((x, y, x + w, y + h, ba))

    if not keep:
        return None

    keep.sort(key=lambda t: -t[4])
    anchor = keep[0]
    ax_c = (anchor[0] + anchor[2]) / 2
    ay_c = (anchor[1] + anchor[3]) / 2
    a_size = max(anchor[2] - anchor[0], anchor[3] - anchor[1])
    D = max(a_size * 3.0, 0.12 * max(Hi, Wi))

    merged = [anchor]
    for c in keep[1:]:
        cx = (c[0] + c[2]) / 2
        cy = (c[1] + c[3]) / 2
        if abs(cx - ax_c) < D and abs(cy - ay_c) < D:
            merged.append(c)

    return (
        float(min(c[0] for c in merged) + mw),
        float(min(c[1] for c in merged) + mh),
        float(max(c[2] for c in merged) + mw),
        float(max(c[3] for c in merged) + mh),
    )


def yolo_bbox(pil_img: Image.Image):
    arr = np.array(pil_img.convert('RGB'))
    res = get_yolo().predict(arr, conf=YOLO_CONF, classes=[PERSON_CLS], verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return tuple(map(float, boxes[areas.argmax()]))


def process(pil_img: Image.Image):
    pil_rgb = pil_img.convert('RGB')
    W, H = pil_rgb.size
    enh = enhance(pil_rgb)

    bb = contour_bbox(enh)
    src = 'contour'
    if bb is None:
        bb = yolo_bbox(enh)
        src = 'yolo'
    if bb is None:
        bb = (0, 0, W, H)
        src = 'whole'

    x1, y1, x2, y2 = bb
    px = (x2 - x1) * PAD_PCT
    py = (y2 - y1) * PAD_PCT
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(W, int(x2 + px)), min(H, int(y2 + py))
    cropped = pil_rgb.crop((cx1, cy1, cx2, cy2))

    cw, ch = cropped.size
    s = max(cw, ch)
    canvas = Image.new('RGB', (s, s), (255, 255, 255))
    canvas.paste(cropped, ((s - cw) // 2, (s - ch) // 2))
    resized = canvas.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)

    return resized, bb, src, (W, H)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/crop')
async def crop(file: UploadFile = File(...)):
    raw = await file.read()
    pil = Image.open(io.BytesIO(raw)).convert('RGB')

    resized, bb, src, (W, H) = process(pil)

    buf = io.BytesIO()
    resized.save(buf, format='PNG')
    cropped_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    return {
        'cropped_png_b64': cropped_b64,
        'bbox': {'x1': bb[0], 'y1': bb[1], 'x2': bb[2], 'y2': bb[3]},
        'orig_w': W,
        'orig_h': H,
        'src': src,
    }

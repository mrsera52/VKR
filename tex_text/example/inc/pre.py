import cv2
import numpy as np

def detect_figure_bbox(img):
    H, W = img.shape[:2]
    margin = (int(0.03 * H), int(0.03 * W))
    inner = img[margin[0]:H-margin[0], margin[1]:W-margin[1]]
    if len(inner.shape) == 3:
        inner = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(inner, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    bw = cv2.medianBlur(bw, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    Hi, Wi = bw.shape
    img_area = Hi * Wi
    candidates = []
    for i in range(1, n_lbl):
        x, y, w, h, area = stats[i, [0, 1, 2, 3, 4]]
        bbox_area = w * h
        if bbox_area < 0.0005 * img_area: continue
        if w > 0.92 * Wi and h > 0.92 * Hi: continue
        if max(w, h) / max(min(w, h), 1) > 12: continue
        if area < 0.005 * bbox_area: continue
        candidates.append((x, y, x + w, y + h, bbox_area))
    if not candidates:
        return (0, 0, W, H)
    candidates.sort(key=lambda t: -t[4])
    anchor = candidates[0]
    ax, ay = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    a_size = max(anchor[2] - anchor[0], anchor[3] - anchor[1])
    D = max(3.0 * a_size, 0.12 * max(Hi, Wi))
    merged = [anchor]
    for c in candidates[1:]:
        cx, cy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
        if abs(cx - ax) < D and abs(cy - ay) < D:
            merged.append(c)
    x1 = min(c[0] for c in merged) + margin[1]
    y1 = min(c[1] for c in merged) + margin[0]
    x2 = max(c[2] for c in merged) + margin[1]
    y2 = max(c[3] for c in merged) + margin[0]
    return (x1, y1, x2, y2)

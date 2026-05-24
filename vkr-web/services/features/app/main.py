from fastapi import FastAPI
from pydantic import BaseModel

BBOX_DIM = 12

app = FastAPI(title='features')


class Bbox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class FeatureRequest(BaseModel):
    bbox: Bbox
    orig_w: float
    orig_h: float
    src: str = 'contour'


class FeatureResponse(BaseModel):
    features: list[float]
    names: list[str]


FEAT_NAMES = ['bbox_w_rel', 'bbox_h_rel', 'bbox_area_rel', 'cx_rel', 'cy_rel',
              'aspect', 'touches_top', 'touches_bottom', 'touches_left', 'touches_right',
              'min_corner_dist', 'is_yolo']


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/compute', response_model=FeatureResponse)
def compute(req: FeatureRequest):
    x1, y1, x2, y2 = req.bbox.x1, req.bbox.y1, req.bbox.x2, req.bbox.y2
    W, H = req.orig_w, req.orig_h
    if W <= 0 or H <= 0:
        return FeatureResponse(features=[0.0] * BBOX_DIM, names=FEAT_NAMES)

    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rw, rh = bw / W, bh / H
    aspect = bh / bw
    touches_top = 1.0 if y1 < 0.05 * H else 0.0
    touches_bottom = 1.0 if y2 > 0.95 * H else 0.0
    touches_left = 1.0 if x1 < 0.05 * W else 0.0
    touches_right = 1.0 if x2 > 0.95 * W else 0.0
    diag = (W ** 2 + H ** 2) ** 0.5
    min_corner = min(((cx - cxx) ** 2 + (cy - cyy) ** 2) ** 0.5
                     for cxx in (0, W) for cyy in (0, H)) / diag
    is_yolo = 1.0 if str(req.src).startswith('yolo') else 0.0

    feats = [rw, rh, rw * rh, cx / W, cy / H, min(aspect, 5.0),
             touches_top, touches_bottom, touches_left, touches_right,
             min_corner, is_yolo]
    return FeatureResponse(features=feats, names=FEAT_NAMES)

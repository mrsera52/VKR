from __future__ import annotations

import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from torchvision import transforms

from app.model import DrawingNet

DATA_DIR = Path('/data')
ENCODERS_PATH = DATA_DIR / 'encoders.pkl'
IMG_SIZE = int(os.getenv('IMG_SIZE', '320'))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

app = FastAPI(title='inference')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[inference] device: {DEVICE}')

if not ENCODERS_PATH.exists():
    raise RuntimeError(f'encoders.pkl не найден в {DATA_DIR}. Запусти build_encoders.')
enc = pickle.loads(ENCODERS_PATH.read_bytes())

CRITERIA_CONFIGS = enc['criteria_configs']
LABEL_DECODERS = enc['label_decoders']
SAFE_COL_MAP = enc['safe_col_map']
NUM_CLASSES_SAFE = enc['num_classes_safe']
TARGET_SAFE = enc['target_safe_names']
NONTARGET_SAFE = enc['nontarget_safe_names']
BBOX_MEAN_T = torch.tensor(enc['bbox_mean'], dtype=torch.float32)
BBOX_STD_T = torch.tensor(enc['bbox_std'], dtype=torch.float32)
BBOX_DIM = int(enc['bbox_dim'])
print(f'[inference] criteria={len(CRITERIA_CONFIGS)}, heads={len(NUM_CLASSES_SAFE)}, '
      f'target={len(TARGET_SAFE)}')

model_files = list(DATA_DIR.glob('*.pth'))
if not model_files:
    raise RuntimeError(f'*.pth не найден в {DATA_DIR}')
MODEL_PATH = model_files[0]
print(f'[inference] loading: {MODEL_PATH.name}')

model = DrawingNet(NUM_CLASSES_SAFE, TARGET_SAFE, NONTARGET_SAFE, bbox_in_dim=BBOX_DIM)
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=False)
model.to(DEVICE).eval()
print(f'[inference] params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'device': str(DEVICE),
        'model': MODEL_PATH.name,
        'criteria': len(CRITERIA_CONFIGS),
    }


@app.post('/predict')
async def predict(image: UploadFile = File(...), features: str = Form(...)):
    feats = json.loads(features)
    if len(feats) != BBOX_DIM:
        return {'error': f'features длина {len(feats)} != {BBOX_DIM}'}

    pil = Image.open(io.BytesIO(await image.read())).convert('RGB')
    img_t = infer_tf(pil).unsqueeze(0).to(DEVICE)

    bbox_t = (torch.tensor(feats, dtype=torch.float32) - BBOX_MEAN_T) / BBOX_STD_T
    bbox_t = bbox_t.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img_t, bbox_t)

    predictions = {}
    confidences = {}
    for cfg in CRITERIA_CONFIGS:
        n = cfg['name']
        sn = SAFE_COL_MAP[n]
        logits = out[sn][0]
        probs = F.softmax(logits, dim=-1)
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
        if cfg['task_info']['type'] == 'binary':
            predictions[n] = idx
        else:
            decoder = LABEL_DECODERS.get(n, {0: ''})
            predictions[n] = decoder.get(idx, '')
        confidences[n] = conf

    return {
        'predictions': predictions,
        'confidences': confidences,
        'model': MODEL_PATH.name,
    }

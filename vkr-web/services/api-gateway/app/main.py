from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.storage import (
    get_analysis,
    get_minio,
    list_history,
    new_id,
    put_image,
    save_analysis,
)

PREPROCESSING_URL = os.getenv('PREPROCESSING_URL', 'http://preprocessing:8001')
FEATURES_URL = os.getenv('FEATURES_URL', 'http://features:8002')
INFERENCE_URL = os.getenv('INFERENCE_URL', 'http://inference:8003')
INTERPRETATION_URL = os.getenv('INTERPRETATION_URL', 'http://interpretation:8004')
REPORT_URL = os.getenv('REPORT_URL', 'http://report:8005')

BASE_DIR = Path(__file__).parent.parent

app = FastAPI(title='vkr-web api-gateway')
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'static')), name='static')
templates = Jinja2Templates(directory=str(BASE_DIR / 'templates'))

HTTP_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('upload.html', {'request': request})


@app.get('/history', response_class=HTMLResponse)
async def history(request: Request):
    items = list_history(limit=50)
    return templates.TemplateResponse('history.html', {'request': request, 'items': items})


@app.get('/report/{analysis_id}', response_class=HTMLResponse)
async def report_view(analysis_id: str):
    a = get_analysis(analysis_id)
    if not a:
        return HTMLResponse('Не найдено', status_code=404)
    return HTMLResponse(a['portrait_html'])


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.post('/analyze', response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)):
    analysis_id = new_id()
    raw = await file.read()

    minio = get_minio()
    original_key = f'{analysis_id}/original.png'
    original_url = put_image(minio, original_key, raw, file.content_type or 'image/png')

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        files = {'file': (file.filename or 'image.png', raw, file.content_type or 'image/png')}
        prep_resp = await client.post(f'{PREPROCESSING_URL}/crop', files=files)
        prep_resp.raise_for_status()
        prep = prep_resp.json()

        cropped_bytes = base64.b64decode(prep['cropped_png_b64'])
        cropped_key = f'{analysis_id}/cropped.png'
        cropped_url = put_image(minio, cropped_key, cropped_bytes)

        feat_resp = await client.post(
            f'{FEATURES_URL}/compute',
            json={
                'bbox': prep['bbox'],
                'orig_w': prep['orig_w'],
                'orig_h': prep['orig_h'],
                'src': prep['src'],
            },
        )
        feat_resp.raise_for_status()
        feats = feat_resp.json()['features']

        inf_resp = await client.post(
            f'{INFERENCE_URL}/predict',
            files={'image': ('cropped.png', cropped_bytes, 'image/png')},
            data={'features': json.dumps(feats)},
        )
        inf_resp.raise_for_status()
        inf = inf_resp.json()
        predictions = inf['predictions']
        confidences = inf.get('confidences', {})
        model_name = inf.get('model', 'DrawingNet')

        interp_resp = await client.post(
            f'{INTERPRETATION_URL}/derive',
            json={'predictions': predictions, 'top_k': 7, 'min_support': 2},
        )
        interp_resp.raise_for_status()
        interp = interp_resp.json()

        active = [
            (crit, val, confidences.get(crit, 0.0))
            for crit, val in predictions.items()
            if val not in (0, None, '', '0')
        ]
        active.sort(key=lambda t: -t[2])
        top10 = active[:10]

        report_payload = {
            'analysis_id': analysis_id,
            'image_url': original_url,
            'cropped_url': cropped_url,
            'bbox_source': prep['src'],
            'model_name': model_name,
            'salient': interp['salient'],
            'weak': interp['weak'],
            'active_predictions_count': interp['active_predictions_count'],
            'total_predictions_count': interp['total_predictions_count'],
            'top_predictions': top10,
        }
        rep_resp = await client.post(f'{REPORT_URL}/render', json=report_payload)
        rep_resp.raise_for_status()
        portrait_html = rep_resp.text

    save_analysis(
        analysis_id=analysis_id,
        original_filename=file.filename,
        minio_original_key=original_key,
        minio_cropped_key=cropped_key,
        bbox_source=prep['src'],
        predictions=predictions,
        trait_support=interp['trait_support'],
        portrait_html=portrait_html,
    )

    return HTMLResponse(portrait_html)

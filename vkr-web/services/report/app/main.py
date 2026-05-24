from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel

TEMPLATES_DIR = Path(__file__).parent.parent / 'templates'

app = FastAPI(title='report')

jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(['html']),
)


class TraitItem(BaseModel):
    trait: str
    support: int
    description: str = ''
    evidence: list[tuple[str, Any]] = []


class RenderRequest(BaseModel):
    analysis_id: str | None = None
    image_url: str | None = None
    cropped_url: str | None = None
    bbox_source: str = 'contour'
    model_name: str = 'DrawingNet'
    salient: list[TraitItem] = []
    weak: list[TraitItem] = []
    active_predictions_count: int = 0
    total_predictions_count: int = 0
    top_predictions: list[tuple[str, Any, float]] = []


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/render', response_class=HTMLResponse)
def render(req: RenderRequest):
    tpl = jinja_env.get_template('portrait.html')
    return tpl.render(r=req)

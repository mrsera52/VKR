from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

DATA_DIR = Path('/data')
CRIT_JSON_PATH = DATA_DIR / 'criteria_interpretations.json'

app = FastAPI(title='interpretation')

if not CRIT_JSON_PATH.exists():
    raise RuntimeError(f'criteria_interpretations.json не найден в {DATA_DIR}')

CRIT_JSON = json.loads(CRIT_JSON_PATH.read_text(encoding='utf-8'))
CRIT_MAP = CRIT_JSON.get('criteria', {})
TRAITS_DESCR = CRIT_JSON.get('traits', {})
print(f'[interpretation] criteria: {len(CRIT_MAP)}, traits: {len(TRAITS_DESCR)}')


class DeriveRequest(BaseModel):
    predictions: dict[str, Any]
    top_k: int = 7
    min_support: int = 2


class EvidenceItem(BaseModel):
    criterion: str
    value: Any
    traits: list[str]


class TraitWithEvidence(BaseModel):
    trait: str
    support: int
    description: str
    evidence: list[tuple[str, Any]]


class DeriveResponse(BaseModel):
    trait_support: dict[str, int]
    salient: list[TraitWithEvidence]
    weak: list[TraitWithEvidence]
    evidence: list[EvidenceItem]
    active_predictions_count: int
    total_predictions_count: int


@app.get('/health')
def health():
    return {'status': 'ok', 'criteria': len(CRIT_MAP), 'traits': len(TRAITS_DESCR)}


@app.post('/derive', response_model=DeriveResponse)
def derive(req: DeriveRequest):
    trait_support: Counter = Counter()
    evidence: list[EvidenceItem] = []

    active_count = 0
    for crit_name, pred in req.predictions.items():
        if pred in (0, None, '', '0'):
            continue
        active_count += 1
        entry = CRIT_MAP.get(crit_name)
        if not entry:
            continue
        if entry.get('kind') == 'binary_presence':
            ts = entry.get('traits', [])
        else:
            ts = entry.get('class_traits', {}).get(str(pred), [])
        if ts:
            evidence.append(EvidenceItem(criterion=crit_name, value=pred, traits=ts))
            for t in ts:
                trait_support[t] += 1

    trait_evidence: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for item in evidence:
        for t in item.traits:
            trait_evidence[t].append((item.criterion, item.value))

    ranked = sorted(trait_support.items(), key=lambda x: -x[1])
    salient_traits = [
        TraitWithEvidence(
            trait=t, support=s,
            description=TRAITS_DESCR.get(t, ''),
            evidence=trait_evidence.get(t, [])[:4],
        )
        for t, s in ranked if s >= req.min_support
    ][:req.top_k]
    weak_traits = [
        TraitWithEvidence(
            trait=t, support=s,
            description=TRAITS_DESCR.get(t, ''),
            evidence=trait_evidence.get(t, [])[:3],
        )
        for t, s in ranked if s < req.min_support
    ][:5]

    return DeriveResponse(
        trait_support=dict(trait_support),
        salient=salient_traits,
        weak=weak_traits,
        evidence=evidence,
        active_predictions_count=active_count,
        total_predictions_count=len(req.predictions),
    )

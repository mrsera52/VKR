from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime
from typing import Any

import psycopg
from minio import Minio

MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS = os.getenv('MINIO_ROOT_USER', 'minioadmin')
MINIO_SECRET = os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'drawings')
MINIO_PUBLIC_BASE = os.getenv('MINIO_PUBLIC_BASE', 'http://localhost:9000')

PG_DSN = os.getenv(
    'PG_DSN',
    'postgresql://vkr:vkr_change_me@postgres:5432/vkr',
)


def get_minio() -> Minio:
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
    return client


def put_image(client: Minio, name: str, data: bytes, content_type: str = 'image/png') -> str:
    client.put_object(MINIO_BUCKET, name, io.BytesIO(data), length=len(data), content_type=content_type)
    return f'{MINIO_PUBLIC_BASE}/{MINIO_BUCKET}/{name}'


def save_analysis(
    analysis_id: str,
    original_filename: str | None,
    minio_original_key: str,
    minio_cropped_key: str,
    bbox_source: str,
    predictions: dict[str, Any],
    trait_support: dict[str, int],
    portrait_html: str,
) -> None:
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO analyses (
                id, created_at, original_filename,
                minio_original_key, minio_cropped_key,
                bbox_source, predictions, trait_support, portrait_html
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                analysis_id,
                datetime.utcnow(),
                original_filename,
                minio_original_key,
                minio_cropped_key,
                bbox_source,
                json.dumps(predictions, ensure_ascii=False),
                json.dumps(trait_support, ensure_ascii=False),
                portrait_html,
            ),
        )


def list_history(limit: int = 50) -> list[dict[str, Any]]:
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, created_at, original_filename, bbox_source, minio_cropped_key,
                   trait_support
            FROM analyses
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {
                'id': str(r[0]),
                'created_at': r[1].isoformat() if r[1] else None,
                'original_filename': r[2],
                'bbox_source': r[3],
                'cropped_url': f'{MINIO_PUBLIC_BASE}/{MINIO_BUCKET}/{r[4]}' if r[4] else None,
                'traits_count': len(r[5]) if isinstance(r[5], dict) else 0,
            }
            for r in rows
        ]


def get_analysis(analysis_id: str) -> dict[str, Any] | None:
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, created_at, original_filename,
                   minio_original_key, minio_cropped_key,
                   bbox_source, portrait_html
            FROM analyses WHERE id = %s
            """,
            (analysis_id,),
        )
        r = cur.fetchone()
        if not r:
            return None
        return {
            'id': str(r[0]),
            'created_at': r[1].isoformat() if r[1] else None,
            'original_filename': r[2],
            'original_url': f'{MINIO_PUBLIC_BASE}/{MINIO_BUCKET}/{r[3]}' if r[3] else None,
            'cropped_url': f'{MINIO_PUBLIC_BASE}/{MINIO_BUCKET}/{r[4]}' if r[4] else None,
            'bbox_source': r[5],
            'portrait_html': r[6],
        }


def new_id() -> str:
    return uuid.uuid4().hex

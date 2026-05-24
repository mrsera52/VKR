CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS analyses (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    original_filename   TEXT,
    minio_original_key  TEXT NOT NULL,
    minio_cropped_key   TEXT NOT NULL,
    bbox_source         TEXT,
    predictions         JSONB,
    trait_support       JSONB,
    portrait_html       TEXT
);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_bbox_source ON analyses (bbox_source);

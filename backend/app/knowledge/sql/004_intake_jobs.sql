-- Durable intake state; complete source/index snapshots remain immutable revisions.
CREATE TABLE knowledge.intake_jobs (
    job_id uuid PRIMARY KEY,
    collection_id uuid NOT NULL REFERENCES knowledge.knowledge_collections,
    identity_hash text NOT NULL CHECK (identity_hash ~ '^[0-9a-f]{64}$'),
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','processing','succeeded','failed')),
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    lease_token uuid,
    lease_until timestamptz,
    base_active_id uuid,
    base_revision_id uuid,
    base_captured boolean NOT NULL DEFAULT false,
    result_revision_id uuid,
    error_code text,
    duplicate boolean NOT NULL DEFAULT false,
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (collection_id,identity_hash),
    CHECK ((status='processing') = (lease_token IS NOT NULL AND lease_until IS NOT NULL)),
    CHECK (status <> 'succeeded' OR result_revision_id IS NOT NULL)
);
CREATE UNIQUE INDEX intake_one_processing_collection
    ON knowledge.intake_jobs(collection_id) WHERE status='processing';
CREATE INDEX intake_queue ON knowledge.intake_jobs(collection_id,created_at,job_id)
    WHERE status='queued';
-- Job payloads are management state, not student-readable knowledge.
REVOKE ALL ON knowledge.intake_jobs FROM fredi_knowledge_reader;

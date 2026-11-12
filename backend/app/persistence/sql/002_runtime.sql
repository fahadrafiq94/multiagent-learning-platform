-- Forward-only runtime integration; 001_application remains unchanged.
ALTER TABLE application.turns ADD COLUMN response_payload jsonb;
ALTER TABLE application.turns ADD CONSTRAINT response_payload_object
    CHECK (response_payload IS NULL OR jsonb_typeof(response_payload)='object');

-- A dedicated session advisory lock admits one HTTP worker per database.
-- This row fences transactions from an old worker after ownership changes.
CREATE TABLE application.runtime_owner (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    worker_id uuid NOT NULL,
    backend_pid integer NOT NULL,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
GRANT SELECT,INSERT,UPDATE ON application.runtime_owner TO fredi_app_runtime;

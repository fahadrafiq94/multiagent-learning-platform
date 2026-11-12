-- Retire semantic approval without changing 001 or discarding source annotations.
-- The migration runner applies the complete upgrade in one transaction.
UPDATE knowledge.knowledge_sections
SET payload = payload || jsonb_build_object(
    'legacy_annotations', COALESCE(payload->'legacy_annotations', '{}'::jsonb)
        || jsonb_build_object('semantic_type', semantic_type, 'review_status', review_status)
);
UPDATE knowledge.knowledge_chunks
SET payload = payload || jsonb_build_object(
    'legacy_annotations', COALESCE(payload->'legacy_annotations', '{}'::jsonb)
        || jsonb_build_object('semantic_type', semantic_type)
);

-- Remove only the old chunk-to-section FK. Other relationships remain intact.
DO $$
DECLARE old_fk record;
BEGIN
    FOR old_fk IN
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'knowledge.knowledge_chunks'::regclass
          AND confrelid = 'knowledge.knowledge_sections'::regclass
          AND contype = 'f'
    LOOP
        EXECUTE format('ALTER TABLE knowledge.knowledge_chunks DROP CONSTRAINT %I', old_fk.conname);
    END LOOP;
END $$;

-- PostgreSQL drops checks/indexes on these columns; no CASCADE is needed.
ALTER TABLE knowledge.knowledge_chunks
    DROP COLUMN semantic_type,
    DROP COLUMN approved_required;
ALTER TABLE knowledge.knowledge_sections
    DROP COLUMN semantic_type,
    DROP COLUMN review_status,
    ADD COLUMN processing_status text NOT NULL DEFAULT 'ready'
        CHECK (processing_status IN ('pending', 'ready', 'failed'));

UPDATE knowledge.knowledge_sections s SET processing_status = 'failed'
FROM knowledge.knowledge_documents d
WHERE s.revision_id = d.revision_id AND s.document_id = d.document_id
  AND d.payload->>'extraction_status' = 'failed';

ALTER TABLE knowledge.knowledge_sections ADD CONSTRAINT section_processing_identity
    UNIQUE (revision_id, document_id, section_id, processing_status);
ALTER TABLE knowledge.knowledge_chunks
    ADD COLUMN processing_required text NOT NULL DEFAULT 'ready'
        CHECK (processing_required = 'ready'),
    ADD CONSTRAINT chunk_ready_source FOREIGN KEY
        (revision_id, document_id, section_id, processing_required)
        REFERENCES knowledge.knowledge_sections
        (revision_id, document_id, section_id, processing_status),
    ADD CONSTRAINT authored_hint_shape CHECK (
        (kind = 'hint' AND hint_level IS NOT NULL AND hint_level BETWEEN 1 AND 4)
        OR (kind <> 'hint' AND hint_level IS NULL)
    );
CREATE INDEX chunk_revision_kind ON knowledge.knowledge_chunks (revision_id, kind);

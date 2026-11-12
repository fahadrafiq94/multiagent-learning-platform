ALTER TABLE knowledge.knowledge_revisions
    ADD COLUMN indexing_status text NOT NULL DEFAULT 'pending'
        CHECK (indexing_status IN ('pending','ready','failed')),
    ADD COLUMN indexing_error text CHECK (indexing_error ~ '^[a-z][a-z0-9_]{0,63}$'),
    ADD CONSTRAINT indexing_failure_code CHECK
        ((indexing_status='failed') = (indexing_error IS NOT NULL));

-- Language-neutral tokenization preserves identifiers across mixed-language courses.
ALTER TABLE knowledge.knowledge_chunks
    ADD COLUMN search_text text GENERATED ALWAYS AS
        (COALESCE(payload->>'context_prefix','') || E'\n' || COALESCE(payload->>'content','')) STORED,
    ADD COLUMN search_vector tsvector GENERATED ALWAYS AS
        (to_tsvector('simple'::regconfig,
            COALESCE(payload->>'context_prefix','') || E'\n' || COALESCE(payload->>'content',''))) STORED;
CREATE INDEX chunk_full_text ON knowledge.knowledge_chunks USING gin(search_vector);
-- pgvector's vector HNSW indexes support up to 2000 dimensions. Larger profiles
-- retain exact vector search; this deployment's measured profile is 768 dimensions.
DO $$ BEGIN
    IF __DIMENSION__ <= 2000 THEN
        CREATE INDEX chunk_cosine ON knowledge.knowledge_chunks
            USING hnsw(embedding vector_cosine_ops) WHERE kind='evidence';
    END IF;
END $$;

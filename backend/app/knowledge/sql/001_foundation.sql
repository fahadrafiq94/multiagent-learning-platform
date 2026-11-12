CREATE TABLE knowledge.knowledge_collections (
    collection_id uuid PRIMARY KEY,
    access_domain text NOT NULL,
    sharing_class text NOT NULL CHECK (sharing_class IN ('shared_instructional', 'owner_private')),
    owner_user_id uuid,
    active_revision_id uuid,
    accepted_report jsonb,
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    CHECK ((sharing_class = 'owner_private') = (owner_user_id IS NOT NULL))
);
CREATE TABLE knowledge.knowledge_revisions (
    revision_id uuid PRIMARY KEY,
    collection_id uuid NOT NULL REFERENCES knowledge.knowledge_collections,
    fingerprint text NOT NULL CHECK (fingerprint ~ '^[0-9a-f]{64}$'),
    embedding_dimension integer NOT NULL CHECK (embedding_dimension = __DIMENSION__),
    version integer NOT NULL DEFAULT 0 CHECK (version >= 0),
    status text NOT NULL DEFAULT 'staging' CHECK (status IN ('staging', 'validated')),
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    UNIQUE (collection_id, revision_id),
    UNIQUE (collection_id, fingerprint)
);
ALTER TABLE knowledge.knowledge_collections ADD CONSTRAINT active_revision_belongs_to_collection
    FOREIGN KEY (collection_id, active_revision_id)
    REFERENCES knowledge.knowledge_revisions (collection_id, revision_id)
    DEFERRABLE INITIALLY DEFERRED;
CREATE TABLE knowledge.knowledge_documents (
    revision_id uuid NOT NULL REFERENCES knowledge.knowledge_revisions,
    document_id uuid NOT NULL,
    source_hash text NOT NULL CHECK (source_hash ~ '^[0-9a-f]{64}$'),
    workflow_authority boolean NOT NULL DEFAULT false,
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    PRIMARY KEY (revision_id, document_id),
    UNIQUE (revision_id, document_id, workflow_authority)
);
CREATE TABLE knowledge.knowledge_sections (
    revision_id uuid NOT NULL,
    document_id uuid NOT NULL,
    section_id uuid NOT NULL,
    semantic_type text NOT NULL CHECK (semantic_type IN (
        'process_reasoning', 'applus_execution', 'shared_process_metadata', 'scenario_reference', 'unknown')),
    review_status text NOT NULL CHECK (review_status IN ('pending', 'needs_review', 'approved')),
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    PRIMARY KEY (revision_id, section_id),
    FOREIGN KEY (revision_id, document_id) REFERENCES knowledge.knowledge_documents,
    UNIQUE (revision_id, document_id, section_id, semantic_type, review_status),
    CHECK (review_status <> 'approved' OR semantic_type <> 'unknown')
);
CREATE TABLE knowledge.process_steps (
    revision_id uuid NOT NULL,
    namespace text NOT NULL,
    step_id text NOT NULL,
    source_document_id uuid NOT NULL,
    authority_required boolean NOT NULL DEFAULT true CHECK (authority_required),
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    PRIMARY KEY (revision_id, namespace, step_id),
    FOREIGN KEY (revision_id, source_document_id, authority_required)
        REFERENCES knowledge.knowledge_documents (revision_id, document_id, workflow_authority)
);
CREATE TABLE knowledge.knowledge_chunks (
    revision_id uuid NOT NULL,
    document_id uuid NOT NULL,
    section_id uuid NOT NULL,
    chunk_id uuid NOT NULL,
    kind text NOT NULL CHECK (kind IN ('evidence', 'support', 'hint')),
    semantic_type text NOT NULL CHECK (semantic_type IN (
        'process_reasoning', 'applus_execution', 'scenario_reference')),
    approved_required text NOT NULL DEFAULT 'approved' CHECK (approved_required = 'approved'),
    namespace text,
    step_id text,
    hint_level integer,
    content_hash text NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    embedding vector(__DIMENSION__),
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    PRIMARY KEY (revision_id, chunk_id),
    FOREIGN KEY (revision_id, document_id, section_id, semantic_type, approved_required)
        REFERENCES knowledge.knowledge_sections
            (revision_id, document_id, section_id, semantic_type, review_status),
    FOREIGN KEY (revision_id, namespace, step_id) REFERENCES knowledge.process_steps,
    CHECK ((namespace IS NULL) = (step_id IS NULL)),
    CHECK ((kind = 'evidence') = (embedding IS NOT NULL)),
    CHECK (embedding IS NULL OR vector_norm(embedding) > 0),
    CHECK ((kind = 'hint' AND hint_level IS NOT NULL AND hint_level BETWEEN 1 AND 4 AND semantic_type = 'process_reasoning'
            AND step_id IS NOT NULL) OR (kind <> 'hint' AND hint_level IS NULL))
);
CREATE INDEX collection_access ON knowledge.knowledge_collections (access_domain, owner_user_id);
CREATE INDEX chunk_eligibility ON knowledge.knowledge_chunks (revision_id, semantic_type, kind);
CREATE INDEX chunk_step ON knowledge.knowledge_chunks (revision_id, namespace, step_id);

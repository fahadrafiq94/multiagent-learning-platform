"""Real read-role PostgreSQL retrieval: ranking, provenance and isolation."""

import hashlib
import json
from uuid import uuid4

import pytest

from app.knowledge.errors import KnowledgeIntegrityError, KnowledgeNotFound
from app.knowledge.indexer import IndexConfig, index_revision
from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.retrieval_schemas import MetadataRequest, RetrievalConfig, RetrievalRequest
from app.knowledge.schemas import (
    AccessScope,
    KnowledgeCollection,
    KnowledgeRevision,
    KnowledgeWriteScope,
    PipelineVersions,
    RevisionData,
    SharingClass,
)
from app.knowledge.search_repository import PostgresSearchRepository
from app.services.model_service.exceptions import ModelEmbeddingError
from app.services.model_service.schemas import ModelEmbeddingResponse
from tests.knowledge.test_chunking_metadata import METADATA_TEXT, source
from tests.knowledge.test_indexer import PROFILE, Embeddings

pytestmark = pytest.mark.asyncio
DOMAIN = "course-test"


class QueryModel(Embeddings):
    def __init__(self, vector=None, *, failure=None, hook=None):
        super().__init__(failure=failure)
        self.vector = vector or [0.2, 0.3, 0.4]
        self.hook = hook

    async def embed(self, request):
        self.requests.append(request)
        if self.hook:
            await self.hook()
        if self.failure in ("provider", "too_long"):
            raise ModelEmbeddingError(
                "ollama",
                details={"private-marker": True, "input_too_long": self.failure == "too_long"},
            )
        vectors = [self.vector for _ in request.texts]
        if self.failure == "count":
            vectors = []
        if self.failure == "dimension":
            vectors = [[1, 2]]
        if self.failure == "nan":
            vectors = [[float("nan"), 0, 0]]
        if self.failure == "zero":
            vectors = [[0, 0, 0]]
        return ModelEmbeddingResponse(
            embeddings=vectors, provider="ollama", model=request.model, latency_ms=1
        )


async def corpus(
    pg,
    *,
    text=METADATA_TEXT,
    authority=True,
    owner=None,
    domain=DOMAIN,
    collection=None,
    vector=None,
    pages=None,
):
    scope = KnowledgeWriteScope(access_domain=domain, user_id=owner, can_manage_knowledge=True)
    collection = collection or KnowledgeCollection(
        name="Test corpus",
        access_domain=domain,
        sharing_class=SharingClass.PRIVATE if owner else SharingClass.SHARED,
        owner_user_id=owner,
    )
    if await pg.writer.get_collection(collection.collection_id, scope) is None:
        await pg.writer.create_collection(collection, scope=scope)
    document, sections = source(pages or [text], authority=authority)
    revision = KnowledgeRevision(
        revision_id=document.revision_id,
        collection_id=collection.collection_id,
        embedding=PROFILE,
        pipeline=PipelineVersions(extractor="retrieval-test", chunker="not-run", policy="general"),
        manifest={document.document_id: document.source_hash},
    )
    await pg.writer.create_revision(revision, scope=scope)
    await pg.writer.write_revision_data(
        revision.revision_id,
        RevisionData(documents=[document], sections=sections),
        scope=scope,
        expected_version=0,
    )
    result = await index_revision(
        revision.revision_id,
        config=IndexConfig(embedding=PROFILE),
        repository=pg.writer,
        model_service=QueryModel(vector),
        scope=scope,
    )
    assert result.status == "ready"
    return collection, result.revision_id, revision.revision_id, scope


def search_request(collection, revision, query="Create order", **kwargs):
    return RetrievalRequest(
        collection_id=collection.collection_id, revision_id=revision, query=query, **kwargs
    )


async def test_sql_hybrid_retrieval_soft_preferences_and_source_citations(postgres_fixture):
    pg = postgres_fixture
    collection, revision, _, scope = await corpus(pg)
    model = QueryModel()
    service = KnowledgeRetrievalService(PostgresSearchRepository(pg.reader_database), model)
    result = await service.search(
        search_request(
            collection,
            revision,
            current_step_id="FLOW-1",
            phase="Planning",
            purpose="business reasoning",
        ),
        scope=scope,
    )
    assert result.status == "found", result.model_dump_json()
    assert result.items[0].reference_id == "FLOW-2"
    assert result.stats.vector_candidates and result.stats.lexical_candidates
    assert result.revision_id == revision and result.sources
    assert model.requests[0].texts == [PROFILE.query_prefix + "Create order"]
    assert model.requests[0].truncate is False
    assert all(item.citations[0].revision_id == revision for item in result.items)
    assert all(item.citations[0].page_spans for item in result.items)
    assert all("embedding" not in item.model_dump() for item in result.items)


async def test_exact_identifiers_cross_step_queries_and_unknown_ids(postgres_fixture):
    collection, revision, _, scope = await corpus(postgres_fixture)
    service = KnowledgeRetrievalService(
        PostgresSearchRepository(postgres_fixture.reader_database), QueryModel()
    )
    exact = await service.search(
        search_request(collection, revision, "flow-2", current_step_id="FLOW-1"), scope=scope
    )
    assert exact.status == "found" and {item.reference_id for item in exact.items} == {"FLOW-2"}
    cross = await service.search(
        search_request(
            collection, revision, "Compare FLOW-1 and FLOW-2", top_k=2, current_step_id="FLOW-99"
        ),
        scope=scope,
    )
    assert {item.reference_id for item in cross.items} == {"FLOW-1", "FLOW-2"}
    unknown = await service.search(search_request(collection, revision, "FLOW-99999"), scope=scope)
    assert unknown.status == "no_evidence" and not unknown.items


async def test_metadata_lookup_uses_exact_source_fields_without_model_calls(postgres_fixture):
    collection, revision, _, scope = await corpus(postgres_fixture)
    service = KnowledgeRetrievalService(PostgresSearchRepository(postgres_fixture.reader_database))
    request = MetadataRequest(
        collection_id=collection.collection_id,
        revision_id=revision,
        step_id="flow-1",
        fields=["name", "next_step_id"],
    )
    result = await service.lookup(request, scope=scope)
    assert result.status == "found" and result.stats.model_calls == 0
    assert {item.field_name: item.content for item in result.items} == {
        "name": "Prepare data",
        "next_step_id": "FLOW-2",
    }
    assert all(item.kind == "metadata" and item.citations[0].field_name for item in result.items)
    assert all(item.citations[0].chunk_id is None for item in result.items)
    missing = await service.lookup(request.model_copy(update={"step_id": "unknown"}), scope=scope)
    assert missing.status == "no_evidence" and missing.reason == "metadata_missing"
    bounded = await service.lookup(request.model_copy(update={"max_chars": 1}), scope=scope)
    assert bounded.status == "no_evidence" and bounded.reason == "budget_exhausted"
    assert bounded.stats.omitted_fields == ["name", "next_step_id"]


async def test_wrong_course_owner_unready_and_unpublished_are_unavailable_before_inference(
    postgres_fixture,
):
    owner = uuid4()
    collection, revision, source_revision, scope = await corpus(postgres_fixture, owner=owner)
    model = QueryModel()
    service = KnowledgeRetrievalService(
        PostgresSearchRepository(postgres_fixture.reader_database), model
    )
    request = search_request(collection, revision)
    for denied in [
        AccessScope(access_domain=DOMAIN),
        AccessScope(access_domain=DOMAIN, user_id=uuid4()),
        AccessScope(access_domain="other", user_id=owner),
    ]:
        result = await service.search(request, scope=denied)
        assert result.status == "unavailable" and not result.items and not result.sources
        metadata = await service.lookup(
            MetadataRequest(
                collection_id=collection.collection_id, revision_id=revision, step_id="FLOW-1"
            ),
            scope=denied,
        )
        assert metadata.status == "unavailable"
    for target in [source_revision, None, uuid4()]:
        result = await service.search(
            request.model_copy(update={"revision_id": target}), scope=scope
        )
        assert result.status == "unavailable"
    # This principal can read both collections, but a revision cannot be admitted
    # under a different collection's identity.
    other_collection, _, _, _ = await corpus(postgres_fixture)
    mismatch = await service.search(
        request.model_copy(update={"collection_id": other_collection.collection_id}), scope=scope
    )
    assert mismatch.status == "unavailable" and not mismatch.items
    assert model.requests == [] and model.inspections == 0


async def test_scope_filter_precedes_vector_limit_and_cannot_hydrate_foreign_ids(postgres_fixture):
    pg = postgres_fixture
    allowed, revision, _, scope = await corpus(
        pg, text="Allowed purchase order evidence.", authority=False, vector=[0.8, 0.6, 0]
    )
    private, private_revision, _, private_scope = await corpus(
        pg, owner=uuid4(), text="Private canary purchase order.", authority=False, vector=[1, 0, 0]
    )
    repository = PostgresSearchRepository(pg.reader_database)
    service = KnowledgeRetrievalService(
        repository, QueryModel([1, 0, 0]), RetrievalConfig(candidate_limit=1)
    )
    result = await service.search(
        search_request(allowed, revision, "purchase order", top_k=1), scope=scope
    )
    assert result.status == "found" and "Allowed" in result.items[0].content
    assert "Private" not in result.model_dump_json()
    private_data = await pg.writer.get_revision_data(private_revision, private_scope)
    pin = await repository.admit(allowed.collection_id, revision, scope)
    with pytest.raises(KnowledgeIntegrityError):
        await repository.evidence(pin, scope, [private_data.chunks[0].chunk_id])
    with pytest.raises(KnowledgeNotFound):
        await repository.candidates(
            pin,
            AccessScope(access_domain="other"),
            search_request(allowed, revision),
            [1, 0, 0],
            [],
            1,
        )


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("provider", "embedding_unavailable"),
        ("too_long", "query_too_long"),
        ("digest", "embedding_profile_mismatch"),
        ("count", "invalid_query_embedding"),
        ("dimension", "invalid_query_embedding"),
        ("zero", "invalid_query_embedding"),
        ("nan", "invalid_query_embedding"),
    ],
)
async def test_embedding_errors_are_distinct_from_no_evidence_and_sanitized(
    postgres_fixture, failure, reason
):
    collection, revision, _, scope = await corpus(postgres_fixture)
    service = KnowledgeRetrievalService(
        PostgresSearchRepository(postgres_fixture.reader_database), QueryModel(failure=failure)
    )
    result = await service.search(search_request(collection, revision), scope=scope)
    assert result.status == "error" and result.reason == reason and not result.items
    assert "private-marker" not in result.model_dump_json()


async def test_no_evidence_budget_limits_and_generic_documents_need_no_step_metadata(
    postgres_fixture,
):
    collection, revision, _, scope = await corpus(
        postgres_fixture, text="General invoice policy.", authority=False
    )
    repository = PostgresSearchRepository(postgres_fixture.reader_database)
    service = KnowledgeRetrievalService(repository, QueryModel())
    result = await service.search(
        search_request(collection, revision, "invoice", current_step_id="missing", phase="other"),
        scope=scope,
    )
    assert result.status == "found" and result.items[0].reference_id is None
    bounded = await service.search(
        search_request(collection, revision, "invoice", max_chars=1), scope=scope
    )
    assert bounded.status == "no_evidence" and bounded.reason == "budget_exhausted"
    unrelated = await KnowledgeRetrievalService(repository, QueryModel([-0.3, 0.2, 0])).search(
        search_request(collection, revision, "galactic unicorn"), scope=scope
    )
    assert unrelated.status == "no_evidence"


async def test_pinned_revision_survives_active_change_and_revocation_rechecks_scope(
    postgres_fixture,
):
    pg = postgres_fixture
    collection, first, _, scope = await corpus(pg)
    _, second, _, _ = await corpus(
        pg, collection=collection, text=METADATA_TEXT + "\nOther source words"
    )
    async with pg.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.knowledge_collections SET active_revision_id=%s "
            "WHERE collection_id=%s",
            (first, collection.collection_id),
        )

    async def change_active():
        async with pg.writer_database.connection() as connection:
            await connection.execute(
                "UPDATE knowledge.knowledge_collections SET active_revision_id=%s "
                "WHERE collection_id=%s",
                (second, collection.collection_id),
            )

    repository = PostgresSearchRepository(pg.reader_database)
    result = await KnowledgeRetrievalService(repository, QueryModel(hook=change_active)).search(
        search_request(collection, None), scope=scope
    )
    assert result.status == "found" and result.revision_id == first
    assert all(c.revision_id == first for item in result.items for c in item.citations)

    async def revoke():
        async with pg.writer_database.connection() as connection:
            await connection.execute(
                "UPDATE knowledge.knowledge_collections SET access_domain='revoked' "
                "WHERE collection_id=%s",
                (collection.collection_id,),
            )

    denied = await KnowledgeRetrievalService(repository, QueryModel(hook=revoke)).search(
        search_request(collection, second), scope=scope
    )
    assert denied.status == "unavailable" and not denied.items


async def test_corrupt_source_proof_is_error_and_query_text_cannot_inject_sql(postgres_fixture):
    pg = postgres_fixture
    collection, revision, _, scope = await corpus(pg)
    service = KnowledgeRetrievalService(PostgresSearchRepository(pg.reader_database), QueryModel())
    injected = await service.search(
        search_request(collection, revision, "x'); DELETE FROM knowledge.knowledge_chunks; --"),
        scope=scope,
    )
    assert injected.status in ("found", "no_evidence")
    original = await pg.writer.get_revision_data(revision, scope)
    async with pg.writer_database.connection() as connection:
        for chunk in original.chunks:
            content = "forged evidence"
            input_hash = hashlib.sha256(
                (PROFILE.document_prefix + chunk.context_prefix + "\n" + content).encode()
            ).hexdigest()
            await connection.execute(
                "UPDATE knowledge.knowledge_chunks SET payload="
                "jsonb_set(jsonb_set(payload,'{content}',%s::jsonb),"
                "'{embedding_input_hash}',to_jsonb(%s::text)),content_hash=%s "
                "WHERE revision_id=%s AND chunk_id=%s",
                (
                    json.dumps(content),
                    input_hash,
                    hashlib.sha256(content.encode()).hexdigest(),
                    revision,
                    chunk.chunk_id,
                ),
            )
    corrupt = await service.search(search_request(collection, revision, "FLOW-1"), scope=scope)
    assert (
        corrupt.status == "error"
        and corrupt.reason == "invalid_stored_evidence"
        and not corrupt.items
    )
    async with pg.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.process_steps SET payload="
            "jsonb_set(payload,'{source_fields}','{}'::jsonb) WHERE revision_id=%s",
            (revision,),
        )
    corrupt_metadata = await service.lookup(
        MetadataRequest(
            collection_id=collection.collection_id,
            revision_id=revision,
            step_id="FLOW-1",
        ),
        scope=scope,
    )
    assert (
        corrupt_metadata.status == "error" and corrupt_metadata.reason == "invalid_stored_evidence"
    )


async def test_database_outage_is_error_instead_of_no_evidence(postgres_fixture):
    collection, revision, _, scope = await corpus(postgres_fixture)
    repository = PostgresSearchRepository(postgres_fixture.reader_database)
    await postgres_fixture.reader_database.close()
    result = await KnowledgeRetrievalService(repository, QueryModel()).search(
        search_request(collection, revision), scope=scope
    )
    assert result.status == "error" and result.reason == "knowledge_unavailable"


async def test_metadata_ambiguity_requires_namespace_without_guessing(postgres_fixture):
    pg = postgres_fixture
    collection, revision, _, scope = await corpus(pg)
    async with pg.writer_database.connection() as connection:
        await connection.execute(
            "INSERT INTO knowledge.process_steps "
            "SELECT revision_id,'alternate',step_id,source_document_id,authority_required,payload "
            "FROM knowledge.process_steps WHERE revision_id=%s AND step_id='FLOW-1'",
            (revision,),
        )
    service = KnowledgeRetrievalService(PostgresSearchRepository(pg.reader_database))
    request = MetadataRequest(
        collection_id=collection.collection_id, revision_id=revision, step_id="FLOW-1"
    )
    ambiguous = await service.lookup(request, scope=scope)
    assert ambiguous.status == "no_evidence" and ambiguous.reason == "metadata_ambiguous"
    resolved = await service.lookup(
        request.model_copy(update={"namespace": "alternate"}), scope=scope
    )
    assert resolved.status == "found" and all(i.namespace == "alternate" for i in resolved.items)


async def test_cross_page_evidence_and_metadata_keep_exact_source_spans(postgres_fixture):
    collection, revision, _, scope = await corpus(
        postgres_fixture,
        pages=[
            "Step ID\nFLOW-1\nStep Name\nCreate cross-page",
            "verification\nPhase\nOperations\n",
        ],
    )
    service = KnowledgeRetrievalService(
        PostgresSearchRepository(postgres_fixture.reader_database), QueryModel()
    )
    result = await service.search(search_request(collection, revision, "FLOW-1"), scope=scope)
    assert result.status == "found" and result.items[0].citations[0].page_end == 2
    metadata = await service.lookup(
        MetadataRequest(
            collection_id=collection.collection_id,
            revision_id=revision,
            step_id="FLOW-1",
            fields=["name"],
        ),
        scope=scope,
    )
    assert (
        metadata.status == "found"
        and metadata.items[0].content == "Create cross-page\nverification"
    )
    assert [span.page for span in metadata.items[0].citations[0].page_spans] == [1, 2]

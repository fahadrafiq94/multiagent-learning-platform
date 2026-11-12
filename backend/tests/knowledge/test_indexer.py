import asyncio
from uuid import uuid4

import pytest

from app.knowledge.chunking import ChunkingConfig
from app.knowledge.errors import (
    KnowledgeConflict,
    KnowledgeIntegrityError,
    KnowledgeNotFound,
    KnowledgePermissionError,
)
from app.knowledge.index_validation import validate_indexed_data
from app.knowledge.indexer import IndexConfig, index_revision
from app.knowledge.schemas import (
    EmbeddingProfile,
    KnowledgeCollection,
    KnowledgeRevision,
    KnowledgeWriteScope,
    PipelineVersions,
    ProcessingStatus,
    RevisionData,
)
from app.services.model_service import ModelService
from app.services.model_service.exceptions import ModelEmbeddingError
from app.services.model_service.schemas import EmbeddingModelInfo, ModelEmbeddingResponse
from tests.knowledge.test_chunking_metadata import METADATA_TEXT, source

pytestmark = pytest.mark.asyncio
SCOPE = KnowledgeWriteScope(access_domain="course-test", can_manage_knowledge=True)
PROFILE = EmbeddingProfile(
    provider="ollama",
    model="fixed",
    revision="test-digest",
    dimension=3,
    document_prefix="document: ",
    query_prefix="query: ",
    normalized=True,
)


class Embeddings(ModelService):
    def __init__(self, *, failure=None, limit=None):
        self.requests = []
        self.inspections = 0
        self.failure = failure
        self.limit = limit

    async def inspect_embedding_model(self, model):
        self.inspections += 1
        return EmbeddingModelInfo(
            provider="ollama",
            model=model,
            revision="wrong" if self.failure == "digest" else PROFILE.revision,
            dimension=3,
            context_length=512,
        )

    async def embed(self, request):
        await asyncio.sleep(0)
        self.requests.append(request)
        if self.failure == "provider":
            raise ModelEmbeddingError("ollama", details={"error": "private-source-marker"})
        if self.limit and any(len(t) > self.limit for t in request.texts):
            raise ModelEmbeddingError("ollama", details={"input_too_long": True})
        vectors = [[0.2, 0.3, 0.4] for _ in request.texts]
        if self.failure == "count":
            vectors = []
        elif self.failure in ("zero", "nan", "dimension"):
            vector = {"zero": [0, 0, 0], "nan": [float("nan"), 1, 1], "dimension": [1, 2]}[
                self.failure
            ]
            vectors = [vector for _ in request.texts]
        return ModelEmbeddingResponse(
            embeddings=vectors, model=request.model, provider="ollama", latency_ms=1
        )


async def extracted_revision(repository, *, authority=False, text=None, collection=None):
    collection = collection or KnowledgeCollection(
        name="General sources", access_domain=SCOPE.access_domain
    )
    if await repository.get_collection(collection.collection_id, SCOPE) is None:
        await repository.create_collection(collection, scope=SCOPE)
    document, sections = source(
        [text or "# General manual\n\n" + "Only proceed after validation. " * 40],
        authority=authority,
    )
    revision = KnowledgeRevision(
        revision_id=document.revision_id,
        collection_id=collection.collection_id,
        embedding=PROFILE,
        pipeline=PipelineVersions(extractor="synthetic-v1", chunker="not-run", policy="general"),
        manifest={document.document_id: document.source_hash},
    )
    await repository.create_revision(revision, scope=SCOPE)
    await repository.write_revision_data(
        revision.revision_id,
        RevisionData(documents=[document], sections=sections),
        scope=SCOPE,
        expected_version=0,
    )
    return collection, revision


async def test_generic_index_round_trip_reuse_and_immutable_completion(repository):
    _, source_revision = await extracted_revision(repository)
    original = await repository.get_revision_data(source_revision.revision_id, SCOPE)
    model = Embeddings()
    config = IndexConfig(embedding=PROFILE)
    result = await index_revision(
        source_revision.revision_id,
        config=config,
        repository=repository,
        model_service=model,
        scope=SCOPE,
    )
    assert result.status == ProcessingStatus.READY and not result.reused and result.chunks > 0
    assert all(request.truncate is False for request in model.requests)
    assert all(t.startswith(PROFILE.document_prefix) for r in model.requests for t in r.texts)
    revision = await repository.get_revision(result.revision_id, SCOPE)
    data = await repository.get_revision_data(result.revision_id, SCOPE)
    validate_indexed_data(revision, data)
    assert data.steps == [] and all(c.step_id is None for c in data.chunks)
    assert await repository.get_revision_data(source_revision.revision_id, SCOPE) == original
    requests = len(model.requests)
    again = await index_revision(
        source_revision.revision_id,
        config=config.model_copy(update={"batch_size": 1}),
        repository=repository,
        model_service=model,
        scope=SCOPE,
    )
    assert again.reused and again.revision_id == result.revision_id
    assert len(model.requests) == requests and model.inspections == 2
    with pytest.raises(KnowledgeConflict):
        await repository.write_revision_data(
            result.revision_id, data, scope=SCOPE, expected_version=revision.version
        )
    with pytest.raises(KnowledgeConflict):
        await repository.record_index_failure(
            result.revision_id,
            scope=SCOPE,
            expected_version=revision.version,
            error_code="too_late",
        )


async def test_canonical_step_metadata_is_source_linked_but_not_an_indexing_gate(repository):
    _, source_revision = await extracted_revision(repository, text=METADATA_TEXT, authority=True)
    result = await index_revision(
        source_revision.revision_id,
        config=IndexConfig(embedding=PROFILE),
        repository=repository,
        model_service=Embeddings(),
        scope=SCOPE,
    )
    assert result.status == "ready" and result.steps == 2 and result.diagnostics == 1
    data = await repository.get_revision_data(result.revision_id, SCOPE)
    assert all(step.source_fields for step in data.steps)
    assert {c.step_id for c in data.chunks} == {"FLOW-1", "FLOW-2"}


@pytest.mark.parametrize("failure", ["provider", "count", "zero", "nan", "dimension", "digest"])
async def test_embedding_failures_are_explicit_atomic_and_retryable(repository, failure):
    _, source_revision = await extracted_revision(repository)
    config = IndexConfig(embedding=PROFILE)
    failed = await index_revision(
        source_revision.revision_id,
        config=config,
        repository=repository,
        model_service=Embeddings(failure=failure),
        scope=SCOPE,
    )
    assert failed.status == "failed" and failed.error_code
    assert "private" not in failed.model_dump_json()
    stored = await repository.get_revision_data(failed.revision_id, SCOPE)
    assert stored.documents and stored.sections and stored.chunks == []
    retry = await index_revision(
        source_revision.revision_id,
        config=config,
        repository=repository,
        model_service=Embeddings(),
        scope=SCOPE,
    )
    assert retry.status == "ready" and retry.revision_id == failed.revision_id


async def test_context_limit_resplits_exact_source_instead_of_truncating(repository):
    _, source_revision = await extracted_revision(repository)
    config = IndexConfig(
        embedding=PROFILE, chunking=ChunkingConfig(max_chars=256, overlap_chars=32)
    )
    model = Embeddings(limit=230)
    result = await index_revision(
        source_revision.revision_id,
        config=config,
        repository=repository,
        model_service=model,
        scope=SCOPE,
    )
    assert result.status == "ready" and result.chunks > 6
    data = await repository.get_revision_data(result.revision_id, SCOPE)
    validate_indexed_data(await repository.get_revision(result.revision_id, SCOPE), data)
    assert all(
        len(PROFILE.document_prefix + c.context_prefix + "\n" + c.content) <= 230
        for c in data.chunks
    )


async def test_concurrent_duplicate_indexes_converge_without_duplicate_records(repository):
    _, source_revision = await extracted_revision(repository)
    results = await asyncio.gather(
        *(
            index_revision(
                source_revision.revision_id,
                config=IndexConfig(embedding=PROFILE),
                repository=repository,
                model_service=Embeddings(),
                scope=SCOPE,
            )
            for _ in range(3)
        )
    )
    assert len({r.revision_id for r in results}) == 1 and all(r.status == "ready" for r in results)
    assert sum(not r.reused for r in results) == 1


async def test_config_or_source_change_creates_distinct_revision_and_preserves_old_index(
    repository,
):
    collection, source_revision = await extracted_revision(repository)
    config = IndexConfig(embedding=PROFILE)
    first = await index_revision(
        source_revision.revision_id,
        config=config,
        repository=repository,
        model_service=Embeddings(),
        scope=SCOPE,
    )
    second = await index_revision(
        source_revision.revision_id,
        config=config.model_copy(
            update={
                "chunking": ChunkingConfig(max_chars=500, overlap_chars=60),
            }
        ),
        repository=repository,
        model_service=Embeddings(),
        scope=SCOPE,
    )
    _, changed = await extracted_revision(
        repository, collection=collection, text="Updated course rules."
    )
    third = await index_revision(
        changed.revision_id,
        config=config,
        repository=repository,
        model_service=Embeddings(),
        scope=SCOPE,
    )
    assert len({first.revision_id, second.revision_id, third.revision_id}) == 3
    assert await repository.get_revision_data(first.revision_id, SCOPE)


async def test_index_scope_checks_and_completeness_validation(repository):
    _, source_revision = await extracted_revision(repository)
    with pytest.raises(KnowledgePermissionError):
        await index_revision(
            source_revision.revision_id,
            config=IndexConfig(embedding=PROFILE),
            repository=repository,
            model_service=Embeddings(),
            scope=SCOPE.model_copy(update={"can_manage_knowledge": False}),
        )
    with pytest.raises(KnowledgeNotFound):
        await index_revision(
            source_revision.revision_id,
            config=IndexConfig(embedding=PROFILE),
            repository=repository,
            model_service=Embeddings(),
            scope=SCOPE.model_copy(update={"access_domain": "other"}),
        )
    result = await index_revision(
        source_revision.revision_id,
        config=IndexConfig(embedding=PROFILE),
        repository=repository,
        model_service=Embeddings(),
        scope=SCOPE,
    )
    revision = await repository.get_revision(result.revision_id, SCOPE)
    data = await repository.get_revision_data(result.revision_id, SCOPE)
    corrupted = data.model_copy(update={"chunks": data.chunks[:-1]})
    with pytest.raises(KnowledgeIntegrityError):
        validate_indexed_data(revision, corrupted)
    bad_vector = data.chunks[0].model_copy(update={"embedding": [2.0, 0.0, 0.0]})
    with pytest.raises(KnowledgeIntegrityError, match="normalization"):
        validate_indexed_data(
            revision, data.model_copy(update={"chunks": [bad_vector, *data.chunks[1:]]})
        )


async def test_new_revision_cannot_claim_index_completion_before_processing(repository):
    _, source_revision = await extracted_revision(repository)
    for status, code in [
        (ProcessingStatus.READY, None),
        (ProcessingStatus.FAILED, "fake_failure"),
        (ProcessingStatus.PENDING, "fake_failure"),
    ]:
        candidate = source_revision.model_copy(
            update={
                "revision_id": uuid4(),
                "indexing_status": status,
                "indexing_error": code,
                "pipeline": source_revision.pipeline.model_copy(update={"policy": uuid4().hex}),
            }
        )
        with pytest.raises(KnowledgeConflict):
            await repository.create_revision(candidate, scope=SCOPE)

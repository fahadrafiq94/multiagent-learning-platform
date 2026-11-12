"""Real PostgreSQL, real PDF extraction, durable jobs and atomic publication."""

import asyncio
from uuid import uuid4

import pytest

from app.knowledge.errors import (
    KnowledgeConflict,
    KnowledgeIntegrityError,
    KnowledgePermissionError,
)
from app.knowledge.extraction import PdfExtractor, failure
from app.knowledge.indexer import IndexConfig
from app.knowledge.intake import IntakeService
from app.knowledge.intake_repository import IntakeRepository
from app.knowledge.intake_schemas import IntakeBinding, IntakeConfig
from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.retrieval_schemas import RetrievalRequest
from app.knowledge.schemas import KnowledgeCollection
from app.knowledge.search_repository import PostgresSearchRepository
from app.knowledge.source_store import FileSourceStore
from tests.knowledge.test_extraction import pdf_bytes
from tests.knowledge.test_indexer import PROFILE, SCOPE, Embeddings

pytestmark = pytest.mark.asyncio


async def setup(kp, tmp_path):
    collection = KnowledgeCollection(name="Generic course", access_domain=SCOPE.access_domain)
    await kp.writer.create_collection(collection, scope=SCOPE)
    binding = IntakeBinding(
        collection_id=collection.collection_id,
        scope=SCOPE,
        manager_user_ids=[uuid4()],
        index=IndexConfig(embedding=PROFILE),
    )
    model = Embeddings()
    service = IntakeService(
        IntakeRepository(kp.writer_database),
        FileSourceStore(tmp_path / "sources"),
        PdfExtractor(),
        model,
        IntakeConfig(bindings=[binding]),
    )
    return collection, binding, model, service


async def submit(service, collection, text, **kwargs):
    return await service.submit(collection.collection_id, pdf_bytes(text), title="Manual", **kwargs)


async def test_new_duplicate_update_and_retrieval(postgres_fixture, tmp_path):
    kp = postgres_fixture
    collection, binding, model, service = await setup(kp, tmp_path)
    first = await submit(service, collection, "Delivery validation is required before invoicing.")
    assert first.status == "queued"
    assert await service.jobs.head(binding) == (None, None)
    ready = await service.run_once(collection.collection_id)
    assert ready.status == "succeeded" and not ready.duplicate
    head, _ = await service.jobs.head(binding)
    assert head == ready.result_revision_id
    old = await kp.reader.get_revision_data(head, binding.scope)
    count = len(model.requests)
    duplicate = await submit(
        service, collection, "Delivery validation is required before invoicing."
    )
    assert duplicate.payload.document_id == first.payload.document_id
    reused = await service.run_once(collection.collection_id)
    assert reused.duplicate and reused.result_revision_id == head
    assert len(model.requests) == count
    updated = await submit(
        service,
        collection,
        "Confirm delivery and payment before invoicing.",
        replace_document_id=first.payload.document_id,
    )
    assert (await service.jobs.head(binding))[0] == head
    result = await service.run_once(collection.collection_id)
    assert result.status == "succeeded" and result.result_revision_id != head
    data = await kp.reader.get_revision_data(result.result_revision_id, binding.scope)
    assert len(data.documents) == 1
    assert data.documents[0].document_id == updated.payload.document_id == first.payload.document_id
    assert old.documents[0].source_hash != data.documents[0].source_hash
    assert (await kp.reader.get_revision_data(head, binding.scope)) == old
    retrieval = KnowledgeRetrievalService(PostgresSearchRepository(kp.reader_database), model)
    found = await retrieval.search(
        RetrievalRequest(collection_id=collection.collection_id, query="payment before invoicing"),
        scope=binding.scope,
    )
    assert found.status == "found" and found.revision_id == result.result_revision_id
    assert any("payment" in item.content for item in found.items)


async def test_two_queued_new_documents_preserve_both_and_deduplicate_concurrent_uploads(
    postgres_fixture, tmp_path
):
    kp = postgres_fixture
    collection, binding, _, service = await setup(kp, tmp_path)
    results = await asyncio.gather(
        *[submit(service, collection, "First independent generic source.") for _ in range(2)]
    )
    assert results[0].job_id == results[1].job_id
    second = await submit(service, collection, "Second source with different content.")
    for _ in range(2):
        assert (await service.run_once(collection.collection_id)).status == "succeeded"
    head, _ = await service.jobs.head(binding)
    data = await kp.reader.get_revision_data(head, binding.scope)
    assert {d.document_id for d in data.documents} == {
        results[0].payload.document_id,
        second.payload.document_id,
    }
    assert await service.run_once(collection.collection_id) is None


async def test_failed_update_preserves_active_and_explicit_retry_recovers(
    postgres_fixture, tmp_path
):
    kp = postgres_fixture
    collection, binding, model, service = await setup(kp, tmp_path)
    first = await submit(service, collection, "Original usable course information.")
    ready = await service.run_once(collection.collection_id)
    update = await submit(
        service,
        collection,
        "Replacement course information.",
        replace_document_id=first.payload.document_id,
    )
    model.failure = "provider"
    failed = await service.run_once(collection.collection_id)
    assert failed.status == "failed" and "private-source-marker" not in failed.model_dump_json()
    assert (await service.jobs.head(binding))[0] == ready.result_revision_id
    # Resource restart: use a fresh service and repository over the same durable state.
    model.failure = None
    fresh = IntakeService(
        IntakeRepository(kp.writer_database),
        service.sources,
        PdfExtractor(),
        model,
        IntakeConfig(bindings=[binding]),
    )
    await fresh.jobs.retry(binding, update.job_id)
    recovered = await fresh.run_once(collection.collection_id)
    assert recovered.status == "succeeded"
    assert (await fresh.jobs.head(binding))[0] != ready.result_revision_id


async def test_expired_worker_is_fenced_and_ready_snapshot_resumes_without_embedding(
    postgres_fixture, tmp_path
):
    kp = postgres_fixture
    collection, binding, model, service = await setup(kp, tmp_path)
    queued = await submit(service, collection, "Stable information for restart recovery.")
    old = await service.jobs.claim(binding)
    assert old.job_id == queued.job_id
    assert await service.jobs.claim(binding) is None
    # Simulate a crash immediately before atomic publication, after indexing succeeded.
    original_finish = service.jobs.finish
    captured = {}

    async def crash(*args, **kwargs):
        captured.update(kwargs)
        raise asyncio.CancelledError

    service.jobs.finish = crash
    with pytest.raises(asyncio.CancelledError):
        await service._process(binding, old)
    service.jobs.finish = original_finish
    before = len(model.requests)
    async with kp.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.intake_jobs SET lease_until=clock_timestamp()-interval '1 second' "
            "WHERE job_id=%s",
            (old.job_id,),
        )
    replacement = await service.jobs.claim(binding)
    assert replacement.job_id == old.job_id and replacement.lease_token != old.lease_token
    with pytest.raises(KnowledgeConflict):
        await service.jobs.finish(binding, old, **captured)
    with pytest.raises(KnowledgeConflict):
        await service.jobs.heartbeat(binding, old)
    result = await service._process(binding, replacement)
    assert result.status == "succeeded" and len(model.requests) == before


async def test_first_failed_extraction_retries_and_scanned_pdf_does_not_publish(
    postgres_fixture, tmp_path
):
    kp = postgres_fixture
    collection, binding, _, service = await setup(kp, tmp_path)
    invalid = await service.submit(collection.collection_id, b"%PDF-invalid", title="Broken")
    failed = await service.run_once(collection.collection_id)
    assert failed.status == "failed"
    assert (await service.jobs.head(binding))[0] is None
    await service.jobs.retry(binding, invalid.job_id)
    assert (await service.run_once(collection.collection_id)).status == "failed"
    blank = await service.submit(collection.collection_id, pdf_bytes(""), title="No text")
    assert blank.job_id != invalid.job_id
    assert (await service.run_once(collection.collection_id)).status == "failed"
    assert (await service.jobs.head(binding))[0] is None


async def test_management_and_reader_permissions_are_enforced(postgres_fixture, tmp_path):
    kp = postgres_fixture
    collection, binding, _, service = await setup(kp, tmp_path)
    with pytest.raises(KnowledgePermissionError):
        await submit(service, collection, "Denied", manager_id=uuid4())
    with pytest.raises(KnowledgeIntegrityError):
        await service.submit(collection.collection_id, b"not a PDF", title="Rejected")
    job = await submit(service, collection, "Allowed source text.")
    denied = binding.model_copy(
        update={"scope": SCOPE.model_copy(update={"access_domain": "other"})}
    )
    with pytest.raises(KnowledgePermissionError):
        await service.jobs.get(denied, job.job_id)
    async with kp.reader_database.connection() as connection:
        allowed = await (
            await connection.execute(
                "SELECT has_table_privilege(current_user,'knowledge.intake_jobs','SELECT') "
                "AS allowed"
            )
        ).fetchone()
        assert not allowed["allowed"]


async def test_concurrent_replacements_do_not_silently_overwrite(postgres_fixture, tmp_path):
    kp = postgres_fixture
    collection, binding, _, service = await setup(kp, tmp_path)
    first = await submit(service, collection, "Initial text.")
    await service.run_once(collection.collection_id)
    one = await submit(
        service, collection, "Version one.", replace_document_id=first.payload.document_id
    )
    two = await submit(
        service, collection, "Version two.", replace_document_id=first.payload.document_id
    )
    assert one.job_id != two.job_id
    result = await service.run_once(collection.collection_id)
    assert result.status == "succeeded"
    loser = await service.run_once(collection.collection_id)
    assert loser.status == "failed" and loser.error_code == "intake_conflict"
    assert (await service.jobs.head(binding))[0] == result.result_revision_id


async def test_transient_extraction_failure_can_resume_after_restart(postgres_fixture, tmp_path):
    collection, binding, _, service = await setup(postgres_fixture, tmp_path)
    queued = await submit(service, collection, "Valid PDF for transient extraction failure.")

    class Transient:
        def extract(self, content):
            return failure("extraction_timeout")

    service.extractor = Transient()
    assert (await service.run_once(collection.collection_id)).status == "failed"
    service.extractor = PdfExtractor()
    await service.jobs.retry(binding, queued.job_id)
    assert (await service.run_once(collection.collection_id)).status == "succeeded"


async def test_lease_exhaustion_and_publication_cannot_drop_documents(postgres_fixture, tmp_path):
    kp = postgres_fixture
    collection, binding, _, service = await setup(kp, tmp_path)
    first = await submit(service, collection, "First preserved document.")
    ready = await service.run_once(collection.collection_id)
    next_job = await submit(service, collection, "Another independent document.")
    processing = await service.jobs.claim(binding)
    with pytest.raises(KnowledgeIntegrityError, match="document set"):
        await service.jobs.finish(binding, processing, revision_id=ready.result_revision_id)
    async with kp.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.intake_jobs SET attempts=5, "
            "lease_until=clock_timestamp()-interval '1 second' WHERE job_id=%s",
            (processing.job_id,),
        )
    assert await service.jobs.claim(binding) is None
    exhausted = await service.jobs.get(binding, next_job.job_id)
    assert exhausted.status == "failed" and exhausted.error_code == "worker_interrupted"
    assert (await service.jobs.head(binding))[0] == ready.result_revision_id
    assert first.payload.document_id != next_job.payload.document_id


async def test_explicit_initial_revision_preserves_existing_unpublished_corpus(
    postgres_fixture, tmp_path
):
    kp = postgres_fixture
    collection, binding, _, service = await setup(kp, tmp_path)
    original = await submit(service, collection, "Existing initial source.")
    ready = await service.run_once(collection.collection_id)
    async with kp.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.knowledge_collections SET active_revision_id=NULL "
            "WHERE collection_id=%s",
            (collection.collection_id,),
        )
    with pytest.raises(KnowledgeConflict, match="explicit initial"):
        await submit(service, collection, "A second source.")
    binding = binding.model_copy(update={"initial_revision_id": ready.result_revision_id})
    service.bindings[collection.collection_id] = binding
    second = await submit(service, collection, "A second source.")
    result = await service.run_once(collection.collection_id)
    assert result.status == "succeeded"
    data = await kp.reader.get_revision_data(result.result_revision_id, binding.scope)
    assert {d.document_id for d in data.documents} == {
        original.payload.document_id,
        second.payload.document_id,
    }

"""Decision execution against actual scoped PostgreSQL, never the main course database."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.knowledge.planner import HybridRetrievalService
from app.knowledge.planner_schemas import PlannerConfig, PlannerRequest, receipt
from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.search_repository import PostgresSearchRepository
from tests.knowledge.test_planner import PlannerModel, proposal
from tests.knowledge.test_retrieval import QueryModel, corpus

pytestmark = pytest.mark.asyncio
CONFIG = PlannerConfig(model="planner-test")


def request(collection, revision, **kwargs):
    return PlannerRequest(
        **{
            "user_id": uuid4(),
            "session_id": uuid4(),
            "collection_id": collection.collection_id,
            "revision_id": revision,
            "question": "Explain FLOW-1 and FLOW-2",
            "specialist": "process_coach",
            "namespace": "workflow",
            **kwargs,
        }
    )


def service(pg, model):
    retrieval = KnowledgeRetrievalService(
        PostgresSearchRepository(pg.reader_database), QueryModel()
    )
    return HybridRetrievalService(retrieval, model, CONFIG)


async def test_combined_metadata_and_vector_search_reader_role_and_source_provenance(
    postgres_fixture,
):
    pg = postgres_fixture
    collection, revision, _, scope = await corpus(pg)
    model = PlannerModel(
        proposal(
            {"action": "metadata_lookup", "step_id": "FLOW-1", "fields": ["name", "next_step_id"]},
            {"action": "document_search", "query": "Create order FLOW-2"},
        )
    )
    req = request(collection, revision, namespace=None)
    result = await service(pg, model).execute(req, scope=scope)
    assert result.status == "found" and result.origin == "model", result.model_dump_json()
    assert {item.kind for item in result.items} == {"metadata", "chunk"}
    assert {item.reference_id for item in result.items} == {"FLOW-1", "FLOW-2"}
    assert sum(t.model_calls for t in result.trace) == 1
    assert all(c.revision_id == revision for item in result.items for c in item.citations)
    async with pg.reader_database.connection() as connection:
        row = await (await connection.execute("SELECT current_user")).fetchone()
        assert row["current_user"] == "fredi_knowledge_reader"


async def test_metadata_evidence_reuse_reloads_cited_field_without_embeddings(postgres_fixture):
    pg = postgres_fixture
    collection, revision, _, scope = await corpus(pg)
    model = PlannerModel(
        proposal(
            {"action": "metadata_lookup", "step_id": "FLOW-1", "fields": ["name"]},
        )
    )
    req = request(collection, revision, question="What is the name of FLOW-1?", namespace=None)
    planner = service(pg, model)
    first = await planner.execute(req, scope=scope)
    assert first.status == "found" and first.items[0].kind == "metadata", first.model_dump_json()
    cached = receipt(first, req, scope, datetime.now(UTC))
    model.output = proposal(
        {"action": "reuse_evidence", "evidence_ids": [str(first.items[0].evidence_id)]},
        reason="The cited name field directly answers the name question.",
    )
    second = await planner.execute(req.model_copy(update={"available": [cached]}), scope=scope)
    assert second.status == "found" and second.origin == "model", second.model_dump_json()
    assert second.items[0].content == first.items[0].content
    assert second.items[0].citations[0].chunk_id is None
    assert all(t.model_calls == 0 for t in second.trace)


async def test_wrong_course_or_owner_denied_before_planner(postgres_fixture):
    pg = postgres_fixture
    collection, revision, _, scope = await corpus(pg, owner=uuid4())
    model = PlannerModel()
    req = request(collection, revision)
    result = await service(pg, model).execute(req, scope=scope)
    assert result.status == "unavailable"  # Scope is owned by a different principal.
    assert not model.requests
    req = req.model_copy(update={"user_id": scope.user_id})
    result = await service(pg, model).execute(
        req, scope=scope.model_copy(update={"access_domain": "wrong"})
    )
    assert result.status == "unavailable" and not model.requests


async def test_permission_revoked_during_planner_never_executes_search(postgres_fixture):
    pg = postgres_fixture
    collection, revision, _, scope = await corpus(pg)

    async def revoke():
        async with pg.writer_database.connection() as connection:
            await connection.execute(
                "UPDATE knowledge.knowledge_collections SET access_domain='revoked' "
                "WHERE collection_id=%s",
                (collection.collection_id,),
            )

    model = PlannerModel(hook=revoke)
    result = await service(pg, model).execute(request(collection, revision), scope=scope)
    assert result.status == "unavailable" and not result.trace and not result.items


async def test_planner_active_pointer_change_preserves_request_revision(postgres_fixture):
    pg = postgres_fixture
    collection, first, _, scope = await corpus(pg)
    _, second, _, _ = await corpus(
        pg, collection=collection, text="Only after delivery may an invoice be created."
    )
    async with pg.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.knowledge_collections SET active_revision_id=%s "
            "WHERE collection_id=%s",
            (first, collection.collection_id),
        )

    async def switch():
        async with pg.writer_database.connection() as connection:
            await connection.execute(
                "UPDATE knowledge.knowledge_collections SET active_revision_id=%s "
                "WHERE collection_id=%s",
                (second, collection.collection_id),
            )

    model = PlannerModel(hook=switch)
    result = await service(pg, model).execute(
        request(collection, None, namespace=None), scope=scope
    )
    assert result.status == "found" and result.revision_id == first
    assert all(c.revision_id == first for item in result.items for c in item.citations)


async def test_general_document_needs_no_step_catalog_or_ingestion_labels(postgres_fixture):
    pg = postgres_fixture
    collection, revision, _, scope = await corpus(
        pg,
        text="Invoices can only be issued after delivery is completed.",
        authority=False,
    )
    model = PlannerModel(
        proposal(
            {"action": "document_search", "query": "invoice after delivery"},
        )
    )
    result = await service(pg, model).execute(
        request(
            collection,
            revision,
            question="Why must delivery happen before invoice?",
            namespace=None,
        ),
        scope=scope,
    )
    assert result.status == "found" and result.origin == "model"
    assert all(item.reference_id is None for item in result.items)
    assert all(item.citations for item in result.items)

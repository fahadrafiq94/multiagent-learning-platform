"""Explicit opt-in: real embeddings and pgvector in a disposable database."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.config.settings import settings
from app.knowledge.extraction import PdfExtractor
from app.knowledge.indexer import IndexConfig
from app.knowledge.intake import IntakeService
from app.knowledge.intake_repository import IntakeRepository
from app.knowledge.intake_schemas import IntakeBinding, IntakeConfig
from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.retrieval_schemas import RetrievalRequest
from app.knowledge.schemas import EmbeddingProfile, KnowledgeCollection
from app.knowledge.search_repository import PostgresSearchRepository
from app.knowledge.source_store import FileSourceStore
from app.services.model_service import create_model_service
from tests.knowledge.test_extraction import pdf_bytes
from tests.knowledge.test_indexer import SCOPE

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("FREDI_INTAKE_REAL_MODEL") != "1",
        reason="Opt-in real embedding service.",
    ),
]


@pytest.mark.parametrize("postgres_fixture", ["real-embedding"], indirect=True)
async def test_pdf_to_real_embeddings_publication_retrieval_and_duplicate(
    postgres_fixture, tmp_path, monkeypatch
):
    kp = postgres_fixture
    model = create_model_service(
        settings.model_copy(
            update={
                "ollama_base_url": os.environ.get(
                    "FREDI_INTAKE_MODEL_URL", "http://127.0.0.1:11434"
                ),
                "model_request_timeout_seconds": 180,
            }
        )
    )
    model_name = os.environ.get("FREDI_INTAKE_EMBEDDING_MODEL", "nomic-embed-text-v2-moe:latest")
    info = await model.inspect_embedding_model(model_name)
    profile = EmbeddingProfile(
        provider=info.provider,
        model=model_name,
        revision=info.revision,
        dimension=info.dimension,
        normalized=True,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )
    collection = KnowledgeCollection(
        name="Isolated real intake test", access_domain=SCOPE.access_domain
    )
    await kp.writer.create_collection(collection, scope=SCOPE)
    binding = IntakeBinding(
        collection_id=collection.collection_id,
        scope=SCOPE,
        manager_user_ids=[uuid4()],
        index=IndexConfig(embedding=profile),
    )
    service = IntakeService(
        IntakeRepository(kp.writer_database),
        FileSourceStore(tmp_path / "sources"),
        PdfExtractor(),
        model,
        IntakeConfig(bindings=[binding]),
    )
    calls = []
    original_embed = model.embed

    async def record(request):
        calls.append(request)
        return await original_embed(request)

    monkeypatch.setattr(model, "embed", record)
    content = pdf_bytes("Delivery confirmation is required before generating the customer invoice.")
    await service.submit(collection.collection_id, content, title="Independent new PDF")
    result = await service.run_once(collection.collection_id)
    assert result.status == "succeeded", result.public()
    assert calls
    data = await kp.reader.get_revision_data(result.result_revision_id, binding.scope)
    assert data.chunks and all(len(c.embedding) == 768 for c in data.chunks)
    retrieval = KnowledgeRetrievalService(PostgresSearchRepository(kp.reader_database), model)
    answer = await retrieval.search(
        RetrievalRequest(
            collection_id=collection.collection_id, query="What must happen before invoicing?"
        ),
        scope=SCOPE,
    )
    assert answer.status in {"found", "no_evidence"}
    direct = await retrieval.search(
        RetrievalRequest(
            collection_id=collection.collection_id,
            query="Delivery confirmation is required before generating the customer invoice.",
        ),
        scope=SCOPE,
    )
    assert direct.status == "found" and "Delivery" in direct.items[0].content
    before = len(calls)
    await service.submit(collection.collection_id, content, title="Renamed upload")
    duplicate = await service.run_once(collection.collection_id)
    assert duplicate.status == "succeeded" and duplicate.duplicate
    assert len(calls) == before
    if report := os.environ.get("FREDI_INTAKE_REAL_REPORT"):
        Path(report).write_text(
            json.dumps(
                {
                    "job_status": result.status,
                    "stored_chunks": len(data.chunks),
                    "embedding_dimension": profile.dimension,
                    "embedding_model": profile.model,
                    "embedding_revision": profile.revision,
                    "paraphrase_status": answer.status,
                    "direct_query_status": direct.status,
                    "duplicate_status": duplicate.status,
                    "duplicate_reused": duplicate.duplicate,
                    "duplicate_embedding_calls": len(calls) - before,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

import json

import pytest

from app.knowledge.errors import KnowledgeIntegrityError
from app.knowledge.indexer import IndexConfig, index_revision
from tests.knowledge.test_indexer import PROFILE, SCOPE, Embeddings, extracted_revision

pytestmark = pytest.mark.asyncio


async def test_pgvector_full_text_fields_indexes_and_reader_roundtrip(postgres_fixture):
    repository = postgres_fixture.writer
    _, source_revision = await extracted_revision(repository)
    result = await index_revision(
        source_revision.revision_id,
        config=IndexConfig(embedding=PROFILE),
        repository=repository,
        model_service=Embeddings(),
        scope=SCOPE,
    )
    data = await postgres_fixture.reader.get_revision_data(result.revision_id, SCOPE)
    assert data.chunks and len(data.chunks[0].embedding) == 3
    async with postgres_fixture.reader_database.connection() as connection:
        cursor = await connection.execute(
            "SELECT count(*) AS total FROM knowledge.knowledge_chunks WHERE revision_id=%s "
            "AND search_vector @@ plainto_tsquery('simple','validation')",
            (result.revision_id,),
        )
        assert (await cursor.fetchone())["total"] > 0
        cursor = await connection.execute(
            "SELECT embedding <=> %s::vector AS distance, vector_dims(embedding) AS dimension, "
            "search_text FROM knowledge.knowledge_chunks WHERE revision_id=%s ORDER BY chunk_id",
            (json.dumps(data.chunks[0].embedding), result.revision_id),
        )
        for row in await cursor.fetchall():
            assert row["dimension"] == 3 and abs(row["distance"]) < 1e-5
            assert "General manual" in row["search_text"]
        cursor = await connection.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname='knowledge'"
        )
        indexes = " ".join(row["indexdef"] for row in await cursor.fetchall())
        assert "USING hnsw" in indexes and "USING gin" in indexes
        cursor = await connection.execute(
            "SELECT active_revision_id FROM knowledge.knowledge_collections"
        )
        assert (await cursor.fetchone())["active_revision_id"] is None
    # An accidental direct SQL edit must not make an empty snapshot look ready.
    async with postgres_fixture.writer_database.connection() as connection:
        await connection.execute(
            "DELETE FROM knowledge.knowledge_chunks WHERE revision_id=%s", (result.revision_id,)
        )
    with pytest.raises(KnowledgeIntegrityError):
        await postgres_fixture.reader.get_revision_data(result.revision_id, SCOPE)


async def test_failed_index_storage_is_atomic_recorded_and_retryable(postgres_fixture):
    repository = postgres_fixture.writer
    _, source_revision = await extracted_revision(repository)
    original = await repository.get_revision_data(source_revision.revision_id, SCOPE)
    async with postgres_fixture.writer_database.connection() as connection:
        await connection.execute(
            "CREATE FUNCTION knowledge.reject_chunk() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'private-source-marker'; END $$; "
            "CREATE TRIGGER reject_chunk BEFORE INSERT ON knowledge.knowledge_chunks "
            "FOR EACH ROW EXECUTE FUNCTION knowledge.reject_chunk()"
        )
    result = await index_revision(
        source_revision.revision_id,
        config=IndexConfig(embedding=PROFILE),
        repository=repository,
        model_service=Embeddings(),
        scope=SCOPE,
    )
    assert result.status == "failed" and result.error_code == "index_storage_unavailable"
    assert "private" not in result.model_dump_json()
    assert (await repository.get_revision_data(result.revision_id, SCOPE)).chunks == []
    assert await repository.get_revision_data(source_revision.revision_id, SCOPE) == original
    async with postgres_fixture.writer_database.connection() as connection:
        await connection.execute("DROP TRIGGER reject_chunk ON knowledge.knowledge_chunks")
    retry = await index_revision(
        source_revision.revision_id,
        config=IndexConfig(embedding=PROFILE),
        repository=repository,
        model_service=Embeddings(),
        scope=SCOPE,
    )
    assert retry.status == "ready" and retry.revision_id == result.revision_id

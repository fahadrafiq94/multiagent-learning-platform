from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from app.knowledge import migrations
from app.knowledge.errors import (
    KnowledgeIntegrityError,
    KnowledgePermissionError,
    KnowledgeUnavailable,
)
from app.knowledge.migrations import migrate
from app.knowledge.schemas import AccessScope, KnowledgeWriteScope
from scripts.migrate_knowledge import local_dsn
from tests.knowledge.conftest import PostgresFixture
from tests.knowledge.test_repository_contract import seed

WRITE_SCOPE = KnowledgeWriteScope(access_domain="course-test", can_manage_knowledge=True)


pytestmark = pytest.mark.asyncio
SCOPE = AccessScope(access_domain="course-test")


async def test_reader_roundtrip_but_not_writes(
    postgres_fixture: PostgresFixture, sample: tuple
) -> None:
    collection, revision, _ = sample
    await seed(postgres_fixture.writer, sample)
    assert await postgres_fixture.reader.get_revision_data(revision.revision_id, SCOPE)
    with pytest.raises(KnowledgePermissionError):
        await postgres_fixture.reader.create_collection(
            collection.model_copy(update={"name": "Denied"}), scope=WRITE_SCOPE
        )


async def test_migration_idempotence_dimension_and_drift(
    postgres_fixture: PostgresFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = postgres_fixture.writer_database
    assert await migrate(database, embedding_dimension=3) == []
    with pytest.raises(KnowledgeIntegrityError):
        await migrate(database, embedding_dimension=4)
    original = migrations.SQL_ROOT / "001_foundation.sql"
    for migration in migrations.SQL_ROOT.glob("*.sql"):
        (tmp_path / migration.name).write_bytes(migration.read_bytes())
    (tmp_path / original.name).write_text(original.read_text() + "\n-- changed\n", encoding="utf-8")
    monkeypatch.setattr(migrations, "SQL_ROOT", tmp_path)
    with pytest.raises(KnowledgeIntegrityError):
        await migrate(database, embedding_dimension=3)


async def test_database_foreign_key_rejects_unready_source(
    postgres_fixture: PostgresFixture, sample: tuple
) -> None:
    await seed(postgres_fixture.writer, sample)
    with pytest.raises(KnowledgeIntegrityError):
        async with postgres_fixture.writer_database.connection() as connection:
            await connection.execute(
                "UPDATE knowledge.knowledge_sections SET processing_status='failed'"
            )
    assert await postgres_fixture.reader.get_revision_data(sample[1].revision_id, SCOPE)


async def test_transaction_rolls_back_after_partial_replacement(
    postgres_fixture: PostgresFixture, sample: tuple
) -> None:
    await seed(postgres_fixture.writer, sample)
    before = await postgres_fixture.reader.get_revision_data(sample[1].revision_id, SCOPE)
    async with postgres_fixture.writer_database.connection() as connection:
        await connection.execute(
            "CREATE FUNCTION knowledge.fail_insert() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'injected transaction failure'; END $$; "
            "CREATE TRIGGER fail_insert BEFORE INSERT ON knowledge.knowledge_chunks "
            "FOR EACH ROW EXECUTE FUNCTION knowledge.fail_insert()"
        )
    with pytest.raises(KnowledgeUnavailable):
        await postgres_fixture.writer.write_revision_data(
            sample[1].revision_id, sample[2], expected_version=1, scope=WRITE_SCOPE
        )
    assert await postgres_fixture.reader.get_revision_data(sample[1].revision_id, SCOPE) == before
    revision = await postgres_fixture.reader.get_revision(sample[1].revision_id, SCOPE)
    assert revision is not None and revision.version == 1


async def test_pool_closed_and_rejects_reuse(postgres_fixture: PostgresFixture) -> None:
    database = postgres_fixture.writer_database
    await database.close()
    await database.close()
    with pytest.raises(KnowledgeUnavailable, match="closed"):
        await database.start()


async def test_corrupt_record_error_does_not_expose_content(
    postgres_fixture: PostgresFixture,
    sample: tuple,
) -> None:
    await seed(postgres_fixture.writer, sample)
    async with postgres_fixture.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.knowledge_chunks SET payload=jsonb_set(payload,'{content}',"
            "'{\"sensitive\":\"private-marker\"}'::jsonb) WHERE kind='evidence'"
        )
    with pytest.raises(KnowledgeIntegrityError) as error:
        await postgres_fixture.reader.get_revision_data(sample[1].revision_id, SCOPE)
    assert "private-marker" not in str(error.value)


async def test_missing_bootstrap_schema_is_reported_explicitly(postgres_fixture):
    database = postgres_fixture.writer_database
    async with database.connection() as connection:
        await connection.execute("ALTER SCHEMA knowledge RENAME TO saved_knowledge")
    try:
        with pytest.raises(KnowledgeIntegrityError, match="schema is missing.*bootstrap"):
            await migrate(database, embedding_dimension=3)
    finally:
        async with database.connection() as connection:
            await connection.execute("ALTER SCHEMA saved_knowledge RENAME TO knowledge")
    assert await migrate(database, embedding_dimension=3) == []


async def test_wrong_bootstrap_owner_is_reported_explicitly(postgres_fixture):
    database = postgres_fixture.writer_database
    async with database.connection() as connection:
        name = connection.info.dbname
    async with await psycopg.AsyncConnection.connect(
        local_dsn(role="admin", database=name).get_secret_value()
    ) as admin:
        await admin.execute("ALTER SCHEMA knowledge OWNER TO fredi_knowledge_admin")
    try:
        with pytest.raises(KnowledgeIntegrityError, match="ownership.*bootstrap"):
            await migrate(database, embedding_dimension=3)
    finally:
        async with await psycopg.AsyncConnection.connect(
            local_dsn(role="admin", database=name).get_secret_value()
        ) as admin:
            await admin.execute("ALTER SCHEMA knowledge OWNER TO fredi_knowledge_writer")


async def test_late_batch_failure_rolls_back_earlier_batches(postgres_fixture, sample):
    await seed(postgres_fixture.writer, sample)
    revision = sample[1]
    before = await postgres_fixture.reader.get_revision_data(revision.revision_id, SCOPE)
    expanded = before.model_copy(deep=True)
    evidence = next(c for c in before.chunks if c.embedding)
    expanded = expanded.model_copy(
        update={
            "chunks": [
                evidence.model_copy(update={"chunk_id": uuid4(), "content": f"Evidence {i}"})
                for i in range(260)
            ]
        }
    )
    # Fail at the final row, after two complete batches have already been sent.
    async with postgres_fixture.writer_database.connection() as connection:
        await connection.execute(
            "CREATE FUNCTION knowledge.fail_late() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN IF NEW.payload->>'content' = 'Evidence 259' THEN "
            "RAISE EXCEPTION 'late batch failure'; END IF; RETURN NEW; END $$; "
            "CREATE TRIGGER fail_late BEFORE INSERT ON knowledge.knowledge_chunks "
            "FOR EACH ROW EXECUTE FUNCTION knowledge.fail_late()"
        )
    with pytest.raises(KnowledgeUnavailable):
        await postgres_fixture.writer.write_revision_data(
            revision.revision_id, expanded, expected_version=1, scope=WRITE_SCOPE
        )
    assert await postgres_fixture.reader.get_revision_data(revision.revision_id, SCOPE) == before
    assert (await postgres_fixture.reader.get_revision(revision.revision_id, SCOPE)).version == 1
    async with postgres_fixture.writer_database.connection() as connection:
        await connection.execute("DROP TRIGGER fail_late ON knowledge.knowledge_chunks")
    await postgres_fixture.writer.write_revision_data(
        revision.revision_id, expanded, expected_version=1, scope=WRITE_SCOPE
    )
    assert (
        len((await postgres_fixture.reader.get_revision_data(revision.revision_id, SCOPE)).chunks)
        == 260
    )

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from psycopg import sql

from app.knowledge import migrations
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.in_memory import InMemoryKnowledgeRepository
from app.knowledge.migrations import migrate
from app.knowledge.postgres import PostgresKnowledgeRepository
from app.knowledge.repository import KnowledgeRepository
from app.knowledge.schemas import (
    ChunkKind,
    EmbeddingProfile,
    KnowledgeChunk,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeRevision,
    KnowledgeSection,
    PipelineVersions,
    ProcessStep,
    RevisionData,
)
from scripts.migrate_knowledge import local_dsn


def pytest_asyncio_loop_factories() -> dict[str, Callable[[], asyncio.AbstractEventLoop]]:
    # Local test hook; the application's global event-loop policy is unchanged.
    return {"selector": asyncio.SelectorEventLoop}


@dataclass
class PostgresFixture:
    writer: PostgresKnowledgeRepository
    reader: PostgresKnowledgeRepository
    writer_database: KnowledgeDatabase
    reader_database: KnowledgeDatabase


@pytest_asyncio.fixture
async def postgres_fixture(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[PostgresFixture]:
    if os.environ.get("FREDI_KNOWLEDGE_REPOSITORY_INTEGRATION") != "1":
        pytest.skip("Opt-in: real PostgreSQL repositories and isolated scratch databases.")
    name = f"fredi_smoke_{uuid4().hex[:16]}"
    admin = await psycopg.AsyncConnection.connect(
        local_dsn(role="admin").get_secret_value(),
        autocommit=True,
    )
    writer_db = KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=local_dsn(database=name)))
    reader_db = KnowledgeDatabase(
        KnowledgeDatabaseConfig(dsn=local_dsn(role="reader", database=name))
    )
    created = False
    try:
        await admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER fredi_knowledge_writer").format(
                sql.Identifier(name),
            )
        )
        created = True
        async with await psycopg.AsyncConnection.connect(
            local_dsn(role="admin", database=name).get_secret_value(),
        ) as bootstrap:
            await bootstrap.execute(
                "CREATE EXTENSION vector; "
                "CREATE SCHEMA knowledge AUTHORIZATION fredi_knowledge_writer; "
                "GRANT USAGE ON SCHEMA knowledge TO fredi_knowledge_reader; "
                "ALTER DEFAULT PRIVILEGES FOR ROLE fredi_knowledge_writer IN SCHEMA knowledge "
                "GRANT SELECT ON TABLES TO fredi_knowledge_reader"
            )
        if getattr(request, "param", None) == "foundation":
            original = migrations.SQL_ROOT / "001_foundation.sql"
            (tmp_path / original.name).write_bytes(original.read_bytes())
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(migrations, "SQL_ROOT", tmp_path)
                assert await migrate(writer_db, embedding_dimension=3) == ["001_foundation"]
        else:
            dimension = 768 if getattr(request, "param", None) == "real-embedding" else 3
            assert await migrate(writer_db, embedding_dimension=dimension) == [
                "001_foundation",
                "002_general_indexing",
                "003_chunk_index",
                "004_intake_jobs",
            ]
        yield PostgresFixture(
            PostgresKnowledgeRepository(writer_db),
            PostgresKnowledgeRepository(reader_db),
            writer_db,
            reader_db,
        )
    finally:
        await reader_db.close()
        await writer_db.close()
        if created:
            # Only this fixture's uniquely generated database is removed; no FORCE/CASCADE.
            await admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
        await admin.close()


@pytest.fixture(params=["memory", "postgres"])
def repository(request: pytest.FixtureRequest) -> KnowledgeRepository:
    if request.param == "memory":
        return InMemoryKnowledgeRepository(embedding_dimension=3)
    real: PostgresFixture = request.getfixturevalue("postgres_fixture")
    return real.writer


@pytest.fixture
def sample() -> tuple[KnowledgeCollection, KnowledgeRevision, RevisionData]:
    collection = KnowledgeCollection(name="Course knowledge", access_domain="course-test")
    document_id = uuid4()
    digest = hashlib.sha256(b"synthetic contract source").hexdigest()
    revision = KnowledgeRevision(
        collection_id=collection.collection_id,
        embedding=EmbeddingProfile(provider="test", model="fixed", revision="v1", dimension=3),
        pipeline=PipelineVersions(extractor="v1", chunker="v1", policy="v1"),
        manifest={document_id: digest},
    )
    document = KnowledgeDocument(
        document_id=document_id,
        revision_id=revision.revision_id,
        title="Synthetic source",
        source_uri=f"sha256:{digest}",
        source_hash=digest,
        workflow_authority=True,
    )
    section = KnowledgeSection(
        revision_id=revision.revision_id,
        document_id=document_id,
        title="Business reasoning",
        text="Delivery must be complete before invoicing.",
        page_start=1,
        page_end=2,
    )
    step = ProcessStep(
        revision_id=revision.revision_id,
        namespace="contract",
        step_id="step-a",
        source_document_id=document_id,
        name="Invoice",
        phase="Sales",
    )
    evidence = KnowledgeChunk(
        revision_id=revision.revision_id,
        document_id=document_id,
        section_id=section.section_id,
        kind=ChunkKind.EVIDENCE,
        content=section.text,
        embedding=[1.0, 0.0, 0.0],
        namespace=step.namespace,
        step_id=step.step_id,
    )
    hint = KnowledgeChunk(
        revision_id=revision.revision_id,
        document_id=document_id,
        section_id=section.section_id,
        kind=ChunkKind.HINT,
        content="Think about delivery before billing.",
        hint_level=2,
        namespace=step.namespace,
        step_id=step.step_id,
    )
    return (
        collection,
        revision,
        RevisionData(
            documents=[document],
            sections=[section],
            steps=[step],
            chunks=[evidence, hint],
        ),
    )

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from enum import Enum
from itertools import batched
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection, sql
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from app.knowledge.database import KnowledgeDatabase
from app.knowledge.errors import KnowledgeConflict, KnowledgeIntegrityError, KnowledgeNotFound
from app.knowledge.index_validation import validate_indexed_data
from app.knowledge.repository import authorize_write, require_write_scope, validate_data
from app.knowledge.schemas import (
    AccessScope,
    KnowledgeChunk,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeModel,
    KnowledgeRevision,
    KnowledgeSection,
    KnowledgeWriteScope,
    ProcessingStatus,
    ProcessStep,
    RevisionData,
    RevisionStatus,
)

# Static SQL mappings only; table/column identifiers never come from a caller.
FIELDS: dict[str, tuple[str, ...]] = {
    "knowledge_collections": ("collection_id", "access_domain", "sharing_class", "owner_user_id"),
    "knowledge_revisions": ("revision_id", "collection_id", "version", "status"),
    "knowledge_documents": ("revision_id", "document_id", "source_hash", "workflow_authority"),
    "knowledge_sections": (
        "revision_id",
        "document_id",
        "section_id",
        "processing_status",
    ),
    "process_steps": ("revision_id", "namespace", "step_id", "source_document_id"),
    "knowledge_chunks": (
        "revision_id",
        "document_id",
        "section_id",
        "chunk_id",
        "kind",
        "namespace",
        "step_id",
        "hint_level",
    ),
}
DATA_MODELS = (
    ("documents", "knowledge_documents", KnowledgeDocument),
    ("sections", "knowledge_sections", KnowledgeSection),
    ("steps", "process_steps", ProcessStep),
    ("chunks", "knowledge_chunks", KnowledgeChunk),
)


def encode_insert(table: str, record: KnowledgeModel) -> tuple[sql.Composed, list[Any]]:
    fields = FIELDS[table]
    columns = list(fields)
    raw = record.model_dump()
    values = [raw[key].value if isinstance(raw[key], Enum) else raw[key] for key in fields]
    exclude = set(fields)
    if isinstance(record, KnowledgeRevision):
        columns += ["fingerprint", "embedding_dimension"]
        values += [record.fingerprint, record.embedding.dimension]
    if isinstance(record, KnowledgeChunk):
        columns += ["content_hash", "embedding"]
        values += [record.content_hash, json.dumps(record.embedding) if record.embedding else None]
        exclude.add("embedding")
    columns.append("payload")
    values.append(Jsonb(record.model_dump(mode="json", exclude=exclude)))
    placeholders = [
        sql.SQL("%s::vector") if key == "embedding" else sql.SQL("%s") for key in columns
    ]
    query = sql.SQL("INSERT INTO knowledge.{} ({}) VALUES ({})").format(
        sql.Identifier(table),
        sql.SQL(",").join(map(sql.Identifier, columns)),
        sql.SQL(",").join(placeholders),
    )
    return query, values


async def insert(
    connection: AsyncConnection[dict[str, Any]], table: str, record: KnowledgeModel
) -> None:
    query, values = encode_insert(table, record)
    await connection.execute(query, values)


async def insert_many(
    connection: AsyncConnection[dict[str, Any]], table: str, records: Sequence[KnowledgeModel]
) -> None:
    # Psycopg pipelines each bounded batch inside the atomic snapshot transaction.
    async with connection.cursor() as cursor:
        for batch in batched(records, 128):
            query, _ = encode_insert(table, batch[0])
            await cursor.executemany(query, [encode_insert(table, row)[1] for row in batch])


def decode[M: KnowledgeModel](model: type[M], table: str, row: dict[str, Any]) -> M:
    payload = dict(row["payload"])
    payload.update({field: row[field] for field in FIELDS[table]})
    if model is KnowledgeRevision:
        for name in ("indexing_status", "indexing_error"):
            if name in row:
                payload[name] = row[name]
    if model is KnowledgeChunk:
        payload["embedding"] = json.loads(row["vector_text"]) if row["vector_text"] else None
    try:
        record = model.model_validate(payload)
    except ValidationError:
        raise KnowledgeIntegrityError("Stored knowledge record is invalid.") from None
    if isinstance(record, KnowledgeChunk) and record.content_hash != row["content_hash"]:
        raise KnowledgeIntegrityError("Stored content hash does not match the chunk.")
    return record


class PostgresKnowledgeRepository:
    def __init__(self, database: KnowledgeDatabase) -> None:
        self.database = database

    async def create_collection(
        self, collection: KnowledgeCollection, *, scope: KnowledgeWriteScope
    ) -> KnowledgeCollection:
        collection = KnowledgeCollection.model_validate(collection.model_dump())
        authorize_write(collection, scope)
        async with self.database.connection() as connection:
            await insert(connection, "knowledge_collections", collection)
        return collection.model_copy(deep=True)

    async def get_collection(
        self,
        collection_id: UUID,
        scope: AccessScope,
    ) -> KnowledgeCollection | None:
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM knowledge.knowledge_collections WHERE collection_id=%s "
                "AND access_domain=%s "
                "AND (sharing_class='shared_instructional' OR owner_user_id=%s)",
                (collection_id, scope.access_domain, scope.user_id),
            )
            row = await cursor.fetchone()
            return decode(KnowledgeCollection, "knowledge_collections", row) if row else None

    async def _authorize_collection(
        self,
        connection: AsyncConnection[dict[str, Any]],
        collection_id: UUID,
        scope: KnowledgeWriteScope,
    ) -> None:
        cursor = await connection.execute(
            "SELECT * FROM knowledge.knowledge_collections WHERE collection_id=%s FOR SHARE",
            (collection_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise KnowledgeNotFound("Collection not found.")
        authorize_write(decode(KnowledgeCollection, "knowledge_collections", row), scope)

    async def create_revision(
        self, revision: KnowledgeRevision, *, scope: KnowledgeWriteScope
    ) -> KnowledgeRevision:
        require_write_scope(scope)
        revision = KnowledgeRevision.model_validate(revision.model_dump())
        if (
            revision.version != 0
            or revision.status != RevisionStatus.STAGING
            or revision.indexing_status != ProcessingStatus.PENDING
            or revision.indexing_error is not None
        ):
            raise KnowledgeConflict("New revisions start at staging version zero.")
        async with self.database.connection() as connection:
            await self._authorize_collection(connection, revision.collection_id, scope)
            await insert(connection, "knowledge_revisions", revision)
        return revision.model_copy(deep=True)

    async def _revision(
        self,
        connection: AsyncConnection[dict[str, Any]],
        revision_id: UUID,
        scope: AccessScope,
    ) -> KnowledgeRevision | None:
        cursor = await connection.execute(
            "SELECT r.* FROM knowledge.knowledge_revisions r "
            "JOIN knowledge.knowledge_collections c USING (collection_id) "
            "WHERE r.revision_id=%s AND c.access_domain=%s "
            "AND (c.sharing_class='shared_instructional' OR c.owner_user_id=%s)",
            (revision_id, scope.access_domain, scope.user_id),
        )
        row = await cursor.fetchone()
        return decode(KnowledgeRevision, "knowledge_revisions", row) if row else None

    async def get_revision(
        self,
        revision_id: UUID,
        scope: AccessScope,
    ) -> KnowledgeRevision | None:
        async with self.database.connection() as connection:
            return await self._revision(connection, revision_id, scope)

    async def write_revision_data(
        self,
        revision_id: UUID,
        data: RevisionData,
        *,
        scope: KnowledgeWriteScope,
        expected_version: int,
        index_complete: bool = False,
    ) -> KnowledgeRevision:
        require_write_scope(scope)
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM knowledge.knowledge_revisions WHERE revision_id=%s FOR UPDATE",
                (revision_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise KnowledgeNotFound("Revision not found.")
            await self._authorize_collection(connection, row["collection_id"], scope)
            if (
                row["version"] != expected_version
                or row["status"] != "staging"
                or row.get("indexing_status") == "ready"
            ):
                raise KnowledgeConflict("Revision is stale or no longer editable.")
            revision = decode(KnowledgeRevision, "knowledge_revisions", row)
            data = validate_data(revision, data)
            if index_complete:
                validate_indexed_data(revision, data)
            # Replace a staging snapshot atomically. Published revisions are never overwritten.
            for _, table, _ in reversed(DATA_MODELS):
                await connection.execute(
                    sql.SQL("DELETE FROM knowledge.{} WHERE revision_id=%s").format(
                        sql.Identifier(table)
                    ),
                    (revision_id,),
                )
            for attribute, table, _ in DATA_MODELS:
                await insert_many(connection, table, getattr(data, attribute))
            if index_complete:
                await connection.execute(
                    "UPDATE knowledge.knowledge_revisions SET version=version+1,"
                    "indexing_status='ready',indexing_error=NULL WHERE revision_id=%s",
                    (revision_id,),
                )
            else:
                await connection.execute(
                    "UPDATE knowledge.knowledge_revisions SET version=version+1 "
                    "WHERE revision_id=%s",
                    (revision_id,),
                )
            return revision.model_copy(
                update={
                    "version": revision.version + 1,
                    "indexing_status": ProcessingStatus.READY
                    if index_complete
                    else revision.indexing_status,
                    "indexing_error": None if index_complete else revision.indexing_error,
                },
                deep=True,
            )

    async def record_index_failure(
        self,
        revision_id: UUID,
        *,
        scope: KnowledgeWriteScope,
        expected_version: int,
        error_code: str,
    ) -> KnowledgeRevision:
        require_write_scope(scope)
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_code):
            raise ValueError("Use a bounded indexing error code.")
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT * FROM knowledge.knowledge_revisions WHERE revision_id=%s FOR UPDATE",
                (revision_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise KnowledgeNotFound("Revision not found.")
            await self._authorize_collection(connection, row["collection_id"], scope)
            if (
                row["version"] != expected_version
                or row["status"] != "staging"
                or row["indexing_status"] == "ready"
            ):
                raise KnowledgeConflict("Revision is stale or already indexed.")
            await connection.execute(
                "UPDATE knowledge.knowledge_revisions SET version=version+1,"
                "indexing_status='failed',indexing_error=%s WHERE revision_id=%s",
                (error_code, revision_id),
            )
            return decode(KnowledgeRevision, "knowledge_revisions", row).model_copy(
                update={
                    "version": row["version"] + 1,
                    "indexing_status": ProcessingStatus.FAILED,
                    "indexing_error": error_code,
                }
            )

    async def get_revision_data(
        self,
        revision_id: UUID,
        scope: AccessScope,
    ) -> RevisionData | None:
        async with self.database.connection() as connection:
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            revision = await self._revision(connection, revision_id, scope)
            if revision is None:
                return None
            payload: dict[str, Any] = {}
            for attribute, table, model in DATA_MODELS:
                select = "*, embedding::text AS vector_text" if model is KnowledgeChunk else "*"
                cursor = await connection.execute(
                    sql.SQL("SELECT {} FROM knowledge.{} WHERE revision_id=%s").format(
                        sql.SQL(select),
                        sql.Identifier(table),
                    ),
                    (revision_id,),
                )
                payload[attribute] = [decode(model, table, row) for row in await cursor.fetchall()]
            data = validate_data(revision, RevisionData(**payload))
            if revision.indexing_status == ProcessingStatus.READY:
                validate_indexed_data(revision, data)
            return data

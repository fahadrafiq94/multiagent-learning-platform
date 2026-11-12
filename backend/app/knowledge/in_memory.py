from __future__ import annotations

import asyncio
import re
from uuid import UUID

from app.knowledge.errors import KnowledgeConflict, KnowledgeIntegrityError, KnowledgeNotFound
from app.knowledge.index_validation import validate_indexed_data
from app.knowledge.repository import authorize_write, require_write_scope, validate_data
from app.knowledge.schemas import (
    AccessScope,
    KnowledgeCollection,
    KnowledgeRevision,
    KnowledgeWriteScope,
    ProcessingStatus,
    RevisionData,
    RevisionStatus,
)


class InMemoryKnowledgeRepository:
    def __init__(self, *, embedding_dimension: int) -> None:
        if not 1 <= embedding_dimension <= 16000:
            raise ValueError("Invalid embedding dimension.")
        self.embedding_dimension = embedding_dimension
        self._collections: dict[UUID, KnowledgeCollection] = {}
        self._revisions: dict[UUID, KnowledgeRevision] = {}
        self._data: dict[UUID, RevisionData] = {}
        self._lock = asyncio.Lock()

    async def create_collection(
        self, collection: KnowledgeCollection, *, scope: KnowledgeWriteScope
    ) -> KnowledgeCollection:
        collection = KnowledgeCollection.model_validate(collection.model_dump())
        async with self._lock:
            authorize_write(collection, scope)
            if collection.collection_id in self._collections:
                raise KnowledgeConflict("Collection already exists.")
            self._collections[collection.collection_id] = collection.model_copy(deep=True)
            return collection.model_copy(deep=True)

    async def get_collection(
        self,
        collection_id: UUID,
        scope: AccessScope,
    ) -> KnowledgeCollection | None:
        async with self._lock:
            row = self._collections.get(collection_id)
            return row.model_copy(deep=True) if row and row.permits(scope) else None

    async def create_revision(
        self, revision: KnowledgeRevision, *, scope: KnowledgeWriteScope
    ) -> KnowledgeRevision:
        require_write_scope(scope)
        revision = KnowledgeRevision.model_validate(revision.model_dump())
        async with self._lock:
            if revision.collection_id not in self._collections:
                raise KnowledgeNotFound("Collection not found.")
            authorize_write(self._collections[revision.collection_id], scope)
            if (
                revision.version != 0
                or revision.status != RevisionStatus.STAGING
                or revision.indexing_status != ProcessingStatus.PENDING
                or revision.indexing_error is not None
            ):
                raise KnowledgeConflict("New revisions start at staging version zero.")
            if revision.embedding.dimension != self.embedding_dimension:
                raise KnowledgeIntegrityError("Embedding dimension does not match the database.")
            if revision.revision_id in self._revisions or any(
                r.collection_id == revision.collection_id and r.fingerprint == revision.fingerprint
                for r in self._revisions.values()
            ):
                raise KnowledgeConflict("Revision already exists.")
            self._revisions[revision.revision_id] = revision.model_copy(deep=True)
            self._data[revision.revision_id] = RevisionData()
            return revision.model_copy(deep=True)

    async def get_revision(
        self,
        revision_id: UUID,
        scope: AccessScope,
    ) -> KnowledgeRevision | None:
        async with self._lock:
            row = self._revisions.get(revision_id)
            if row and self._collections[row.collection_id].permits(scope):
                return row.model_copy(deep=True)
            return None

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
        async with self._lock:
            revision = self._revisions.get(revision_id)
            if revision is None:
                raise KnowledgeNotFound("Revision not found.")
            authorize_write(self._collections[revision.collection_id], scope)
            if (
                revision.version != expected_version
                or revision.status != RevisionStatus.STAGING
                or revision.indexing_status == "ready"
            ):
                raise KnowledgeConflict("Revision was modified by another writer.")
            validated = validate_data(revision, data)
            if index_complete:
                validate_indexed_data(revision, validated)
            updated = revision.model_copy(
                update={
                    "version": revision.version + 1,
                    "indexing_status": ProcessingStatus.READY
                    if index_complete
                    else revision.indexing_status,
                    "indexing_error": None if index_complete else revision.indexing_error,
                },
                deep=True,
            )
            self._data[revision_id] = validated.model_copy(deep=True)
            self._revisions[revision_id] = updated
            return updated.model_copy(deep=True)

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
        async with self._lock:
            revision = self._revisions.get(revision_id)
            if revision is None:
                raise KnowledgeNotFound("Revision not found.")
            authorize_write(self._collections[revision.collection_id], scope)
            if (
                revision.version != expected_version
                or revision.status != RevisionStatus.STAGING
                or revision.indexing_status == "ready"
            ):
                raise KnowledgeConflict("Revision is stale or already indexed.")
            updated = revision.model_copy(
                update={
                    "version": revision.version + 1,
                    "indexing_status": ProcessingStatus.FAILED,
                    "indexing_error": error_code,
                }
            )
            self._revisions[revision_id] = updated
            return updated.model_copy(deep=True)

    async def get_revision_data(
        self,
        revision_id: UUID,
        scope: AccessScope,
    ) -> RevisionData | None:
        async with self._lock:
            row = self._revisions.get(revision_id)
            if row and self._collections[row.collection_id].permits(scope):
                if row.indexing_status == ProcessingStatus.READY:
                    validate_indexed_data(row, self._data[revision_id])
                return self._data[revision_id].model_copy(deep=True)
            return None

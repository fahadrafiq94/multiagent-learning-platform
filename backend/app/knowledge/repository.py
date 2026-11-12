from __future__ import annotations

import hashlib
from typing import Protocol
from uuid import UUID

from app.knowledge.errors import KnowledgeIntegrityError, KnowledgePermissionError
from app.knowledge.schemas import (
    AccessScope,
    KnowledgeCollection,
    KnowledgeRevision,
    KnowledgeWriteScope,
    ProcessingStatus,
    RevisionData,
)


class KnowledgeRepository(Protocol):
    async def create_collection(
        self, collection: KnowledgeCollection, *, scope: KnowledgeWriteScope
    ) -> KnowledgeCollection: ...

    async def get_collection(
        self,
        collection_id: UUID,
        scope: AccessScope,
    ) -> KnowledgeCollection | None: ...

    async def create_revision(
        self, revision: KnowledgeRevision, *, scope: KnowledgeWriteScope
    ) -> KnowledgeRevision: ...

    async def get_revision(
        self,
        revision_id: UUID,
        scope: AccessScope,
    ) -> KnowledgeRevision | None: ...

    async def write_revision_data(
        self,
        revision_id: UUID,
        data: RevisionData,
        *,
        scope: KnowledgeWriteScope,
        expected_version: int,
        index_complete: bool = False,
    ) -> KnowledgeRevision: ...

    async def record_index_failure(
        self,
        revision_id: UUID,
        *,
        scope: KnowledgeWriteScope,
        expected_version: int,
        error_code: str,
    ) -> KnowledgeRevision: ...

    async def get_revision_data(
        self,
        revision_id: UUID,
        scope: AccessScope,
    ) -> RevisionData | None: ...


def require_write_scope(scope: KnowledgeWriteScope) -> None:
    if not isinstance(scope, KnowledgeWriteScope) or not scope.can_manage_knowledge:
        raise KnowledgePermissionError("Knowledge editing permission is required.")


def authorize_write(collection: KnowledgeCollection, scope: KnowledgeWriteScope) -> None:
    require_write_scope(scope)
    if not collection.permits(scope):
        raise KnowledgePermissionError("Knowledge collection editing is not permitted.")


def validate_data(revision: KnowledgeRevision, data: RevisionData) -> RevisionData:
    """Revalidate nested mutable lists and complete references before atomic writes."""
    data = RevisionData.model_validate(data.model_dump())
    documents = {d.document_id: d for d in data.documents}
    sections = {s.section_id: s for s in data.sections}
    steps = {(s.namespace, s.step_id): s for s in data.steps}
    chunks = {c.chunk_id: c for c in data.chunks}
    for indexed, sequence in (
        (documents, data.documents),
        (sections, data.sections),
        (steps, data.steps),
        (chunks, data.chunks),
    ):
        if len(indexed) != len(sequence):
            raise KnowledgeIntegrityError("Duplicate identity in revision data.")
    if any(record.revision_id != revision.revision_id for record in data.documents):
        raise KnowledgeIntegrityError("Cross-revision data is not allowed.")
    if any(record.revision_id != revision.revision_id for record in data.sections):
        raise KnowledgeIntegrityError("Cross-revision data is not allowed.")
    if any(record.revision_id != revision.revision_id for record in data.steps):
        raise KnowledgeIntegrityError("Cross-revision data is not allowed.")
    if any(record.revision_id != revision.revision_id for record in data.chunks):
        raise KnowledgeIntegrityError("Cross-revision data is not allowed.")
    for document in data.documents:
        if revision.manifest.get(document.document_id) != document.source_hash:
            raise KnowledgeIntegrityError("Document does not match the revision manifest.")
    for section in data.sections:
        if section.document_id not in documents:
            raise KnowledgeIntegrityError("Section source document is missing.")
        if section.parent_section_id is not None:
            parent = sections.get(section.parent_section_id)
            if (
                parent is None
                or parent.document_id != section.document_id
                or parent.parent_section_id is not None
                or parent.section_id == section.section_id
            ):
                raise KnowledgeIntegrityError("Section split parent is invalid.")
            if section.parent_text_hash != hashlib.sha256(parent.text.encode()).hexdigest():
                raise KnowledgeIntegrityError("Section split context hash is invalid.")
        pages = documents[section.document_id].pages
        if pages and not section.page_spans:
            raise KnowledgeIntegrityError("Extracted sections require page provenance.")
        covered = 0
        previous_page = 0
        for span in section.page_spans:
            if span.page > len(pages):
                raise KnowledgeIntegrityError("Section page provenance is invalid.")
            page = pages[span.page - 1]
            fragment = page.text[span.start : span.end]
            if (
                span.end > len(page.text)
                or section.text[span.section_start : span.section_start + len(fragment)] != fragment
                or not section.page_start <= span.page <= section.page_end
                or span.section_start < covered
                or span.page <= previous_page
                or section.text[covered : span.section_start].strip("\n")
            ):
                raise KnowledgeIntegrityError("Section text does not match page provenance.")
            covered = span.section_start + len(fragment)
            previous_page = span.page
        if section.page_spans and section.text[covered:].strip("\n"):
            raise KnowledgeIntegrityError("Section text lacks page provenance.")
    for step in data.steps:
        authority = documents.get(step.source_document_id)
        if authority is None or not authority.workflow_authority:
            raise KnowledgeIntegrityError("Canonical facts require an authoritative source.")
        targets = list(step.prerequisites)
        if step.next_step_id:
            targets.append(step.next_step_id)
        if any((step.namespace, target) not in steps for target in targets):
            raise KnowledgeIntegrityError("Process-step reference is missing.")
        for source in step.source_fields.values():
            fact_section = sections.get(source.section_id)
            if fact_section is None or fact_section.document_id != step.source_document_id:
                raise KnowledgeIntegrityError("Step fact source does not belong to its document.")
    for chunk in data.chunks:
        linked_section = sections.get(chunk.section_id)
        if linked_section is None or linked_section.document_id != chunk.document_id:
            raise KnowledgeIntegrityError("Chunk section/document reference is invalid.")
        if linked_section.processing_status != ProcessingStatus.READY:
            raise KnowledgeIntegrityError("Chunks require technically ready source sections.")
        if chunk.step_id and (chunk.namespace, chunk.step_id) not in steps:
            raise KnowledgeIntegrityError("Chunk step reference is missing.")
        if chunk.embedding is not None and len(chunk.embedding) != revision.embedding.dimension:
            raise KnowledgeIntegrityError("Embedding dimension does not match the revision.")
        if any(target not in chunks for target in chunk.support_ids):
            raise KnowledgeIntegrityError("Chunk support reference is missing.")
    return RevisionData(
        documents=sorted(data.documents, key=lambda row: str(row.document_id)),
        sections=sorted(data.sections, key=lambda row: str(row.section_id)),
        steps=sorted(data.steps, key=lambda row: (row.namespace, row.step_id)),
        chunks=sorted(data.chunks, key=lambda row: str(row.chunk_id)),
    )

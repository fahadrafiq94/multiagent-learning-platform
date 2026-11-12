"""Explicit registered imports into staging; no classification or publication."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid5

from pydantic import Field, model_validator

from app.knowledge.errors import (
    KnowledgeConflict,
    KnowledgeIntegrityError,
    KnowledgeNotFound,
    KnowledgePermissionError,
    KnowledgeUnavailable,
)
from app.knowledge.extraction import (
    EXTRACTOR_VERSION,
    MAX_SOURCE_BYTES,
    DocumentExtractor,
    StructureProfile,
    split_sections,
)
from app.knowledge.repository import KnowledgeRepository, authorize_write
from app.knowledge.schemas import (
    EmbeddingProfile,
    ExtractionDiagnostic,
    ExtractionStatus,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeModel,
    KnowledgeRevision,
    KnowledgeWriteScope,
    PipelineVersions,
    RevisionData,
)
from app.knowledge.source_store import SourceStore


class RegisteredSource(KnowledgeModel):
    document_id: UUID
    path: str = Field(min_length=1, max_length=1000)
    title: str = Field(min_length=1, max_length=255)
    workflow_authority: bool = False
    profile: StructureProfile = Field(default_factory=StructureProfile)


class ImportRegistration(KnowledgeModel):
    collection: KnowledgeCollection
    embedding: EmbeddingProfile
    sources: list[RegisteredSource] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_sources(self) -> ImportRegistration:
        if len({s.document_id for s in self.sources}) != len(self.sources):
            raise ValueError("Registered document identities must be unique.")
        return self


class ImportOutcome(KnowledgeModel):
    revision_id: UUID
    reused: bool
    documents: int
    sections: int
    diagnostics: int
    failed_documents: int


async def import_registered(
    registration: ImportRegistration,
    *,
    source_root: Path,
    repository: KnowledgeRepository,
    source_store: SourceStore,
    extractor: DocumentExtractor,
    scope: KnowledgeWriteScope,
    retry_failed: bool = False,
) -> ImportOutcome:
    authorize_write(registration.collection, scope)
    root = source_root.resolve()
    prepared = []
    # Preflight every input before creating a database revision. Filenames locate
    # registered inputs only; stable IDs and bytes determine revision identity.
    for source in registration.sources:
        path = (root / source.path).resolve()
        if not path.is_relative_to(root):
            raise KnowledgeIntegrityError("Registered source escapes its allowed directory.")
        try:
            with path.open("rb") as stream:
                content = stream.read(MAX_SOURCE_BYTES + 1)
        except FileNotFoundError:
            raise KnowledgeNotFound("Registered source file is missing.") from None
        except PermissionError:
            raise KnowledgePermissionError("Registered source file cannot be read.") from None
        except OSError:
            raise KnowledgeUnavailable("Registered source file could not be opened.") from None
        if not content or len(content) > MAX_SOURCE_BYTES:
            raise KnowledgeIntegrityError("Registered source is empty or too large.")
        reference = source_store.put(content)
        prepared.append((source, reference, content))
    config = sorted(
        [s.model_dump(mode="json", exclude={"path"}) for s in registration.sources],
        key=lambda s: s["document_id"],
    )
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    revision = KnowledgeRevision(
        collection_id=registration.collection.collection_id,
        embedding=registration.embedding,
        pipeline=PipelineVersions(
            extractor=f"{EXTRACTOR_VERSION}:{config_hash}",
            chunker="not-run",
            policy="general-indexing-v2",
        ),
        manifest={s.document_id: ref.source_hash for s, ref, _ in prepared},
    )
    revision = revision.model_copy(
        update={
            "revision_id": uuid5(registration.collection.collection_id, revision.fingerprint),
        }
    )
    existing_collection = await repository.get_collection(
        registration.collection.collection_id, scope
    )
    if existing_collection is None:
        try:
            await repository.create_collection(registration.collection, scope=scope)
        except KnowledgeConflict:
            existing_collection = await repository.get_collection(
                registration.collection.collection_id, scope
            )
            if existing_collection is None:
                raise KnowledgePermissionError(
                    "Collection is unavailable to this import scope."
                ) from None
    if existing_collection is not None and existing_collection != registration.collection:
        raise KnowledgeConflict("Collection registration changed; use an explicit migration.")
    existing = await repository.get_revision(revision.revision_id, scope)
    if existing is None:
        try:
            existing = await repository.create_revision(revision, scope=scope)
        except KnowledgeConflict:
            existing = await repository.get_revision(revision.revision_id, scope)
            if existing is None:
                raise
    # A version-zero revision can be resumed after an interrupted extraction.
    # Completed imports are immutable to this command, including later processing.
    if existing.version > 0:
        stored = await repository.get_revision_data(existing.revision_id, scope)
        if stored is None:
            raise KnowledgeIntegrityError("Imported revision snapshot is missing.")
        if not retry_failed or not any(
            d.extraction_status == ExtractionStatus.FAILED for d in stored.documents
        ):
            return outcome(existing.revision_id, stored, reused=True)
    documents = []
    sections = []
    for source, reference, content in prepared:
        extracted = await asyncio.to_thread(extractor.extract, content)
        parsed, diagnostics = (
            split_sections(
                extracted.pages,
                revision_id=revision.revision_id,
                document_id=source.document_id,
                profile=source.profile,
            )
            if not extracted.failed
            else ([], [])
        )
        all_diagnostics = [*extracted.diagnostics, *diagnostics]
        if len(all_diagnostics) > 1000:
            all_diagnostics = [
                *all_diagnostics[:999],
                ExtractionDiagnostic(code="diagnostics_truncated"),
            ]
        status = (
            ExtractionStatus.FAILED
            if extracted.failed or not parsed
            else ExtractionStatus.EXTRACTED
        )
        documents.append(
            KnowledgeDocument(
                document_id=source.document_id,
                revision_id=revision.revision_id,
                title=source.title,
                source_uri=reference.uri,
                source_hash=reference.source_hash,
                workflow_authority=source.workflow_authority,
                extraction_status=status,
                pages=extracted.pages,
                diagnostics=all_diagnostics,
            )
        )
        sections.extend(parsed)
    data = RevisionData(documents=documents, sections=sections)
    try:
        await repository.write_revision_data(
            revision.revision_id, data, scope=scope, expected_version=existing.version
        )
    except KnowledgeConflict:
        stored = await repository.get_revision_data(revision.revision_id, scope)
        completed = await repository.get_revision(revision.revision_id, scope)
        if stored is None or completed is None or completed.version == 0:
            raise
        return outcome(revision.revision_id, stored, reused=True)
    return outcome(revision.revision_id, data, reused=False)


def outcome(revision_id: UUID, data: RevisionData, *, reused: bool) -> ImportOutcome:
    return ImportOutcome(
        revision_id=revision_id,
        reused=reused,
        documents=len(data.documents),
        sections=len(data.sections),
        diagnostics=sum(len(d.diagnostics) for d in data.documents),
        failed_documents=sum(
            d.extraction_status == ExtractionStatus.FAILED for d in data.documents
        ),
    )

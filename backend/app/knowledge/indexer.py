"""Source-independent indexing into an isolated, atomically completed revision."""

import asyncio
import hashlib
import json
import math
from collections.abc import Callable
from itertools import batched
from uuid import UUID, uuid5

from pydantic import Field, ValidationError

from app.knowledge.chunking import (
    CHUNKER_VERSION,
    ChunkDraft,
    ChunkingConfig,
    chunk_section,
    draft_slice,
    heading_at,
)
from app.knowledge.errors import (
    KnowledgeConflict,
    KnowledgeIntegrityError,
    KnowledgeNotFound,
    KnowledgeUnavailable,
)
from app.knowledge.metadata import METADATA_VERSION, MetadataConfig, extract_steps
from app.knowledge.repository import KnowledgeRepository, require_write_scope
from app.knowledge.schemas import (
    ChunkKind,
    EmbeddingProfile,
    KnowledgeChunk,
    KnowledgeModel,
    KnowledgeRevision,
    KnowledgeWriteScope,
    PipelineVersions,
    ProcessingStatus,
    RevisionData,
)
from app.services.model_service import ModelService
from app.services.model_service.exceptions import ModelEmbeddingError, ModelServiceError
from app.services.model_service.schemas import ModelEmbeddingRequest


class IndexConfig(KnowledgeModel):
    embedding: EmbeddingProfile
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    batch_size: int = Field(default=8, ge=1, le=32)


class IndexOutcome(KnowledgeModel):
    revision_id: UUID
    source_revision_id: UUID
    reused: bool
    status: ProcessingStatus
    documents: int
    sections: int
    chunks: int
    steps: int
    diagnostics: int
    error_code: str | None = None
    embedding_requests: int = 0


class IndexFailure(Exception):
    def __init__(self, code: str):
        self.code = code


def clone_extraction(data: RevisionData, revision_id: UUID) -> RevisionData:
    identities = {s.section_id: uuid5(revision_id, str(s.section_id)) for s in data.sections}
    return RevisionData(
        documents=[
            d.model_copy(update={"revision_id": revision_id}, deep=True) for d in data.documents
        ],
        sections=[
            s.model_copy(
                update={
                    "revision_id": revision_id,
                    "section_id": identities[s.section_id],
                    "parent_section_id": identities[s.parent_section_id]
                    if s.parent_section_id is not None
                    else None,
                },
                deep=True,
            )
            for s in data.sections
        ],
    )


async def index_revision(
    source_revision_id: UUID,
    *,
    config: IndexConfig,
    repository: KnowledgeRepository,
    model_service: ModelService,
    scope: KnowledgeWriteScope,
    progress: Callable[[int, int], None] | None = None,
) -> IndexOutcome:
    require_write_scope(scope)
    source = await repository.get_revision(source_revision_id, scope)
    original = await repository.get_revision_data(source_revision_id, scope)
    if source is None or original is None:
        raise KnowledgeNotFound("Source revision is unavailable to this scope.")
    reread = await repository.get_revision(source_revision_id, scope)
    if reread is None or reread.version != source.version:
        raise KnowledgeConflict("Source revision changed while reading its snapshot.")
    # Operational batching never changes evidence identity. Source snapshot version
    # does: concurrent extraction edits cannot silently reuse a different snapshot.
    identity = config.model_dump(mode="json", exclude={"batch_size"})
    identity.update(source_revision_id=str(source_revision_id), source_version=source.version)
    config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    revision = KnowledgeRevision(
        collection_id=source.collection_id,
        embedding=config.embedding,
        manifest=source.manifest,
        pipeline=PipelineVersions(
            extractor=source.pipeline.extractor,
            chunker=CHUNKER_VERSION + ":" + config.chunking.fingerprint,
            policy=METADATA_VERSION + ":" + config_hash,
        ),
    )
    revision = revision.model_copy(
        update={"revision_id": uuid5(source.collection_id, revision.fingerprint)}
    )
    try:
        saved = await repository.create_revision(revision, scope=scope)
    except KnowledgeConflict:
        existing = await repository.get_revision(revision.revision_id, scope)
        if existing is None:
            raise
        saved = existing
    prepared = clone_extraction(original, revision.revision_id)
    if saved.version == 0:
        try:
            saved = await repository.write_revision_data(
                revision.revision_id,
                prepared,
                scope=scope,
                expected_version=0,
            )
        except KnowledgeConflict:
            existing = await repository.get_revision(revision.revision_id, scope)
            if existing is None:
                raise
            saved = existing

    def outcome(
        current: KnowledgeRevision, data: RevisionData, *, reused=False, calls=0
    ) -> IndexOutcome:
        return IndexOutcome(
            revision_id=current.revision_id,
            source_revision_id=source_revision_id,
            reused=reused,
            status=current.indexing_status,
            error_code=current.indexing_error,
            documents=len(data.documents),
            sections=len(data.sections),
            chunks=len(data.chunks),
            steps=len(data.steps),
            diagnostics=sum(len(d.diagnostics) for d in data.documents),
            embedding_requests=calls,
        )

    if saved.indexing_status == ProcessingStatus.READY:
        stored = await repository.get_revision_data(saved.revision_id, scope)
        if stored is None:
            raise KnowledgeIntegrityError("Indexed snapshot is missing.")
        return outcome(saved, stored, reused=True)
    calls = 0
    try:

        async def check_profile():
            info = await model_service.inspect_embedding_model(config.embedding.model)
            if (
                info.provider != config.embedding.provider
                or info.revision != config.embedding.revision
                or info.dimension != config.embedding.dimension
            ):
                raise IndexFailure("embedding_profile_mismatch")

        await check_profile()
        documents = {d.document_id: d for d in prepared.documents}
        sections = {s.section_id: s for s in prepared.sections}
        steps, diagnostics = extract_steps(
            prepared.documents,
            prepared.sections,
            revision_id=revision.revision_id,
            collection_id=revision.collection_id,
            config=config.metadata,
        )
        step_keys = {s.step_id: s.namespace for s in steps}
        drafts = []
        heading_context: dict[UUID, list[str]] = {}
        for section in sorted(prepared.sections, key=lambda s: (str(s.document_id), s.ordinal)):
            document = documents[section.document_id]
            if document.extraction_status != "extracted" or section.processing_status != "ready":
                continue
            inherited = (
                [] if section.reference_id else heading_context.get(document.document_id, [])
            )
            drafts.extend(chunk_section(document, section, config.chunking, inherited))
            heading_context[document.document_id] = heading_at(
                section, len(section.text), inherited
            )[1:]
            if len(drafts) > config.chunking.max_chunks:
                raise IndexFailure("chunk_count_limit")
        if not drafts:
            raise IndexFailure("no_usable_content")
        output: list[KnowledgeChunk] = []

        async def embed(items: list[ChunkDraft]):
            nonlocal calls
            texts = [
                config.embedding.document_prefix + item.context_prefix + "\n" + item.content
                for item in items
            ]
            calls += 1
            try:
                response = await model_service.embed(
                    ModelEmbeddingRequest(
                        model=config.embedding.model,
                        texts=texts,
                        truncate=False,
                    )
                )
            except ModelEmbeddingError as exc:
                if not exc.details or not exc.details.get("input_too_long"):
                    raise
                if len(items) > 1:
                    for item in items:
                        await embed([item])
                    return
                item = items[0]
                if item.end - item.start <= 32:
                    raise IndexFailure("embedding_context_too_small") from None
                middle = (item.start + item.end) // 2
                boundary = item.content.rfind("\n", 0, len(item.content) // 2)
                if boundary > len(item.content) // 4:
                    middle = item.start + boundary + 1
                section = sections[item.section_id]
                for left, right in [(item.start, middle), (middle, item.end)]:
                    if section.text[left:right].strip():
                        await embed(
                            [
                                draft_slice(
                                    documents[item.document_id],
                                    section,
                                    left,
                                    right,
                                    config.chunking,
                                    item.inherited_headings,
                                )
                            ]
                        )
                return
            if (
                len(response.embeddings) != len(items)
                or response.provider != config.embedding.provider
                or response.model != config.embedding.model
            ):
                raise IndexFailure("invalid_embedding_response")
            for item, text, vector in zip(items, texts, response.embeddings, strict=True):
                if (
                    len(vector) != config.embedding.dimension
                    or not all(math.isfinite(v) for v in vector)
                    or not any(vector)
                ):
                    raise IndexFailure("invalid_embedding_vector")
                try:
                    norm = math.sqrt(math.fsum(v * v for v in vector))
                except OverflowError:
                    raise IndexFailure("invalid_embedding_vector") from None
                if not math.isfinite(norm) or norm == 0:
                    raise IndexFailure("invalid_embedding_vector")
                if config.embedding.normalized:
                    vector = [v / norm for v in vector]
                namespace = (
                    step_keys.get(item.reference_id) if item.reference_id is not None else None
                )
                output.append(
                    KnowledgeChunk(
                        chunk_id=item.identity(revision.revision_id),
                        revision_id=revision.revision_id,
                        document_id=item.document_id,
                        section_id=item.section_id,
                        kind=ChunkKind.EVIDENCE,
                        content=item.content,
                        context_prefix=item.context_prefix,
                        heading_path=item.heading_path,
                        source_start=item.start,
                        source_end=item.end,
                        page_spans=item.page_spans,
                        ordinal=len(output),
                        reference_id=item.reference_id,
                        namespace=namespace,
                        step_id=item.reference_id if namespace else None,
                        embedding=vector,
                        embedding_input_hash=hashlib.sha256(text.encode()).hexdigest(),
                    )
                )
                if len(output) > config.chunking.max_chunks:
                    raise IndexFailure("chunk_count_limit")
            if progress is not None:
                progress(len(output), len(drafts))

        for batch in batched(drafts, config.batch_size):
            await embed(list(batch))
        await check_profile()
        source_after = await repository.get_revision(source_revision_id, scope)
        if source_after is None or source_after.version != source.version:
            raise IndexFailure("source_changed")
        enriched_documents = []
        for document in prepared.documents:
            extra = diagnostics.get(document.document_id, [])
            combined = [*document.diagnostics, *extra]
            enriched_documents.append(document.model_copy(update={"diagnostics": combined[:1000]}))
        complete = prepared.model_copy(
            update={"documents": enriched_documents, "steps": steps, "chunks": output}
        )
        saved = await repository.write_revision_data(
            revision.revision_id,
            complete,
            scope=scope,
            expected_version=saved.version,
            index_complete=True,
        )
        return outcome(saved, complete, calls=calls)
    except KnowledgeConflict:
        winner = await repository.get_revision(revision.revision_id, scope)
        if winner is not None and winner.indexing_status == ProcessingStatus.READY:
            winner_data = await repository.get_revision_data(winner.revision_id, scope)
            if winner_data is None:
                raise KnowledgeIntegrityError("Indexed snapshot is missing.") from None
            return outcome(
                winner,
                winner_data,
                reused=True,
                calls=calls,
            )
        raise
    except (
        IndexFailure,
        ModelServiceError,
        ValidationError,
        KnowledgeIntegrityError,
        KnowledgeUnavailable,
    ) as exc:
        code = (
            exc.code
            if isinstance(exc, IndexFailure)
            else (
                "index_storage_unavailable"
                if isinstance(exc, KnowledgeUnavailable)
                else (
                    "embedding_failed"
                    if isinstance(exc, ModelServiceError)
                    else "invalid_index_data"
                )
            )
        )
        saved = await repository.record_index_failure(
            revision.revision_id,
            scope=scope,
            expected_version=saved.version,
            error_code=code,
        )
        return outcome(saved, prepared, calls=calls)
    except asyncio.CancelledError:
        # Pending/failed snapshots have no partial embeddings; a later run can resume.
        raise

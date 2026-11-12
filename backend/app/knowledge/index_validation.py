"""Technical completeness/provenance gates for index snapshots, never content approval."""

import hashlib
import math
from uuid import UUID

from app.knowledge.chunking import sliced_spans
from app.knowledge.errors import KnowledgeIntegrityError
from app.knowledge.schemas import ChunkKind, KnowledgeRevision, RevisionData


def validate_indexed_data(revision: KnowledgeRevision, data: RevisionData) -> None:
    usable = {
        s.section_id: s
        for s in data.sections
        if s.processing_status == "ready"
        and any(
            d.document_id == s.document_id and d.extraction_status == "extracted"
            for d in data.documents
        )
    }
    if not usable or not data.chunks:
        raise KnowledgeIntegrityError("An index requires usable source text and evidence.")
    coverage: dict[UUID, list[tuple[int, int]]] = {key: [] for key in usable}
    for chunk in data.chunks:
        section = usable.get(chunk.section_id)
        if (
            section is None
            or chunk.kind != ChunkKind.EVIDENCE
            or chunk.source_start is None
            or chunk.source_end is None
        ):
            raise KnowledgeIntegrityError("Indexed chunks require exact usable-source offsets.")
        if (
            chunk.content != section.text[chunk.source_start : chunk.source_end]
            or chunk.source_end > len(section.text)
            or chunk.page_spans != sliced_spans(section, chunk.source_start, chunk.source_end)
            or not chunk.page_spans
        ):
            raise KnowledgeIntegrityError("Indexed chunk provenance does not match its source.")
        text = revision.embedding.document_prefix + chunk.context_prefix + "\n" + chunk.content
        if chunk.embedding_input_hash != hashlib.sha256(text.encode()).hexdigest():
            raise KnowledgeIntegrityError("Embedding input provenance is invalid.")
        vector = chunk.embedding
        if vector is None:
            raise KnowledgeIntegrityError("Indexed evidence requires an embedding.")
        if revision.embedding.normalized and not math.isclose(
            math.sqrt(math.fsum(v * v for v in vector)),
            1,
            abs_tol=1e-5,
        ):
            raise KnowledgeIntegrityError("Embedding normalization does not match its profile.")
        coverage[section.section_id].append((chunk.source_start, chunk.source_end))
    for key, spans in coverage.items():
        section = usable[key]
        covered = 0
        for start, end in sorted(spans):
            if section.text[covered:start].strip():
                raise KnowledgeIntegrityError("Usable section text is missing from the index.")
            covered = max(covered, end)
        if section.text[covered:].strip():
            raise KnowledgeIntegrityError("Usable section text is missing from the index.")
    for step in data.steps:
        if not step.source_fields:
            raise KnowledgeIntegrityError("Indexed step facts require source provenance.")
        for source in step.source_fields.values():
            section = usable.get(source.section_id)
            if (
                section is None
                or section.document_id != step.source_document_id
                or source.text != section.text[source.start : source.end]
                or source.end > len(section.text)
                or source.page_spans != sliced_spans(section, source.start, source.end)
            ):
                raise KnowledgeIntegrityError("Step fact provenance does not match its source.")

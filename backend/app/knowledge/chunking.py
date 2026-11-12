"""Deterministic structural chunks with exact source slices and page provenance."""

import hashlib
import json
import re
from uuid import UUID, uuid5

from pydantic import Field, model_validator

from app.knowledge.schemas import (
    KnowledgeDocument,
    KnowledgeModel,
    KnowledgeSection,
    Label,
    PageSpan,
)

CHUNKER_VERSION = "contextual-sections-v1"


class ChunkingConfig(KnowledgeModel):
    max_chars: int = Field(default=1000, ge=128, le=3000)
    overlap_chars: int = Field(default=120, ge=0, le=1000)
    context_chars: int = Field(default=160, ge=0, le=512)
    max_chunks: int = Field(default=20000, ge=1, le=100000)

    @model_validator(mode="after")
    def bounds(self):
        if self.overlap_chars * 2 >= self.max_chars:
            raise ValueError("Overlap must be less than half the chunk size.")
        return self

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()


class ChunkDraft(KnowledgeModel):
    document_id: UUID
    section_id: UUID
    start: int
    end: int
    content: str
    context_prefix: str
    heading_path: list[Label]
    page_spans: list[PageSpan]
    reference_id: Label | None = None
    inherited_headings: list[Label] = Field(default_factory=list)

    def identity(self, revision_id: UUID) -> UUID:
        return uuid5(revision_id, f"{self.document_id}:{self.section_id}:{self.start}:{self.end}")


def sliced_spans(section: KnowledgeSection, start: int, end: int) -> list[PageSpan]:
    spans = []
    for span in section.page_spans:
        left = max(start, span.section_start)
        right = min(end, span.section_start + span.end - span.start)
        if right > left:
            spans.append(
                PageSpan(
                    page=span.page,
                    start=span.start + left - span.section_start,
                    end=span.start + right - span.section_start,
                    section_start=left - start,
                )
            )
    return spans


def heading_at(
    section: KnowledgeSection, position: int, inherited: list[str] | None = None
) -> list[str]:
    # Recognize syntax/structure, never subject matter or target agents.
    headings = [section.title, *(inherited or [])]
    for line in section.text[:position].splitlines():
        stripped = line.strip()
        match = re.match(r"^(#{1,6})\s+(.{1,200})$", stripped)
        numbered = re.match(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z][^.!?]{1,120}$", stripped)
        if match:
            level = len(match[1])
            headings = [section.title, *headings[1:level], match[2].strip()]
        elif numbered:
            headings = [section.title, stripped]
    return headings[:20]


def draft_slice(
    document: KnowledgeDocument,
    section: KnowledgeSection,
    start: int,
    end: int,
    config: ChunkingConfig,
    inherited_headings: list[str] | None = None,
) -> ChunkDraft:
    headings = heading_at(section, start, inherited_headings)
    # Include an opening heading when the chunk starts at it.
    first = section.text[start:end].splitlines()[0] if start < end else ""
    if re.match(r"^#{1,6}\s+", first.strip()):
        headings = [*headings, re.sub(r"^#{1,6}\s+", "", first.strip())[:200]]
    prefix = "\n".join([document.title, *dict.fromkeys(headings)])[: config.context_chars]
    return ChunkDraft(
        document_id=document.document_id,
        section_id=section.section_id,
        start=start,
        end=end,
        content=section.text[start:end],
        context_prefix=prefix,
        heading_path=headings,
        page_spans=sliced_spans(section, start, end),
        reference_id=section.reference_id,
        inherited_headings=inherited_headings or [],
    )


def chunk_section(
    document: KnowledgeDocument,
    section: KnowledgeSection,
    config: ChunkingConfig,
    inherited_headings: list[str] | None = None,
) -> list[ChunkDraft]:
    result = []
    start = 0
    while start < len(section.text):
        end = min(len(section.text), start + config.max_chars)
        if end < len(section.text):
            # Preserve complete lines/table rows wherever they fit. Long unbroken
            # lines fall back to word boundaries; no source text is discarded.
            boundary = section.text.rfind("\n", start, end)
            if boundary > start:
                end = boundary + 1
            else:
                boundary = section.text.rfind(" ", start + config.max_chars // 2, end)
                if boundary > start:
                    end = boundary + 1
        if section.text[start:end].strip():
            result.append(draft_slice(document, section, start, end, config, inherited_headings))
        if end == len(section.text):
            break
        next_start = end if end - start <= config.overlap_chars else end - config.overlap_chars
        # Start overlap at a complete line/word without skipping past the last end.
        boundary = section.text.find("\n", next_start, end)
        if boundary >= 0 and boundary + 1 < end:
            next_start = boundary + 1
        else:
            boundary = section.text.find(" ", next_start, end)
            if boundary >= 0 and boundary + 1 < end:
                next_start = boundary + 1
        start = next_start
    return result

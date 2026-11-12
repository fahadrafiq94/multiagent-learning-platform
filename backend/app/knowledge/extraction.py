"""Bounded, page-aware extraction. Structural references never assign agent eligibility."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any, Protocol
from uuid import UUID, uuid5

from pydantic import Field, model_validator

from app.knowledge.schemas import (
    ExtractedPage,
    ExtractionDiagnostic,
    KnowledgeModel,
    KnowledgeSection,
    Label,
    PageSpan,
)

MAX_SOURCE_BYTES = 25 * 1024 * 1024
EXTRACTOR_VERSION = "pdf-sections-v2"


class ExtractionResult(KnowledgeModel):
    pages: list[ExtractedPage] = Field(default_factory=list, max_length=500)
    diagnostics: list[ExtractionDiagnostic] = Field(default_factory=list, max_length=1000)
    failed: bool = False


class DocumentExtractor(Protocol):
    def extract(self, content: bytes) -> ExtractionResult: ...


class PdfExtractor:
    """Parsing runs in a disposable process with a wall-clock timeout and size caps.

    This is not an OS sandbox or an enforced memory limit. Deploy untrusted bulk
    uploads in resource-limited workers; this checkpoint accepts registered inputs.
    """

    def __init__(self, *, timeout_seconds: float = 60) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Extraction timeout must be positive.")
        self.timeout_seconds = timeout_seconds

    def extract(self, content: bytes) -> ExtractionResult:
        if not content or len(content) > MAX_SOURCE_BYTES:
            return failure("source_size_limit")
        try:
            process = subprocess.run(
                [sys.executable, "-m", "app.knowledge.pdf_worker"],
                input=content,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
                # Pin child stdio to UTF-8 instead of the host locale (e.g.
                # cp1252 on Windows); the worker emits UTF-8 bytes.
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            if process.returncode or len(process.stdout) > 16 * 1024 * 1024:
                return failure("extraction_worker_failed")
            return ExtractionResult.model_validate_json(process.stdout)
        except subprocess.TimeoutExpired:
            return failure("extraction_timeout")
        except (OSError, ValueError):
            return failure("extraction_worker_failed")


def failure(code: str) -> ExtractionResult:
    return ExtractionResult(failed=True, diagnostics=[ExtractionDiagnostic(code=code)])


class StructureProfile(KnowledgeModel):
    """Trusted registration data; no source-name or agent-routing rules."""

    reference_label: str = Field(default="Step ID", min_length=1, max_length=100)
    expected_references: list[Label] = Field(default_factory=list, max_length=500)

    @model_validator(mode="before")
    @classmethod
    def historical_profile(cls, values: Any) -> Any:
        # Earlier registrations required a hint ladder. Accept them as historical
        # input, but no general document is blocked for missing authored hints.
        if isinstance(values, dict):
            return {k: v for k, v in values.items() if k != "required_hint_levels"}
        return values


def split_sections(
    pages: list[ExtractedPage],
    *,
    revision_id: UUID,
    document_id: UUID,
    profile: StructureProfile,
) -> tuple[list[KnowledgeSection], list[ExtractionDiagnostic]]:
    text = "\n".join(page.text for page in pages)
    # Declared structural label + an identifier on this or the next line. Both
    # identifiers and profile expectations are data, never hardcoded course IDs.
    pattern = re.compile(
        r"(?mi)^[ \t]*"
        + re.escape(profile.reference_label)
        + r"[ \t]*:?[ \t]*(?:\r?\n[ \t]*)?([A-Za-z0-9][A-Za-z0-9_.-]{0,100})[ \t]*$"
    )
    matches = list(pattern.finditer(text))
    starts = [(m.start(), m.group(1)) for m in matches]
    if not starts:
        # General PDFs need no course references. Blank-line paragraph boundaries
        # provide neutral sections where extraction preserves them; otherwise a
        # bounded continuous section retains cross-page text for later review.
        starts = [(m.end(), None) for m in re.finditer(r"\n[ \t]*\n", text)]
    if not starts or starts[0][0] > 0:
        starts.insert(0, (0, None))
    page_offsets: list[tuple[int, ExtractedPage]] = []
    offset = 0
    for page in pages:
        page_offsets.append((offset, page))
        offset += len(page.text) + 1
    sections: list[KnowledgeSection] = []
    diagnostics: list[ExtractionDiagnostic] = []
    references = [m.group(1) for m in matches]
    for ref in sorted(set(references)):
        if references.count(ref) > 1:
            diagnostics.append(ExtractionDiagnostic(code="duplicate_reference", reference_id=ref))
    for ref in profile.expected_references:
        if ref not in references:
            diagnostics.append(ExtractionDiagnostic(code="missing_reference", reference_id=ref))
    if profile.expected_references:
        for ref in sorted(set(references) - set(profile.expected_references)):
            diagnostics.append(ExtractionDiagnostic(code="unexpected_reference", reference_id=ref))
    for index, (start, ref) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        # Sections are extraction records, not retrieval chunks. Bound very large
        # records; each part retains its structural reference and exact page spans.
        for part_start in range(start, end, 50000):
            part_end = min(end, part_start + 50000)
            body = text[part_start:part_end]
            if not body.strip():
                continue
            spans = []
            for page_start, page in page_offsets:
                left, right = (
                    max(part_start, page_start),
                    min(part_end, page_start + len(page.text)),
                )
                if right > left:
                    spans.append(
                        PageSpan(
                            page=page.number,
                            start=left - page_start,
                            end=right - page_start,
                            section_start=left - part_start,
                        )
                    )
            if not spans:
                continue
            ordinal = len(sections)
            sections.append(
                KnowledgeSection(
                    section_id=uuid5(document_id, f"{revision_id}:{ordinal}"),
                    revision_id=revision_id,
                    document_id=document_id,
                    title=ref or f"Document section {ordinal + 1}",
                    text=body,
                    page_start=spans[0].page,
                    page_end=spans[-1].page,
                    reference_id=ref,
                    ordinal=ordinal,
                    page_spans=spans,
                )
            )
    if len(diagnostics) > 1000:
        diagnostics = [*diagnostics[:999], ExtractionDiagnostic(code="diagnostics_truncated")]
    return sections, diagnostics

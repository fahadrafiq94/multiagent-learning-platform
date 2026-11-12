from __future__ import annotations

import hashlib
import json
import math
import struct
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def preserve_legacy(values: Any, names: tuple[str, ...]) -> Any:
    """Keep retired payload annotations readable, without using them as policy."""
    if not isinstance(values, dict):
        return values
    values = dict(values)
    legacy = dict(values.get("legacy_annotations", {}))
    for name in names:
        if name in values:
            legacy[name] = values.pop(name)
    if legacy:
        values["legacy_annotations"] = legacy
    return values


class KnowledgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkKind(StrEnum):
    EVIDENCE = "evidence"
    SUPPORT = "support"
    HINT = "hint"


class SharingClass(StrEnum):
    SHARED = "shared_instructional"
    PRIVATE = "owner_private"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class RevisionStatus(StrEnum):
    STAGING = "staging"
    VALIDATED = "validated"


class AccessScope(KnowledgeModel):
    """Trusted server-side access context, never arbitrary HTTP metadata."""

    access_domain: Label
    user_id: UUID | None = None


class KnowledgeWriteScope(AccessScope):
    """Trusted service grant; never deserialize this from student/model input.

    Read eligibility alone grants no editing permission. An authenticated service
    or operator command must explicitly grant management of this access domain.
    Private collections additionally require the matching owner identity.
    """

    can_manage_knowledge: bool = False


class KnowledgeCollection(KnowledgeModel):
    collection_id: UUID = Field(default_factory=uuid4)
    name: Label
    access_domain: Label
    sharing_class: SharingClass = SharingClass.SHARED
    owner_user_id: UUID | None = None
    course_profile: Label | None = None

    @model_validator(mode="after")
    def ownership(self) -> KnowledgeCollection:
        if (self.sharing_class == SharingClass.PRIVATE) != (self.owner_user_id is not None):
            raise ValueError("Only private collections require an owner.")
        return self

    def permits(self, scope: AccessScope) -> bool:
        return self.access_domain == scope.access_domain and (
            self.sharing_class == SharingClass.SHARED or self.owner_user_id == scope.user_id
        )


class EmbeddingProfile(KnowledgeModel):
    provider: Label
    model: Label
    revision: Label
    dimension: int = Field(ge=1, le=16000)
    metric: Literal["cosine"] = "cosine"
    normalized: bool = False
    query_prefix: str = Field(default="", max_length=1000)
    document_prefix: str = Field(default="", max_length=1000)


class PipelineVersions(KnowledgeModel):
    extractor: Label
    chunker: Label
    policy: Label
    legacy_versions: dict[Literal["classifier", "taxonomy"], Label] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def historical_versions(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        values = dict(values)
        legacy = dict(values.get("legacy_versions", {}))
        for name in ("classifier", "taxonomy"):
            if name in values:
                legacy[name] = values.pop(name)
        if legacy:
            values["legacy_versions"] = legacy
        return values


class KnowledgeRevision(KnowledgeModel):
    revision_id: UUID = Field(default_factory=uuid4)
    collection_id: UUID
    embedding: EmbeddingProfile
    pipeline: PipelineVersions
    manifest: dict[UUID, Digest]
    version: int = Field(default=0, ge=0)
    status: RevisionStatus = RevisionStatus.STAGING
    indexing_status: ProcessingStatus = ProcessingStatus.PENDING
    indexing_error: Label | None = None

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={
                "revision_id",
                "version",
                "status",
                "indexing_status",
                "indexing_error",
            },
        )
        # Historical revision identity includes its original opaque pipeline markers.
        # Flatten them back to reproduce the stored fingerprint without a classifier.
        legacy = payload["pipeline"].pop("legacy_versions")
        payload["pipeline"].update(legacy)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ExtractionStatus(StrEnum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    FAILED = "failed"


class ExtractionDiagnostic(KnowledgeModel):
    code: Label
    reference_id: Label | None = None
    page: int | None = Field(default=None, ge=1)


class ExtractedPage(KnowledgeModel):
    number: int = Field(ge=1)
    text: str = Field(max_length=100000)

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


class PageSpan(KnowledgeModel):
    page: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    section_start: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> PageSpan:
        if self.end <= self.start:
            raise ValueError("Empty or reversed page span.")
        return self


class KnowledgeDocument(KnowledgeModel):
    document_id: UUID = Field(default_factory=uuid4)
    revision_id: UUID
    title: Label
    source_uri: Label
    source_hash: Digest
    media_type: Label = "application/pdf"
    language: Label | None = None
    supplied_version: Label | None = None
    workflow_authority: bool = False
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    pages: list[ExtractedPage] = Field(default_factory=list, max_length=500)
    diagnostics: list[ExtractionDiagnostic] = Field(default_factory=list, max_length=1000)

    legacy_annotations: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def historical_status(cls, values: Any) -> Any:
        if not isinstance(values, dict) or values.get("extraction_status") != "needs_review":
            return values
        values = preserve_legacy(values, ("extraction_status",))
        usable = any(
            (page.get("text", "") if isinstance(page, dict) else page.text).strip()
            for page in values.get("pages", [])
        )
        values["extraction_status"] = "extracted" if usable else "failed"
        return values

    @model_validator(mode="after")
    def page_sequence(self) -> KnowledgeDocument:
        if [p.number for p in self.pages] != list(range(1, len(self.pages) + 1)):
            raise ValueError("Extracted pages must be contiguous and ordered.")
        if sum(len(p.text) for p in self.pages) > 2000000:
            raise ValueError("Extracted document text exceeds its limit.")
        return self


class KnowledgeSection(KnowledgeModel):
    section_id: UUID = Field(default_factory=uuid4)
    revision_id: UUID
    document_id: UUID
    title: Label
    text: str = Field(min_length=1, max_length=100000)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    processing_status: ProcessingStatus = ProcessingStatus.READY
    reference_id: Label | None = None
    ordinal: int = Field(default=0, ge=0)
    page_spans: list[PageSpan] = Field(default_factory=list)
    parent_section_id: UUID | None = None
    parent_text_hash: Digest | None = None
    legacy_annotations: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def historical_annotations(cls, values: Any) -> Any:
        return preserve_legacy(
            values,
            (
                "semantic_type",
                "review_status",
                "confidence",
                "evidence_spans",
                "review_note",
                "classification",
            ),
        )

    @model_validator(mode="after")
    def provenance(self) -> KnowledgeSection:
        if (self.parent_section_id is None) != (self.parent_text_hash is None):
            raise ValueError("Split sections require both parent identity and context hash.")
        if self.page_end < self.page_start:
            raise ValueError("Section page range is reversed.")
        return self


class FactSource(KnowledgeModel):
    section_id: UUID
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=100000)
    page_spans: list[PageSpan] = Field(min_length=1)


class ProcessStep(KnowledgeModel):
    revision_id: UUID
    namespace: Label
    step_id: Label
    source_document_id: UUID
    name: Label
    course_label: Label | None = None
    phase: Label | None = None
    business_object: Label | None = None
    prerequisites: dict[Label, str] = Field(default_factory=dict)
    next_step_id: Label | None = None
    source_fields: dict[Label, FactSource] = Field(default_factory=dict)


class KnowledgeChunk(KnowledgeModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    revision_id: UUID
    document_id: UUID
    section_id: UUID
    kind: ChunkKind
    legacy_annotations: dict[str, Any] = Field(default_factory=dict)
    content: str = Field(min_length=1, max_length=4000)
    namespace: Label | None = None
    step_id: Label | None = None
    reference_id: Label | None = None
    hint_level: int | None = Field(default=None, ge=1, le=4)
    support_ids: list[UUID] = Field(default_factory=list)
    embedding: list[float] | None = None
    context_prefix: str = Field(default="", max_length=512)
    heading_path: list[Label] = Field(default_factory=list, max_length=20)
    source_start: int | None = Field(default=None, ge=0)
    source_end: int | None = Field(default=None, gt=0)
    page_spans: list[PageSpan] = Field(default_factory=list)
    ordinal: int = Field(default=0, ge=0)
    embedding_input_hash: Digest | None = None

    @model_validator(mode="before")
    @classmethod
    def historical_annotations(cls, values: Any) -> Any:
        return preserve_legacy(values, ("semantic_type",))

    @field_validator("embedding")
    @classmethod
    def single_precision(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        try:
            rounded = [struct.unpack("!f", struct.pack("!f", item))[0] for item in value]
        except (OverflowError, struct.error):
            raise ValueError("Embedding components must fit single precision.") from None
        if not all(math.isfinite(item) for item in rounded):
            raise ValueError("Embedding components must be finite.")
        return rounded

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()

    @model_validator(mode="after")
    def shape(self) -> KnowledgeChunk:
        if (self.source_start is None) != (self.source_end is None):
            raise ValueError("Chunk source offsets require both start and end.")
        if self.source_start is not None and (
            self.source_end is None or self.source_end <= self.source_start
        ):
            raise ValueError("Chunk source offsets are reversed.")
        if (self.namespace is None) != (self.step_id is None):
            raise ValueError("Step identity requires both namespace and ID.")
        if self.kind == ChunkKind.EVIDENCE:
            if not self.embedding or not all(math.isfinite(x) for x in self.embedding):
                raise ValueError("Evidence needs a finite embedding.")
            if not any(x != 0 for x in self.embedding):
                raise ValueError("Zero vectors cannot support cosine distance.")
        elif self.embedding is not None:
            raise ValueError("Only evidence is embedded.")
        if self.kind == ChunkKind.HINT:
            if self.hint_level is None:
                raise ValueError("Authored hints require a level.")
        elif self.hint_level is not None:
            raise ValueError("Only hint records carry a hint level.")
        if self.chunk_id in self.support_ids or len(set(self.support_ids)) != len(self.support_ids):
            raise ValueError("Support links must be unique and cannot point to self.")
        return self


class RevisionData(KnowledgeModel):
    documents: list[KnowledgeDocument] = Field(default_factory=list)
    sections: list[KnowledgeSection] = Field(default_factory=list)
    steps: list[ProcessStep] = Field(default_factory=list)
    chunks: list[KnowledgeChunk] = Field(default_factory=list)

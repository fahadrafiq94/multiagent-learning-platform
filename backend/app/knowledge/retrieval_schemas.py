"""Bounded retrieval requests and source-linked evidence, independent of agent names."""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.knowledge.schemas import Digest, KnowledgeModel, KnowledgeRevision, Label, PageSpan


class RetrievalStatus(StrEnum):
    FOUND = "found"
    NO_EVIDENCE = "no_evidence"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class RetrievalConfig(KnowledgeModel):
    candidate_limit: int = Field(default=40, ge=1, le=100)
    min_vector_similarity: float = Field(default=0.50, ge=-1, le=1)
    lexical_vector_floor: float = Field(default=0.35, ge=-1, le=1)
    rrf_constant: int = Field(default=60, ge=1, le=1000)
    max_citations_per_item: int = Field(default=8, ge=1, le=20)


class RetrievalRequest(KnowledgeModel):
    collection_id: UUID
    # None resolves the active revision. Explicit ready revisions are for trusted
    # admissions/operator evaluation; publication is handled in Checkpoint 9.
    revision_id: UUID | None = None
    query: str = Field(min_length=1, max_length=2000)
    purpose: str = Field(default="", max_length=600)
    current_step_id: Label | None = None
    namespace: Label | None = None
    phase: Label | None = None
    reference_ids: list[Label] = Field(default_factory=list, max_length=20)
    top_k: int = Field(default=4, ge=1, le=20)
    max_chars: int = Field(default=8000, ge=1, le=24000)

    @model_validator(mode="after")
    def nonblank(self):
        if not self.query.strip():
            raise ValueError("Retrieval query must not be blank.")
        return self


MetadataField = Literal[
    "step_id", "name", "course_label", "phase", "business_object", "prerequisites", "next_step_id"
]


def metadata_fields() -> list[MetadataField]:
    return [
        "step_id",
        "name",
        "phase",
        "business_object",
        "prerequisites",
        "next_step_id",
        "course_label",
    ]


class MetadataRequest(KnowledgeModel):
    collection_id: UUID
    revision_id: UUID | None = None
    namespace: Label | None = None
    step_id: Label
    fields: list[
        Literal[
            "step_id",
            "name",
            "course_label",
            "phase",
            "business_object",
            "prerequisites",
            "next_step_id",
        ]
    ] = Field(
        default_factory=metadata_fields,
        min_length=1,
        max_length=7,
    )
    max_chars: int = Field(default=8000, ge=1, le=24000)


class RevisionPin(KnowledgeModel):
    revision: KnowledgeRevision
    # From trusted collection admission, never student-supplied metadata.
    cache_owner_user_id: UUID | None = None


class SearchCandidate(KnowledgeModel):
    chunk_id: UUID
    vector_similarity: float = Field(ge=-1.001, le=1.001, allow_inf_nan=False)
    lexical_score: float = Field(ge=0, allow_inf_nan=False)
    vector_rank: int | None = Field(default=None, ge=1)
    lexical_rank: int | None = Field(default=None, ge=1)
    identifier_rank: int | None = Field(default=None, ge=1)
    reference_id: Label | None = None
    step_id: Label | None = None
    current_step_match: bool = False
    phase_match: bool = False
    purpose_score: float = Field(default=0, ge=0, allow_inf_nan=False)


class EvidenceCitation(KnowledgeModel):
    revision_id: UUID
    document_id: UUID
    section_id: UUID
    chunk_id: UUID | None = None
    field_name: Label | None = None
    title: Label
    source_hash: Digest
    reference_id: Label | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    page_spans: list[PageSpan] = Field(min_length=1, max_length=500)


class RetrievedEvidence(KnowledgeModel):
    evidence_id: UUID
    kind: Literal["chunk", "metadata"]
    content: str = Field(min_length=1, max_length=100000)
    context_prefix: str = Field(default="", max_length=512)
    reference_id: Label | None = None
    namespace: Label | None = None
    field_name: Label | None = None
    citations: list[EvidenceCitation] = Field(min_length=1, max_length=20)
    score: float = Field(default=0, allow_inf_nan=False)
    vector_similarity: float | None = Field(default=None, ge=-1.001, le=1.001, allow_inf_nan=False)
    lexical_score: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    source_start: int | None = None
    source_end: int | None = None


class SourceBundle(KnowledgeModel):
    document_id: UUID
    revision_id: UUID
    title: Label
    evidence_ids: list[UUID]
    pages: list[int]


class RetrievalStats(KnowledgeModel):
    candidate_count: int = 0
    vector_candidates: int = 0
    lexical_candidates: int = 0
    identifier_candidates: int = 0
    returned_count: int = 0
    deduplicated: int = 0
    budget_omissions: int = 0
    omitted_fields: list[str] = Field(default_factory=list)
    omitted_references: list[str] = Field(default_factory=list)
    model_calls: int = 0
    ranker_version: str = "rrf-context-v1"


class RetrievalResult(KnowledgeModel):
    status: RetrievalStatus
    reason: Label
    collection_id: UUID
    revision_id: UUID | None = None
    revision_version: int | None = None
    items: list[RetrievedEvidence] = Field(default_factory=list)
    sources: list[SourceBundle] = Field(default_factory=list)
    stats: RetrievalStats = Field(default_factory=RetrievalStats)

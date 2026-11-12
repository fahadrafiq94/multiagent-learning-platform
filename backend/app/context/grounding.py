"""Neutral request-scoped evidence contracts; no retrieval/provider dependencies."""

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.context.schemas import ContextModel, MessageSourceReference


class EvidenceContext(ContextModel):
    evidence_id: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=100000)
    context_prefix: str = Field(default="", max_length=512)
    references: list[MessageSourceReference] = Field(min_length=1, max_length=20)


class GroundingBudget(ContextModel):
    max_context_tokens: int = Field(default=8192, ge=256, le=262144)
    reserved_output_tokens: int = Field(default=2048, ge=64, le=16384)
    characters_per_token: float = Field(default=4, gt=0, le=16)
    message_overhead_tokens: int = Field(default=32, ge=0, le=1024)
    reasoning_enabled: bool | None = None

    @model_validator(mode="after")
    def valid_budget(self):
        if self.reserved_output_tokens >= self.max_context_tokens:
            raise ValueError("Output reservation must leave input capacity.")
        return self


class GroundingContext(GroundingBudget):
    status: Literal["skipped", "clarify", "found", "no_evidence", "unavailable", "error"]
    items: list[EvidenceContext] = Field(default_factory=list, max_length=12)
    current_message_id: UUID | None = None

    @model_validator(mode="after")
    def found_evidence(self):
        if self.status != "found" and self.items:
            raise ValueError("Only a found result may contain evidence.")
        return self

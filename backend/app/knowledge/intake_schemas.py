"""Trusted deployment policy and durable PDF intake contracts."""

from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.knowledge.extraction import StructureProfile
from app.knowledge.indexer import IndexConfig
from app.knowledge.schemas import Digest, KnowledgeModel, KnowledgeWriteScope, Label


class IntakeBinding(KnowledgeModel):
    collection_id: UUID
    scope: KnowledgeWriteScope
    manager_user_ids: list[UUID] = Field(min_length=1, max_length=1000)
    index: IndexConfig
    initial_revision_id: UUID | None = None

    @model_validator(mode="after")
    def management(self):
        if not self.scope.can_manage_knowledge:
            raise ValueError("Intake requires an explicit management grant.")
        return self


class IntakeConfig(KnowledgeModel):
    bindings: list[IntakeBinding] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def unique_collections(self):
        if len({b.collection_id for b in self.bindings}) != len(self.bindings):
            raise ValueError("Collection intake bindings must be unambiguous.")
        return self

    @classmethod
    def read(cls, path: Path):
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Intake configuration is too large.")
        return cls.model_validate_json(path.read_bytes())


class IntakePayload(KnowledgeModel):
    document_id: UUID
    source_hash: Digest
    title: Label
    replace: bool = False
    expected_source_hash: Digest | None = None
    workflow_authority: bool = False
    profile: StructureProfile = Field(default_factory=StructureProfile)
    index: IndexConfig


class IntakeJob(KnowledgeModel):
    job_id: UUID
    collection_id: UUID
    status: Literal["queued", "processing", "succeeded", "failed"]
    attempts: int
    base_active_id: UUID | None = None
    base_revision_id: UUID | None = None
    result_revision_id: UUID | None = None
    duplicate: bool = False
    error_code: str | None = None
    lease_token: UUID | None = None
    created_at: datetime
    updated_at: datetime
    payload: IntakePayload

    def public(self) -> dict:
        return self.model_dump(mode="json", exclude={"payload", "lease_token"}) | {
            "document_id": str(self.payload.document_id)
        }

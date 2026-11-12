"""Request-scoped planning contracts. None of these models grant access or persist state."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from app.agents.schemas import AgentName
from app.context.agent_views import RoutingFacts
from app.context.schemas import AgentContext
from app.knowledge.retrieval_schemas import MetadataField, RetrievedEvidence, SourceBundle
from app.knowledge.schemas import AccessScope, KnowledgeModel, Label

ActionName = Literal["skip", "reuse_evidence", "metadata_lookup", "document_search", "clarify"]


class PlannerAction(KnowledgeModel):
    action: ActionName
    query: str | None = Field(default=None, min_length=1, max_length=2000)
    step_id: Label | None = None
    fields: list[MetadataField] = Field(default_factory=list, max_length=7)
    evidence_ids: list[UUID] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def action_arguments(self):
        if self.action == "document_search":
            valid = bool(self.query and self.query.strip()) and not (
                self.step_id or self.fields or self.evidence_ids
            )
        elif self.action == "metadata_lookup":
            valid = bool(self.step_id and self.fields) and not (self.query or self.evidence_ids)
        elif self.action == "reuse_evidence":
            valid = bool(self.evidence_ids) and not (self.query or self.step_id or self.fields)
        else:
            valid = not (self.query or self.step_id or self.fields or self.evidence_ids)
        if not valid:
            raise ValueError("Arguments do not match the retrieval action.")
        return self


class RetrievalPlan(KnowledgeModel):
    need: Literal["none", "documents", "ambiguous"]
    reason: str = Field(min_length=1, max_length=400)
    actions: list[PlannerAction] = Field(min_length=1, max_length=3)
    clarification: str | None = Field(default=None, min_length=1, max_length=300)

    @model_validator(mode="after")
    def consistent_actions(self):
        names = [a.action for a in self.actions]
        if self.need == "none":
            valid = names == ["skip"] and self.clarification is None
        elif self.need == "ambiguous":
            valid = names == ["clarify"] and bool(self.clarification and self.clarification.strip())
        else:
            valid = not {"skip", "clarify"}.intersection(names) and self.clarification is None
        if not valid or not self.reason.strip():
            raise ValueError("Plan need, actions and explanation must agree.")
        return self


class EvidenceReceipt(KnowledgeModel):
    """Trusted server-side prior evidence handle; never accept this from a student or LLM."""

    user_id: UUID
    session_id: UUID
    scope: AccessScope
    collection_id: UUID
    revision_id: UUID
    created_at: AwareDatetime
    items: list[RetrievedEvidence] = Field(min_length=1, max_length=12)


class PlannerRequest(KnowledgeModel):
    user_id: UUID
    session_id: UUID
    collection_id: UUID
    revision_id: UUID | None = None
    specialist: AgentName
    question: str = Field(min_length=1, max_length=2000)
    current_step_id: Label | None = None
    namespace: Label | None = None
    phase: Label | None = None
    objective: str = Field(default="", max_length=800)
    blocker: str = Field(default="", max_length=800)
    conversation: str = Field(default="", max_length=4000)
    summary: str = Field(default="", max_length=1500)
    scenario_facts: RoutingFacts | None = None
    # Trusted operational flags, not a keyword classifier or model-proposed permission.
    enabled: bool = True
    onboarding_required: bool = False
    # Explicit caller assertion used ONLY when semantic planning fails.
    clear_course_query: bool = False
    max_chars: int = Field(default=8000, ge=0, le=24000)
    max_items: int = Field(default=12, ge=1, le=20)
    available: list[EvidenceReceipt] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def nonblank(self):
        if not self.question.strip():
            raise ValueError("Question must not be blank.")
        return self

    @classmethod
    def from_context(
        cls, context: AgentContext, *, specialist: AgentName, collection_id: UUID, **kwargs
    ) -> "PlannerRequest":
        """Project bounded existing context; source chunks/history are never sufficient by fiat."""
        state = context.session_context
        history = []
        for message in context.recent_messages:
            if message.role == "user" and message.content == context.current_message:
                continue
            history.append(f"{message.role}: {message.content[:1000]}")
        return cls(
            user_id=context.user_id,
            session_id=context.session_id,
            collection_id=collection_id,
            specialist=specialist,
            question=context.current_message,
            current_step_id=state.current_step_id if state else None,
            phase=state.current_process_phase if state else None,
            objective=(state.current_objective or "")[:800] if state else "",
            blocker=(state.active_blocker or "")[:800] if state else "",
            conversation="\n".join(history)[-4000:],
            summary=(
                (context.session_summary.overview or "")[:1500] if context.session_summary else ""
            ),
            scenario_facts=RoutingFacts.from_card(context.scenario_card),
            **kwargs,
        )


class PlannerConfig(KnowledgeModel):
    model: Label
    timeout_seconds: float = Field(default=60, gt=0, le=180)
    max_tokens: int = Field(default=650, ge=100, le=2000)
    max_prompt_chars: int = Field(default=18000, ge=4000, le=32000)
    evidence_preview_chars: int = Field(default=5000, ge=0, le=8000)
    reuse_ttl_seconds: int = Field(default=300, ge=1, le=3600)


class ActionTrace(KnowledgeModel):
    action: ActionName
    status: str
    reason: Label
    returned_count: int = 0
    model_calls: int = 0


class PlannerResult(KnowledgeModel):
    status: Literal["skipped", "clarify", "found", "no_evidence", "unavailable", "error"]
    reason: Label
    collection_id: UUID
    revision_id: UUID | None = None
    plan: RetrievalPlan | None = None
    origin: Literal["gate", "model", "fallback"] = "gate"
    planner_version: Literal["hybrid-v1"] = "hybrid-v1"
    planner_model: Label | None = None
    planner_failure: Label | None = None
    planner_failure_detail: Label | None = None
    planner_calls: int = 0
    planner_latency_ms: float = 0
    elapsed_ms: float = 0
    trace: list[ActionTrace] = Field(default_factory=list)
    items: list[RetrievedEvidence] = Field(default_factory=list)
    sources: list[SourceBundle] = Field(default_factory=list)
    budget_omissions: int = 0
    rejected_receipts: int = 0


def receipt(result: PlannerResult, request: PlannerRequest, scope: AccessScope, now: datetime):
    """Called only by trusted application code after successful evidence execution."""
    if result.status != "found" or result.revision_id is None or not result.items:
        raise ValueError("Only found evidence can be reused.")
    return EvidenceReceipt(
        user_id=request.user_id,
        session_id=request.session_id,
        scope=AccessScope(access_domain=scope.access_domain, user_id=scope.user_id),
        collection_id=result.collection_id,
        revision_id=result.revision_id,
        created_at=now,
        items=result.items[:12],
    )

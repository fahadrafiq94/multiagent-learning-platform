"""Trusted course bindings and request-scoped grounding after specialist routing."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from pydantic import AnyHttpUrl, Field, ValidationError, model_validator

from app.agents.schemas import AgentName
from app.context.grounding import EvidenceContext, GroundingBudget, GroundingContext
from app.context.schemas import (
    AgentContext,
    MessageSourceReference,
    PendingEvidenceReceipt,
    SessionContextUpdate,
)
from app.knowledge.errors import KnowledgeError, KnowledgeNotFound
from app.knowledge.planner import HybridRetrievalService
from app.knowledge.planner_schemas import (
    EvidenceReceipt,
    PlannerConfig,
    PlannerRequest,
    PlannerResult,
)
from app.knowledge.retrieval_schemas import MetadataRequest, RetrievedEvidence, RevisionPin
from app.knowledge.schemas import AccessScope, KnowledgeModel, Label
from app.services.session_context_service import SessionContextService


class CourseBinding(KnowledgeModel):
    user_id: UUID
    collection_id: UUID
    access_domain: Label
    namespace: Label | None = None
    # Trusted pilot admission only; None follows the active collection pointer.
    revision_id: UUID | None = None


class RagRuntimeConfig(KnowledgeModel):
    planner: PlannerConfig
    bindings: list[CourseBinding] = Field(min_length=1, max_length=10000)
    prompt: GroundingBudget = Field(default_factory=GroundingBudget)
    planner_url: AnyHttpUrl | None = None
    embedding_url: AnyHttpUrl | None = None
    max_evidence_chars: int = Field(default=8000, ge=1, le=24000)

    @model_validator(mode="after")
    def unique_users(self):
        if len({binding.user_id for binding in self.bindings}) != len(self.bindings):
            raise ValueError("Each pilot user must have one unambiguous trusted course binding.")
        return self

    @classmethod
    def read(cls, path: Path):
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("RAG configuration exceeds its limit.")
        return cls.model_validate_json(path.read_bytes())


@dataclass
class GroundingAdmission:
    context: GroundingContext
    trace: dict
    scope: AccessScope | None = None
    pin: RevisionPin | None = None
    clarification: str | None = None
    binding: CourseBinding | None = None


def project_evidence(item: RetrievedEvidence) -> EvidenceContext:
    return EvidenceContext(
        evidence_id=str(item.evidence_id),
        content=item.content,
        context_prefix=item.context_prefix,
        references=[
            MessageSourceReference(
                document_id=str(c.document_id),
                revision_id=str(c.revision_id),
                chunk_id=str(c.chunk_id) if c.chunk_id else None,
                kind=item.kind,
                evidence_id=str(item.evidence_id),
                field_name=item.field_name,
                reference_id=item.reference_id,
                namespace=item.namespace,
                source_hash=c.source_hash,
                title=c.title,
                section=str(c.section_id),
                page_start=c.page_start,
                page_end=c.page_end,
            )
            for c in item.citations
        ],
    )


class SpecialistGroundingService:
    def __init__(
        self,
        planner: HybridRetrievalService,
        config: RagRuntimeConfig,
        session_context_service: (SessionContextService | None) = None,
    ):
        self.planner = planner
        self.config = config
        self.session_context_service = session_context_service
        self.bindings = {binding.user_id: binding for binding in config.bindings}

    def _context(self, status, items=(), message_id=None) -> GroundingContext:
        return GroundingContext(
            **self.config.prompt.model_dump(),
            status=status,
            items=[project_evidence(item) for item in items],
            current_message_id=message_id,
        )

    async def _rehydrate_refs(self, refs, pin, scope) -> dict:
        """Reread archived references against the live pinned revision (bounded to 12)."""
        repository = self.planner.retrieval.repository
        items: dict = {}
        for ref in list(refs)[:24]:
            try:
                if UUID(ref.revision_id) != pin.revision.revision_id:
                    continue
                if ref.kind == "chunk":
                    fresh = await repository.evidence(pin, scope, [UUID(ref.chunk_id)])
                elif ref.reference_id and ref.field_name:
                    fresh, _ = await repository.metadata(
                        pin,
                        scope,
                        MetadataRequest(
                            collection_id=pin.revision.collection_id,
                            revision_id=pin.revision.revision_id,
                            namespace=ref.namespace,
                            step_id=ref.reference_id,
                            fields=[ref.field_name],
                        ),
                    )
                else:
                    continue
                for item in fresh:
                    if ref.evidence_id and str(item.evidence_id) != ref.evidence_id:
                        continue
                    items[item.evidence_id] = item
                    if len(items) == 12:
                        break
            except (ValueError, TypeError, KnowledgeNotFound):
                continue
            if len(items) == 12:
                break
        return items

    async def _clear_pending_receipt(self, context) -> None:
        service = self.session_context_service
        if service is None:
            return
        try:
            await service.update(
                user_id=context.user_id,
                session_id=context.session_id,
                changes=SessionContextUpdate(pending_evidence_receipt=None),
            )
        except Exception:
            # Consuming the bridge must never break the turn; any leftover
            # receipt is revalidated against the live revision before use.
            pass

    async def _prior_receipt(self, context, scope, pin):
        now = datetime.now(UTC)
        receipt = (
            context.session_context.pending_evidence_receipt
            if context.session_context is not None
            else None
        )
        if receipt is not None:
            await self._clear_pending_receipt(context)
            fresh = (
                receipt.created_at.tzinfo is not None
                and 0
                <= (now - receipt.created_at).total_seconds()
                <= self.config.planner.reuse_ttl_seconds
            )
            if fresh and receipt.references:
                items = await self._rehydrate_refs(receipt.references, pin, scope)
                if items:
                    return EvidenceReceipt(
                        user_id=context.user_id,
                        session_id=context.session_id,
                        scope=scope,
                        collection_id=pin.revision.collection_id,
                        revision_id=pin.revision.revision_id,
                        created_at=receipt.created_at,
                        items=list(items.values()),
                    )
        checked = 0
        for message in reversed(context.recent_messages):
            if message.created_at.tzinfo is None:
                continue
            age = (now - message.created_at).total_seconds()
            if message.role != "assistant" or not 0 <= age <= self.config.planner.reuse_ttl_seconds:
                continue
            if not message.source_references:
                continue
            checked += 1
            if checked > 2:
                break
            items = await self._rehydrate_refs(message.source_references, pin, scope)
            if items:
                return EvidenceReceipt(
                    user_id=context.user_id,
                    session_id=context.session_id,
                    scope=scope,
                    collection_id=pin.revision.collection_id,
                    revision_id=pin.revision.revision_id,
                    created_at=message.created_at,
                    items=list(items.values()),
                )
        return None

    async def prepare(
        self,
        context: AgentContext,
        specialist: AgentName,
        *,
        current_message_id: UUID | None = None,
    ) -> GroundingAdmission:
        context = AgentContext.model_validate(context.model_dump())
        binding = self.bindings.get(context.user_id)
        trace = {"status": "unavailable", "reason": "knowledge_unavailable", "planner_calls": 0}
        if binding is None:
            return GroundingAdmission(
                self._context("unavailable", message_id=current_message_id), trace
            )
        scope = AccessScope(access_domain=binding.access_domain, user_id=context.user_id)
        repository = self.planner.retrieval.repository
        pin = None
        try:
            pin = await repository.admit(binding.collection_id, binding.revision_id, scope)
            if pin is None:
                return GroundingAdmission(
                    self._context("unavailable", message_id=current_message_id), trace
                )
            available = await self._prior_receipt(context, scope, pin)
            request = PlannerRequest.from_context(
                context,
                specialist=specialist,
                collection_id=binding.collection_id,
                revision_id=pin.revision.revision_id,
                namespace=binding.namespace,
                max_chars=self.config.max_evidence_chars,
                available=[available] if available else [],
            )
            result: PlannerResult = await self.planner.execute(request, scope=scope)
            if result.collection_id != binding.collection_id or (
                result.revision_id is not None and result.revision_id != pin.revision.revision_id
            ):
                raise ValueError("Grounding result must match its admitted revision.")
            trace = {
                "status": result.status,
                "reason": result.reason,
                "origin": result.origin,
                "planner_calls": result.planner_calls,
                "planner_latency_ms": result.planner_latency_ms,
                "planner_failure": result.planner_failure,
                "actions": [a.action for a in result.trace],
                "embedding_calls": sum(a.model_calls for a in result.trace),
                "selected_count": len(result.items),
                "budget_omissions": result.budget_omissions,
            }
            return GroundingAdmission(
                self._context(result.status, result.items, current_message_id),
                trace,
                scope,
                pin,
                result.plan.clarification if result.status == "clarify" and result.plan else None,
                binding,
            )
        except (KnowledgeError, ValidationError, ValueError):
            return GroundingAdmission(
                self._context("error", message_id=current_message_id),
                {"status": "error", "reason": "grounding_preparation_failed", "planner_calls": 0},
            )

    async def authorize(self, admission: GroundingAdmission) -> None:
        if admission.context.items and admission.pin and admission.scope:
            if (
                admission.scope.user_id is None
                or self.bindings.get(admission.scope.user_id) != admission.binding
            ):
                raise KnowledgeNotFound("Course binding is no longer available.")
            await self.planner.retrieval.repository.check_pin(admission.pin, admission.scope)

    async def record_unverified_receipt(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        admission: GroundingAdmission,
    ) -> None:
        """
        Bridge retrieved-but-uncited evidence to the next turn.

        The answer was grounded but failed verification, so the student
        received no citations; the same evidence may still answer the
        student's follow-up. The receipt is app-owned session state,
        never prompt content. Best-effort: a persistence failure only
        forfeits the bridge, never the already-completed turn.
        """

        service = self.session_context_service
        if service is None or admission.context.status != "found":
            return
        references = [
            reference for item in admission.context.items for reference in item.references
        ][:12]
        if not references:
            return
        try:
            await service.update(
                user_id=user_id,
                session_id=session_id,
                changes=SessionContextUpdate(
                    pending_evidence_receipt=PendingEvidenceReceipt(
                        created_at=datetime.now(UTC),
                        references=references,
                    )
                ),
            )
        except Exception:
            pass

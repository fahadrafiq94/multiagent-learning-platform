"""Hybrid semantic planning and read-only execution; no chat graph/state mutations."""

import asyncio
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from app.knowledge.errors import KnowledgeError, KnowledgeNotFound
from app.knowledge.planner_prompt import planner_chat
from app.knowledge.planner_schemas import (
    ActionTrace,
    PlannerAction,
    PlannerConfig,
    PlannerRequest,
    PlannerResult,
    RetrievalPlan,
)
from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.retrieval_ranking import item_chars, references, source_bundles
from app.knowledge.retrieval_schemas import (
    MetadataRequest,
    RetrievalRequest,
    RetrievedEvidence,
    RevisionPin,
)
from app.knowledge.schemas import AccessScope
from app.services.model_service import ModelService
from app.services.model_service.exceptions import ModelServiceError, ModelTimeoutError

PURPOSES = {
    "scenario": "Understand the business scenario, goals and constraints.",
    "process_coach": "Explain process concepts, relationships and business prerequisites.",
    "ap_plus_navigator": "Find navigation, entry steps and verification conditions.",
}


class InvalidPlan(ValueError):
    pass


class HybridRetrievalService:
    def __init__(
        self,
        retrieval: KnowledgeRetrievalService,
        model_service: ModelService,
        config: PlannerConfig,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.retrieval = retrieval
        self.model_service = model_service
        self.config = config
        self.now = now

    def _search(
        self,
        request: PlannerRequest,
        pin: RevisionPin,
        query: str,
        covered_refs: set[str] | None = None,
    ) -> RetrievalRequest:
        original = RetrievalRequest(collection_id=request.collection_id, query=request.question)
        return RetrievalRequest(
            collection_id=request.collection_id,
            revision_id=pin.revision.revision_id,
            query=query,
            purpose=PURPOSES[request.specialist],
            current_step_id=request.current_step_id,
            namespace=request.namespace,
            phase=request.phase,
            # Preserve still-uncovered references across the combined operation. An
            # earlier metadata/reuse hit must not dominate a search for a different gap.
            reference_ids=[
                ref for ref in references(original) if ref not in (covered_refs or set())
            ],
            top_k=min(request.max_items, 20),
            max_chars=max(1, request.max_chars),
        )

    async def _rehydrate(
        self, pin: RevisionPin, scope: AccessScope, item: RetrievedEvidence
    ) -> RetrievedEvidence | None:
        """Only fresh repository text/provenance may reach a planner or specialist."""
        if any(c.revision_id != pin.revision.revision_id for c in item.citations):
            return None
        if item.kind == "chunk":
            fresh = await self.retrieval.repository.evidence(pin, scope, [item.evidence_id])
        elif item.reference_id and item.field_name:
            lookup = MetadataRequest(
                collection_id=pin.revision.collection_id,
                revision_id=pin.revision.revision_id,
                namespace=item.namespace,
                step_id=item.reference_id,
                fields=[item.field_name],
            )
            fresh, _ = await self.retrieval.repository.metadata(pin, scope, lookup)
        else:
            return None
        return next(
            (candidate for candidate in fresh if candidate.evidence_id == item.evidence_id), None
        )

    async def _available(self, request, scope, pin):
        available = {}
        rejected = 0
        for handle in request.available:
            age = (self.now() - handle.created_at).total_seconds()
            if (
                handle.user_id != request.user_id
                or handle.session_id != request.session_id
                or handle.scope != scope
                or handle.collection_id != request.collection_id
                or handle.revision_id != pin.revision.revision_id
                or not 0 <= age <= self.config.reuse_ttl_seconds
            ):
                rejected += 1
                continue
            for item in handle.items:
                try:
                    fresh = await self._rehydrate(pin, scope, item)
                except (KnowledgeNotFound, ValidationError):
                    fresh = None
                if fresh is not None:
                    available[fresh.evidence_id] = fresh
                else:
                    rejected += 1
        return available, rejected

    def _validate_plan(self, plan, request, available):
        # Only IDs actually shown in the bounded prompt are eligible for reuse.
        chat = planner_chat(request, list(available.values()), self.config)
        shown = {
            entry["evidence_id"]
            for entry in json.loads(chat.messages[1].content)["available_evidence"]
        }
        refs = references(
            RetrievalRequest(collection_id=request.collection_id, query=request.question)
        )
        allowed_steps = {value.casefold() for value in refs}
        if request.current_step_id:
            allowed_steps.add(request.current_step_id.casefold())
        for item in available.values():
            if item.reference_id:
                allowed_steps.add(item.reference_id.casefold())
        for action in plan.actions:
            if action.action == "reuse_evidence" and any(
                str(eid) not in shown for eid in action.evidence_ids
            ):
                raise InvalidPlan("unseen_evidence")
            if (
                action.action == "metadata_lookup"
                and action.step_id.casefold() not in allowed_steps
                and re.search(
                    r"(?<!\w)" + re.escape(action.step_id) + r"(?!\w)",
                    request.question,
                    re.IGNORECASE,
                )
                is None
            ):
                raise InvalidPlan("ungrounded_metadata_id")
        if refs and plan.need == "ambiguous":
            # An explicit identifier target in the question must not be dismissed as
            # ambiguous; a stale conversation anchor must not suppress retrieval.
            raise InvalidPlan("clarify_despite_reference")
        if refs and plan.need == "documents":
            if not any(action.action == "document_search" for action in plan.actions):
                covered = {a.step_id.casefold() for a in plan.actions if a.step_id}
                for action in plan.actions:
                    for eid in action.evidence_ids:
                        if available[eid].reference_id:
                            covered.add(available[eid].reference_id.casefold())
                # This is only identifier coverage, not a claim of semantic sufficiency.
                if not set(refs).issubset(covered):
                    raise InvalidPlan("omitted_references")

    def _fallback(self, request: PlannerRequest) -> RetrievalPlan:
        refs = references(
            RetrievalRequest(collection_id=request.collection_id, query=request.question)
        )
        # Any identifier target in the question is a search target: searching the
        # student's own words is bounded and read-only, unlike a stale clarification.
        if request.clear_course_query or refs:
            return RetrievalPlan(
                need="documents",
                reason="Semantic planning failed; search the original trusted course request.",
                actions=[PlannerAction(action="document_search", query=request.question)],
            )
        return RetrievalPlan(
            need="ambiguous",
            reason="Semantic planning failed and a useful course target is not established.",
            actions=[PlannerAction(action="clarify")],
            clarification="Which course topic or process step should I help you with?",
        )

    async def execute(self, request: PlannerRequest, *, scope: AccessScope) -> PlannerResult:
        request = PlannerRequest.model_validate(request.model_dump())
        scope = AccessScope(access_domain=scope.access_domain, user_id=scope.user_id)
        started = time.perf_counter()
        values: dict[str, Any] = {
            "collection_id": request.collection_id,
            "planner_model": self.config.model,
        }

        def result(status, reason, **extra) -> PlannerResult:
            payload = {**values, **extra}
            if status in {"unavailable", "error"}:
                # Reasons/queries can quote private text read before revocation.
                # Suppress derived content too, while retaining safe diagnostics.
                payload.update(plan=None, items=[], sources=[])
            return PlannerResult(
                **payload,
                status=status,
                reason=reason,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        # Cheap operational checks precede SQL, source hydration and inference.
        if not request.enabled:
            return result("skipped", "retrieval_disabled")
        if request.onboarding_required:
            return result("skipped", "onboarding_required")
        if not request.max_chars:
            return result("no_evidence", "budget_exhausted")
        if scope.user_id is not None and scope.user_id != request.user_id:
            return result("unavailable", "revision_unavailable")
        pin = None
        try:
            pin = await self.retrieval.repository.admit(
                request.collection_id, request.revision_id, scope
            )
            if pin is None:
                return result("unavailable", "revision_unavailable")
            values["revision_id"] = pin.revision.revision_id
            available, rejected = await self._available(request, scope, pin)
            values["rejected_receipts"] = rejected
            values["origin"] = "model"
            planner_started = time.perf_counter()
            try:
                chat = planner_chat(request, list(available.values()), self.config)
                values["planner_calls"] = 1
                async with asyncio.timeout(self.config.timeout_seconds):
                    response = await self.model_service.generate(chat)
                if len(response.content) > 8000:
                    raise InvalidPlan("response_too_large")
                plan = RetrievalPlan.model_validate_json(response.content)
                self._validate_plan(plan, request, available)
            except (ModelServiceError, TimeoutError, ValidationError, ValueError) as exc:
                values["origin"] = "fallback"
                values["planner_failure"] = (
                    "planner_timeout"
                    if isinstance(exc, (TimeoutError, ModelTimeoutError))
                    else "planner_unavailable"
                    if isinstance(exc, ModelServiceError)
                    else "invalid_plan"
                )
                values["planner_failure_detail"] = (
                    str(exc)
                    if isinstance(exc, InvalidPlan)
                    else exc.errors(include_input=False)[0]["type"]
                    if isinstance(exc, ValidationError)
                    else exc.code
                    if isinstance(exc, ModelServiceError)
                    else "timeout"
                    if isinstance(exc, TimeoutError)
                    else "input_budget"
                )
                plan = self._fallback(request)
            values["planner_latency_ms"] = round((time.perf_counter() - planner_started) * 1000, 2)
            values["plan"] = plan
            # No transaction is held while generating. Check any permission change now.
            await self.retrieval.repository.check_pin(pin, scope)
            if plan.need == "none":
                return result("skipped", "no_external_knowledge_needed")
            if plan.need == "ambiguous":
                return result("clarify", "target_unclear")

            selected: list[RetrievedEvidence] = []
            trace = []
            omissions = 0
            for action in plan.actions:
                remaining = request.max_chars - sum(item_chars(item) for item in selected)
                if remaining <= 0 or len(selected) >= request.max_items:
                    omissions += 1
                    trace.append(
                        ActionTrace(
                            action=action.action, status="not_executed", reason="budget_exhausted"
                        )
                    )
                    continue
                if action.action == "reuse_evidence":
                    items = []
                    for eid in dict.fromkeys(action.evidence_ids):
                        fresh = await self._rehydrate(pin, scope, available[eid])
                        if fresh is None:
                            raise KnowledgeNotFound("Evidence is no longer available.")
                        items.append(fresh)
                    outcome = ActionTrace(
                        action=action.action,
                        status="found",
                        reason="fresh_sources",
                        returned_count=len(items),
                    )
                else:
                    if action.action == "metadata_lookup":
                        retrieved = await self.retrieval.lookup(
                            MetadataRequest(
                                collection_id=request.collection_id,
                                revision_id=pin.revision.revision_id,
                                namespace=request.namespace,
                                step_id=action.step_id,
                                fields=action.fields,
                                max_chars=remaining,
                            ),
                            scope=scope,
                        )
                    else:
                        assert action.query is not None  # Validated action contract.
                        covered = {
                            item.reference_id.casefold() for item in selected if item.reference_id
                        }
                        search = self._search(request, pin, action.query, covered)
                        search = search.model_copy(update={"max_chars": remaining})
                        retrieved = await self.retrieval.search(search, scope=scope)
                    items = retrieved.items
                    omissions += retrieved.stats.budget_omissions
                    outcome = ActionTrace(
                        action=action.action,
                        status=retrieved.status,
                        reason=retrieved.reason,
                        returned_count=len(items),
                        model_calls=retrieved.stats.model_calls,
                    )
                trace.append(outcome)
                values["trace"] = trace
                if outcome.status in ("error", "unavailable"):
                    return result(outcome.status, outcome.reason)
                for item in items:
                    duplicate = next(
                        (
                            i
                            for i, old in enumerate(selected)
                            if old.kind == item.kind
                            and old.content == item.content
                            and old.field_name == item.field_name
                            and old.namespace == item.namespace
                        ),
                        None,
                    )
                    candidate = list(selected)
                    if duplicate is None:
                        candidate.append(item)
                    else:
                        old = candidate[duplicate]
                        citations = list(old.citations)
                        citations.extend(c for c in item.citations if c not in citations)
                        candidate[duplicate] = old.model_copy(update={"citations": citations[:20]})
                    if (
                        len(candidate) > request.max_items
                        or sum(map(item_chars, candidate)) > request.max_chars
                    ):
                        omissions += 1
                    else:
                        selected = candidate
            await self.retrieval.repository.check_pin(pin, scope)
            return result(
                "found" if selected else "no_evidence",
                "evidence_selected"
                if selected
                else "budget_exhausted"
                if omissions
                else "no_matches",
                items=selected,
                sources=source_bundles(selected),
                budget_omissions=omissions,
            )
        except KnowledgeNotFound:
            return result("unavailable", "revision_unavailable")
        except (KnowledgeError, ValidationError):
            return result("error", "knowledge_unavailable")

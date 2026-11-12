"""App-owned pending-evidence bridge: single-turn reuse of retrieved-but-uncited refs."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.context.grounding import EvidenceContext, GroundingContext
from app.context.schemas import (
    AgentContext,
    ConversationMessage,
    MessageSourceReference,
    PendingEvidenceReceipt,
    SessionContext,
)
from app.knowledge.errors import KnowledgeNotFound
from app.knowledge.integration import (
    CourseBinding,
    GroundingAdmission,
    RagRuntimeConfig,
    SpecialistGroundingService,
    project_evidence,
)
from app.knowledge.planner_schemas import PlannerConfig, PlannerResult
from app.knowledge.retrieval_schemas import RevisionPin
from app.knowledge.schemas import AccessScope, KnowledgeRevision, PipelineVersions
from tests.knowledge.test_indexer import PROFILE
from tests.knowledge.test_retrieval_ranking import evidence

NOW = datetime(2026, 11, 4, tzinfo=UTC)
USER_ID = UUID(int=10)
SESSION_ID = UUID(int=11)
COLLECTION_ID = UUID(int=12)
REVISION_ID = UUID(int=50)


@pytest.fixture(autouse=True)
def receipt_clock(monkeypatch):
    class ReceiptClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is not None else NOW.replace(tzinfo=None)

    monkeypatch.setattr("app.knowledge.integration.datetime", ReceiptClock)


def chunk_ref(number: int = 1, revision: UUID = REVISION_ID) -> MessageSourceReference:
    return MessageSourceReference(
        document_id=str(UUID(int=100 + number)),
        revision_id=str(revision),
        chunk_id=str(UUID(int=number)),
        kind="chunk",
        evidence_id=str(UUID(int=number)),
        page_start=1,
        page_end=1,
        source_hash="a" * 64,
    )


async def setup(
    pending: PendingEvidenceReceipt | None = None,
    assistant_refs: Sequence[MessageSourceReference] = (),
):
    repository = AsyncMock()
    repository.admit.return_value = RevisionPin(
        revision=KnowledgeRevision(
            revision_id=REVISION_ID,
            collection_id=COLLECTION_ID,
            embedding=PROFILE,
            pipeline=PipelineVersions(extractor="v1", chunker="v1", policy="v1"),
            manifest={},
            version=2,
            indexing_status="ready",
        )
    )
    repository.evidence.return_value = [evidence(1)]
    planner = AsyncMock()
    planner.retrieval.repository = repository
    planner.execute.return_value = PlannerResult(
        status="found",
        reason="matches",
        collection_id=COLLECTION_ID,
        revision_id=REVISION_ID,
    )
    context = AgentContext(
        user_id=USER_ID,
        session_id=SESSION_ID,
        current_message="Why is delivery required before invoice?",
        session_context=SessionContext(
            user_id=USER_ID,
            session_id=SESSION_ID,
            pending_evidence_receipt=pending,
        ),
        recent_messages=[
            ConversationMessage(
                user_id=USER_ID,
                session_id=SESSION_ID,
                role="assistant",
                content="Earlier grounded answer.",
                created_at=NOW,
                source_references=list(assistant_refs),
            )
        ],
    )
    config = RagRuntimeConfig(
        planner=PlannerConfig(model="test-planner"),
        bindings=[
            CourseBinding(
                user_id=USER_ID,
                collection_id=COLLECTION_ID,
                access_domain="test",
            )
        ],
    )
    session_service = AsyncMock()
    service = SpecialistGroundingService(planner, config, session_service)
    scope = AccessScope(access_domain="test", user_id=USER_ID)
    return service, planner, repository, session_service, scope, context


@pytest.mark.asyncio
async def test_pending_receipt_is_reused_and_consumed_once():
    service, planner, _, session_service, scope, context = await setup(
        pending=PendingEvidenceReceipt(created_at=NOW, references=[chunk_ref(1)]),
    )

    admission = await service.prepare(context, "process_coach")

    assert admission.context.status == "found"
    available = planner.execute.await_args.args[0].available
    assert len(available) == 1
    assert available[0].revision_id == REVISION_ID
    assert str(available[0].items[0].evidence_id) == str(UUID(int=1))
    # The receipt is cleared on the admitting turn, whatever the planner decides.
    assert session_service.update.await_args.kwargs["changes"].pending_evidence_receipt is None


@pytest.mark.asyncio
async def test_stale_pending_receipt_is_dropped_but_still_cleared():
    service, planner, _, session_service, scope, context = await setup(
        pending=PendingEvidenceReceipt(
            created_at=NOW - timedelta(seconds=301), references=[chunk_ref(1)]
        ),
    )

    await service.prepare(context, "process_coach")

    assert planner.execute.await_args.args[0].available == []
    session_service.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_revision_mismatch_falls_back_to_archived_message_refs():
    service, planner, _, session_service, scope, context = await setup(
        pending=PendingEvidenceReceipt(
            created_at=NOW, references=[chunk_ref(1, revision=UUID(int=51))]
        ),
        assistant_refs=[chunk_ref(1)],
    )

    await service.prepare(context, "process_coach")

    available = planner.execute.await_args.args[0].available
    assert len(available) == 1
    assert available[0].created_at == NOW
    session_service.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_pending_and_no_archived_refs_yields_no_available():
    service, planner, _, session_service, scope, context = await setup()

    await service.prepare(context, "process_coach")

    assert planner.execute.await_args.args[0].available == []
    session_service.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_unverified_receipt_persists_bounded_references():
    service, _, _, session_service, _, _ = await setup()
    admission = GroundingAdmission(
        context=GroundingContext(
            status="found",
            items=[
                EvidenceContext(
                    evidence_id=str(UUID(int=i)),
                    content="Source evidence.",
                    references=[chunk_ref(i), chunk_ref(i + 20)],
                )
                for i in range(1, 13)
            ],
        ),
        trace={"status": "found"},
    )

    await service.record_unverified_receipt(
        user_id=USER_ID, session_id=SESSION_ID, admission=admission
    )

    changes = session_service.update.await_args.kwargs
    assert changes["user_id"] == USER_ID and changes["session_id"] == SESSION_ID
    receipt = changes["changes"].pending_evidence_receipt
    assert len(receipt.references) == 12
    assert receipt.created_at.tzinfo is not None


@pytest.mark.asyncio
async def test_record_unverified_receipt_ignores_non_found_admissions():
    service, _, _, session_service, _, _ = await setup()
    admission = GroundingAdmission(
        context=GroundingContext(status="no_evidence"),
        trace={"status": "no_evidence"},
    )

    await service.record_unverified_receipt(
        user_id=USER_ID, session_id=SESSION_ID, admission=admission
    )

    session_service.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_and_clear_failures_never_break_the_turn():
    service, _, _, session_service, scope, context = await setup(
        pending=PendingEvidenceReceipt(created_at=NOW, references=[chunk_ref(1)]),
    )
    session_service.update.side_effect = RuntimeError("database unavailable")

    await service.prepare(context, "process_coach")

    admission = GroundingAdmission(
        context=GroundingContext(status="found", items=[project_evidence(evidence(1))]),
        trace={"status": "found"},
    )
    await service.record_unverified_receipt(
        user_id=USER_ID, session_id=SESSION_ID, admission=admission
    )


@pytest.mark.asyncio
async def test_consumed_receipt_is_not_available_on_a_second_admission():
    service, planner, _, session_service, _, context = await setup(
        pending=PendingEvidenceReceipt(created_at=NOW, references=[chunk_ref(1)])
    )

    async def clear(**kwargs):
        context.session_context.pending_evidence_receipt = kwargs[
            "changes"
        ].pending_evidence_receipt

    session_service.update.side_effect = clear
    await service.prepare(context, "process_coach")
    assert len(planner.execute.await_args.args[0].available) == 1
    await service.prepare(context, "process_coach")
    assert planner.execute.await_args.args[0].available == []
    session_service.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_revoked_pending_reference_is_consumed_without_reusing_evidence():
    service, planner, repository, session_service, _, context = await setup(
        pending=PendingEvidenceReceipt(created_at=NOW, references=[chunk_ref(1)])
    )
    repository.evidence.side_effect = KnowledgeNotFound("Access revoked")
    await service.prepare(context, "process_coach")
    assert planner.execute.await_args.args[0].available == []
    session_service.update.assert_awaited_once()
    assert repository.evidence.await_args.args[1].user_id == USER_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["user_id", "session_id"])
async def test_pending_receipt_cannot_be_imported_from_another_owner_or_session(identity):
    service, planner, repository, _, _, context = await setup(
        pending=PendingEvidenceReceipt(created_at=NOW, references=[chunk_ref(1)])
    )
    context = context.model_copy(
        update={
            "session_context": context.session_context.model_copy(update={identity: UUID(int=999)})
        }
    )
    with pytest.raises(ValidationError):
        await service.prepare(context, "process_coach")
    repository.admit.assert_not_awaited()
    planner.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_receipt_is_absent_from_agent_prompts_and_llm_update_fields():
    from app.agents.context_prompt import context_view_to_prompt_text
    from app.context.agent_views import ContextTarget
    from app.context.policies import AgentContextPolicyService
    from app.context.state_updates import SessionStateUpdateProposal

    _, _, _, _, _, context = await setup(
        pending=PendingEvidenceReceipt(created_at=NOW, references=[chunk_ref(1)])
    )
    for target in ContextTarget:
        view = AgentContextPolicyService().select(context=context, target=target)
        prompt = context_view_to_prompt_text(view)
        assert "pending_evidence_receipt" not in prompt
        assert str(UUID(int=101)) not in prompt
    with pytest.raises(ValidationError):
        SessionStateUpdateProposal.model_validate(
            {
                "operations": [
                    {
                        "field": "pending_evidence_receipt",
                        "action": "set",
                        "value": "forged",
                    }
                ]
            }
        )

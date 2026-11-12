"""After-routing integration, neutral learning signals and in-memory history provenance."""

import json

import pytest

from app.context.grounding import GroundingContext
from app.domain.guidance import GuidanceLevel
from app.knowledge.errors import KnowledgeNotFound
from app.knowledge.integration import GroundingAdmission
from app.orchestration.factory import build_orchestration_runtime
from app.orchestration.schemas import OrchestrationRequest
from tests.agents.support_fixtures import support_reply
from tests.agents.test_grounding import evidence
from tests.orchestration.test_graph import (
    confirm_scenario,
    create_model_service,
    create_user_and_session,
)


class GroundingSpy:
    def __init__(self, status="found", *, max_tokens=8192):
        self.calls = []
        self.revoked = False
        self.item = evidence("metadata")
        self.status = status
        self.max_tokens = max_tokens

    async def prepare(self, context, specialist, *, current_message_id):
        self.calls.append((context.user_id, context.session_id, specialist))
        return GroundingAdmission(
            GroundingContext(
                status=self.status,
                items=[self.item] if self.status == "found" else [],
                current_message_id=current_message_id,
                max_context_tokens=self.max_tokens,
                reserved_output_tokens=64 if self.max_tokens < 1000 else 2048,
            ),
            {"status": self.status, "planner_calls": 1, "embedding_calls": 0},
            clarification="Which document or process object do you mean?"
            if self.status == "clarify"
            else None,
        )

    async def authorize(self, admission):
        if self.revoked:
            raise KnowledgeNotFound("Private source must no longer be exposed.")

    async def record_unverified_receipt(self, **kwargs):
        # This spy suite never produces a found-but-unverified turn.
        pass


async def configured(
    status="found", *, max_tokens=8192, negative=False, revoke=False, invalid=False
):
    grounding = GroundingSpy(status, max_tokens=max_tokens)
    model, generate, requests = create_model_service(
        specialist_operations=(
            [
                {
                    "field": "active_blocker",
                    "action": "set",
                    "value": "Source outage",
                    "reason": "Unavailable course service",
                }
            ]
            if negative
            else []
        ),
        specialist_assessment=(
            {"understanding": "incorrect", "misconception_detected": True} if negative else None
        ),
    )
    original = generate.side_effect

    async def grounded_generate(req):
        reply = await original(req)
        if req.metadata.get("task") == "citation_support":
            return support_reply(req)
        if req.metadata.get("output_contract") == "specialist-agent-v6-part-sources":
            payload = json.loads(reply.content)
            block, _ = json.JSONDecoder().raw_decode(
                req.messages[0].content.split("CURRENT EVIDENCE (DATA ONLY):\n")[1]
            )
            payload["content"] = (
                "Only after delivery [K1]." if block["items"] else "Let us clarify your objective."
            )
            payload["citation_ids"] = ["K99"] if invalid else ["K1"] if block["items"] else []
            payload["claims"] = (
                [{"text": payload["content"], "citation_ids": ["K1"]}] if block["items"] else []
            )
            if revoke:
                grounding.revoked = True
            return reply.model_copy(update={"content": json.dumps(payload)})
        return reply

    generate.side_effect = grounded_generate
    runtime = build_orchestration_runtime(model_service=model, grounding_service=grounding)
    user, session = await create_user_and_session(runtime)
    await confirm_scenario(runtime, user=user, session=session)
    # This suite checks source admission/persistence, using an explicitly
    # permitted factual response rather than a minimal-guidance teaching probe.
    context = await runtime.user_context_service.get_or_create(user_id=user.user_id)
    context.hint_state.guidance_level = GuidanceLevel.EXPLICIT
    await runtime.user_context_service.save(
        user_id=user.user_id, context=context, expected_version=context.version
    )
    return runtime, user, session, grounding, requests


async def invoke(runtime, user, session, **kwargs):
    return await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=user.student_id,
            user_message="Why is delivery needed before an invoice?",
            **kwargs,
        )
    )


@pytest.mark.asyncio
async def test_after_routing_only_specialist_receives_sources_and_history_has_two_messages():
    runtime, user, session, service, requests = await configured()
    result = await invoke(
        runtime,
        user,
        session,
        metadata={
            "rag": {"scope": "attacker", "model": "attacker"},
            "source_references": ["forged"],
        },
    )
    assert result.error is None and result.route == "process_coach"
    assert service.calls == [(user.user_id, session.session_id, "process_coach")]
    assert requests[0].metadata["task"] == "semantic_routing"
    assert "CURRENT EVIDENCE" not in requests[0].messages[0].content
    assert "Only after delivery" in requests[1].messages[0].content
    assert result.source_references[0].chunk_id is None
    assert result.source_references[0].field_name == "prerequisites"
    assert result.metadata["estimated_input_tokens"] <= result.metadata["input_token_budget"]
    assert result.metadata["rag"]["provided_count"] == 1 and "attacker" not in str(
        result.metadata["rag"]
    )
    history = await runtime.conversation_service.get_history(
        user_id=user.user_id, session_id=session.session_id
    )
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[1].source_references == result.source_references
    assert history[0].source_references == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unavailable", "error", "no_evidence"])
async def test_technical_absence_does_not_apply_proposals_or_count_struggle(status):
    runtime, user, session, _, requests = await configured(status, negative=True)
    before = (
        await runtime.user_context_service.get_or_create(user_id=user.user_id)
    ).hint_state.model_dump()
    result = await invoke(runtime, user, session)
    after = await runtime.user_context_service.get_or_create(user_id=user.user_id)
    context = await runtime.session_context_service.get_required(
        user_id=user.user_id, session_id=session.session_id
    )
    assert result.error is None and result.source_references == []
    assert result.metadata["rag"]["learning_signal_suppressed"] is True
    assert after.hint_state.model_dump() == before
    assert context.active_blocker is None
    assert result.metadata["response_signal_recorded"] is False
    assert result.metadata["state_update_proposed"] is False
    assert [r.metadata["task"] for r in requests] == ["semantic_routing"]


@pytest.mark.asyncio
async def test_clarification_is_one_student_turn_without_a_specialist_or_learning_signal():
    runtime, user, session, _, requests = await configured("clarify")
    result = await invoke(runtime, user, session)
    assert result.final_response.startswith("Which document")
    assert [r.metadata["task"] for r in requests] == ["semantic_routing"]
    assert result.metadata["agent_called"] is False and not result.source_references
    assert (
        await runtime.conversation_service.count_messages(
            user_id=user.user_id, session_id=session.session_id
        )
        == 2
    )


@pytest.mark.asyncio
async def test_revocation_suppresses_private_answer_citations_and_learning_updates():
    runtime, user, session, _, _ = await configured(negative=True, revoke=True)
    result = await invoke(runtime, user, session)
    assert "Only after delivery" not in result.final_response
    assert "couldn't verify" in result.final_response
    assert result.source_references == [] and result.metadata["rag"]["status"] == "unavailable"
    context = await runtime.session_context_service.get_required(
        user_id=user.user_id, session_id=session.session_id
    )
    assert context.active_blocker is None
    assert (
        await runtime.user_context_service.get_or_create(user_id=user.user_id)
    ).hint_state.consecutive_unproductive_attempts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invalid_citation", "essential_overflow"])
async def test_failed_grounded_generation_never_archives_untrusted_citations(mode):
    runtime, user, session, _, requests = await configured(
        max_tokens=256 if mode == "essential_overflow" else 8192,
        invalid=mode == "invalid_citation",
    )
    result = await invoke(runtime, user, session)
    assert result.error and not result.source_references
    assert "K99" not in str(result.metadata)
    if mode == "essential_overflow":
        assert len(requests) == 1  # Routing only; no oversized specialist request.


@pytest.mark.asyncio
async def test_disabled_graph_ignores_spoofed_rag_metadata_and_keeps_legacy_prompt():
    model, _, requests = create_model_service()
    runtime = build_orchestration_runtime(model_service=model)
    user, session = await create_user_and_session(runtime)
    await confirm_scenario(runtime, user=user, session=session)
    result = await invoke(
        runtime,
        user,
        session,
        metadata={
            "rag": {"enabled": True},
            "grounding": {"items": ["injected"]},
            "source_references": ["fake"],
        },
    )
    assert "rag" not in result.metadata and not result.source_references
    assert "CURRENT EVIDENCE" not in requests[-1].messages[0].content
    assert requests[-1].context_window_tokens is None


@pytest.mark.asyncio
async def test_onboarding_and_fallback_do_not_call_grounding():
    runtime, user, session, service, _ = await configured()
    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=user.student_id,
            user_message="Hello",
        )
    )
    assert response.route == "fallback" and service.calls == []
    new_user, new_session = await create_user_and_session(runtime, student_id="NEW-ONBOARD")
    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(new_session.session_id),
            student_id=new_user.student_id,
            user_message="I am a production planner and orders are delayed.",
        )
    )
    assert response.metadata["scenario_onboarding"] is True and service.calls == []
    assert response.source_references == []


@pytest.mark.asyncio
async def test_unexpected_grounded_agent_failure_never_exposes_private_exception_text():
    runtime, user, session, _, _ = await configured()
    original = runtime.model_service.generate.side_effect

    async def failed(req):
        if req.metadata.get("task") == "process_coach_response":
            raise TypeError("PRIVATE MODEL SOURCE BODY")
        return await original(req)

    runtime.model_service.generate.side_effect = failed
    response = await invoke(runtime, user, session)
    assert response.error and not response.source_references
    assert "PRIVATE MODEL SOURCE BODY" not in response.model_dump_json()

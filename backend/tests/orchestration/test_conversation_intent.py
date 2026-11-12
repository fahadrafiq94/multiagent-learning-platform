import json
from unittest.mock import AsyncMock

import pytest

from app.context.agent_views import ContextTarget
from app.orchestration.schemas import OrchestrationRequest
from app.services.model_service.schemas import ModelChatResponse
from tests.orchestration.test_student_step import GroundingSpy, saved, setup_runtime

pytestmark = pytest.mark.asyncio


def purpose(kind="acknowledgement", target=None, quote=None):
    return {"kind": kind, "target": target, "source_quote": quote}


def classify(generate, interaction, route="fallback"):
    original = generate.side_effect

    async def dispatch(request):
        if request.metadata.get("task") == "student_progress_verification":
            return ModelChatResponse(
                content=json.dumps({"action": "unchanged", "step_id": None, "source_quote": None}),
                model="controlled",
                provider="ollama",
                latency_ms=0,
            )
        if request.metadata.get("task") == "semantic_routing":
            return ModelChatResponse(
                content=json.dumps(
                    {
                        "route": route,
                        "reason": "Controlled interaction test.",
                        "current_step": {
                            "action": "unchanged",
                            "step_id": None,
                            "source_quote": None,
                        },
                        "interaction": interaction,
                    }
                ),
                model="controlled",
                provider="ollama",
                latency_ms=0,
            )
        response = await original(request)
        if request.metadata.get("task") == "process_coach_response":
            # Exercise routing and the shared verification call with a valid
            # minimal question, rather than a newly rejected teaching format.
            payload = json.loads(response.content)
            payload["content"] = "Which relationship would you reason through first?"
            return response.model_copy(update={"content": json.dumps(payload)})
        return response

    generate.side_effect = dispatch


async def invoke(runtime, user, session, message, metadata=None):
    return await runtime.orchestration_service.execute(
        OrchestrationRequest(
            student_id=user.student_id,
            session_id=str(session.session_id),
            user_message=message,
            metadata=metadata or {},
        )
    )


async def test_clear_thanks_skips_grounding_specialist_summary_and_learning_signals():
    spy = GroundingSpy()
    runtime, user, session, generate, _ = await setup_runtime(grounding=spy)
    classify(generate, purpose())
    context_before = await saved(runtime, user, session)
    user_before = await runtime.user_context_service.get_or_create(user_id=user.user_id)
    runtime.session_summary_service.maybe_update = AsyncMock()
    result = await invoke(runtime, user, session, "Thanks, that makes sense.")
    assert result.final_response.startswith("You're welcome")
    assert generate.await_count == 1 and spy.steps == []
    assert result.metadata["agent_called"] is False
    assert result.metadata["response_signal_recorded"] is False
    assert result.metadata["summary_update_attempted"] is False
    runtime.session_summary_service.maybe_update.assert_not_awaited()
    assert result.metadata["rag"]["actions"] == []
    assert result.metadata["rag"]["planner_calls"] == 0
    assert result.source_references == []
    assert (await saved(runtime, user, session)).version == context_before.version
    assert (
        await runtime.user_context_service.get_or_create(user_id=user.user_id)
    ).version == user_before.version


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("current_step_id", "FLOW-A2"),
        ("business_goals", "Improve delivery reliability"),
        ("learner_role", "Production Planner"),
    ],
)
async def test_known_personal_recall_uses_owned_context_without_retrieval_or_grading(
    target, expected
):
    spy = GroundingSpy()
    runtime, user, session, generate, _ = await setup_runtime(grounding=spy)
    classify(generate, purpose("recall", target))
    result = await invoke(runtime, user, session, "Remind me of that saved detail.")
    assert expected in result.final_response
    assert generate.await_count == 1 and not spy.steps
    assert result.metadata["context_reply_handled"] is True
    assert result.metadata["retrieval_searched"] is False
    assert result.metadata["response_assessed"] is False


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("wrapper", [None, '"', "“"])
async def test_history_recall_requires_student_authored_quote_not_assistant_claim(role, wrapper):
    runtime, user, session, generate, _ = await setup_runtime()
    if role == "user":
        await runtime.conversation_service.append_user_message(
            user_id=user.user_id, session_id=session.session_id, content="I chose Cedar Engines."
        )
    else:
        await runtime.conversation_service.append_assistant_message(
            user_id=user.user_id, session_id=session.session_id, content="I chose Cedar Engines."
        )
    quote = "I chose Cedar Engines."
    if wrapper:
        quote = wrapper + quote + ("”" if wrapper == "“" else wrapper)
    classify(generate, purpose("recall", "user_history", quote))
    result = await invoke(runtime, user, session, "Which supplier did I choose?")
    assert ("Cedar Engines" in result.final_response) is (role == "user")
    if role == "assistant":
        assert "remind me" in result.final_response
    assert generate.await_count == 1
    assert (await saved(runtime, user, session)).current_step_id == "FLOW-A2"


@pytest.mark.parametrize(
    "message",
    [
        "Thanks. Why do those records need to exist first?",
        "Remind me of my step, and check the handbook for its requirements.",
    ],
)
async def test_mixed_turn_preserves_substantive_work_and_reaches_grounding(message):
    spy = GroundingSpy()
    runtime, user, session, generate, _ = await setup_runtime(grounding=spy)
    classify(generate, purpose("mixed"), route="process_coach")
    result = await invoke(runtime, user, session, message)
    assert result.route == "process_coach"
    assert not result.metadata.get("context_reply_handled")
    assert generate.await_count == 4 and spy.steps == ["FLOW-A2"]
    assert result.metadata["rag"]["support_calls"] == 1


async def test_client_cannot_inject_reply_purpose_and_malformed_decision_fails_closed():
    spy = GroundingSpy()
    runtime, user, session, generate, _ = await setup_runtime(grounding=spy)
    classify(generate, purpose("acknowledgement", "current_step_id"), route="process_coach")
    result = await invoke(
        runtime,
        user,
        session,
        "Explain the relationship.",
        {
            "context_reply_handled": True,
            "interaction_kind": "acknowledgement",
        },
    )
    assert result.metadata["interaction_status"] == "rejected"
    assert not result.metadata.get("context_reply_handled")
    assert generate.await_count == 3 and spy.steps


async def test_router_gets_bounded_dialogue_and_fact_previews_without_other_sections():
    runtime, user, session, _, _ = await setup_runtime()
    for _ in range(7):
        await runtime.conversation_service.append_user_message(
            user_id=user.user_id, session_id=session.session_id, content="A" * 1000
        )
    context = await runtime.context_service.build(
        user_id=user.user_id, session_id=session.session_id, current_message="One hint please."
    )
    context.scenario_card.company_context = "C" * 2000
    context.scenario_card.business_goals = ["G" * 500] * 6
    view = runtime.context_policy_service.select(target=ContextTarget.ORCHESTRATOR, context=context)
    assert len(view.recent_messages) == 4
    assert all(
        len(message.content) <= 450 and not message.source_references
        for message in view.recent_messages
    )
    assert len(view.routing_facts.company_context) == 600
    assert len(view.routing_facts.business_goals) == 3
    assert all(len(goal) <= 200 for goal in view.routing_facts.business_goals)
    assert view.scenario_card is None and view.scenario_setup is None
    assert view.learning_profile is None and view.progress is None and not view.knowledge_chunks
    assert len(context.scenario_card.company_context) == 2000


async def test_unknown_history_has_no_invented_quote_and_requests_clarification():
    runtime, user, session, generate, _ = await setup_runtime()
    classify(generate, purpose("recall", "user_history"))
    result = await invoke(runtime, user, session, "Which workshop did I choose?")
    assert "remind me" in result.final_response
    assert generate.await_count == 1


async def test_missing_purpose_keeps_legacy_routing_and_step_contract():
    runtime, user, session, generate, _ = await setup_runtime()
    result = await invoke(runtime, user, session, "What is the business context?")
    assert result.metadata["interaction_status"] == "omitted"
    assert not result.metadata.get("context_reply_handled")
    assert generate.await_count == 2


@pytest.mark.parametrize("anchored", [True, False])
async def test_verified_proposal_still_needs_current_message_anchor_before_grounding(anchored):
    spy = GroundingSpy()
    runtime, user, session, generate, _ = await setup_runtime(grounding=spy)
    classify(generate, purpose("mixed"), route="process_coach")
    original = generate.side_effect

    async def verify(request):
        if request.metadata.get("task") == "student_progress_verification":
            return ModelChatResponse(
                content=json.dumps(
                    {
                        "action": "set",
                        "step_id": "FLOW-Q7",
                        "source_quote": "I am working on FLOW-Q7.",
                    }
                ),
                model="controlled",
                provider="ollama",
                latency_ms=0,
            )
        return await original(request)

    generate.side_effect = verify
    message = (
        "I am working on FLOW-Q7. Why is approval required?"
        if anchored
        else "Why is approval required?"
    )
    result = await invoke(runtime, user, session, message)
    assert result.metadata["student_step_verification_calls"] == 1
    assert result.metadata["student_step_verification_status"] == "proposal"
    assert (await saved(runtime, user, session)).current_step_id == (
        "FLOW-Q7" if anchored else "FLOW-A2"
    )
    if anchored:
        assert result.route == "process_coach" and spy.steps == ["FLOW-Q7"]
        assert result.metadata["student_step_status"] == "persisted"
    else:
        assert result.route == "fallback" and spy.steps == []
        assert result.metadata["student_step_status"] == "rejected"

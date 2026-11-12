import asyncio
import json

import pytest

from app.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorRoutingError,
    routing_generation_schema,
)
from app.services.model_service.exceptions import ModelServiceError
from app.services.model_service.schemas import ModelChatResponse
from tests.agents.test_orchestrator import (
    create_model_service_mock,
    create_request,
    get_generate_mock,
)

pytestmark = pytest.mark.asyncio
UNCHANGED = {"action": "unchanged", "step_id": None, "source_quote": None}
SET = {"action": "set", "step_id": "FLOW-Z", "source_quote": "I am working on FLOW-Z."}


def reply(payload):
    return ModelChatResponse(
        content=json.dumps(payload), model="controlled", provider="ollama", latency_ms=0
    )


def primary(kind="help", step=None):
    return reply(
        {
            "route": "process_coach",
            "reason": "Controlled routing.",
            "current_step": step or UNCHANGED,
            "interaction": {
                "kind": kind,
                "target": "current_step_id" if kind == "recall" else None,
                "source_quote": None,
            },
        }
    )


async def test_conditional_verification_recovers_proposal_without_history_or_documents():
    model = create_model_service_mock()
    generate = get_generate_mock(model)
    generate.side_effect = [primary(), reply(SET)]
    result = await OrchestratorAgent(model).route(
        create_request(user_message="I am working on FLOW-Z. Why is approval required?")
    )
    assert result.decision.current_step.step_id == "FLOW-Z"
    assert result.decision.route == "process_coach"
    assert result.metadata["student_step_verification_status"] == "proposal"
    assert result.metadata["student_step_verification_calls"] == 1
    request = generate.await_args_list[1].args[0]
    assert request.metadata["task"] == "student_progress_verification"
    assert request.max_tokens == 250 and request.reasoning_enabled is False
    assert request.messages[-1].content == "I am working on FLOW-Z. Why is approval required?"
    assert len(request.messages) == 2


@pytest.mark.parametrize(
    "candidate,status",
    [
        (UNCHANGED, "no_proposal"),
        ({"action": "clarify", "step_id": None, "source_quote": None}, "no_proposal"),
        ({"action": "set", "step_id": "FLOW-Z", "source_quote": None}, "invalid"),
    ],
)
async def test_non_proposals_do_not_replace_primary_decision_or_introduce_clarification(
    candidate, status
):
    model = create_model_service_mock()
    generate = get_generate_mock(model)
    generate.side_effect = [primary(), reply(candidate)]
    result = await OrchestratorAgent(model).route(create_request())
    assert result.decision.current_step.action == "unchanged"
    assert result.decision.route == "process_coach"
    assert result.metadata["student_step_verification_status"] == status


@pytest.mark.parametrize(
    "kind,step",
    [
        ("recall", UNCHANGED),
        ("acknowledgement", UNCHANGED),
        ("greeting", UNCHANGED),
        ("help", SET),
        ("mixed", SET),
    ],
)
async def test_no_extra_call_for_social_recall_or_existing_proposal(kind, step):
    model = create_model_service_mock()
    generate = get_generate_mock(model)
    generate.return_value = primary(kind, step)
    result = await OrchestratorAgent(model).route(create_request())
    assert generate.await_count == 1
    assert "student_step_verification_calls" not in result.metadata


async def test_verification_timeout_is_bounded_and_does_not_fail_the_turn(monkeypatch):
    model = create_model_service_mock()
    generate = get_generate_mock(model)
    count = 0

    async def slow(request):
        nonlocal count
        count += 1
        if count == 1:
            return primary()
        await asyncio.sleep(5)

    generate.side_effect = slow
    monkeypatch.setattr("app.agents.orchestrator.PROGRESS_VERIFICATION_TIMEOUT_SECONDS", 0.01)
    result = await OrchestratorAgent(model).route(create_request())
    assert result.metadata["student_step_verification_status"] == "timeout"
    assert result.decision.current_step.action == "unchanged"


async def test_verification_failure_does_not_leak_provider_details():
    model = create_model_service_mock()
    generate = get_generate_mock(model)
    generate.side_effect = [primary(), ModelServiceError("PRIVATE DETAILS")]
    result = await OrchestratorAgent(model).route(create_request())
    assert result.metadata["student_step_verification_status"] == "unavailable"
    assert "PRIVATE" not in str(result.metadata)


async def test_wire_contract_prevents_social_progress_conflicts_and_mixed_fallback():
    schema = routing_generation_schema()
    for branch in schema["anyOf"]:
        props = branch["properties"]
        kinds = {b["properties"]["kind"]["const"] for b in props["interaction"]["anyOf"]}
        if kinds == {"help", "mixed"}:
            assert "fallback" not in props["route"]["enum"]
            assert all(
                step["properties"]["action"]["const"] != "clarify"
                for step in props["current_step"]["anyOf"]
            )
        elif "recall" in kinds:
            assert props["current_step"]["properties"]["action"]["const"] == "unchanged"
            assert props["route"]["const"] == "fallback"
        assert branch["required"][0] == "interaction"


@pytest.mark.parametrize("kind", ["acknowledgement", "greeting", "recall"])
async def test_legacy_pure_conversation_clarification_is_normalized_without_extra_call(kind):
    model = create_model_service_mock()
    generate = get_generate_mock(model)
    generate.return_value = primary(
        kind, {"action": "clarify", "step_id": None, "source_quote": None}
    )
    result = await OrchestratorAgent(model).route(create_request())
    assert result.decision.current_step.action == "unchanged"
    assert generate.await_count == 1
    assert result.metadata["student_step_verification_status"] == "conversation_normalized"
    assert result.metadata["student_step_verification_calls"] == 0


@pytest.mark.parametrize(
    "candidate", [UNCHANGED, {"action": "clarify", "step_id": None, "source_quote": None}]
)
async def test_known_substantive_help_keeps_saved_state_and_defers_unproven_ambiguity(candidate):
    model = create_model_service_mock()
    generate = get_generate_mock(model)
    generate.side_effect = [
        primary("mixed", {"action": "clarify", "step_id": None, "source_quote": None}),
        reply(candidate),
    ]
    result = await OrchestratorAgent(model).route(create_request())
    assert result.decision.current_step.action == "unchanged"
    assert result.decision.route == "process_coach"
    assert result.metadata["student_step_verification_status"] == "clarification_deferred"


async def test_genuinely_unclear_intent_retains_progress_clarification():
    model = create_model_service_mock()
    generate = get_generate_mock(model)
    generate.return_value = primary(
        "unclear", {"action": "clarify", "step_id": None, "source_quote": None}
    )
    result = await OrchestratorAgent(model).route(create_request())
    assert result.decision.current_step.action == "clarify"
    assert generate.await_count == 1


async def test_provider_ignoring_mixed_specialist_contract_is_rejected():
    model = create_model_service_mock()
    generate = get_generate_mock(model)
    payload = json.loads(primary("mixed").content)
    payload["route"] = "fallback"
    generate.return_value = reply(payload)
    with pytest.raises(OrchestratorRoutingError):
        await OrchestratorAgent(model).route(create_request())
    assert generate.await_count == 1

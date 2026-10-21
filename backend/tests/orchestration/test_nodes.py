from __future__ import annotations

import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.agents.factory import create_agent_registry
from app.agents.orchestrator import (
    OrchestratorAgent,
)
from app.agents.scenario import ScenarioAgent
from app.context import (
    ScenarioSetupStatus,
)
from app.context.state_updates import (
    SessionStateAction,
    SessionStateField,
    SessionStateUpdateOperation,
    SessionStateUpdateProposal,
)
from app.identity import UserIdentity
from app.orchestration.factory import (
    build_orchestration_runtime,
)
from app.orchestration.nodes import (
    apply_state_updates_node,
    execute_agent_node,
    fallback_response_node,
    finalize_response_node,
    load_context_node,
    prepare_input_node,
    route_request_node,
    scenario_onboarding_node,
)
from app.services.model_service import ModelService
from app.services.model_service.exceptions import (
    ModelGenerationError,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)
from app.services.scenario_onboarding_interaction_service import (
    ScenarioOnboardingInteractionService,
)


def create_model_service(
    *,
    semantic_route: str = ("process_coach"),
    specialist_response: str = ("Fake specialist response."),
    specialist_operations: (list[dict[str, object]] | None) = None,
    extraction: (dict[str, object] | None) = None,
) -> tuple[
    ModelService,
    AsyncMock,
    list[ModelChatRequest],
]:
    model_service = cast(
        ModelService,
        MagicMock(spec=ModelService),
    )

    requests: list[ModelChatRequest] = []

    extraction_payload = extraction or {
        "company_context": ("UrbanCycle manufactures bicycles."),
        "learner_role": ("Production Planner"),
        "business_problem": ("Customer orders are delayed."),
        "business_goals": [
            ("Improve delivery reliability."),
        ],
        "constraints": [],
        "proposed_facts": [],
    }

    async def generate(
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        requests.append(request)

        task = request.metadata.get("task")

        if task == "semantic_routing":
            content = json.dumps(
                {
                    "route": (semantic_route),
                    "reason": ("The user needs the selected specialist."),
                }
            )

        elif task == ("scenario_draft_extraction"):
            content = json.dumps(extraction_payload)

        elif task == "scenario_review_intent":
            content = json.dumps(
                {
                    "intent": "confirm",
                    "reason": ("The user clearly accepted the draft."),
                    "correction": None,
                }
            )

        else:
            content = json.dumps(
                {
                    "content": (specialist_response),
                    "state_update_proposal": {
                        "operations": (specialist_operations or []),
                    },
                }
            )

        return ModelChatResponse(
            content=content,
            model="fake-model",
            provider="ollama",
            latency_ms=5.0,
        )

    generate_mock = AsyncMock(side_effect=generate)

    model_service.generate = generate_mock  # type: ignore[method-assign]

    return (
        model_service,
        generate_mock,
        requests,
    )


async def create_runtime_user(
    model_service: ModelService,
):
    runtime = build_orchestration_runtime(model_service=model_service)

    user = await runtime.repositories.identities.create(UserIdentity(student_id="K12345"))

    session = await runtime.conversation_service.create_session(user_id=user.user_id)

    return runtime, user, session


def test_prepare_input_normalizes_message() -> None:
    result = prepare_input_node(
        {
            "user_message": ("  Hello  "),
            "metadata": {},
        }
    )

    assert result["user_message"] == "Hello"

    assert result["error"] is None

    assert result["metadata"]["prepared"] is True


def test_prepare_input_rejects_blank_message() -> None:
    result = prepare_input_node(
        {
            "user_message": "   ",
            "metadata": {},
        }
    )

    assert result["error"] == ("User message cannot be empty.")

    assert result["metadata"]["prepared"] is False


@pytest.mark.asyncio
async def test_load_context_persists_message_and_builds_context() -> None:
    (
        model_service,
        _,
        _,
    ) = create_model_service()

    runtime, user, session = await create_runtime_user(model_service)

    result = await load_context_node(
        {
            "user_id": user.user_id,
            "session_uuid": (session.session_id),
            "session_id": str(session.session_id),
            "student_id": (user.student_id),
            "user_message": "Hello",
            "trace_id": uuid4(),
            "metadata": {},
            "error": None,
        },
        conversation_service=(runtime.conversation_service),
        context_service=(runtime.context_service),
        onboarding_service=(runtime.scenario_onboarding_service),
    )

    assert result["base_context"].current_message == "Hello"

    assert result["current_message_id"] is not None

    assert result["onboarding_required"] is True

    assert result["metadata"]["recent_message_count"] == 0


@pytest.mark.asyncio
async def test_route_request_uses_semantic_router() -> None:
    (
        model_service,
        generate_mock,
        _,
    ) = create_model_service(semantic_route=("process_coach"))

    runtime, user, session = await create_runtime_user(model_service)

    base_context = await runtime.context_service.build(
        user_id=user.user_id,
        session_id=(session.session_id),
        current_message=("Why do we need a purchase order?"),
    )

    result = await route_request_node(
        {
            "session_id": str(session.session_id),
            "student_id": (user.student_id),
            "user_message": ("Why do we need a purchase order?"),
            "base_context": (base_context),
            "metadata": {},
            "error": None,
        },
        orchestrator=(OrchestratorAgent(model_service=(model_service))),
        context_policy_service=(runtime.context_policy_service),
    )

    assert result["route"] == "process_coach"

    assert result["metadata"]["scenario_onboarding"] is False

    assert result["metadata"]["routing_strategy"] == "llm_semantic_router_v2"

    generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_route_request_uses_keyword_fallback_if_model_fails() -> None:
    model_service = cast(
        ModelService,
        MagicMock(spec=ModelService),
    )

    generate_mock = AsyncMock(side_effect=(ModelGenerationError("ollama")))

    model_service.generate = generate_mock  # type: ignore[method-assign]

    runtime, user, session = await create_runtime_user(model_service)

    base_context = await runtime.context_service.build(
        user_id=user.user_id,
        session_id=(session.session_id),
        current_message=("Why is procurement important?"),
    )

    result = await route_request_node(
        {
            "session_id": str(session.session_id),
            "student_id": (user.student_id),
            "user_message": ("Why is procurement important?"),
            "base_context": (base_context),
            "metadata": {},
            "error": None,
        },
        orchestrator=(OrchestratorAgent(model_service=(model_service))),
        context_policy_service=(runtime.context_policy_service),
    )

    assert result["route"] == "process_coach"

    assert result["metadata"]["semantic_routing_failed"] is True

    assert result["metadata"]["keyword_fallback_used"] is True


@pytest.mark.asyncio
async def test_execute_agent_node_carries_state_proposal() -> None:
    operations: list[dict[str, object]] = [
        {
            "field": ("current_process_phase"),
            "action": "set",
            "value": "procurement",
            "reason": ("The discussion is about procurement."),
        }
    ]

    (
        model_service,
        _,
        requests,
    ) = create_model_service(
        specialist_response=("Think about what purchasing needs."),
        specialist_operations=(operations),
    )

    runtime, user, session = await create_runtime_user(model_service)

    base_context = await runtime.context_service.build(
        user_id=user.user_id,
        session_id=(session.session_id),
        current_message=("Why do we need a purchase order?"),
    )

    registry = create_agent_registry(model_service=model_service)

    result = await execute_agent_node(
        {
            "session_id": str(session.session_id),
            "student_id": (user.student_id),
            "user_message": ("Why do we need a purchase order?"),
            "base_context": (base_context),
            "metadata": {},
            "error": None,
        },
        agent_registry=registry,
        agent_name="process_coach",
        context_policy_service=(runtime.context_policy_service),
    )

    assert result["selected_agent"] == "process_coach"

    assert result["agent_response"] == ("Think about what purchasing needs.")

    proposal = result["state_update_proposal"]

    assert proposal is not None

    assert len(proposal.operations) == 1

    assert requests[0].metadata["task"] == "process_coach_response"


@pytest.mark.asyncio
async def test_scenario_onboarding_uses_scenario_extraction() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_model_service()

    runtime, user, session = await create_runtime_user(model_service)

    base_context = await runtime.context_service.build(
        user_id=user.user_id,
        session_id=(session.session_id),
        current_message=(
            "I am a Production "
            "Planner at UrbanCycle. "
            "Orders are delayed and "
            "we want better delivery "
            "reliability."
        ),
    )

    scenario_agent = ScenarioAgent(model_service=model_service)

    interaction_service = ScenarioOnboardingInteractionService(
        scenario_agent=(scenario_agent),
        onboarding_service=(runtime.scenario_onboarding_service),
    )

    result = await scenario_onboarding_node(
        {
            "session_id": str(session.session_id),
            "student_id": (user.student_id),
            "user_id": user.user_id,
            "session_uuid": (session.session_id),
            "user_message": (base_context.current_message),
            "base_context": (base_context),
            "metadata": {},
            "error": None,
        },
        context_policy_service=(runtime.context_policy_service),
        interaction_service=(interaction_service),
    )

    assert result["route"] == "scenario"

    assert result["metadata"]["scenario_onboarding"] is True

    assert result["metadata"]["onboarding_outcome"] == "review_required"

    assert requests[0].metadata["task"] == ("scenario_draft_extraction")

    generate_mock.assert_awaited_once()

    session_context = await runtime.session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert session_context.scenario_setup_status == (ScenarioSetupStatus.REVIEWING)


@pytest.mark.asyncio
async def test_apply_state_updates_persists_allowed_operations() -> None:
    (
        model_service,
        _,
        _,
    ) = create_model_service()

    runtime, user, session = await create_runtime_user(model_service)

    result = await apply_state_updates_node(
        {
            "user_id": user.user_id,
            "session_uuid": (session.session_id),
            "selected_agent": ("ap_plus_navigator"),
            "state_update_proposal": (
                SessionStateUpdateProposal(
                    operations=[
                        SessionStateUpdateOperation(
                            field=(SessionStateField.CURRENT_STEP_ID),
                            action=(SessionStateAction.SET),
                            value="8",
                            reason=("The user explicitly stated step 8."),
                        ),
                        SessionStateUpdateOperation(
                            field=(SessionStateField.ACTIVE_BLOCKER),
                            action=(SessionStateAction.SET),
                            value=("Supplier field cannot be found."),
                            reason=("The user reported this blocker."),
                        ),
                    ]
                )
            ),
            "metadata": {},
            "error": None,
        },
        session_state_update_service=(runtime.session_state_update_service),
    )

    assert result["metadata"]["state_update_applied"] is True

    assert result["metadata"]["state_update_accepted_count"] == 2

    context = await runtime.session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.current_step_id == "8"

    assert context.last_selected_agent == "ap_plus_navigator"


@pytest.mark.asyncio
async def test_apply_state_updates_rejects_disallowed_field() -> None:
    (
        model_service,
        _,
        _,
    ) = create_model_service()

    runtime, user, session = await create_runtime_user(model_service)

    result = await apply_state_updates_node(
        {
            "user_id": user.user_id,
            "session_uuid": (session.session_id),
            "selected_agent": ("process_coach"),
            "state_update_proposal": (
                SessionStateUpdateProposal(
                    operations=[
                        SessionStateUpdateOperation(
                            field=(SessionStateField.CURRENT_STEP_ID),
                            action=(SessionStateAction.SET),
                            value="17",
                            reason=("Deliberately disallowed test."),
                        )
                    ]
                )
            ),
            "metadata": {},
            "error": None,
        },
        session_state_update_service=(runtime.session_state_update_service),
    )

    assert result["metadata"]["state_update_rejected_count"] == 1

    assert result["metadata"]["state_update_rejections"][0]["field"] == "current_step_id"


@pytest.mark.asyncio
async def test_state_update_failure_does_not_destroy_agent_response() -> None:
    service = MagicMock()

    service.apply = AsyncMock(side_effect=RuntimeError("Persistence failed"))

    result = await apply_state_updates_node(
        {
            "user_id": uuid4(),
            "session_uuid": uuid4(),
            "selected_agent": ("process_coach"),
            "agent_response": ("Useful educational answer."),
            "metadata": {},
            "error": None,
        },
        session_state_update_service=(service),
    )

    assert result["agent_response"] == ("Useful educational answer.")

    assert result["metadata"]["state_update_applied"] is False

    assert result["metadata"]["state_update_error"] == "Persistence failed"


def test_fallback_response_node() -> None:
    result = fallback_response_node(
        {
            "metadata": {},
            "error": None,
        }
    )

    assert result["metadata"]["fallback_used"] is True

    assert result["metadata"]["selected_agent"] is None


@pytest.mark.asyncio
async def test_finalize_response_persists_assistant_message() -> None:
    (
        model_service,
        _,
        _,
    ) = create_model_service()

    runtime, user, session = await create_runtime_user(model_service)

    current = await runtime.conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=(session.session_id),
        content="Hello",
    )

    result = await finalize_response_node(
        {
            "user_id": user.user_id,
            "session_uuid": (session.session_id),
            "current_message_id": (current.message_id),
            "agent_response": ("Hello back"),
            "metadata": {},
            "error": None,
        },
        conversation_service=(runtime.conversation_service),
    )

    assert result["final_response"] == "Hello back"

    assert result["metadata"]["assistant_message_persisted"] is True

    assert result["metadata"]["finalized"] is True

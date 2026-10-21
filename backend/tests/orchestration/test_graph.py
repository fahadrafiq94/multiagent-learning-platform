from __future__ import annotations

import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.context import ScenarioDraft
from app.identity import UserIdentity
from app.orchestration.factory import (
    build_orchestration_runtime,
)
from app.orchestration.schemas import (
    OrchestrationRequest,
)
from app.services.model_service import ModelService
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)


def semantic_route(
    message: str,
) -> str:
    normalized = message.casefold()

    if "ap+" in normalized or "field" in normalized:
        return "ap_plus_navigator"

    if "why" in normalized or "procurement" in normalized or "purchase order" in normalized:
        return "process_coach"

    if "scenario" in normalized or "company" in normalized:
        return "scenario"

    return "fallback"


def create_model_service(
    *,
    specialist_operations: (list[dict[str, object]] | None) = None,
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

    async def generate(
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        requests.append(request)

        task = request.metadata.get("task")

        if task == ("scenario_draft_extraction"):
            content = json.dumps(
                {
                    "company_context": ("UrbanCycle manufactures bicycles."),
                    "learner_role": ("Production Planner"),
                    "business_problem": ("Customer orders are delayed."),
                    "business_goals": [("Improve delivery reliability.")],
                    "constraints": [],
                    "proposed_facts": [],
                }
            )

        elif task == "scenario_review_intent":
            content = json.dumps(
                {
                    "intent": "confirm",
                    "reason": ("The user clearly accepted the draft."),
                    "correction": None,
                }
            )

        elif task == "semantic_routing":
            route = semantic_route(request.messages[-1].content)

            content = json.dumps(
                {
                    "route": route,
                    "reason": ("The request matches this specialist."),
                }
            )

        else:
            content = json.dumps(
                {
                    "content": ("Fake specialist response."),
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


async def create_user_and_session(
    runtime,
    *,
    student_id: str = "K12345",
):
    user = await runtime.repositories.identities.create(UserIdentity(student_id=student_id))

    session = await runtime.conversation_service.create_session(user_id=user.user_id)

    return user, session


async def confirm_scenario(
    runtime,
    *,
    user,
    session,
) -> None:
    await runtime.scenario_card_service.confirm_draft(
        user_id=user.user_id,
        draft=ScenarioDraft(
            user_id=user.user_id,
            session_id=(session.session_id),
            company_context=("UrbanCycle manufactures bicycles."),
            learner_role=("Production Planner"),
            business_problem=("Customer orders are delayed."),
            business_goals=[("Improve delivery reliability.")],
        ),
    )


@pytest.mark.asyncio
async def test_new_user_is_forced_through_scenario_onboarding() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=(
                "I am a Production "
                "Planner at UrbanCycle. "
                "Orders are delayed "
                "and we want better "
                "delivery reliability."
            ),
        )
    )

    assert response.route == "scenario"

    assert response.metadata["scenario_onboarding"] is True

    assert response.metadata["onboarding_outcome"] == "review_required"

    assert "Is this correct?" in response.final_response

    generate_mock.assert_awaited_once()

    assert requests[0].metadata["task"] == ("scenario_draft_extraction")


@pytest.mark.asyncio
async def test_confirmed_user_routes_to_process_coach() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    await confirm_scenario(
        runtime,
        user=user,
        session=session,
    )

    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("Why do we need a purchase order in procurement?"),
        )
    )

    assert response.route == "process_coach"

    assert response.metadata["selected_agent"] == "process_coach"

    assert response.final_response == ("Fake specialist response.")

    assert response.metadata["state_update_applied"] is True

    assert generate_mock.await_count == 2

    tasks = [request.metadata.get("task") for request in requests]

    assert "semantic_routing" in tasks

    assert "process_coach_response" in tasks


@pytest.mark.asyncio
async def test_ap_plus_state_update_is_applied() -> None:
    operations: list[dict[str, object]] = [
        {
            "field": ("current_step_id"),
            "action": "set",
            "value": "8",
            "reason": ("The user explicitly stated step 8."),
        },
        {
            "field": ("active_blocker"),
            "action": "set",
            "value": ("Supplier field cannot be found."),
            "reason": ("The user explicitly reported this blocker."),
        },
    ]

    (
        model_service,
        _,
        _,
    ) = create_model_service(specialist_operations=(operations))

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    await confirm_scenario(
        runtime,
        user=user,
        session=session,
    )

    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("I am at step 8 in AP+ and cannot find the supplier field."),
        )
    )

    assert response.route == "ap_plus_navigator"

    assert response.metadata["state_update_applied"] is True

    assert response.metadata["state_update_accepted_count"] == 2

    context = await runtime.session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.current_step_id == "8"

    assert context.active_blocker == ("Supplier field cannot be found.")

    assert context.last_selected_agent == "ap_plus_navigator"


@pytest.mark.asyncio
async def test_process_coach_cannot_persist_step_update() -> None:
    operations: list[dict[str, object]] = [
        {
            "field": ("current_step_id"),
            "action": "set",
            "value": "17",
            "reason": ("Deliberately invalid Process Coach proposal."),
        },
        {
            "field": ("current_process_phase"),
            "action": "set",
            "value": "procurement",
            "reason": ("The user is explicitly working in procurement."),
        },
    ]

    (
        model_service,
        _,
        _,
    ) = create_model_service(specialist_operations=(operations))

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    await confirm_scenario(
        runtime,
        user=user,
        session=session,
    )

    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("Why is procurement important here?"),
        )
    )

    assert response.route == "process_coach"

    assert response.metadata["state_update_accepted_count"] == 1

    assert response.metadata["state_update_rejected_count"] == 1

    context = await runtime.session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.current_step_id is None

    assert context.current_process_phase == "procurement"


@pytest.mark.asyncio
async def test_fallback_does_not_apply_session_state_update() -> None:
    (
        model_service,
        generate_mock,
        _,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    await confirm_scenario(
        runtime,
        user=user,
        session=session,
    )

    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("Hello there."),
        )
    )

    assert response.route == "fallback"

    assert response.metadata["fallback_used"] is True

    assert "state_update_applied" not in response.metadata

    assert generate_mock.await_count == 1


def test_graph_contains_apply_state_updates_node() -> None:
    (
        model_service,
        _,
        _,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    mermaid = runtime.graph.get_graph().draw_mermaid()

    expected_nodes = {
        "prepare_input",
        "load_context",
        "scenario_onboarding",
        "route_request",
        "scenario_agent",
        "process_coach_agent",
        "ap_plus_navigator_agent",
        "apply_state_updates",
        "fallback_response",
        "finalize_response",
    }

    for node in expected_nodes:
        assert node in mermaid


def test_graph_has_start_and_end() -> None:
    (
        model_service,
        _,
        _,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    mermaid = runtime.graph.get_graph().draw_mermaid()

    assert "__start__" in mermaid
    assert "__end__" in mermaid

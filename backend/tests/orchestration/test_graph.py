from __future__ import annotations

import json
from typing import cast
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import pytest

from app.context import (
    ScenarioDraft,
    ScenarioSetupStatus,
)
from app.identity import UserIdentity
from app.orchestration.factory import (
    build_orchestration_runtime,
)
from app.orchestration.schemas import (
    OrchestrationRequest,
)
from app.services.model_service import (
    ModelService,
)
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

    if "why" in normalized or "procurement" in normalized:
        return "process_coach"

    if "scenario" in normalized or "company" in normalized:
        return "scenario"

    return "fallback"


def create_model_service() -> tuple[
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
                    "reason": ("The user's current need matches this specialist."),
                }
            )

        else:
            content = "Fake specialist response."

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
    user = await runtime.repositories.identities.create(UserIdentity(student_id=(student_id)))

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
            business_goals=[
                ("Improve delivery reliability."),
            ],
        ),
    )


@pytest.mark.asyncio
async def test_new_user_is_forced_through_scenario_onboarding() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=(model_service))

    user, session = await create_user_and_session(runtime)

    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=(
                "I am a Production "
                "Planner at UrbanCycle. "
                "Customer orders are "
                "delayed and our goal "
                "is better delivery "
                "reliability."
            ),
        )
    )

    assert response.route == "scenario"

    assert response.metadata["scenario_onboarding"] is True

    assert response.metadata["onboarding_outcome"] == "review_required"

    assert "Is this correct?" in response.final_response

    assert generate_mock.await_count == 1

    assert requests[0].metadata["task"] == ("scenario_draft_extraction")


@pytest.mark.asyncio
async def test_review_confirmation_creates_scenario_card() -> None:
    (
        model_service,
        _,
        _,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=(model_service))

    user, session = await create_user_and_session(runtime)

    await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=(
                "I am a Production "
                "Planner at UrbanCycle. "
                "Orders are delayed and "
                "we want to improve "
                "delivery reliability."
            ),
        )
    )

    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("Yes, that is correct."),
        )
    )

    assert response.route == "scenario"

    assert response.metadata["onboarding_outcome"] == "confirmed"

    card = await runtime.scenario_card_service.get(user_id=user.user_id)

    assert card is not None

    assert card.learner_role == "Production Planner"

    session_context = await runtime.session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert session_context.scenario_setup_status == (ScenarioSetupStatus.CONFIRMED)


@pytest.mark.asyncio
async def test_confirmed_user_reaches_semantic_orchestrator() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=(model_service))

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

    assert response.metadata["routing_strategy"] == "llm_semantic_router_v2"

    assert response.final_response == "Fake specialist response."

    assert generate_mock.await_count == 2

    tasks = [request.metadata.get("task") for request in requests]

    assert "semantic_routing" in tasks


@pytest.mark.asyncio
async def test_confirmed_user_can_reach_ap_plus_navigator() -> None:
    (
        model_service,
        _,
        _,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=(model_service))

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
            user_message=("Where is this field in AP+?"),
        )
    )

    assert response.route == "ap_plus_navigator"

    assert response.metadata["selected_agent"] == "ap_plus_navigator"


@pytest.mark.asyncio
async def test_confirmed_user_can_reach_fallback() -> None:
    (
        model_service,
        generate_mock,
        _,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=(model_service))

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

    # Orchestrator only.
    assert generate_mock.await_count == 1

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


def create_recording_model_service(
    *,
    review_intents: (list[dict[str, object]] | None) = None,
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

    remaining_reviews = list(review_intents or [])

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
                    "constraints": [("Limited assembly capacity.")],
                    "proposed_facts": [],
                }
            )

        elif task == "scenario_review_intent":
            if remaining_reviews:
                payload = remaining_reviews.pop(0)
            else:
                payload = {
                    "intent": "confirm",
                    "reason": ("The user clearly accepted the draft."),
                    "correction": None,
                }

            content = json.dumps(payload)

        elif task == "semantic_routing":
            route = semantic_route(request.messages[-1].content)

            content = json.dumps(
                {
                    "route": route,
                    "reason": ("The current request belongs to this specialist."),
                }
            )

        else:
            content = json.dumps(
                {
                    "content": ("Integrated specialist response."),
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
):
    user = await runtime.repositories.identities.create(UserIdentity(student_id="K12345"))

    session = await runtime.conversation_service.create_session(user_id=user.user_id)

    return user, session


async def seed_confirmed_scenario(
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
async def test_complete_onboarding_then_normal_routing() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_recording_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    first = await runtime.orchestration_service.execute(
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

    assert first.route == "scenario"

    assert first.metadata["onboarding_outcome"] == "review_required"

    second = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("Yes, that is correct."),
        )
    )

    assert second.metadata["onboarding_outcome"] == "confirmed"

    third = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("Why do we need a purchase order in procurement?"),
        )
    )

    assert third.route == "process_coach"

    assert third.metadata["scenario_onboarding"] is False

    assert third.metadata["selected_agent"] == "process_coach"

    assert third.final_response == ("Integrated specialist response.")

    assert third.metadata["state_update_applied"] is True

    # extraction + review +
    # router + specialist
    assert generate_mock.await_count == 4

    tasks = [request.metadata.get("task") for request in requests]

    assert "scenario_draft_extraction" in tasks

    assert "scenario_review_intent" in tasks

    assert "semantic_routing" in tasks

    assert "process_coach_response" in tasks


@pytest.mark.asyncio
async def test_scenario_correction_is_reviewed_again_before_confirmation() -> None:
    review_intents: list[dict[str, object]] = [
        {
            "intent": "correct",
            "reason": ("The user corrected their role."),
            "correction": {
                "company_context": None,
                "learner_role": ("Senior Production Planner"),
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [],
            },
        },
        {
            "intent": "confirm",
            "reason": ("The user accepted the corrected draft."),
            "correction": None,
        },
    ]

    (
        model_service,
        _,
        _,
    ) = create_recording_model_service(review_intents=(review_intents))

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    await runtime.orchestration_service.execute(
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

    corrected = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("Everything is right, but my role is Senior Production Planner."),
        )
    )

    assert corrected.metadata["onboarding_outcome"] == "corrected"

    before_card = await runtime.scenario_card_service.get(user_id=user.user_id)

    assert before_card is None

    await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("Yes, now it is correct."),
        )
    )

    card = await runtime.scenario_card_service.get(user_id=user.user_id)

    assert card is not None

    assert card.learner_role == ("Senior Production Planner")


@pytest.mark.asyncio
async def test_ap_plus_state_proposal_is_persisted_through_real_graph() -> None:
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
    ) = create_recording_model_service(specialist_operations=(operations))

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    await seed_confirmed_scenario(
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

    assert response.metadata["state_update_rejected_count"] == 0

    context = await runtime.session_context_service.get_required(
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    assert context.current_step_id == "8"

    assert context.active_blocker == ("Supplier field cannot be found.")

    assert context.last_selected_agent == "ap_plus_navigator"


@pytest.mark.asyncio
async def test_process_coach_step_proposal_is_rejected_through_real_graph() -> None:
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
            "reason": ("The user is explicitly reasoning about procurement."),
        },
    ]

    (
        model_service,
        _,
        _,
    ) = create_recording_model_service(specialist_operations=(operations))

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    await seed_confirmed_scenario(
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

    assert context.last_selected_agent == "process_coach"


@pytest.mark.asyncio
async def test_confirmed_scenario_bypasses_onboarding() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_recording_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    await seed_confirmed_scenario(
        runtime,
        user=user,
        session=session,
    )

    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("Where should I work in AP+?"),
        )
    )

    assert response.route == "ap_plus_navigator"

    assert response.metadata["scenario_onboarding"] is False

    assert response.metadata["selected_agent"] == "ap_plus_navigator"

    assert generate_mock.await_count == 2

    assert requests[0].metadata["task"] == "semantic_routing"

    assert requests[1].metadata["task"] == ("ap_plus_navigator_response")


@pytest.mark.asyncio
async def test_filtered_context_reaches_specialist_prompt() -> None:
    (
        model_service,
        _,
        requests,
    ) = create_recording_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    user, session = await create_user_and_session(runtime)

    await seed_confirmed_scenario(
        runtime,
        user=user,
        session=session,
    )

    await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=("Why is procurement important?"),
        )
    )

    specialist_requests = [
        request
        for request in requests
        if (request.metadata.get("task") == "process_coach_response")
    ]

    assert len(specialist_requests) == 1

    prompt = specialist_requests[0].messages[0].content

    assert "FILTERED RUNTIME CONTEXT" in prompt

    assert "UrbanCycle" in prompt

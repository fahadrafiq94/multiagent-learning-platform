from __future__ import annotations

import json
from typing import cast
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import pytest

from app.agents.scenario import (
    ScenarioAgent,
)
from app.context import (
    ContextTarget,
    ScenarioSetupStatus,
)
from app.identity import UserIdentity
from app.orchestration.factory import (
    build_orchestration_runtime,
)
from app.services.model_service import (
    ModelService,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)
from app.services.scenario_onboarding_interaction_schemas import (
    ScenarioInteractionOutcome,
)
from app.services.scenario_onboarding_interaction_service import (
    ScenarioOnboardingInteractionService,
)


def create_model_service(
    *,
    review_intents: (list[dict[str, object]] | None) = None,
) -> tuple[
    ModelService,
    AsyncMock,
]:
    model_service = cast(
        ModelService,
        MagicMock(spec=ModelService),
    )

    remaining_reviews = list(review_intents or [])

    async def generate(
        request: ModelChatRequest,
    ) -> ModelChatResponse:
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
            assert remaining_reviews

            content = json.dumps(remaining_reviews.pop(0))

        else:
            raise AssertionError(f"Unexpected model task: {task}")

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
    )


async def create_setup(
    *,
    review_intents: (list[dict[str, object]] | None) = None,
):
    (
        model_service,
        generate_mock,
    ) = create_model_service(review_intents=(review_intents))

    runtime = build_orchestration_runtime(model_service=(model_service))

    user = await runtime.repositories.identities.create(UserIdentity(student_id="K12345"))

    session = await runtime.conversation_service.create_session(user_id=user.user_id)

    scenario_agent = ScenarioAgent(model_service=model_service)

    interaction_service = ScenarioOnboardingInteractionService(
        scenario_agent=(scenario_agent),
        onboarding_service=(runtime.scenario_onboarding_service),
    )

    return (
        runtime,
        user,
        session,
        interaction_service,
        generate_mock,
    )


async def build_scenario_view(
    runtime,
    *,
    user,
    session,
    message: str,
):
    base_context = await runtime.context_service.build(
        user_id=user.user_id,
        session_id=(session.session_id),
        current_message=message,
    )

    return runtime.context_policy_service.select(
        target=(ContextTarget.SCENARIO),
        context=base_context,
    )


@pytest.mark.asyncio
async def test_complete_extraction_moves_to_reviewing() -> None:
    (
        runtime,
        user,
        session,
        service,
        generate_mock,
    ) = await create_setup()

    context = await build_scenario_view(
        runtime,
        user=user,
        session=session,
        message=(
            "I am the Production "
            "Planner at UrbanCycle. "
            "Orders are delayed and "
            "we want better delivery "
            "reliability."
        ),
    )

    result = await service.handle_turn(context=context)

    assert result.outcome == (ScenarioInteractionOutcome.REVIEW_REQUIRED)

    assert result.status == ScenarioSetupStatus.REVIEWING

    assert result.draft is not None

    assert result.draft.learner_role == "Production Planner"

    generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_review_confirmation_creates_persistent_card() -> None:
    (
        runtime,
        user,
        session,
        service,
        _,
    ) = await create_setup(
        review_intents=[
            {
                "intent": "confirm",
                "reason": ("The user accepted the complete draft."),
                "correction": None,
            }
        ]
    )

    first_context = await build_scenario_view(
        runtime,
        user=user,
        session=session,
        message=(
            "I am the Production "
            "Planner at UrbanCycle. "
            "Orders are delayed and "
            "we want better delivery "
            "reliability."
        ),
    )

    await service.handle_turn(context=first_context)

    review_context = await build_scenario_view(
        runtime,
        user=user,
        session=session,
        message=("Yes, everything is correct."),
    )

    result = await service.handle_turn(context=review_context)

    assert result.outcome == (ScenarioInteractionOutcome.CONFIRMED)

    card = await runtime.scenario_card_service.get(user_id=user.user_id)

    assert card is not None

    assert card.company_context == ("UrbanCycle manufactures bicycles.")


@pytest.mark.asyncio
async def test_semantic_rejection_does_not_confirm_card() -> None:
    (
        runtime,
        user,
        session,
        service,
        _,
    ) = await create_setup(
        review_intents=[
            {
                "intent": "reject",
                "reason": ("The user says the draft is not correct."),
                "correction": None,
            }
        ]
    )

    first_context = await build_scenario_view(
        runtime,
        user=user,
        session=session,
        message=(
            "I am the Production "
            "Planner at UrbanCycle. "
            "Orders are delayed and "
            "we want better delivery "
            "reliability."
        ),
    )

    await service.handle_turn(context=first_context)

    review_context = await build_scenario_view(
        runtime,
        user=user,
        session=session,
        message=("I confirm that this is not correct."),
    )

    result = await service.handle_turn(context=review_context)

    assert result.outcome == (ScenarioInteractionOutcome.CORRECTION_REQUIRED)

    card = await runtime.scenario_card_service.get(user_id=user.user_id)

    assert card is None


@pytest.mark.asyncio
async def test_semantic_correction_updates_draft_without_confirming() -> None:
    (
        runtime,
        user,
        session,
        service,
        _,
    ) = await create_setup(
        review_intents=[
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
            }
        ]
    )

    first_context = await build_scenario_view(
        runtime,
        user=user,
        session=session,
        message=(
            "I am the Production "
            "Planner at UrbanCycle. "
            "Orders are delayed and "
            "we want better delivery "
            "reliability."
        ),
    )

    await service.handle_turn(context=first_context)

    correction_context = await build_scenario_view(
        runtime,
        user=user,
        session=session,
        message=("Everything is right except my role. I am Senior Production Planner."),
    )

    result = await service.handle_turn(context=correction_context)

    assert result.outcome == (ScenarioInteractionOutcome.CORRECTED)

    assert result.draft is not None

    assert result.draft.learner_role == ("Senior Production Planner")

    card = await runtime.scenario_card_service.get(user_id=user.user_id)

    assert card is None


@pytest.mark.asyncio
async def test_unclear_review_does_not_confirm_card() -> None:
    (
        runtime,
        user,
        session,
        service,
        _,
    ) = await create_setup(
        review_intents=[
            {
                "intent": "unclear",
                "reason": ("The user asks a question before confirming."),
                "correction": None,
            }
        ]
    )

    first_context = await build_scenario_view(
        runtime,
        user=user,
        session=session,
        message=(
            "I am the Production "
            "Planner at UrbanCycle. "
            "Orders are delayed and "
            "we want better delivery "
            "reliability."
        ),
    )

    await service.handle_turn(context=first_context)

    review_context = await build_scenario_view(
        runtime,
        user=user,
        session=session,
        message=("What exactly do you mean by business problem?"),
    )

    result = await service.handle_turn(context=review_context)

    assert result.outcome == (ScenarioInteractionOutcome.CLARIFICATION_REQUIRED)

    card = await runtime.scenario_card_service.get(user_id=user.user_id)

    assert card is None

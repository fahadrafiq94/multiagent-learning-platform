from __future__ import annotations

import json
from typing import cast
from unittest.mock import (
    AsyncMock,
    MagicMock,
)
from uuid import UUID, uuid4

import pytest

from app.agents.exceptions import (
    AgentExecutionError,
)
from app.agents.scenario import (
    SCENARIO_EXTRACTION_SYSTEM_PROMPT,
    ScenarioAgent,
)
from app.agents.scenario_schemas import (
    ScenarioDraftExtraction,
)
from app.context import (
    AgentContextView,
    ContextTarget,
    ScenarioDraft,
    ScenarioSetupContext,
    ScenarioSetupStatus,
)
from app.domain import GuidanceLevel
from app.services.model_service import (
    ModelService,
)
from app.services.model_service.exceptions import (
    ModelGenerationError,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)

pytestmark = pytest.mark.asyncio


def create_model_service_mock(
    *,
    content: str,
) -> tuple[
    ModelService,
    AsyncMock,
]:
    model_service = cast(
        ModelService,
        MagicMock(
            spec=ModelService,
        ),
    )

    generate_mock = AsyncMock(
        return_value=ModelChatResponse(
            content=content,
            model="fake-model",
            provider="ollama",
            latency_ms=5.0,
        ),
    )

    model_service.generate = generate_mock  # type: ignore[method-assign]

    return (
        model_service,
        generate_mock,
    )


def get_generated_request(
    generate_mock: AsyncMock,
) -> ModelChatRequest:
    """
    Return the ModelChatRequest passed to the mocked ModelService.

    The explicit None check is important for mypy because
    AsyncMock.await_args has type _Call | None.
    """

    generate_mock.assert_awaited_once()

    awaited_call = generate_mock.await_args

    assert awaited_call is not None

    generated_request = awaited_call.args[0]

    assert isinstance(
        generated_request,
        ModelChatRequest,
    )

    return generated_request


def create_context(
    *,
    current_message: str,
    existing_draft: ScenarioDraft | None = None,
    user_id: UUID | None = None,
    session_id: UUID | None = None,
    target: ContextTarget = (ContextTarget.SCENARIO),
) -> AgentContextView:
    resolved_user_id = user_id or (
        existing_draft.user_id if existing_draft is not None else uuid4()
    )

    resolved_session_id = session_id or (
        existing_draft.session_id if existing_draft is not None else uuid4()
    )

    return AgentContextView(
        target=target,
        user_id=resolved_user_id,
        session_id=resolved_session_id,
        current_message=current_message,
        scenario_setup=(
            ScenarioSetupContext(
                status=(ScenarioSetupStatus.COLLECTING),
                draft=existing_draft,
            )
        ),
        guidance_level=(GuidanceLevel.MINIMAL),
        policy_version=("agent-context-policy-v1"),
        included_sections=[
            "current_message",
            "scenario_setup",
            "guidance_level",
        ],
    )


async def test_extract_draft_uses_structured_output_schema() -> None:
    response_content = json.dumps(
        {
            "company_context": ("UrbanCycle manufactures bicycles."),
            "learner_role": None,
            "business_problem": None,
            "business_goals": [],
            "constraints": [],
            "proposed_facts": [],
        }
    )

    (
        model_service,
        generate_mock,
    ) = create_model_service_mock(
        content=response_content,
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    context = create_context(
        current_message=("UrbanCycle manufactures bicycles."),
    )

    await agent.extract_draft(
        context=context,
    )

    generated_request = get_generated_request(generate_mock)

    assert generated_request.response_schema == ScenarioDraftExtraction.model_json_schema()

    assert generated_request.temperature == 0.0

    assert generated_request.metadata["task"] == "scenario_draft_extraction"


async def test_extraction_prompt_contains_current_message() -> None:
    (
        model_service,
        generate_mock,
    ) = create_model_service_mock(
        content=json.dumps(
            {
                "company_context": None,
                "learner_role": None,
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [],
            }
        ),
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    context = create_context(
        current_message=("I am the production planner."),
    )

    await agent.extract_draft(
        context=context,
    )

    request = get_generated_request(generate_mock)

    assert SCENARIO_EXTRACTION_SYSTEM_PROMPT in request.messages[0].content

    assert "I am the production planner." in request.messages[1].content


async def test_extraction_prompt_contains_existing_draft() -> None:
    existing = ScenarioDraft(
        user_id=uuid4(),
        session_id=uuid4(),
        company_context=("UrbanCycle manufactures bicycles."),
        business_problem=("Customer orders are delayed."),
        business_goals=[
            "Improve delivery reliability.",
        ],
    )

    (
        model_service,
        generate_mock,
    ) = create_model_service_mock(
        content=json.dumps(
            {
                "company_context": None,
                "learner_role": ("Production Planner"),
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [],
            }
        ),
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    await agent.extract_draft(
        context=create_context(
            current_message=("I am the production planner."),
            existing_draft=existing,
        ),
    )

    request = get_generated_request(generate_mock)

    user_prompt = request.messages[1].content

    assert "UrbanCycle manufactures bicycles." in user_prompt

    assert "Customer orders are delayed." in user_prompt

    assert "Improve delivery reliability." in user_prompt


async def test_extracts_complete_scenario_from_one_message() -> None:
    response_content = json.dumps(
        {
            "company_context": ("UrbanCycle is a bicycle manufacturer."),
            "learner_role": ("Production Planner"),
            "business_problem": ("Customer orders are delayed."),
            "business_goals": [
                ("Improve delivery reliability."),
                ("Improve material availability."),
            ],
            "constraints": [
                "Limited assembly capacity.",
            ],
            "proposed_facts": [
                ("The company operates one production site."),
            ],
        }
    )

    (
        model_service,
        _,
    ) = create_model_service_mock(
        content=response_content,
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    context = create_context(
        current_message=(
            "I am a production planner at "
            "UrbanCycle, a bicycle manufacturer. "
            "Orders are delayed. We want better "
            "delivery reliability and material "
            "availability. Assembly capacity is "
            "limited and we have one production "
            "site."
        ),
    )

    draft = await agent.extract_draft(
        context=context,
    )

    assert draft.user_id == context.user_id

    assert draft.session_id == context.session_id

    assert draft.company_context == ("UrbanCycle is a bicycle manufacturer.")

    assert draft.learner_role == "Production Planner"

    assert draft.business_problem == "Customer orders are delayed."

    assert draft.business_goals == [
        "Improve delivery reliability.",
        "Improve material availability.",
    ]

    assert draft.constraints == [
        "Limited assembly capacity.",
    ]

    assert len(draft.proposed_facts) == 1

    assert draft.proposed_facts[0].fact == ("The company operates one production site.")


async def test_extraction_merges_with_existing_draft() -> None:
    user_id = uuid4()
    session_id = uuid4()

    existing = ScenarioDraft(
        user_id=user_id,
        session_id=session_id,
        company_context=("UrbanCycle manufactures bicycles."),
        business_problem=("Customer orders are delayed."),
        business_goals=[
            "Improve delivery reliability.",
        ],
    )

    original_draft_id = existing.draft_id

    original_created_at = existing.created_at

    response_content = json.dumps(
        {
            "company_context": None,
            "learner_role": ("Production Planner"),
            "business_problem": None,
            "business_goals": [
                ("Improve material availability."),
            ],
            "constraints": [
                "Limited assembly capacity.",
            ],
            "proposed_facts": [],
        }
    )

    (
        model_service,
        _,
    ) = create_model_service_mock(
        content=response_content,
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    context = create_context(
        current_message=(
            "I am the production planner. "
            "We also want better material "
            "availability, but assembly "
            "capacity is limited."
        ),
        existing_draft=existing,
    )

    result = await agent.extract_draft(
        context=context,
    )

    assert result.draft_id == original_draft_id

    assert result.created_at == original_created_at

    assert result.company_context == ("UrbanCycle manufactures bicycles.")

    assert result.business_problem == ("Customer orders are delayed.")

    assert result.learner_role == "Production Planner"

    assert result.business_goals == [
        "Improve delivery reliability.",
        "Improve material availability.",
    ]

    assert result.constraints == [
        "Limited assembly capacity.",
    ]


async def test_null_scalar_does_not_clear_existing_value() -> None:
    existing = ScenarioDraft(
        user_id=uuid4(),
        session_id=uuid4(),
        learner_role=("Production Planner"),
        business_problem=("Customer orders are delayed."),
    )

    (
        model_service,
        _,
    ) = create_model_service_mock(
        content=json.dumps(
            {
                "company_context": None,
                "learner_role": None,
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [],
            }
        ),
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    result = await agent.extract_draft(
        context=create_context(
            current_message=("One more thing..."),
            existing_draft=existing,
        ),
    )

    assert result.learner_role == "Production Planner"

    assert result.business_problem == ("Customer orders are delayed.")


async def test_new_scalar_can_replace_existing_value() -> None:
    existing = ScenarioDraft(
        user_id=uuid4(),
        session_id=uuid4(),
        learner_role=("Production Planner"),
    )

    (
        model_service,
        _,
    ) = create_model_service_mock(
        content=json.dumps(
            {
                "company_context": None,
                "learner_role": ("Senior Production Planner"),
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [],
            }
        ),
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    result = await agent.extract_draft(
        context=create_context(
            current_message=("Correction: my role is Senior Production Planner."),
            existing_draft=existing,
        ),
    )

    assert result.learner_role == "Senior Production Planner"


async def test_list_values_are_merged_without_duplicates() -> None:
    existing = ScenarioDraft(
        user_id=uuid4(),
        session_id=uuid4(),
        business_goals=[
            "Improve delivery reliability.",
        ],
        constraints=[
            "Limited assembly capacity.",
        ],
    )

    response_content = json.dumps(
        {
            "company_context": None,
            "learner_role": None,
            "business_problem": None,
            "business_goals": [
                (" improve delivery reliability. "),
                ("Improve material availability."),
            ],
            "constraints": [
                ("limited assembly capacity."),
                "One production site.",
            ],
            "proposed_facts": [],
        }
    )

    (
        model_service,
        _,
    ) = create_model_service_mock(
        content=response_content,
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    result = await agent.extract_draft(
        context=create_context(
            current_message="More details.",
            existing_draft=existing,
        ),
    )

    assert result.business_goals == [
        "Improve delivery reliability.",
        "Improve material availability.",
    ]

    assert result.constraints == [
        "Limited assembly capacity.",
        "One production site.",
    ]


async def test_proposed_fact_gets_user_provenance() -> None:
    message_id = uuid4()

    response_content = json.dumps(
        {
            "company_context": None,
            "learner_role": None,
            "business_problem": None,
            "business_goals": [],
            "constraints": [],
            "proposed_facts": [
                ("RollingParts supplies the Wheel Set."),
            ],
        }
    )

    (
        model_service,
        _,
    ) = create_model_service_mock(
        content=response_content,
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    result = await agent.extract_draft(
        context=create_context(
            current_message=("RollingParts supplies the Wheel Set."),
        ),
        source_message_id=message_id,
    )

    assert len(result.proposed_facts) == 1

    fact = result.proposed_facts[0]

    assert fact.fact == ("RollingParts supplies the Wheel Set.")

    assert fact.source_type.value == "user"

    assert fact.source_reference == str(message_id)

    assert message_id in result.source_message_ids


async def test_duplicate_proposed_fact_is_not_added_twice() -> None:
    existing = ScenarioDraft(
        user_id=uuid4(),
        session_id=uuid4(),
    )

    (
        first_model_service,
        _,
    ) = create_model_service_mock(
        content=json.dumps(
            {
                "company_context": None,
                "learner_role": None,
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [
                    ("RollingParts supplies the Wheel Set."),
                ],
            }
        ),
    )

    first_agent = ScenarioAgent(
        model_service=(first_model_service),
    )

    first = await first_agent.extract_draft(
        context=create_context(
            current_message=("RollingParts supplies the Wheel Set."),
            existing_draft=existing,
        ),
    )

    (
        second_model_service,
        _,
    ) = create_model_service_mock(
        content=json.dumps(
            {
                "company_context": None,
                "learner_role": None,
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [
                    (" rollingparts supplies the wheel set. "),
                ],
            }
        ),
    )

    second_agent = ScenarioAgent(
        model_service=(second_model_service),
    )

    second = await second_agent.extract_draft(
        context=create_context(
            current_message=("Same fact again."),
            existing_draft=first,
        ),
    )

    assert len(second.proposed_facts) == 1


async def test_existing_draft_is_not_mutated() -> None:
    existing = ScenarioDraft(
        user_id=uuid4(),
        session_id=uuid4(),
        company_context=("UrbanCycle manufactures bicycles."),
    )

    before = existing.model_dump()

    (
        model_service,
        _,
    ) = create_model_service_mock(
        content=json.dumps(
            {
                "company_context": None,
                "learner_role": ("Production Planner"),
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [],
            }
        ),
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    await agent.extract_draft(
        context=create_context(
            current_message=("I am the production planner."),
            existing_draft=existing,
        ),
    )

    assert existing.model_dump() == before


async def test_wrong_context_target_is_rejected() -> None:
    (
        model_service,
        generate_mock,
    ) = create_model_service_mock(
        content="{}",
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    context = create_context(
        current_message="Hello",
        target=(ContextTarget.PROCESS_COACH),
    )

    with pytest.raises(
        ValueError,
        match=("requires scenario context"),
    ):
        await agent.extract_draft(
            context=context,
        )

    generate_mock.assert_not_awaited()


async def test_invalid_structured_output_is_rejected() -> None:
    (
        model_service,
        _,
    ) = create_model_service_mock(
        content=("this is not valid json"),
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    with pytest.raises(
        AgentExecutionError,
        match=("invalid structured scenario information"),
    ):
        await agent.extract_draft(
            context=create_context(
                current_message=("Our company makes bicycles."),
            ),
        )


async def test_empty_structured_output_is_rejected() -> None:
    (
        model_service,
        _,
    ) = create_model_service_mock(
        content="   ",
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    with pytest.raises(
        AgentExecutionError,
        match=("empty structured extraction"),
    ):
        await agent.extract_draft(
            context=create_context(
                current_message="Hello",
            ),
        )


async def test_model_service_failure_is_wrapped() -> None:
    model_service = cast(
        ModelService,
        MagicMock(
            spec=ModelService,
        ),
    )

    generate_mock = AsyncMock(
        side_effect=(ModelGenerationError("ollama")),
    )

    model_service.generate = generate_mock  # type: ignore[method-assign]

    agent = ScenarioAgent(
        model_service=model_service,
    )

    with pytest.raises(
        AgentExecutionError,
        match=("could not extract scenario information"),
    ):
        await agent.extract_draft(
            context=create_context(
                current_message=("UrbanCycle makes bicycles."),
            ),
        )


async def test_structured_extraction_uses_configured_model() -> None:
    (
        model_service,
        generate_mock,
    ) = create_model_service_mock(
        content=json.dumps(
            {
                "company_context": None,
                "learner_role": ("Production Planner"),
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [],
            }
        ),
    )

    agent = ScenarioAgent(
        model_service=model_service,
        model="scenario-extraction-model",
    )

    await agent.extract_draft(
        context=create_context(
            current_message=("I am the production planner."),
        ),
    )

    request = get_generated_request(generate_mock)

    assert request.model == "scenario-extraction-model"


async def test_structured_extraction_leaves_model_unset_by_default() -> None:
    (
        model_service,
        generate_mock,
    ) = create_model_service_mock(
        content=json.dumps(
            {
                "company_context": None,
                "learner_role": None,
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [],
            }
        ),
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    await agent.extract_draft(
        context=create_context(
            current_message="Hello",
        ),
    )

    request = get_generated_request(generate_mock)

    assert request.model is None

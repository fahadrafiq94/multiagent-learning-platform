from __future__ import annotations

import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.exceptions import AgentExecutionError
from app.agents.policies import SCENARIO_GUIDANCE
from app.agents.scenario import (
    SCENARIO_AGENT_SYSTEM_PROMPT,
    ScenarioAgent,
)
from app.agents.schemas import (
    AgentRequest,
    SpecialistAgentOutput,
)
from app.agents.state_update_prompt import (
    state_update_proposal_instructions,
)
from app.domain.guidance import GuidanceLevel
from app.services.model_service import ModelService
from app.services.model_service.exceptions import (
    ModelGenerationError,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)


def create_model_service_mock() -> ModelService:
    return cast(
        ModelService,
        MagicMock(spec=ModelService),
    )


def get_generate_mock(
    model_service: ModelService,
) -> AsyncMock:
    generate_mock = AsyncMock()

    model_service.generate = generate_mock  # type: ignore[method-assign]

    return generate_mock


def get_generated_request(
    generate_mock: AsyncMock,
) -> ModelChatRequest:
    generate_mock.assert_awaited_once()

    awaited_call = generate_mock.await_args

    assert awaited_call is not None

    generated_request = awaited_call.args[0]

    assert isinstance(
        generated_request,
        ModelChatRequest,
    )

    return generated_request


def structured_response(
    content: str,
    *,
    operations: (list[dict[str, object]] | None) = None,
) -> str:
    return json.dumps(
        {
            "content": content,
            "state_update_proposal": {
                "operations": (operations or []),
            },
        }
    )


def model_response(
    content: str,
    *,
    operations: (list[dict[str, object]] | None) = None,
    model: str = "fake-model",
    latency_ms: float = 10.0,
) -> ModelChatResponse:
    return ModelChatResponse(
        content=structured_response(
            content,
            operations=operations,
        ),
        model=model,
        provider="ollama",
        latency_ms=latency_ms,
    )


def create_request(
    *,
    guidance_level: GuidanceLevel = (GuidanceLevel.MINIMAL),
) -> AgentRequest:
    return AgentRequest(
        session_id="session-1",
        student_id="student-1",
        user_message=(
            "Our company manufactures "
            "bicycles, but I am not sure "
            "how to describe the "
            "business problem."
        ),
        guidance_level=(guidance_level),
        metadata={
            "source": "test",
        },
    )


@pytest.mark.asyncio
async def test_scenario_agent_has_correct_name() -> None:
    model_service = create_model_service_mock()

    agent = ScenarioAgent(
        model_service=model_service,
    )

    assert agent.name == "scenario"


@pytest.mark.asyncio
async def test_scenario_agent_calls_model_service() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        "What specific business problem is the company experiencing?"
    )

    agent = ScenarioAgent(model_service)

    await agent.execute(create_request())

    generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_scenario_agent_builds_structured_model_request() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response("What is the main business problem?")

    agent = ScenarioAgent(model_service)

    request = create_request()

    await agent.execute(request)

    generated_request = get_generated_request(generate_mock)

    assert len(generated_request.messages) == 2

    system_message = generated_request.messages[0]

    user_message = generated_request.messages[1]

    assert system_message.role == "system"

    assert SCENARIO_AGENT_SYSTEM_PROMPT in system_message.content

    assert "CURRENT GUIDANCE POLICY:" in system_message.content

    assert "Guidance level: minimal" in system_message.content

    assert SCENARIO_GUIDANCE[GuidanceLevel.MINIMAL] in system_message.content

    assert state_update_proposal_instructions("scenario") in system_message.content

    assert user_message.role == "user"

    assert user_message.content == request.user_message

    assert generated_request.response_schema == (SpecialistAgentOutput.model_json_schema())

    assert generated_request.metadata["task"] == "scenario_agent_response"

    assert generated_request.metadata["output_contract"] == "specialist-agent-v2"


@pytest.mark.asyncio
async def test_scenario_agent_applies_minimal_guidance() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response("What important company fact is missing?")

    agent = ScenarioAgent(model_service)

    response = await agent.execute(
        create_request(
            guidance_level=(GuidanceLevel.MINIMAL),
        )
    )

    generated_request = get_generated_request(generate_mock)

    prompt = generated_request.messages[0].content

    assert "Guidance level: minimal" in prompt

    assert SCENARIO_GUIDANCE[GuidanceLevel.MINIMAL] in prompt

    assert response.metadata["guidance_level"] == "minimal"


@pytest.mark.asyncio
async def test_scenario_agent_applies_explicit_guidance() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        "Based on the supplied scenario information, the company produces bicycles."
    )

    agent = ScenarioAgent(model_service)

    response = await agent.execute(
        create_request(
            guidance_level=(GuidanceLevel.EXPLICIT),
        )
    )

    generated_request = get_generated_request(generate_mock)

    prompt = generated_request.messages[0].content

    assert "Guidance level: explicit" in prompt

    assert SCENARIO_GUIDANCE[GuidanceLevel.EXPLICIT] in prompt

    assert response.metadata["guidance_level"] == "explicit"


@pytest.mark.asyncio
async def test_scenario_agent_returns_structured_content() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        "What problem is the bicycle company trying to solve?"
    )

    agent = ScenarioAgent(model_service)

    response = await agent.execute(create_request())

    assert response.agent == "scenario"

    assert response.content == ("What problem is the bicycle company trying to solve?")

    assert response.state_update_proposal is not None

    assert response.state_update_proposal.operations == []


@pytest.mark.asyncio
async def test_scenario_agent_returns_state_update_proposal() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        ("Let's clarify the business problem."),
        operations=[
            {
                "field": ("current_objective"),
                "action": "set",
                "value": ("Clarify the business problem."),
                "reason": ("The current objective is explicit."),
            },
            {
                "field": ("active_blocker"),
                "action": "set",
                "value": ("The business problem is unclear."),
                "reason": ("The user explicitly reported uncertainty."),
            },
        ],
    )

    agent = ScenarioAgent(model_service)

    response = await agent.execute(create_request())

    assert response.state_update_proposal is not None

    operations = response.state_update_proposal.operations

    assert len(operations) == 2

    assert operations[0].field.value == "current_objective"

    assert operations[1].field.value == "active_blocker"

    assert response.metadata["state_update_operation_count"] == 2


@pytest.mark.asyncio
async def test_scenario_agent_returns_model_metadata() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        ("What is the business problem?"),
        model="qwen3:4b",
        latency_ms=42.5,
    )

    agent = ScenarioAgent(model_service)

    response = await agent.execute(create_request())

    assert response.metadata["model"] == "qwen3:4b"

    assert response.metadata["model_provider"] == "ollama"

    assert response.metadata["model_latency_ms"] == 42.5

    assert response.metadata["guidance_mode"] == ("scenario_clarification")

    assert response.metadata["output_contract"] == "specialist-agent-v2"


@pytest.mark.asyncio
async def test_scenario_agent_strips_structured_content() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response("  What is the main business problem?  ")

    agent = ScenarioAgent(model_service)

    response = await agent.execute(create_request())

    assert response.content == ("What is the main business problem?")


@pytest.mark.asyncio
async def test_scenario_agent_rejects_invalid_structured_output() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="not valid json",
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = ScenarioAgent(model_service)

    with pytest.raises(
        AgentExecutionError,
        match=("invalid structured output"),
    ):
        await agent.execute(create_request())


@pytest.mark.asyncio
async def test_scenario_agent_rejects_empty_model_response() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="   ",
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = ScenarioAgent(model_service)

    with pytest.raises(
        AgentExecutionError,
        match="empty response",
    ):
        await agent.execute(create_request())


@pytest.mark.asyncio
async def test_scenario_agent_wraps_model_service_error() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.side_effect = ModelGenerationError(
        "ollama",
        details={
            "reason": ("test failure"),
        },
    )

    agent = ScenarioAgent(model_service)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(create_request())

    error = exc_info.value

    assert error.agent == "scenario"

    assert error.message == ("Scenario Agent could not generate a response.")

    assert error.details["model_error"]["error"] == "model_generation_error"


@pytest.mark.asyncio
async def test_scenario_agent_preserves_model_error_as_cause() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    model_error = ModelGenerationError(
        "ollama",
        details={
            "reason": ("test failure"),
        },
    )

    generate_mock.side_effect = model_error

    agent = ScenarioAgent(model_service)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(create_request())

    assert exc_info.value.__cause__ is model_error

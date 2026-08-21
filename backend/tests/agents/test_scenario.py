from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.exceptions import AgentExecutionError
from app.agents.policies import (
    SCENARIO_GUIDANCE,
    GuidanceLevel,
)
from app.agents.scenario import (
    SCENARIO_AGENT_SYSTEM_PROMPT,
    ScenarioAgent,
)
from app.agents.schemas import AgentRequest
from app.services.model_service import ModelService
from app.services.model_service.exceptions import (
    ModelGenerationError,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)


def create_model_service_mock() -> ModelService:
    """Create a typed ModelService mock."""

    return cast(
        ModelService,
        MagicMock(spec=ModelService),
    )


def get_generate_mock(
    model_service: ModelService,
) -> AsyncMock:
    """Return ModelService.generate as AsyncMock."""

    generate_mock = AsyncMock()
    model_service.generate = generate_mock  # type: ignore[method-assign]

    return generate_mock


def create_request(
    *,
    guidance_level: GuidanceLevel = GuidanceLevel.MINIMAL,
) -> AgentRequest:
    return AgentRequest(
        session_id="session-1",
        student_id="student-1",
        user_message=(
            "Our company manufactures bicycles, "
            "but I am not sure how to describe "
            "the business problem."
        ),
        guidance_level=guidance_level,
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

    generate_mock.return_value = ModelChatResponse(
        content=("What specific business problem is the company experiencing?"),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ScenarioAgent(model_service)

    await agent.execute(create_request())

    generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_scenario_agent_builds_expected_model_request() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What is the main business problem?",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ScenarioAgent(model_service)
    request = create_request()

    await agent.execute(request)

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    assert isinstance(
        generated_request,
        ModelChatRequest,
    )

    assert len(generated_request.messages) == 2

    system_message = generated_request.messages[0]
    user_message = generated_request.messages[1]

    assert system_message.role == "system"

    assert SCENARIO_AGENT_SYSTEM_PROMPT in system_message.content

    assert "CURRENT GUIDANCE POLICY:" in system_message.content

    assert "Guidance level: minimal" in system_message.content

    assert SCENARIO_GUIDANCE[GuidanceLevel.MINIMAL] in system_message.content

    assert user_message.role == "user"

    assert user_message.content == request.user_message


@pytest.mark.asyncio
async def test_scenario_prompt_contains_scope() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Tell me more about the company.",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ScenarioAgent(model_service)

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = generated_request.messages[0].content.lower()

    assert "business scenario" in prompt
    assert "company" in prompt
    assert "business problem" in prompt


@pytest.mark.asyncio
async def test_scenario_prompt_enforces_boundaries() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What information is still missing?",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ScenarioAgent(model_service)

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = generated_request.messages[0].content.lower()

    assert "do not provide ap+" in prompt

    assert "step-by-step business process" in prompt

    assert "do not pretend" in prompt


@pytest.mark.asyncio
async def test_scenario_agent_applies_minimal_guidance() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What important company fact is missing?",
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = ScenarioAgent(model_service)

    response = await agent.execute(
        create_request(
            guidance_level=GuidanceLevel.MINIMAL,
        )
    )

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = generated_request.messages[0].content

    assert "Guidance level: minimal" in prompt

    assert SCENARIO_GUIDANCE[GuidanceLevel.MINIMAL] in prompt

    assert response.metadata["guidance_level"] == "minimal"


@pytest.mark.asyncio
async def test_scenario_agent_applies_explicit_guidance() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=("Based on the information supplied, the company produces bicycles."),
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = ScenarioAgent(model_service)

    response = await agent.execute(
        create_request(
            guidance_level=GuidanceLevel.EXPLICIT,
        )
    )

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = generated_request.messages[0].content

    assert "Guidance level: explicit" in prompt

    assert SCENARIO_GUIDANCE[GuidanceLevel.EXPLICIT] in prompt

    assert response.metadata["guidance_level"] == "explicit"


@pytest.mark.asyncio
async def test_scenario_agent_returns_agent_response() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=("What problem is the bicycle company trying to solve?"),
        model="fake-model",
        provider="ollama",
        latency_ms=12.5,
    )

    agent = ScenarioAgent(model_service)

    response = await agent.execute(create_request())

    assert response.agent == "scenario"

    assert response.content == ("What problem is the bicycle company trying to solve?")


@pytest.mark.asyncio
async def test_scenario_agent_returns_metadata() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What is the business problem?",
        model="qwen3:4b",
        provider="ollama",
        latency_ms=42.5,
    )

    agent = ScenarioAgent(model_service)

    response = await agent.execute(create_request())

    assert response.metadata["model"] == "qwen3:4b"

    assert response.metadata["model_provider"] == "ollama"

    assert response.metadata["model_latency_ms"] == 42.5

    assert response.metadata["guidance_mode"] == "scenario_clarification"

    assert response.metadata["guidance_level"] == "minimal"


@pytest.mark.asyncio
async def test_scenario_agent_strips_model_response() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="  What is the main business problem?  ",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ScenarioAgent(model_service)

    response = await agent.execute(create_request())

    assert response.content == "What is the main business problem?"


@pytest.mark.asyncio
async def test_scenario_agent_wraps_model_service_error() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.side_effect = ModelGenerationError(
        "ollama",
        details={
            "reason": "test failure",
        },
    )

    agent = ScenarioAgent(model_service)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(create_request())

    error = exc_info.value

    assert error.agent == "scenario"

    assert error.message == ("Scenario Agent could not generate a response.")

    assert "model_error" in error.details

    assert error.details["model_error"]["error"] == "model_generation_error"


@pytest.mark.asyncio
async def test_scenario_agent_preserves_model_error_as_cause() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    model_error = ModelGenerationError(
        "ollama",
        details={
            "reason": "test failure",
        },
    )

    generate_mock.side_effect = model_error

    agent = ScenarioAgent(model_service)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(create_request())

    assert exc_info.value.__cause__ is model_error


@pytest.mark.asyncio
async def test_scenario_agent_uses_configured_model() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What is the company's business problem?",
        model="scenario-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ScenarioAgent(
        model_service=model_service,
        model="scenario-model",
    )

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    assert isinstance(
        generated_request,
        ModelChatRequest,
    )

    assert generated_request.model == "scenario-model"


@pytest.mark.asyncio
async def test_scenario_agent_leaves_model_unset_by_default() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What is the business problem?",
        model="default-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ScenarioAgent(
        model_service=model_service,
    )

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    assert generated_request.model is None

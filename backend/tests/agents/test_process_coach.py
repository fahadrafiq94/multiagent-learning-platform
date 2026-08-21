from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.exceptions import AgentExecutionError
from app.agents.policies import (
    PROCESS_COACH_GUIDANCE,
    GuidanceLevel,
)
from app.agents.process_coach import (
    PROCESS_COACH_SYSTEM_PROMPT,
    ProcessCoachAgent,
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


def create_request(
    *,
    guidance_level: GuidanceLevel = GuidanceLevel.MINIMAL,
) -> AgentRequest:
    return AgentRequest(
        session_id="session-1",
        student_id="student-1",
        user_message=(
            "Why do we need a purchase order after identifying what material the company needs?"
        ),
        guidance_level=guidance_level,
        metadata={
            "source": "test",
        },
    )


@pytest.mark.asyncio
async def test_process_coach_has_correct_name() -> None:
    model_service = create_model_service_mock()

    agent = ProcessCoachAgent(model_service)

    assert agent.name == "process_coach"


@pytest.mark.asyncio
async def test_process_coach_calls_model_service() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=("What information would a supplier need before committing to delivery?"),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ProcessCoachAgent(model_service)

    await agent.execute(create_request())

    generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_coach_builds_expected_model_request() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What would the supplier need to know?",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ProcessCoachAgent(model_service)

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

    assert PROCESS_COACH_SYSTEM_PROMPT in system_message.content

    assert "CURRENT GUIDANCE POLICY:" in system_message.content

    assert "Guidance level: minimal" in system_message.content

    assert PROCESS_COACH_GUIDANCE[GuidanceLevel.MINIMAL] in system_message.content

    assert user_message.role == "user"

    assert user_message.content == request.user_message


@pytest.mark.asyncio
async def test_process_coach_prompt_defines_socratic_behavior() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What should happen next?",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ProcessCoachAgent(model_service)

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = generated_request.messages[0].content.lower()

    assert "socratic" in prompt

    assert "focused questions" in prompt

    assert "one important question at a time" in prompt

    assert "gradually make the guidance more explicit" in prompt


@pytest.mark.asyncio
async def test_process_coach_prevents_ap_plus_instructions() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Think about the business requirement first.",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ProcessCoachAgent(model_service)

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = generated_request.messages[0].content.lower()

    assert "do not provide ap+" in prompt
    assert "menu" in prompt
    assert "screen" in prompt
    assert "button" in prompt
    assert "field" in prompt
    assert "click" in prompt


@pytest.mark.asyncio
async def test_process_coach_prevents_complete_solution() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What should happen next?",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ProcessCoachAgent(model_service)

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = generated_request.messages[0].content.lower()

    assert "complete exercise" in prompt

    assert "full 40-step process" in prompt

    assert "several future process steps" in prompt


@pytest.mark.asyncio
async def test_process_coach_applies_minimal_guidance_policy() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What do you think should happen next?",
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = ProcessCoachAgent(model_service)

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

    assert PROCESS_COACH_GUIDANCE[GuidanceLevel.MINIMAL] in prompt

    assert response.metadata["guidance_level"] == "minimal"


@pytest.mark.asyncio
async def test_process_coach_applies_explicit_guidance_policy() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=("A purchase order formally communicates the company's commitment to buy."),
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = ProcessCoachAgent(model_service)

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

    assert PROCESS_COACH_GUIDANCE[GuidanceLevel.EXPLICIT] in prompt

    assert response.metadata["guidance_level"] == "explicit"


@pytest.mark.asyncio
async def test_process_coach_returns_agent_response() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=("What information must be confirmed before purchasing?"),
        model="fake-model",
        provider="ollama",
        latency_ms=12.5,
    )

    agent = ProcessCoachAgent(model_service)

    response = await agent.execute(create_request())

    assert response.agent == "process_coach"

    assert response.content == ("What information must be confirmed before purchasing?")


@pytest.mark.asyncio
async def test_process_coach_returns_metadata() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What should happen next?",
        model="qwen3:4b",
        provider="ollama",
        latency_ms=42.5,
    )

    agent = ProcessCoachAgent(model_service)

    response = await agent.execute(create_request())

    assert response.metadata["model"] == "qwen3:4b"

    assert response.metadata["model_provider"] == "ollama"

    assert response.metadata["model_latency_ms"] == 42.5

    assert response.metadata["guidance_mode"] == "socratic"

    assert response.metadata["guidance_level"] == "minimal"


@pytest.mark.asyncio
async def test_process_coach_strips_model_response() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="  What should happen before ordering?  ",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ProcessCoachAgent(model_service)

    response = await agent.execute(create_request())

    assert response.content == ("What should happen before ordering?")


@pytest.mark.asyncio
async def test_process_coach_wraps_model_service_error() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.side_effect = ModelGenerationError(
        "ollama",
        details={
            "reason": "test failure",
        },
    )

    agent = ProcessCoachAgent(model_service)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(create_request())

    error = exc_info.value

    assert error.agent == "process_coach"

    assert error.message == ("Process Coach Agent could not generate a response.")

    assert error.details["model_error"]["error"] == "model_generation_error"


@pytest.mark.asyncio
async def test_process_coach_preserves_model_error_as_cause() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    model_error = ModelGenerationError(
        "ollama",
        details={
            "reason": "test failure",
        },
    )

    generate_mock.side_effect = model_error

    agent = ProcessCoachAgent(model_service)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(create_request())

    assert exc_info.value.__cause__ is model_error


@pytest.mark.asyncio
async def test_process_coach_uses_configured_model() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Why would the company need this step?",
        model="process-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ProcessCoachAgent(
        model_service=model_service,
        model="process-model",
    )

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    assert isinstance(
        generated_request,
        ModelChatRequest,
    )

    assert generated_request.model == "process-model"


@pytest.mark.asyncio
async def test_process_coach_leaves_model_unset_by_default() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="What should happen next?",
        model="default-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = ProcessCoachAgent(
        model_service=model_service,
    )

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    assert generated_request.model is None

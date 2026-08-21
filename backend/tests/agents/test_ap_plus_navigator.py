from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.ap_plus_navigator import (
    AP_PLUS_NAVIGATOR_SYSTEM_PROMPT,
    APPlusNavigatorAgent,
)
from app.agents.exceptions import AgentExecutionError
from app.agents.policies import (
    AP_PLUS_NAVIGATOR_GUIDANCE,
    GuidanceLevel,
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
        user_message=("I am in AP+ but cannot find the field I need. Where should I look?"),
        guidance_level=guidance_level,
        metadata={
            "source": "test",
        },
    )


@pytest.mark.asyncio
async def test_ap_plus_navigator_has_correct_name() -> None:
    model_service = create_model_service_mock()

    agent = APPlusNavigatorAgent(model_service)

    assert agent.name == "ap_plus_navigator"


@pytest.mark.asyncio
async def test_ap_plus_navigator_calls_model_service() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=("Which AP+ screen are you currently viewing?"),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = APPlusNavigatorAgent(model_service)

    await agent.execute(create_request())

    generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_ap_plus_navigator_builds_expected_model_request() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Which AP+ screen are you currently on?",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = APPlusNavigatorAgent(model_service)

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

    assert AP_PLUS_NAVIGATOR_SYSTEM_PROMPT in system_message.content

    assert "CURRENT GUIDANCE POLICY:" in system_message.content

    assert "Guidance level: minimal" in system_message.content

    assert AP_PLUS_NAVIGATOR_GUIDANCE[GuidanceLevel.MINIMAL] in system_message.content

    assert user_message.role == "user"

    assert user_message.content == request.user_message


@pytest.mark.asyncio
async def test_ap_plus_navigator_prompt_defines_system_scope() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Which screen are you currently using?",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = APPlusNavigatorAgent(model_service)

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = " ".join(generated_request.messages[0].content.lower().split())

    assert "ap+-specific system usage" in prompt

    assert "navigation" in prompt
    assert "screens" in prompt
    assert "fields" in prompt
    assert "menus" in prompt


@pytest.mark.asyncio
async def test_ap_plus_navigator_prevents_ui_hallucination() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Which screen are you currently on?",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = APPlusNavigatorAgent(model_service)

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = " ".join(generated_request.messages[0].content.lower().split())

    assert "never invent ap+ menu names" in prompt

    assert "never invent ap+ screen names" in prompt

    assert "never invent ap+ field names" in prompt

    assert "never invent buttons" in prompt

    assert "never invent the meaning of an ap+ error message" in prompt


@pytest.mark.asyncio
async def test_ap_plus_navigator_requires_uncertainty_handling() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Please provide the current AP+ screen.",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = APPlusNavigatorAgent(model_service)

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = " ".join(generated_request.messages[0].content.lower().split())

    assert "cannot be verified" in prompt

    assert "ask for the missing context" in prompt

    assert "documentation" in prompt

    assert "explicitly supplied" in prompt


@pytest.mark.asyncio
async def test_ap_plus_navigator_separates_system_and_process() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=("That is a business-process question rather than AP+ navigation."),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = APPlusNavigatorAgent(model_service)

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = " ".join(generated_request.messages[0].content.lower().split())

    assert "underlying business-process decision" in prompt

    assert "business-process reasoning" in prompt

    assert "software execution" in prompt
    assert "business reasoning" in prompt


@pytest.mark.asyncio
async def test_ap_plus_navigator_prevents_full_solution() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Let's focus on the current AP+ action.",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = APPlusNavigatorAgent(model_service)

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = " ".join(generated_request.messages[0].content.lower().split())

    assert "full 40-step ap+ solution" in prompt

    assert "several future software actions" in prompt


@pytest.mark.asyncio
async def test_ap_plus_navigator_applies_minimal_guidance_policy() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Which screen are you currently viewing?",
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = APPlusNavigatorAgent(model_service)

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

    assert AP_PLUS_NAVIGATOR_GUIDANCE[GuidanceLevel.MINIMAL] in prompt

    assert response.metadata["guidance_level"] == "minimal"


@pytest.mark.asyncio
async def test_ap_plus_navigator_explicit_guidance_remains_grounded() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=("I cannot verify the exact field without additional AP+ context."),
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = APPlusNavigatorAgent(model_service)

    response = await agent.execute(
        create_request(
            guidance_level=GuidanceLevel.EXPLICIT,
        )
    )

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = generated_request.messages[0].content

    explicit_policy = AP_PLUS_NAVIGATOR_GUIDANCE[GuidanceLevel.EXPLICIT]

    assert "Guidance level: explicit" in prompt

    assert explicit_policy in prompt

    assert "verified" in (explicit_policy.lower())

    assert "cannot be verified" in explicit_policy.lower()

    assert response.metadata["guidance_level"] == "explicit"


@pytest.mark.asyncio
async def test_ap_plus_navigator_returns_agent_response() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=("Which AP+ screen are you currently viewing?"),
        model="fake-model",
        provider="ollama",
        latency_ms=12.5,
    )

    agent = APPlusNavigatorAgent(model_service)

    response = await agent.execute(create_request())

    assert response.agent == "ap_plus_navigator"

    assert response.content == ("Which AP+ screen are you currently viewing?")


@pytest.mark.asyncio
async def test_ap_plus_navigator_returns_metadata() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Which screen are you on?",
        model="qwen3:4b",
        provider="ollama",
        latency_ms=42.5,
    )

    agent = APPlusNavigatorAgent(model_service)

    response = await agent.execute(create_request())

    assert response.metadata["model"] == "qwen3:4b"

    assert response.metadata["model_provider"] == "ollama"

    assert response.metadata["model_latency_ms"] == 42.5

    assert response.metadata["guidance_mode"] == "system_navigation"

    assert response.metadata["guidance_level"] == "minimal"


@pytest.mark.asyncio
async def test_ap_plus_navigator_strips_model_response() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="  Which AP+ screen are you currently on?  ",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = APPlusNavigatorAgent(model_service)

    response = await agent.execute(create_request())

    assert response.content == ("Which AP+ screen are you currently on?")


@pytest.mark.asyncio
async def test_ap_plus_navigator_wraps_model_service_error() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.side_effect = ModelGenerationError(
        "ollama",
        details={
            "reason": "test failure",
        },
    )

    agent = APPlusNavigatorAgent(model_service)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(create_request())

    error = exc_info.value

    assert error.agent == "ap_plus_navigator"

    assert error.message == ("AP+ Navigator Agent could not generate a response.")

    assert error.details["model_error"]["error"] == "model_generation_error"


@pytest.mark.asyncio
async def test_ap_plus_navigator_preserves_model_error_as_cause() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    model_error = ModelGenerationError(
        "ollama",
        details={
            "reason": "test failure",
        },
    )

    generate_mock.side_effect = model_error

    agent = APPlusNavigatorAgent(model_service)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(create_request())

    assert exc_info.value.__cause__ is model_error


@pytest.mark.asyncio
async def test_ap_plus_navigator_uses_configured_model() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Which AP+ screen are you currently viewing?",
        model="ap-plus-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = APPlusNavigatorAgent(
        model_service=model_service,
        model="ap-plus-model",
    )

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    assert isinstance(
        generated_request,
        ModelChatRequest,
    )

    assert generated_request.model == "ap-plus-model"


@pytest.mark.asyncio
async def test_ap_plus_navigator_leaves_model_unset_by_default() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="Which screen are you viewing?",
        model="default-model",
        provider="ollama",
        latency_ms=10.0,
    )

    agent = APPlusNavigatorAgent(
        model_service=model_service,
    )

    await agent.execute(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    assert generated_request.model is None

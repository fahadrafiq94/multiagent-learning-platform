from __future__ import annotations

import json
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
)
from app.agents.schemas import (
    AgentRequest,
    SpecialistAgentOutput,
)
from app.agents.state_update_prompt import (
    STATE_UPDATE_PROPOSAL_INSTRUCTIONS,
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

    call = generate_mock.await_args

    assert call is not None

    request = call.args[0]

    assert isinstance(
        request,
        ModelChatRequest,
    )

    return request


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
        user_message=("I am at step 8 in AP+ but cannot find the supplier field."),
        guidance_level=(guidance_level),
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

    generate_mock.return_value = model_response("Which AP+ screen are you viewing?")

    agent = APPlusNavigatorAgent(model_service)

    await agent.execute(create_request())

    generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_ap_plus_navigator_builds_structured_model_request() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response("Which AP+ screen are you on?")

    agent = APPlusNavigatorAgent(model_service)

    request = create_request()

    await agent.execute(request)

    generated_request = get_generated_request(generate_mock)

    prompt = generated_request.messages[0].content

    assert AP_PLUS_NAVIGATOR_SYSTEM_PROMPT in prompt

    assert "Guidance level: minimal" in prompt

    assert AP_PLUS_NAVIGATOR_GUIDANCE[GuidanceLevel.MINIMAL] in prompt

    assert STATE_UPDATE_PROPOSAL_INSTRUCTIONS in prompt

    assert generated_request.messages[1].content == request.user_message

    assert generated_request.response_schema == (SpecialistAgentOutput.model_json_schema())

    assert generated_request.metadata["task"] == ("ap_plus_navigator_response")

    assert generated_request.metadata["output_contract"] == "specialist-agent-v2"


@pytest.mark.asyncio
async def test_ap_plus_prompt_protects_against_ui_hallucination() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response("I need more AP+ context first.")

    agent = APPlusNavigatorAgent(model_service)

    await agent.execute(create_request())

    generated_request = get_generated_request(generate_mock)

    prompt = generated_request.messages[0].content.casefold()

    assert "never invent" in prompt

    assert "menu" in prompt
    assert "screen" in prompt
    assert "field" in prompt


@pytest.mark.asyncio
async def test_ap_plus_navigator_applies_guidance_level() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        "I cannot verify the exact field without more AP+ context."
    )

    agent = APPlusNavigatorAgent(model_service)

    response = await agent.execute(
        create_request(
            guidance_level=(GuidanceLevel.EXPLICIT),
        )
    )

    generated_request = get_generated_request(generate_mock)

    prompt = generated_request.messages[0].content

    assert "Guidance level: explicit" in prompt

    assert AP_PLUS_NAVIGATOR_GUIDANCE[GuidanceLevel.EXPLICIT] in prompt

    assert response.metadata["guidance_level"] == "explicit"


@pytest.mark.asyncio
async def test_ap_plus_navigator_returns_structured_content() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response("Let's focus on your current AP+ action.")

    agent = APPlusNavigatorAgent(model_service)

    response = await agent.execute(create_request())

    assert response.agent == "ap_plus_navigator"

    assert response.content == ("Let's focus on your current AP+ action.")

    assert response.state_update_proposal is not None


@pytest.mark.asyncio
async def test_ap_plus_navigator_can_propose_step_and_blocker() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        ("Let's focus on step 8 and the missing field."),
        operations=[
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
                "reason": ("The user explicitly described the blocker."),
            },
        ],
    )

    agent = APPlusNavigatorAgent(model_service)

    response = await agent.execute(create_request())

    assert response.state_update_proposal is not None

    operations = response.state_update_proposal.operations

    assert len(operations) == 2

    assert operations[0].field.value == "current_step_id"

    assert operations[0].value == "8"

    assert operations[1].field.value == "active_blocker"

    assert response.metadata["state_update_operation_count"] == 2


@pytest.mark.asyncio
async def test_ap_plus_navigator_returns_model_metadata() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        ("Which screen are you on?"),
        model="qwen3:4b",
        latency_ms=42.5,
    )

    agent = APPlusNavigatorAgent(model_service)

    response = await agent.execute(create_request())

    assert response.metadata["model"] == "qwen3:4b"

    assert response.metadata["model_provider"] == "ollama"

    assert response.metadata["model_latency_ms"] == 42.5

    assert response.metadata["guidance_mode"] == "system_navigation"


@pytest.mark.asyncio
async def test_ap_plus_navigator_strips_content() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response("  Which AP+ screen are you currently on?  ")

    agent = APPlusNavigatorAgent(model_service)

    response = await agent.execute(create_request())

    assert response.content == ("Which AP+ screen are you currently on?")


@pytest.mark.asyncio
async def test_ap_plus_navigator_rejects_invalid_structured_output() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="invalid-json",
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = APPlusNavigatorAgent(model_service)

    with pytest.raises(
        AgentExecutionError,
        match=("invalid structured output"),
    ):
        await agent.execute(create_request())


@pytest.mark.asyncio
async def test_ap_plus_navigator_rejects_empty_response() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=" ",
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = APPlusNavigatorAgent(model_service)

    with pytest.raises(
        AgentExecutionError,
        match="empty response",
    ):
        await agent.execute(create_request())


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

    assert exc_info.value.agent == "ap_plus_navigator"

    assert exc_info.value.details["model_error"]["error"] == "model_generation_error"


@pytest.mark.asyncio
async def test_ap_plus_navigator_preserves_model_error_as_cause() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    error = ModelGenerationError("ollama")

    generate_mock.side_effect = error

    agent = APPlusNavigatorAgent(model_service)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(create_request())

    assert exc_info.value.__cause__ is error

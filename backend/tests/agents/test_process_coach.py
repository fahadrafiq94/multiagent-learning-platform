from __future__ import annotations

import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.exceptions import AgentExecutionError
from app.agents.policies import (
    PROCESS_COACH_GUIDANCE,
)
from app.agents.process_coach import (
    PROCESS_COACH_SYSTEM_PROMPT,
    ProcessCoachAgent,
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

    awaited_call = generate_mock.await_args

    assert awaited_call is not None

    request = awaited_call.args[0]

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
        user_message=(
            "Why do we need a purchase order after identifying what material the company needs?"
        ),
        guidance_level=(guidance_level),
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

    generate_mock.return_value = model_response(
        "What information would a supplier need before committing to delivery?"
    )

    agent = ProcessCoachAgent(model_service)

    await agent.execute(create_request())

    generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_coach_builds_structured_model_request() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response("What would the supplier need to know?")

    agent = ProcessCoachAgent(model_service)

    request = create_request()

    await agent.execute(request)

    generated_request = get_generated_request(generate_mock)

    assert len(generated_request.messages) == 2

    system_message = generated_request.messages[0]

    assert PROCESS_COACH_SYSTEM_PROMPT in system_message.content

    assert "Guidance level: minimal" in system_message.content

    assert PROCESS_COACH_GUIDANCE[GuidanceLevel.MINIMAL] in system_message.content

    assert STATE_UPDATE_PROPOSAL_INSTRUCTIONS in system_message.content

    assert generated_request.messages[1].content == request.user_message

    assert generated_request.response_schema == (SpecialistAgentOutput.model_json_schema())

    assert generated_request.metadata["task"] == "process_coach_response"

    assert generated_request.metadata["output_contract"] == "specialist-agent-v2"


@pytest.mark.asyncio
async def test_process_coach_prompt_is_socratic() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response("What should happen next?")

    agent = ProcessCoachAgent(model_service)

    await agent.execute(create_request())

    request = get_generated_request(generate_mock)

    prompt = request.messages[0].content.casefold()

    assert "socratic" in prompt

    assert "business-process reasoning" in prompt or "business process" in prompt

    assert "do not provide ap+" in prompt


@pytest.mark.asyncio
async def test_process_coach_applies_guidance_level() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        "A purchase order formalizes the purchase commitment."
    )

    agent = ProcessCoachAgent(model_service)

    response = await agent.execute(
        create_request(
            guidance_level=(GuidanceLevel.EXPLICIT),
        )
    )

    request = get_generated_request(generate_mock)

    prompt = request.messages[0].content

    assert "Guidance level: explicit" in prompt

    assert PROCESS_COACH_GUIDANCE[GuidanceLevel.EXPLICIT] in prompt

    assert response.metadata["guidance_level"] == "explicit"


@pytest.mark.asyncio
async def test_process_coach_returns_structured_content() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        "What information must be confirmed before purchasing?"
    )

    agent = ProcessCoachAgent(model_service)

    response = await agent.execute(create_request())

    assert response.agent == "process_coach"

    assert response.content == ("What information must be confirmed before purchasing?")

    assert response.state_update_proposal is not None

    assert response.state_update_proposal.operations == []


@pytest.mark.asyncio
async def test_process_coach_returns_state_update_proposal() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        ("Let's focus on the procurement reasoning."),
        operations=[
            {
                "field": ("current_process_phase"),
                "action": "set",
                "value": "procurement",
                "reason": ("The current discussion is explicitly about procurement."),
            },
            {
                "field": ("active_blocker"),
                "action": "set",
                "value": ("Supplier selection is unclear."),
                "reason": ("The user identified supplier selection as the blocker."),
            },
        ],
    )

    agent = ProcessCoachAgent(model_service)

    response = await agent.execute(create_request())

    assert response.state_update_proposal is not None

    assert len(response.state_update_proposal.operations) == 2

    assert response.metadata["state_update_operation_count"] == 2


@pytest.mark.asyncio
async def test_process_coach_model_metadata() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response(
        "What should happen next?",
        model="qwen3:4b",
        latency_ms=42.5,
    )

    agent = ProcessCoachAgent(model_service)

    response = await agent.execute(create_request())

    assert response.metadata["model"] == "qwen3:4b"

    assert response.metadata["model_provider"] == "ollama"

    assert response.metadata["model_latency_ms"] == 42.5

    assert response.metadata["guidance_mode"] == "socratic"


@pytest.mark.asyncio
async def test_process_coach_strips_content() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = model_response("  What should happen before ordering?  ")

    agent = ProcessCoachAgent(model_service)

    response = await agent.execute(create_request())

    assert response.content == ("What should happen before ordering?")


@pytest.mark.asyncio
async def test_process_coach_rejects_invalid_structured_output() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="not-json",
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = ProcessCoachAgent(model_service)

    with pytest.raises(
        AgentExecutionError,
        match=("invalid structured output"),
    ):
        await agent.execute(create_request())


@pytest.mark.asyncio
async def test_process_coach_rejects_empty_response() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=" ",
        model="fake-model",
        provider="ollama",
        latency_ms=1.0,
    )

    agent = ProcessCoachAgent(model_service)

    with pytest.raises(
        AgentExecutionError,
        match="empty response",
    ):
        await agent.execute(create_request())


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

    assert exc_info.value.agent == "process_coach"

    assert exc_info.value.details["model_error"]["error"] == "model_generation_error"


@pytest.mark.asyncio
async def test_process_coach_preserves_model_error_as_cause() -> None:
    model_service = create_model_service_mock()

    generate_mock = get_generate_mock(model_service)

    error = ModelGenerationError("ollama")

    generate_mock.side_effect = error

    agent = ProcessCoachAgent(model_service)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(create_request())

    assert exc_info.value.__cause__ is error

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.orchestrator import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    OrchestratorAgent,
    OrchestratorRoutingError,
)
from app.agents.routing import (
    OrchestratorRequest,
    RoutingDecision,
)
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
    """Return ModelService.generate as an AsyncMock."""

    generate_mock = AsyncMock()

    model_service.generate = generate_mock  # type: ignore[method-assign]

    return generate_mock


def create_request(
    *,
    user_message: str = ("I know the supplier must be recorded, but I do not know where to do it."),
) -> OrchestratorRequest:
    return OrchestratorRequest(
        session_id="session-1",
        student_id="student-1",
        user_message=user_message,
        metadata={
            "source": "test",
        },
    )


@pytest.mark.asyncio
async def test_orchestrator_calls_model_service() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"ap_plus_navigator","reason":"The student needs AP+ system guidance."}'),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    await orchestrator.route(create_request())

    generate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_builds_expected_model_request() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"ap_plus_navigator","reason":"The student needs AP+ system guidance."}'),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    request = create_request()

    await orchestrator.route(request)

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

    assert ORCHESTRATOR_SYSTEM_PROMPT in system_message.content

    assert user_message.role == "user"

    assert user_message.content == request.user_message


@pytest.mark.asyncio
async def test_orchestrator_requests_structured_routing_output() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"scenario","reason":"The student needs scenario clarification."}'),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    await orchestrator.route(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    assert generated_request.response_schema is not None

    expected_schema = RoutingDecision.model_json_schema()

    assert generated_request.response_schema == expected_schema


@pytest.mark.asyncio
async def test_orchestrator_prompt_defines_all_routes() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"fallback","reason":"More information is required."}'),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    await orchestrator.route(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = " ".join(generated_request.messages[0].content.lower().split())

    assert "scenario" in prompt
    assert "process_coach" in prompt
    assert "ap_plus_navigator" in prompt
    assert "fallback" in prompt


@pytest.mark.asyncio
async def test_orchestrator_prompt_requires_semantic_routing() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"process_coach","reason":"The student needs conceptual reasoning."}'),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    await orchestrator.route(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = " ".join(generated_request.messages[0].content.lower().split())

    assert "route according to meaning, not keyword matching" in prompt


@pytest.mark.asyncio
async def test_orchestrator_does_not_answer_student_question() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=(
            '{"route":"process_coach","reason":"The student needs business-process reasoning."}'
        ),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    await orchestrator.route(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    prompt = " ".join(generated_request.messages[0].content.lower().split())

    assert "you do not answer the student's question yourself" in prompt


@pytest.mark.asyncio
async def test_orchestrator_returns_scenario_decision() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"scenario","reason":"The student needs scenario clarification."}'),
        model="fake-model",
        provider="ollama",
        latency_ms=11.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    result = await orchestrator.route(create_request())

    assert result.decision.route == "scenario"

    assert result.decision.reason == ("The student needs scenario clarification.")


@pytest.mark.asyncio
async def test_orchestrator_returns_process_coach_decision() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=(
            '{"route":"process_coach","reason":"The student needs business-process reasoning."}'
        ),
        model="fake-model",
        provider="ollama",
        latency_ms=11.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    result = await orchestrator.route(create_request())

    assert result.decision.route == "process_coach"


@pytest.mark.asyncio
async def test_orchestrator_returns_ap_plus_decision() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"ap_plus_navigator","reason":"The student needs AP+ navigation help."}'),
        model="fake-model",
        provider="ollama",
        latency_ms=11.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    result = await orchestrator.route(create_request())

    assert result.decision.route == "ap_plus_navigator"


@pytest.mark.asyncio
async def test_orchestrator_accepts_fallback_decision() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"fallback","reason":"The immediate need cannot yet be determined."}'),
        model="fake-model",
        provider="ollama",
        latency_ms=11.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    result = await orchestrator.route(create_request())

    assert result.decision.route == "fallback"


@pytest.mark.asyncio
async def test_orchestrator_returns_model_metadata() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"scenario","reason":"The student needs scenario clarification."}'),
        model="qwen3:4b",
        provider="ollama",
        latency_ms=42.5,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    result = await orchestrator.route(create_request())

    assert result.metadata["model"] == "qwen3:4b"

    assert result.metadata["model_provider"] == "ollama"

    assert result.metadata["model_latency_ms"] == 42.5

    assert result.metadata["routing_strategy"] == "llm_semantic_router_v1"


@pytest.mark.asyncio
async def test_orchestrator_rejects_invalid_json_response() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content="not valid json",
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    with pytest.raises(OrchestratorRoutingError) as exc_info:
        await orchestrator.route(create_request())

    error = exc_info.value

    assert error.message == ("Semantic router returned an invalid routing decision.")

    assert "validation_error" in error.details


@pytest.mark.asyncio
async def test_orchestrator_rejects_unknown_route() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"unknown_agent","reason":"Invalid target."}'),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    with pytest.raises(OrchestratorRoutingError):
        await orchestrator.route(create_request())


@pytest.mark.asyncio
async def test_orchestrator_rejects_missing_reason() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"scenario"}'),
        model="fake-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    with pytest.raises(OrchestratorRoutingError):
        await orchestrator.route(create_request())


@pytest.mark.asyncio
async def test_orchestrator_wraps_model_service_error() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.side_effect = ModelGenerationError(
        "ollama",
        details={
            "reason": "test failure",
        },
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    with pytest.raises(OrchestratorRoutingError) as exc_info:
        await orchestrator.route(create_request())

    error = exc_info.value

    assert error.message == ("Semantic routing model call failed.")

    assert error.details["model_error"]["error"] == "model_generation_error"


@pytest.mark.asyncio
async def test_orchestrator_preserves_model_error_as_cause() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    model_error = ModelGenerationError(
        "ollama",
        details={
            "reason": "test failure",
        },
    )

    generate_mock.side_effect = model_error

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    with pytest.raises(OrchestratorRoutingError) as exc_info:
        await orchestrator.route(create_request())

    assert exc_info.value.__cause__ is model_error


def test_orchestrator_routing_error_to_dict() -> None:
    error = OrchestratorRoutingError(
        "Routing failed.",
        details={
            "reason": "test",
        },
    )

    assert error.to_dict() == {
        "error": "orchestrator_routing_error",
        "message": "Routing failed.",
        "details": {
            "reason": "test",
        },
    }


@pytest.mark.asyncio
async def test_orchestrator_uses_configured_model() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"scenario","reason":"The student needs scenario clarification."}'),
        model="router-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
        model="router-model",
    )

    await orchestrator.route(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    assert isinstance(
        generated_request,
        ModelChatRequest,
    )

    assert generated_request.model == "router-model"


@pytest.mark.asyncio
async def test_orchestrator_leaves_model_unset_by_default() -> None:
    model_service = create_model_service_mock()
    generate_mock = get_generate_mock(model_service)

    generate_mock.return_value = ModelChatResponse(
        content=('{"route":"fallback","reason":"More information is required."}'),
        model="default-model",
        provider="ollama",
        latency_ms=10.0,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    await orchestrator.route(create_request())

    generated_request = (
        generate_mock.await_args.args[0]  # type: ignore[union-attr]
    )

    assert generated_request.model is None

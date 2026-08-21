from __future__ import annotations

import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.ap_plus_navigator import (
    AP_PLUS_NAVIGATOR_SYSTEM_PROMPT,
)
from app.agents.factory import create_agent_registry
from app.agents.orchestrator import OrchestratorAgent
from app.agents.policies import (
    AP_PLUS_NAVIGATOR_GUIDANCE,
    PROCESS_COACH_GUIDANCE,
    SCENARIO_GUIDANCE,
    GuidanceLevel,
)
from app.agents.process_coach import (
    PROCESS_COACH_SYSTEM_PROMPT,
)
from app.agents.routing import RoutingTarget
from app.agents.scenario import (
    SCENARIO_AGENT_SYSTEM_PROMPT,
)
from app.orchestration.graph import (
    build_orchestration_graph,
)
from app.orchestration.schemas import (
    OrchestrationRequest,
)
from app.orchestration.service import (
    OrchestrationService,
)
from app.services.model_service import ModelService
from app.services.model_service.exceptions import (
    ModelGenerationError,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)


def _semantic_route_for_message(
    user_message: str,
) -> RoutingTarget:
    """Return a deterministic fake semantic-model decision."""

    normalized = " ".join(user_message.lower().split())

    if (
        "ap+" in normalized
        or "where can i find" in normalized
        or "where should i enter" in normalized
    ):
        return "ap_plus_navigator"

    if (
        "why" in normalized
        or "procurement" in normalized
        or "business process" in normalized
        or "purchase order" in normalized
    ):
        return "process_coach"

    if "company" in normalized or "scenario" in normalized or "business problem" in normalized:
        return "scenario"

    return "fallback"


def _routing_reason(
    route: RoutingTarget,
) -> str:
    """Return a valid fake routing reason for one semantic route."""

    reasons: dict[RoutingTarget, str] = {
        "scenario": "The student needs business-scenario clarification.",
        "process_coach": "The student needs business-process reasoning.",
        "ap_plus_navigator": "The student needs AP+ system guidance.",
        "fallback": "The student's immediate need cannot yet be determined.",
    }

    return reasons[route]


def create_recording_model_service(
    *,
    response_content: str = "Fake integrated model response.",
) -> tuple[
    ModelService,
    AsyncMock,
    list[ModelChatRequest],
]:
    """Create a fake ModelService that records every generation request."""

    model_service_mock = MagicMock(
        spec=ModelService,
    )

    requests: list[ModelChatRequest] = []

    async def generate(
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        requests.append(request)

        if request.response_schema is not None:
            user_message = request.messages[-1].content

            route = _semantic_route_for_message(user_message)

            content = json.dumps(
                {
                    "route": route,
                    "reason": _routing_reason(route),
                }
            )
        else:
            content = response_content

        return ModelChatResponse(
            content=content,
            model="fake-model",
            provider="ollama",
            latency_ms=5.0,
        )

    generate_mock = AsyncMock(
        side_effect=generate,
    )

    model_service_mock.generate = generate_mock

    model_service = cast(
        ModelService,
        model_service_mock,
    )

    return (
        model_service,
        generate_mock,
        requests,
    )


def create_orchestration_service(
    model_service: ModelService,
) -> OrchestrationService:
    """Create the real Sprint 3 stack above the ModelService boundary."""

    agent_registry = create_agent_registry(
        model_service=model_service,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    graph = build_orchestration_graph(
        agent_registry=agent_registry,
        orchestrator=orchestrator,
    )

    return OrchestrationService(
        graph=graph,
    )


def _orchestrator_requests(
    requests: list[ModelChatRequest],
) -> list[ModelChatRequest]:
    return [request for request in requests if request.response_schema is not None]


def _agent_requests(
    requests: list[ModelChatRequest],
) -> list[ModelChatRequest]:
    return [request for request in requests if request.response_schema is None]


@pytest.mark.asyncio
async def test_scenario_request_executes_real_scenario_agent() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_recording_model_service(
        response_content=("What business problem is the bicycle company trying to solve?"),
    )

    service = create_orchestration_service(model_service)

    response = await service.execute(
        OrchestrationRequest(
            session_id="integration-scenario",
            student_id="student-1",
            user_message=(
                "Our company manufactures bicycles and I need help defining the business problem."
            ),
        )
    )

    assert response.route == "scenario"
    assert response.metadata["selected_agent"] == "scenario"
    assert response.metadata["agent_called"] is True
    assert response.metadata["routing_strategy"] == "llm_semantic_router_v1"
    assert response.metadata["keyword_fallback_used"] is False

    assert response.final_response == (
        "What business problem is the bicycle company trying to solve?"
    )

    assert generate_mock.await_count == 2
    assert len(requests) == 2

    assert len(_orchestrator_requests(requests)) == 1

    agent_requests = _agent_requests(requests)

    assert len(agent_requests) == 1

    system_prompt = agent_requests[0].messages[0].content

    assert SCENARIO_AGENT_SYSTEM_PROMPT in system_prompt
    assert "Guidance level: minimal" in system_prompt

    assert SCENARIO_GUIDANCE[GuidanceLevel.MINIMAL] in system_prompt


@pytest.mark.asyncio
async def test_process_request_executes_real_process_coach() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_recording_model_service(
        response_content=(
            "What information should be confirmed before the company places an order?"
        ),
    )

    service = create_orchestration_service(model_service)

    response = await service.execute(
        OrchestrationRequest(
            session_id="integration-process",
            student_id="student-1",
            user_message=("Why is a purchase order needed in the procurement process?"),
        )
    )

    assert response.route == "process_coach"
    assert response.metadata["selected_agent"] == "process_coach"
    assert response.metadata["agent_called"] is True

    assert response.metadata["agent_metadata"]["guidance_mode"] == "socratic"

    assert response.metadata["agent_metadata"]["guidance_level"] == "minimal"

    assert generate_mock.await_count == 2
    assert len(requests) == 2

    agent_requests = _agent_requests(requests)

    assert len(agent_requests) == 1

    system_prompt = agent_requests[0].messages[0].content

    assert PROCESS_COACH_SYSTEM_PROMPT in system_prompt
    assert "Guidance level: minimal" in system_prompt

    assert PROCESS_COACH_GUIDANCE[GuidanceLevel.MINIMAL] in system_prompt

    assert "Socratic" in system_prompt


@pytest.mark.asyncio
async def test_ap_plus_request_executes_real_navigator() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_recording_model_service(
        response_content="Which AP+ screen are you currently viewing?",
    )

    service = create_orchestration_service(model_service)

    response = await service.execute(
        OrchestrationRequest(
            session_id="integration-ap-plus",
            student_id="student-1",
            user_message=("Where can I find the supplier field in AP+?"),
        )
    )

    assert response.route == "ap_plus_navigator"
    assert response.metadata["selected_agent"] == "ap_plus_navigator"

    assert response.metadata["agent_metadata"]["guidance_mode"] == "system_navigation"

    assert generate_mock.await_count == 2
    assert len(requests) == 2

    agent_requests = _agent_requests(requests)

    assert len(agent_requests) == 1

    system_prompt = agent_requests[0].messages[0].content

    assert AP_PLUS_NAVIGATOR_SYSTEM_PROMPT in system_prompt

    assert AP_PLUS_NAVIGATOR_GUIDANCE[GuidanceLevel.MINIMAL] in system_prompt

    prompt_lower = system_prompt.lower()

    assert "never invent ap+ menu names" in prompt_lower
    assert "never invent ap+ screen names" in prompt_lower
    assert "never invent ap+ field names" in prompt_lower


@pytest.mark.asyncio
async def test_fallback_uses_only_orchestrator_model_call() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_recording_model_service()

    service = create_orchestration_service(model_service)

    response = await service.execute(
        OrchestrationRequest(
            session_id="integration-fallback",
            student_id="student-1",
            user_message="Hello there.",
        )
    )

    assert response.route == "fallback"
    assert response.metadata["fallback_used"] is True
    assert response.metadata["selected_agent"] is None
    assert response.metadata["keyword_fallback_used"] is True
    assert response.metadata["keyword_route"] == "fallback"

    assert generate_mock.await_count == 1
    assert len(requests) == 1

    assert len(_orchestrator_requests(requests)) == 1

    assert _agent_requests(requests) == []

    assert "business scenario" in response.final_response
    assert "business-process reasoning" in response.final_response
    assert "AP+" in response.final_response


@pytest.mark.asyncio
async def test_exactly_one_agent_model_call_occurs_per_routed_request() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_recording_model_service()

    service = create_orchestration_service(model_service)

    await service.execute(
        OrchestrationRequest(
            session_id="integration-one-call",
            student_id="student-1",
            user_message=("Why is procurement important in this business process?"),
        )
    )

    assert generate_mock.await_count == 2

    assert len(_orchestrator_requests(requests)) == 1

    assert len(_agent_requests(requests)) == 1


@pytest.mark.asyncio
async def test_default_guidance_level_reaches_agent() -> None:
    (
        model_service,
        _,
        requests,
    ) = create_recording_model_service()

    service = create_orchestration_service(model_service)

    response = await service.execute(
        OrchestrationRequest(
            session_id="integration-guidance",
            student_id="student-1",
            user_message=("Why do we need a purchase order in the business process?"),
        )
    )

    agent_requests = _agent_requests(requests)

    assert len(agent_requests) == 1

    system_prompt = agent_requests[0].messages[0].content

    assert "Guidance level: minimal" in system_prompt

    assert response.metadata["agent_metadata"]["guidance_level"] == "minimal"


@pytest.mark.asyncio
async def test_model_failure_isolated_as_controlled_agent_error() -> None:
    model_service_mock = MagicMock(
        spec=ModelService,
    )

    generate_mock = AsyncMock(
        side_effect=ModelGenerationError(
            "ollama",
            details={
                "reason": "integration failure",
            },
        )
    )

    model_service_mock.generate = generate_mock

    model_service = cast(
        ModelService,
        model_service_mock,
    )

    service = create_orchestration_service(model_service)

    response = await service.execute(
        OrchestrationRequest(
            session_id="integration-error",
            student_id="student-1",
            user_message=("I need help defining my company scenario."),
        )
    )

    assert generate_mock.await_count == 2
    assert response.route == "scenario"

    assert response.metadata["keyword_fallback_trigger"] == "orchestrator_error"

    assert response.metadata["semantic_routing_failed"] is True

    assert response.error == ("Scenario Agent could not generate a response.")

    assert response.metadata["selected_agent"] == "scenario"

    assert response.metadata["agent_error"]["error"] == "agent_execution_error"

    model_error = response.metadata["agent_error"]["details"]["model_error"]

    assert model_error["error"] == "model_generation_error"

    assert response.final_response == (
        "I could not process the message because: Scenario Agent could not generate a response."
    )


@pytest.mark.asyncio
async def test_student_identity_survives_full_orchestration_flow() -> None:
    (
        model_service,
        _,
        requests,
    ) = create_recording_model_service()

    service = create_orchestration_service(model_service)

    response = await service.execute(
        OrchestrationRequest(
            session_id="session-123",
            student_id="student-456",
            user_message="Help me define my company scenario.",
        )
    )

    assert response.session_id == "session-123"
    assert response.student_id == "student-456"

    orchestrator_requests = _orchestrator_requests(requests)

    assert len(orchestrator_requests) == 1

    assert orchestrator_requests[0].messages[-1].content == "Help me define my company scenario."


@pytest.mark.asyncio
async def test_agent_model_metadata_survives_full_flow() -> None:
    (
        model_service,
        _,
        _,
    ) = create_recording_model_service()

    service = create_orchestration_service(model_service)

    response = await service.execute(
        OrchestrationRequest(
            session_id="metadata-test",
            student_id="student-1",
            user_message=("Why is procurement part of the business process?"),
        )
    )

    agent_metadata = response.metadata["agent_metadata"]

    assert agent_metadata["model"] == "fake-model"
    assert agent_metadata["model_provider"] == "ollama"
    assert agent_metadata["model_latency_ms"] == 5.0
    assert agent_metadata["guidance_level"] == "minimal"

    orchestrator_metadata = response.metadata["orchestrator_metadata"]

    assert orchestrator_metadata["model"] == "fake-model"
    assert orchestrator_metadata["model_provider"] == "ollama"

    assert orchestrator_metadata["routing_strategy"] == "llm_semantic_router_v1"

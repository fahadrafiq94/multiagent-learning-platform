from __future__ import annotations

import json
from collections.abc import Iterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.agents.factory import create_agent_registry
from app.agents.orchestrator import OrchestratorAgent
from app.agents.routing import RoutingTarget
from app.api.routes import (
    orchestration as orchestration_routes,
)
from app.main import app
from app.orchestration.graph import (
    build_orchestration_graph,
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
    """Simulate semantic routing at the ModelService boundary."""

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
    reasons: dict[RoutingTarget, str] = {
        "scenario": "The student needs business-scenario clarification.",
        "process_coach": "The student needs business-process reasoning.",
        "ap_plus_navigator": "The student needs AP+ system guidance.",
        "fallback": "The student's immediate need cannot yet be determined.",
    }

    return reasons[route]


def create_model_service(
    *,
    content: str = "Integrated API response.",
) -> tuple[
    ModelService,
    AsyncMock,
    list[ModelChatRequest],
]:
    """Create the ModelService test boundary for the full HTTP stack."""

    model_service_mock = MagicMock(
        spec=ModelService,
    )

    requests: list[ModelChatRequest] = []

    async def generate(
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        requests.append(request)

        if request.response_schema is not None:
            route = _semantic_route_for_message(request.messages[-1].content)

            response_content = json.dumps(
                {
                    "route": route,
                    "reason": _routing_reason(route),
                }
            )
        else:
            response_content = content

        return ModelChatResponse(
            content=response_content,
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


def build_service(
    model_service: ModelService,
) -> OrchestrationService:
    """Build the real orchestration stack for API integration tests."""

    registry = create_agent_registry(
        model_service=model_service,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    graph = build_orchestration_graph(
        agent_registry=registry,
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


@pytest.fixture
def integrated_client() -> Iterator[
    tuple[
        TestClient,
        AsyncMock,
        list[ModelChatRequest],
    ]
]:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_model_service(
        content="Scenario integration response.",
    )

    service = build_service(model_service)

    app.dependency_overrides[orchestration_routes.get_orchestration_service] = lambda: service

    with TestClient(app) as client:
        yield (
            client,
            generate_mock,
            requests,
        )

    app.dependency_overrides.clear()


def test_http_request_runs_complete_agent_stack(
    integrated_client: tuple[
        TestClient,
        AsyncMock,
        list[ModelChatRequest],
    ],
) -> None:
    (
        client,
        generate_mock,
        requests,
    ) = integrated_client

    response = client.post(
        "/orchestration/invoke",
        json={
            "session_id": "api-integration-1",
            "student_id": "student-1",
            "user_message": ("I need help defining my company and business problem."),
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["route"] == "scenario"
    assert body["metadata"]["selected_agent"] == "scenario"
    assert body["metadata"]["agent_called"] is True

    assert body["metadata"]["routing_strategy"] == "llm_semantic_router_v1"

    assert body["final_response"] == "Scenario integration response."

    assert generate_mock.await_count == 2
    assert len(requests) == 2

    assert len(_orchestrator_requests(requests)) == 1

    assert len(_agent_requests(requests)) == 1


def test_http_process_request_uses_process_coach(
    integrated_client: tuple[
        TestClient,
        AsyncMock,
        list[ModelChatRequest],
    ],
) -> None:
    (
        client,
        generate_mock,
        requests,
    ) = integrated_client

    response = client.post(
        "/orchestration/invoke",
        json={
            "session_id": "api-process",
            "student_id": "student-1",
            "user_message": ("Why is a purchase order needed in the procurement process?"),
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["route"] == "process_coach"

    assert body["metadata"]["selected_agent"] == "process_coach"

    assert body["metadata"]["agent_metadata"]["guidance_mode"] == "socratic"

    assert generate_mock.await_count == 2

    assert len(_agent_requests(requests)) == 1


def test_http_ap_plus_request_uses_navigator(
    integrated_client: tuple[
        TestClient,
        AsyncMock,
        list[ModelChatRequest],
    ],
) -> None:
    (
        client,
        generate_mock,
        requests,
    ) = integrated_client

    response = client.post(
        "/orchestration/invoke",
        json={
            "session_id": "api-ap-plus",
            "student_id": "student-1",
            "user_message": ("Where can I find this field in AP+?"),
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["route"] == "ap_plus_navigator"

    assert body["metadata"]["selected_agent"] == "ap_plus_navigator"

    assert body["metadata"]["agent_metadata"]["guidance_mode"] == "system_navigation"

    assert generate_mock.await_count == 2

    assert len(_agent_requests(requests)) == 1


def test_http_fallback_uses_only_orchestrator_model_call(
    integrated_client: tuple[
        TestClient,
        AsyncMock,
        list[ModelChatRequest],
    ],
) -> None:
    (
        client,
        generate_mock,
        requests,
    ) = integrated_client

    response = client.post(
        "/orchestration/invoke",
        json={
            "session_id": "api-fallback",
            "student_id": "student-1",
            "user_message": "Hello there.",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["route"] == "fallback"
    assert body["metadata"]["fallback_used"] is True
    assert body["metadata"]["selected_agent"] is None
    assert body["metadata"]["keyword_fallback_used"] is True
    assert body["metadata"]["keyword_route"] == "fallback"

    assert generate_mock.await_count == 1
    assert len(requests) == 1

    assert len(_orchestrator_requests(requests)) == 1

    assert _agent_requests(requests) == []


def test_http_model_failure_returns_controlled_response() -> None:
    model_service_mock = MagicMock(
        spec=ModelService,
    )

    generate_mock = AsyncMock(
        side_effect=ModelGenerationError(
            "ollama",
            details={
                "reason": "API integration failure",
            },
        )
    )

    model_service_mock.generate = generate_mock

    model_service = cast(
        ModelService,
        model_service_mock,
    )

    service = build_service(model_service)

    app.dependency_overrides[orchestration_routes.get_orchestration_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/orchestration/invoke",
                json={
                    "session_id": "api-model-error",
                    "student_id": "student-1",
                    "user_message": ("Help me define my company scenario."),
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert generate_mock.await_count == 2
    assert response.status_code == 200

    body = response.json()

    assert body["route"] == "scenario"

    assert body["metadata"]["keyword_fallback_trigger"] == "orchestrator_error"

    assert body["metadata"]["semantic_routing_failed"] is True

    assert body["error"] == ("Scenario Agent could not generate a response.")

    assert body["metadata"]["agent_error"]["error"] == "agent_execution_error"


def test_http_invalid_request_still_returns_422() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_model_service()

    service = build_service(model_service)

    app.dependency_overrides[orchestration_routes.get_orchestration_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/orchestration/invoke",
                json={
                    "session_id": "api-invalid",
                    "user_message": "   ",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422

    generate_mock.assert_not_awaited()

    assert requests == []

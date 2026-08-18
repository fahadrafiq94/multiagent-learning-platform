from __future__ import annotations

from collections.abc import Iterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.agents.factory import create_agent_registry
from app.api.routes import (
    orchestration as orchestration_routes,
)
from app.main import app
from app.orchestration.graph import build_orchestration_graph
from app.orchestration.service import OrchestrationService
from app.services.model_service import ModelService
from app.services.model_service.exceptions import (
    ModelGenerationError,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)


def create_model_service(
    *,
    content: str = "Integrated API response.",
) -> tuple[
    ModelService,
    AsyncMock,
    list[ModelChatRequest],
]:
    model_service = cast(
        ModelService,
        MagicMock(spec=ModelService),
    )

    requests: list[ModelChatRequest] = []

    async def generate(
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        requests.append(request)

        return ModelChatResponse(
            content=content,
            model="fake-model",
            provider="ollama",
            latency_ms=5.0,
        )

    generate_mock = AsyncMock(side_effect=generate)

    model_service.generate = generate_mock  # type: ignore[method-assign]

    return (
        model_service,
        generate_mock,
        requests,
    )


def build_service(
    model_service: ModelService,
) -> OrchestrationService:
    registry = create_agent_registry(
        model_service=model_service,
    )

    graph = build_orchestration_graph(
        agent_registry=registry,
    )

    return OrchestrationService(
        graph=graph,
    )


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

    assert body["final_response"] == "Scenario integration response."

    generate_mock.assert_awaited_once()

    assert len(requests) == 1


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

    generate_mock.assert_awaited_once()

    assert len(requests) == 1


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

    generate_mock.assert_awaited_once()

    assert len(requests) == 1


def test_http_fallback_does_not_invoke_model(
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

    generate_mock.assert_not_awaited()

    assert requests == []


def test_http_model_failure_returns_controlled_response() -> None:
    model_service = cast(
        ModelService,
        MagicMock(spec=ModelService),
    )

    generate_mock = AsyncMock(
        side_effect=ModelGenerationError(
            "ollama",
            details={
                "reason": "API integration failure",
            },
        )
    )

    model_service.generate = generate_mock  # type: ignore[method-assign]

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

    # The graph executed correctly and converted the expected
    # model failure into application state, so this remains 200.
    assert response.status_code == 200

    body = response.json()

    assert body["route"] == "scenario"

    assert body["error"] == ("Scenario Agent could not generate a response.")

    assert body["metadata"]["agent_error"]["error"] == "agent_execution_error"


def test_http_invalid_request_still_returns_422() -> None:
    model_service, _, _ = create_model_service()

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

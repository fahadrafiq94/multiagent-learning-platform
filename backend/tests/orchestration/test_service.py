from __future__ import annotations

from typing import Any

import pytest

from app.orchestration.schemas import OrchestrationRequest
from app.orchestration.service import (
    OrchestrationService,
    OrchestrationServiceError,
)


class FakeGraph:
    def __init__(self) -> None:
        self.received_input: dict[str, Any] | None = None

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.received_input = input

        return {
            **input,
            "model_response": "Fake model response",
            "final_response": "Fake model response",
            "error": None,
            "metadata": {
                **input.get("metadata", {}),
                "prepared": True,
                "model_called": True,
                "finalized": True,
            },
        }


class FailingGraph:
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("Graph failed")


class MissingResponseGraph:
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            **input,
            "error": None,
            "metadata": {},
        }


@pytest.mark.asyncio
async def test_orchestration_service_invokes_graph() -> None:
    graph = FakeGraph()
    service = OrchestrationService(graph=graph)

    request = OrchestrationRequest(
        session_id="session-1",
        student_id="student-1",
        user_message="Hello",
        metadata={"source": "test"},
    )

    response = await service.execute(request)

    assert graph.received_input is not None
    assert graph.received_input["session_id"] == "session-1"
    assert graph.received_input["student_id"] == "student-1"
    assert graph.received_input["user_message"] == "Hello"

    assert response.session_id == "session-1"
    assert response.student_id == "student-1"
    assert response.final_response == "Fake model response"
    assert response.error is None
    assert response.metadata["model_called"] is True


@pytest.mark.asyncio
async def test_orchestration_service_preserves_graph_error() -> None:
    class ErrorResultGraph:
        async def ainvoke(
            self,
            input: dict[str, Any],
            config: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return {
                **input,
                "error": "Model unavailable",
                "final_response": ("I could not process the message because: Model unavailable"),
                "metadata": {
                    "finalized": True,
                },
            }

    service = OrchestrationService(graph=ErrorResultGraph())

    response = await service.execute(
        OrchestrationRequest(
            session_id="session-1",
            user_message="Hello",
        )
    )

    assert response.error == "Model unavailable"
    assert response.final_response == ("I could not process the message because: Model unavailable")


@pytest.mark.asyncio
async def test_orchestration_service_wraps_unexpected_graph_error() -> None:
    service = OrchestrationService(graph=FailingGraph())

    with pytest.raises(
        OrchestrationServiceError,
        match="workflow failed unexpectedly",
    ):
        await service.execute(
            OrchestrationRequest(
                session_id="session-1",
                user_message="Hello",
            )
        )


@pytest.mark.asyncio
async def test_orchestration_service_rejects_missing_final_response() -> None:
    service = OrchestrationService(graph=MissingResponseGraph())

    with pytest.raises(
        OrchestrationServiceError,
        match="did not produce a final response",
    ):
        await service.execute(
            OrchestrationRequest(
                session_id="session-1",
                user_message="Hello",
            )
        )

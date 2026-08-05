from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routes import orchestration as orchestration_routes
from app.main import app
from app.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
)
from app.orchestration.service import OrchestrationServiceError


class FakeOrchestrationService:
    async def execute(
        self,
        request: OrchestrationRequest,
    ) -> OrchestrationResponse:
        return OrchestrationResponse(
            session_id=request.session_id,
            student_id=request.student_id,
            final_response=f"Processed: {request.user_message}",
            route=None,
            route_reason=None,
            error=None,
            metadata={
                **request.metadata,
                "prepared": True,
                "model_called": True,
                "finalized": True,
            },
        )


class FailingOrchestrationService:
    async def execute(
        self,
        request: OrchestrationRequest,
    ) -> OrchestrationResponse:
        raise OrchestrationServiceError("The orchestration workflow failed unexpectedly.")


def override_orchestration_service() -> FakeOrchestrationService:
    return FakeOrchestrationService()


client = TestClient(app)


def setup_function() -> None:
    app.dependency_overrides[orchestration_routes.get_orchestration_service] = (
        override_orchestration_service
    )


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_invoke_orchestration_route() -> None:
    response = client.post(
        "/orchestration/invoke",
        json={
            "session_id": "session-1",
            "student_id": "student-1",
            "user_message": "Hello orchestration",
            "metadata": {
                "source": "api-test",
            },
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["session_id"] == "session-1"
    assert body["student_id"] == "student-1"
    assert body["final_response"] == "Processed: Hello orchestration"
    assert body["error"] is None
    assert body["metadata"]["prepared"] is True
    assert body["metadata"]["model_called"] is True


def test_invoke_orchestration_route_without_student_id() -> None:
    response = client.post(
        "/orchestration/invoke",
        json={
            "session_id": "session-1",
            "user_message": "Hello",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["student_id"] is None
    assert body["final_response"] == "Processed: Hello"


def test_invoke_orchestration_route_rejects_blank_message() -> None:
    response = client.post(
        "/orchestration/invoke",
        json={
            "session_id": "session-1",
            "user_message": "   ",
        },
    )

    assert response.status_code == 422


def test_invoke_orchestration_route_rejects_missing_session_id() -> None:
    response = client.post(
        "/orchestration/invoke",
        json={
            "user_message": "Hello",
        },
    )

    assert response.status_code == 422


def test_invoke_orchestration_route_handles_service_failure() -> None:
    def override_failing_service() -> FailingOrchestrationService:
        return FailingOrchestrationService()

    app.dependency_overrides[orchestration_routes.get_orchestration_service] = (
        override_failing_service
    )

    response = client.post(
        "/orchestration/invoke",
        json={
            "session_id": "session-1",
            "user_message": "Hello",
        },
    )

    assert response.status_code == 500

    body = response.json()

    assert body["detail"]["error"] == "orchestration_service_error"
    assert body["detail"]["message"] == "The orchestration workflow failed unexpectedly."

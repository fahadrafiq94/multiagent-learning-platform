from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import pytest

from app.identity import UserIdentity
from app.observability.events import (
    REQUEST_COMPLETED_EVENT,
    REQUEST_EVENT_LOGGER_NAME,
)
from app.orchestration.schemas import (
    OrchestrationRequest,
)
from app.orchestration.service import (
    OrchestrationService,
    OrchestrationServiceError,
)
from app.repositories import (
    create_in_memory_repositories,
)


class FakeGraph:
    def __init__(self) -> None:
        self.received_input: dict[str, Any] | None = None

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: (dict[str, Any] | None) = None,
    ) -> dict[str, Any]:
        self.received_input = input

        return {
            **input,
            "final_response": ("Fake graph response"),
            "route": "process_coach",
            "route_reason": ("Fake routing reason."),
            "error": None,
            "metadata": {
                **input.get(
                    "metadata",
                    {},
                ),
                "finalized": True,
            },
        }


class FailingGraph:
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: (dict[str, Any] | None) = None,
    ) -> dict[str, Any]:
        raise RuntimeError("Graph failed")


class MissingResponseGraph:
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: (dict[str, Any] | None) = None,
    ) -> dict[str, Any]:
        return {
            **input,
            "error": None,
            "metadata": {},
        }


class ErrorResultGraph:
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: (dict[str, Any] | None) = None,
    ) -> dict[str, Any]:
        return {
            **input,
            "error": ("Model unavailable"),
            "final_response": ("I could not process the message because: Model unavailable"),
            "metadata": {
                "finalized": True,
            },
        }


async def create_identity_repository():
    repositories = create_in_memory_repositories()

    identity = await repositories.identities.create(UserIdentity(student_id="K12345"))

    return (
        repositories.identities,
        identity,
    )


@pytest.mark.asyncio
async def test_orchestration_service_resolves_identity_and_invokes_graph() -> None:
    (
        identity_repository,
        identity,
    ) = await create_identity_repository()

    graph = FakeGraph()

    service = OrchestrationService(
        graph=graph,
        identity_repository=(identity_repository),
    )

    session_id = str(uuid4())

    response = await service.execute(
        OrchestrationRequest(
            session_id=session_id,
            student_id="K12345",
            user_message="Hello",
            metadata={
                "source": "test",
            },
        )
    )

    assert graph.received_input is not None

    assert graph.received_input["student_id"] == "K12345"

    assert graph.received_input["user_id"] == identity.user_id

    assert str(graph.received_input["session_uuid"]) == session_id

    assert graph.received_input["user_message"] == "Hello"

    assert "trace_id" in graph.received_input

    assert response.final_response == "Fake graph response"

    assert response.route == "process_coach"


@pytest.mark.asyncio
async def test_orchestration_service_preserves_graph_error() -> None:
    (
        identity_repository,
        _,
    ) = await create_identity_repository()

    service = OrchestrationService(
        graph=ErrorResultGraph(),
        identity_repository=(identity_repository),
    )

    response = await service.execute(
        OrchestrationRequest(
            session_id=str(uuid4()),
            student_id="K12345",
            user_message="Hello",
        )
    )

    assert response.error == "Model unavailable"

    assert response.final_response == ("I could not process the message because: Model unavailable")


@pytest.mark.asyncio
async def test_orchestration_service_wraps_unexpected_graph_error() -> None:
    (
        identity_repository,
        _,
    ) = await create_identity_repository()

    service = OrchestrationService(
        graph=FailingGraph(),
        identity_repository=(identity_repository),
    )

    with pytest.raises(
        OrchestrationServiceError,
        match=("workflow failed unexpectedly"),
    ):
        await service.execute(
            OrchestrationRequest(
                session_id=str(uuid4()),
                student_id="K12345",
                user_message="Hello",
            )
        )


@pytest.mark.asyncio
async def test_orchestration_service_rejects_missing_final_response() -> None:
    (
        identity_repository,
        _,
    ) = await create_identity_repository()

    service = OrchestrationService(
        graph=MissingResponseGraph(),
        identity_repository=(identity_repository),
    )

    with pytest.raises(
        OrchestrationServiceError,
        match=("did not produce a final response"),
    ):
        await service.execute(
            OrchestrationRequest(
                session_id=str(uuid4()),
                student_id="K12345",
                user_message="Hello",
            )
        )


@pytest.mark.asyncio
async def test_orchestration_service_rejects_unknown_student_id() -> None:
    repositories = create_in_memory_repositories()

    service = OrchestrationService(
        graph=FakeGraph(),
        identity_repository=(repositories.identities),
    )

    with pytest.raises(
        OrchestrationServiceError,
        match=("No FREDi user identity"),
    ):
        await service.execute(
            OrchestrationRequest(
                session_id=str(uuid4()),
                student_id="UNKNOWN",
                user_message="Hello",
            )
        )


def _request_events(
    caplog: pytest.LogCaptureFixture,
) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == REQUEST_EVENT_LOGGER_NAME]


def _event_payload(
    record: logging.LogRecord,
) -> dict[str, Any]:
    payload = getattr(record, "event", None)

    assert isinstance(payload, dict)

    return payload


@pytest.mark.asyncio
async def test_orchestration_service_emits_single_request_event_on_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    (
        identity_repository,
        identity,
    ) = await create_identity_repository()

    service = OrchestrationService(
        graph=FakeGraph(),
        identity_repository=(identity_repository),
    )

    session_id = str(uuid4())

    with caplog.at_level(
        logging.INFO,
        logger=(REQUEST_EVENT_LOGGER_NAME),
    ):
        await service.execute(
            OrchestrationRequest(
                session_id=session_id,
                student_id="K12345",
                user_message="Hello",
            )
        )

    events = _request_events(caplog)

    assert len(events) == 1

    assert events[0].msg == REQUEST_COMPLETED_EVENT

    event = _event_payload(events[0])

    assert event["outcome"] == "completed"

    assert event["error"] is None

    assert event["student_id"] == "K12345"

    assert event["session_id"] == session_id

    assert event["user_id"] == str(identity.user_id)

    assert event["route"] == "process_coach"

    assert event["route_reason"] == ("Fake routing reason.")

    # Duration is wall-clock and clock-resolution dependent (coarse
    # on some virtualized Windows hosts); assert it is recorded, not
    # its magnitude.
    assert event["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_orchestration_service_emits_error_event_and_still_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    (
        identity_repository,
        _,
    ) = await create_identity_repository()

    service = OrchestrationService(
        graph=FailingGraph(),
        identity_repository=(identity_repository),
    )

    with caplog.at_level(
        logging.INFO,
        logger=(REQUEST_EVENT_LOGGER_NAME),
    ):
        with pytest.raises(
            OrchestrationServiceError,
            match=("workflow failed unexpectedly"),
        ):
            await service.execute(
                OrchestrationRequest(
                    session_id=str(uuid4()),
                    student_id="K12345",
                    user_message="Hello",
                )
            )

    events = _request_events(caplog)

    assert len(events) == 1

    event = _event_payload(events[0])

    assert event["outcome"] == "error"

    assert event["error"] == "Graph failed"

    assert event["student_id"] == "K12345"

    assert event["route"] is None

    assert event["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_orchestration_service_requires_valid_session_uuid() -> None:
    (
        identity_repository,
        _,
    ) = await create_identity_repository()

    service = OrchestrationService(
        graph=FakeGraph(),
        identity_repository=(identity_repository),
    )

    with pytest.raises(
        OrchestrationServiceError,
        match=("session_id must be a valid"),
    ):
        await service.execute(
            OrchestrationRequest(
                session_id=("not-a-uuid"),
                student_id="K12345",
                user_message="Hello",
            )
        )

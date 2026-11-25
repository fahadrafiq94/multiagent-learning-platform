from __future__ import annotations

import logging
import time
from typing import (
    Any,
    Protocol,
    cast,
)
from uuid import UUID, uuid4

from app.identity import UserIdentity
from app.identity.schemas import UserStatus
from app.logging.logger import logger
from app.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
)
from app.orchestration.state import (
    OrchestrationState,
)
from app.repositories.identity import (
    UserIdentityRepository,
)
from app.repositories.turns import TurnStore
from app.services.durable_turn_service import DurableTurnService


class AsyncGraph(Protocol):
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: (dict[str, Any] | None) = None,
    ) -> dict[str, Any]: ...


class OrchestrationServiceError(Exception):
    pass


class OrchestrationClientError(OrchestrationServiceError):
    """
    Base class for errors caused by the request itself, not by a
    failure inside the workflow.

    The API layer maps these to 4xx responses so bad client input
    is never reported as a server error.
    """

    http_status_code = 400


class UnknownStudentError(OrchestrationClientError):
    """No FREDi identity exists for the supplied student_id."""

    http_status_code = 404


class InvalidSessionIdError(OrchestrationClientError):
    """The supplied session_id is not a valid session UUID."""

    http_status_code = 422


class MissingRequestIdError(OrchestrationClientError):
    http_status_code = 422


class UnauthorizedStudentError(OrchestrationClientError):
    http_status_code = 403


class OrchestrationService:
    def __init__(
        self,
        *,
        graph: AsyncGraph,
        identity_repository: (UserIdentityRepository),
        turn_store: TurnStore | None = None,
    ) -> None:
        self._graph = graph
        self._identity_repository = identity_repository
        self._turn_service = DurableTurnService(turn_store) if turn_store is not None else None

    def _emit_request_event(
        self,
        state: dict[str, Any],
        *,
        duration_ms: float,
    ) -> None:
        # Best-effort (4B.10): an observability failure must never fail
        # or mask the real request outcome.
        try:
            # Imported lazily: app.observability.events imports
            # orchestration state symbols, so a module-level import
            # here would cycle through app.orchestration.__init__.
            from app.observability.events import (
                REQUEST_COMPLETED_EVENT,
                REQUEST_EVENT_LOGGER_NAME,
                build_request_event,
            )

            event = build_request_event(
                state,
                duration_ms=(duration_ms),
            )

            logging.getLogger(REQUEST_EVENT_LOGGER_NAME).info(
                REQUEST_COMPLETED_EVENT,
                extra={"event": event.log_payload()},
            )
        except Exception:  # noqa: BLE001 - emission is best-effort
            pass

    async def execute(
        self,
        request: OrchestrationRequest,
        *,
        trusted_user_id: UUID | None = None,
    ) -> OrchestrationResponse:
        if trusted_user_id is None:
            identity = await self._identity_repository.get_by_student_id(
                student_id=request.student_id
            )
        else:
            identity = await self._identity_repository.get_by_user_id(trusted_user_id)
            if (
                identity is None
                or identity.student_id != request.student_id
                or identity.status != UserStatus.ACTIVE
            ):
                raise UnauthorizedStudentError("Current user is not authorized for this request.")

        if identity is None:
            raise UnknownStudentError("No FREDi user identity exists for the supplied student_id.")

        try:
            session_uuid = UUID(request.session_id)
        except ValueError as exc:
            raise InvalidSessionIdError("session_id must be a valid FREDi session UUID.") from exc

        if self._turn_service is not None:
            if request.request_id is None:
                raise MissingRequestIdError("request_id is required for durable requests.")
            return await self._turn_service.execute(
                request,
                user_id=identity.user_id,
                session_id=session_uuid,
                run=lambda message_id, trace_id, replaced_message_ids: self._execute_graph(
                    request,
                    identity,
                    session_uuid,
                    current_message_id=message_id,
                    trace_id=trace_id,
                    replaced_message_ids=replaced_message_ids,
                ),
            )
        if request.edit_message_id is not None:
            raise OrchestrationClientError("Message regeneration requires PostgreSQL storage.")
        return await self._execute_graph(request, identity, session_uuid)

    async def _execute_graph(
        self,
        request: OrchestrationRequest,
        identity: UserIdentity,
        session_uuid: UUID,
        *,
        current_message_id: UUID | None = None,
        trace_id: UUID | None = None,
        replaced_message_ids: tuple[UUID, ...] = (),
    ) -> OrchestrationResponse:
        trace_id = trace_id or uuid4()

        metadata = {
            **{
                k: v
                for k, v in request.metadata.items()
                if k not in {"rag", "grounding", "source_references"}
            },
            "trace_id": str(trace_id),
            "request_id": request.request_id,
        }

        initial_state: OrchestrationState = {
            "session_id": (request.session_id),
            "student_id": (request.student_id),
            "user_id": (identity.user_id),
            "session_uuid": (session_uuid),
            "trace_id": trace_id,
            "user_message": (request.user_message),
            "metadata": metadata,
            "error": None,
        }
        if replaced_message_ids:
            initial_state["replaced_message_ids"] = replaced_message_ids
        if current_message_id is not None:
            initial_state["turn_managed"] = True
            initial_state["current_message_id"] = current_message_id

        logger.info(
            "Orchestration_started",
            session_id=(request.session_id),
            student_id=(request.student_id),
            trace_id=str(trace_id),
        )

        started_at = time.monotonic()

        try:
            raw_result = await self._graph.ainvoke(
                cast(
                    dict[str, Any],
                    initial_state,
                )
            )
        except Exception as exc:
            logger.exception(
                "Orchestration_failed",
                session_id=(request.session_id),
                student_id=(request.student_id),
                trace_id=str(trace_id),
                error=str(exc),
            )

            self._emit_request_event(
                {
                    **dict(initial_state),
                    "error": str(exc),
                },
                duration_ms=(time.monotonic() - started_at) * 1000,
            )

            raise OrchestrationServiceError(
                "The orchestration workflow failed unexpectedly."
            ) from exc

        result = cast(
            OrchestrationState,
            raw_result,
        )

        final_response = result.get(
            "final_response",
            "",
        )

        if not final_response:
            raise OrchestrationServiceError(
                "The orchestration workflow did not produce a final response."
            )

        self._emit_request_event(
            dict(raw_result),
            duration_ms=(time.monotonic() - started_at) * 1000,
        )

        return OrchestrationResponse(
            session_id=result.get(
                "session_id",
                request.session_id,
            ),
            student_id=result.get(
                "student_id",
                request.student_id,
            ),
            final_response=(final_response),
            route=result.get("route"),
            route_reason=result.get("route_reason"),
            error=result.get("error"),
            source_references=result.get("source_references", [])
            if not result.get("error")
            else [],
            metadata=dict(
                result.get(
                    "metadata",
                    {},
                )
            ),
        )

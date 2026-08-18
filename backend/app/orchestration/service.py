from __future__ import annotations

from typing import Any, Protocol, cast

from app.agents.policies import GuidanceLevel
from app.logging.logger import logger
from app.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
)
from app.orchestration.state import OrchestrationState


class AsyncGraph(Protocol):
    """Minimal compiled-graph contract required by OrchestrationService."""

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the graph asynchronously."""
        ...


class OrchestrationServiceError(Exception):
    """Raised when orchestration execution fails unexpectedly."""


class OrchestrationService:
    """Application-facing service for invoking the LangGraph workflow.

    API routes and other application components should depend on this service,
    not on the compiled LangGraph instance directly.
    """

    def __init__(self, graph: AsyncGraph) -> None:
        self._graph = graph

    async def execute(
        self,
        request: OrchestrationRequest,
    ) -> OrchestrationResponse:
        """Execute the orchestration graph for one student message."""

        initial_state: OrchestrationState = {
            "session_id": request.session_id,
            "student_id": request.student_id,
            "user_message": request.user_message,
            "guidance_level": GuidanceLevel.MINIMAL,
            "metadata": dict(request.metadata),
            "error": None,
        }

        logger.info(
            "Orchestration_started",
            session_id=request.session_id,
            student_id=request.student_id,
        )

        try:
            raw_result = await self._graph.ainvoke(cast(dict[str, Any], initial_state))
        except Exception as exc:
            logger.exception(
                "Orchestration_failed",
                session_id=request.session_id,
                student_id=request.student_id,
                error=str(exc),
            )

            raise OrchestrationServiceError(
                "The orchestration workflow failed unexpectedly."
            ) from exc

        result = cast(OrchestrationState, raw_result)

        response = self._build_response(
            request=request,
            result=result,
        )

        logger.info(
            "Orchestration_completed",
            session_id=response.session_id,
            student_id=response.student_id,
            route=response.route,
            has_error=response.error is not None,
        )

        return response

    def _build_response(
        self,
        *,
        request: OrchestrationRequest,
        result: OrchestrationState,
    ) -> OrchestrationResponse:
        final_response = result.get("final_response", "")

        if not final_response:
            raise OrchestrationServiceError(
                "The orchestration workflow did not produce a final response."
            )

        return OrchestrationResponse(
            session_id=result.get("session_id", request.session_id),
            student_id=result.get("student_id", request.student_id),
            final_response=final_response,
            route=result.get("route"),
            route_reason=result.get("route_reason"),
            error=result.get("error"),
            metadata=dict(result.get("metadata", {})),
        )

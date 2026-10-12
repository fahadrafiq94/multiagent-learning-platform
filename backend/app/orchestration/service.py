from __future__ import annotations

from typing import (
    Any,
    Protocol,
    cast,
)
from uuid import UUID, uuid4

from app.domain.guidance import (
    GuidanceLevel,
)
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


class AsyncGraph(Protocol):
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: (dict[str, Any] | None) = None,
    ) -> dict[str, Any]: ...


class OrchestrationServiceError(Exception):
    pass


class OrchestrationService:
    def __init__(
        self,
        *,
        graph: AsyncGraph,
        identity_repository: (UserIdentityRepository),
    ) -> None:
        self._graph = graph
        self._identity_repository = identity_repository

    async def execute(
        self,
        request: OrchestrationRequest,
    ) -> OrchestrationResponse:
        identity = await self._identity_repository.get_by_student_id(
            student_id=(request.student_id),
        )

        if identity is None:
            raise OrchestrationServiceError(
                "No FREDi user identity exists for the supplied student_id."
            )

        try:
            session_uuid = UUID(request.session_id)
        except ValueError as exc:
            raise OrchestrationServiceError(
                "session_id must be a valid FREDi session UUID."
            ) from exc

        trace_id = uuid4()

        metadata = {
            **dict(request.metadata),
            "trace_id": str(trace_id),
        }

        initial_state: OrchestrationState = {
            "session_id": (request.session_id),
            "student_id": (request.student_id),
            "user_id": (identity.user_id),
            "session_uuid": (session_uuid),
            "trace_id": trace_id,
            "user_message": (request.user_message),
            "guidance_level": (GuidanceLevel.MINIMAL),
            "metadata": metadata,
            "error": None,
        }

        logger.info(
            "Orchestration_started",
            session_id=(request.session_id),
            student_id=(request.student_id),
            trace_id=str(trace_id),
        )

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
            metadata=dict(
                result.get(
                    "metadata",
                    {},
                )
            ),
        )

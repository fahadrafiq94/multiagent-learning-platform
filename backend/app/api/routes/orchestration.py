from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.orchestration import (
    OrchestrationRequest,
    OrchestrationResponse,
    OrchestrationService,
    OrchestrationServiceError,
    create_orchestration_service,
)

router = APIRouter(
    prefix="/orchestration",
    tags=["orchestration"],
)


def get_orchestration_service() -> OrchestrationService:
    """Create the configured orchestration service.

    This dependency can be overridden in tests.
    """
    return create_orchestration_service()


OrchestrationServiceDep = Annotated[
    OrchestrationService,
    Depends(get_orchestration_service),
]


@router.post(
    "/invoke",
    response_model=OrchestrationResponse,
    status_code=status.HTTP_200_OK,
)
async def invoke_orchestration(
    request: OrchestrationRequest,
    orchestration_service: OrchestrationServiceDep,
) -> OrchestrationResponse:
    """Execute the LangGraph orchestration workflow for one user message."""

    try:
        return await orchestration_service.execute(request)
    except OrchestrationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "orchestration_service_error",
                "message": str(exc),
            },
        ) from exc

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.orchestration import (
    OrchestrationClientError,
    OrchestrationRequest,
    OrchestrationResponse,
    OrchestrationService,
    OrchestrationServiceError,
    create_orchestration_service,
)
from app.services.model_service.exceptions import ModelServiceError

router = APIRouter(
    prefix="/orchestration",
    tags=["orchestration"],
)


def get_orchestration_service() -> OrchestrationService:
    """Create the configured orchestration service.

    This dependency can be overridden in tests.

    Runtime construction (including model-service creation) happens
    here, so a misconfigured provider fails with a clean 503 instead
    of an unhandled 500.
    """
    try:
        return create_orchestration_service()
    except ModelServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=exc.to_dict(),
        ) from exc


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
    except OrchestrationClientError as exc:
        # Client-side input problems are 4xx, never server errors.
        raise HTTPException(
            status_code=exc.http_status_code,
            detail={
                "error": "orchestration_client_error",
                "message": str(exc),
            },
        ) from exc
    except OrchestrationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "orchestration_service_error",
                "message": str(exc),
            },
        ) from exc

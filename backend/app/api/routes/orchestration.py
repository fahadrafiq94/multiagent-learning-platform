from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import authorize_invocation, get_runtime
from app.identity import UserIdentity
from app.orchestration import (
    OrchestrationClientError,
    OrchestrationRequest,
    OrchestrationResponse,
    OrchestrationService,
    OrchestrationServiceError,
)
from app.services.model_service.exceptions import ModelServiceError

router = APIRouter(
    prefix="/orchestration",
    tags=["orchestration"],
)


def get_orchestration_service(request: Request) -> OrchestrationService:
    """Return the lifespan-owned orchestration service.

    This dependency can be overridden in tests.

    Startup constructs the model service and configured repositories.
    Requests without a ready lifespan runtime fail with 503.
    """
    try:
        return get_runtime(request).orchestration_service
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
    current_user: Annotated[UserIdentity, Depends(authorize_invocation)],
) -> OrchestrationResponse:
    """Execute the LangGraph orchestration workflow for one user message."""

    try:
        return await orchestration_service.execute(request, trusted_user_id=current_user.user_id)
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

"""Trusted identity and lifespan-owned runtime dependencies."""

from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config.settings import settings
from app.identity import UserIdentity
from app.identity.browser_session import browser_configured, cookie_name, require_csrf
from app.identity.pilot_tokens import verify_token
from app.identity.schemas import UserStatus
from app.orchestration.factory import OrchestrationRuntime
from app.orchestration.schemas import OrchestrationRequest

bearer = HTTPBearer(auto_error=False)


def get_runtime(request: Request) -> OrchestrationRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(503, detail="Application runtime is not ready.")
    return cast(OrchestrationRuntime, runtime)


RuntimeDep = Annotated[OrchestrationRuntime, Depends(get_runtime)]


async def get_current_user(
    request: Request,
    runtime: RuntimeDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> UserIdentity:
    if settings.application_pilot_auth_secret is None:
        raise HTTPException(503, detail="Pilot authentication is not configured.")
    try:
        if credentials is not None and credentials.scheme.lower() == "bearer":
            credential = credentials.credentials
        elif request.headers.get("authorization"):
            raise ValueError
        else:
            credential = request.cookies.get(cookie_name(), "")
            if not credential:
                raise ValueError
            browser_configured()
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                require_csrf(request, credential)
        user_id = verify_token(credential, settings.application_pilot_auth_secret)
    except ValueError:
        raise HTTPException(
            401, detail="Invalid or expired credential.", headers={"WWW-Authenticate": "Bearer"}
        ) from None
    user = await runtime.repositories.identities.get_by_user_id(user_id)
    if user is None or user.status != UserStatus.ACTIVE:
        raise HTTPException(
            401, detail="Invalid or expired credential.", headers={"WWW-Authenticate": "Bearer"}
        )
    return user


CurrentUserDep = Annotated[UserIdentity, Depends(get_current_user)]


async def authorize_invocation(request: OrchestrationRequest, user: CurrentUserDep) -> UserIdentity:
    if request.student_id != user.student_id:
        raise HTTPException(403, detail="Request identity does not match the current user.")
    return user

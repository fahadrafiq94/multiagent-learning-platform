"""Exchange existing pilot credentials for a bounded HttpOnly browser cookie."""

import json
import time

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from app.api.dependencies import CurrentUserDep, RuntimeDep
from app.api.routes.identity import CurrentUserView
from app.config.settings import settings
from app.identity.browser_session import cookie_name, csrf_token, require_csrf, require_origin
from app.identity.pilot_tokens import verify_token_and_expiry
from app.identity.schemas import UserStatus

router = APIRouter(prefix="/auth", tags=["browser authentication"])


class BrowserSessionView(BaseModel):
    user: CurrentUserView
    csrf_token: str


LOGIN_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["credential"],
                    "additionalProperties": False,
                    "properties": {"credential": {"type": "string", "maxLength": 1024}},
                }
            }
        },
    }
}


def session_view(user, credential):
    return BrowserSessionView(
        user=CurrentUserView(user_id=user.user_id, student_id=user.student_id),
        csrf_token=csrf_token(credential),
    )


@router.post("/session", response_model=BrowserSessionView, openapi_extra=LOGIN_SCHEMA)
async def login(request: Request, response: Response, runtime: RuntimeDep):
    require_origin(request)
    if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
        raise HTTPException(415, detail="JSON request required.")
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 4096:
            raise HTTPException(413, detail="Credential request is too large.")
    try:
        body = json.loads(raw)
        if not isinstance(body, dict) or set(body) != {"credential"}:
            raise ValueError
        credential = body["credential"]
        if not isinstance(credential, str) or len(credential) > 1024:
            raise ValueError
        secret = settings.application_pilot_auth_secret
        if secret is None:
            raise ValueError
        uid, expires = verify_token_and_expiry(credential, secret)
    except (ValueError, TypeError, UnicodeError):
        raise HTTPException(401, detail="Invalid or expired credential.") from None
    user = await runtime.repositories.identities.get_by_user_id(uid)
    if user is None or user.status != UserStatus.ACTIVE:
        raise HTTPException(401, detail="Invalid or expired credential.")
    remaining = expires - int(time.time())
    if remaining <= 0:
        raise HTTPException(401, detail="Invalid or expired credential.")
    response.set_cookie(
        cookie_name(),
        credential,
        max_age=remaining,
        httponly=True,
        secure=settings.student_browser_secure_cookie,
        samesite="strict",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return session_view(user, credential)


@router.get("/session", response_model=BrowserSessionView)
async def session(request: Request, response: Response, user: CurrentUserDep):
    if request.headers.get("authorization"):
        raise HTTPException(401, detail="Browser session required.")
    credential = request.cookies.get(cookie_name())
    if not credential:
        raise HTTPException(401, detail="Browser session required.")
    response.headers["Cache-Control"] = "no-store"
    return session_view(user, credential)


@router.post("/logout", status_code=204)
async def logout(request: Request):
    require_origin(request)
    credential = request.cookies.get(cookie_name())
    if credential:
        require_csrf(request, credential)
    response = Response(status_code=204, headers={"Cache-Control": "no-store"})
    response.delete_cookie(
        cookie_name(),
        path="/",
        secure=settings.student_browser_secure_cookie,
        httponly=True,
        samesite="strict",
    )
    return response

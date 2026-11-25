"""Same-origin browser cookie/CSRF boundary; existing bearer auth stays separate."""

import hashlib
import hmac

from fastapi import HTTPException, Request

from app.config.settings import settings


def browser_configured() -> str:
    origin = settings.student_browser_origin
    if (
        not origin
        or settings.application_pilot_auth_secret is None
        or (
            settings.app_env == "production"
            and (not origin.startswith("https://") or not settings.student_browser_secure_cookie)
        )
    ):
        raise HTTPException(503, detail="Student browser access is not configured.")
    return origin


def cookie_name() -> str:
    return "__Host-fredi_pilot" if settings.student_browser_secure_cookie else "fredi_pilot_dev"


def csrf_token(credential: str) -> str:
    secret = settings.application_pilot_auth_secret
    if secret is None:
        raise HTTPException(503, detail="Student browser access is not configured.")
    return hmac.new(
        secret.get_secret_value().encode(),
        b"fredi-browser-csrf-v1\x00" + credential.encode(),
        hashlib.sha256,
    ).hexdigest()


def require_origin(request: Request) -> None:
    if request.headers.get("origin") != browser_configured():
        raise HTTPException(403, detail="Same-origin browser request required.")


def require_csrf(request: Request, credential: str) -> None:
    require_origin(request)
    supplied = request.headers.get("x-fredi-csrf", "")
    if len(supplied) != 64 or not hmac.compare_digest(
        supplied.encode(), csrf_token(credential).encode()
    ):
        raise HTTPException(403, detail="Invalid browser request token.")

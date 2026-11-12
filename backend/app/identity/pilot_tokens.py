"""Short-lived signed pilot credentials, bound to internal user UUIDs."""

import base64
import hashlib
import hmac
import json
import time
from uuid import UUID

from pydantic import SecretStr


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def issue_token(user_id: UUID, secret: SecretStr, *, lifetime_seconds: int = 3600) -> str:
    if not 1 <= lifetime_seconds <= 86400:
        raise ValueError("Pilot tokens must expire within 24 hours.")
    payload = _encode(
        json.dumps(
            {
                "sub": str(user_id),
                "exp": int(time.time()) + lifetime_seconds,
                "aud": "fredi-local-pilot",
                "v": 1,
            },
            separators=(",", ":"),
        ).encode()
    )
    signature = hmac.new(
        secret.get_secret_value().encode(), payload.encode(), hashlib.sha256
    ).digest()
    return payload + "." + _encode(signature)


def verify_token(token: str, secret: SecretStr) -> UUID:
    try:
        if len(token) > 1024:
            raise ValueError
        payload, signature = token.split(".")
        expected = _encode(
            hmac.new(secret.get_secret_value().encode(), payload.encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise ValueError
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if (
            claims["aud"] != "fredi-local-pilot"
            or claims["v"] != 1
            or type(claims["exp"]) is not int
            or claims["exp"] <= time.time()
        ):
            raise ValueError
        return UUID(claims["sub"])
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise ValueError("Invalid or expired pilot credential.") from None

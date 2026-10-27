from __future__ import annotations

from uuid import uuid4

from app.identity import (
    UserIdentity,
)
from app.orchestration.factory import (
    get_orchestration_runtime,
)
from app.repositories.exceptions import (
    DuplicateRecordError,
)


async def ensure_students(
    student_ids: list[str],
) -> None:
    """
    Seed local identities so smoke requests pass the service
    boundary (unknown student_id is a 404-class client error).

    Identities created by an earlier run are kept; the in-memory
    runtime is process-local and duplicates are simply ignored.
    """

    identities = get_orchestration_runtime().repositories.identities

    for student_id in dict.fromkeys(student_ids):
        try:
            await identities.create(
                UserIdentity(student_id=student_id),
            )

        except DuplicateRecordError:
            pass


def new_session_id(
    prefix: str,
) -> str:
    """A fresh valid session UUID for one smoke request."""

    return f"{prefix}-{uuid4().hex[:12]}"

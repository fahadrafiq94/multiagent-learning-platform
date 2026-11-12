"""Local operator identity and protected configuration; never a student-facing API."""

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

from app.context import UserContext
from app.identity import ExternalIdentity, IdentityProvider, UserIdentity
from app.persistence.database import ApplicationDatabase
from app.persistence.records import decode, insert
from app.repositories.exceptions import RepositoryIntegrityError
from scripts.knowledge_infra import InfrastructureError, secure_credentials_file

OPERATOR_INSTANCE = "knowledge-management"


class OperatorError(ValueError):
    """Authored operator guidance safe to display without raw provider/DB errors."""


def local_owner() -> str:
    """Use an OS-provided identity, not USERNAME/USER environment variables."""
    if os.name != "nt":
        get_uid = getattr(os, "getuid", None)
        if get_uid is None:
            raise InfrastructureError("Could not identify the local operator account.")
        return f"uid:{get_uid()}@{socket.gethostname()}"
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise InfrastructureError("Could not identify the local operator account.") from None
    value = result.stdout.strip()
    if result.returncode or not value.startswith("S-1-") or len(value) > 200:
        raise InfrastructureError("Could not identify the local operator account.")
    return value


async def provision_operator(database: ApplicationDatabase, owner: str) -> UUID:
    """Reuse only our OS-bound operator link, never an arbitrary student's name."""
    async with database.connection() as connection:
        lock_key = int.from_bytes(
            hashlib.sha256((OPERATOR_INSTANCE + owner).encode()).digest()[:8], signed=True
        )
        await connection.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
        row = await (
            await connection.execute(
                "SELECT i.* FROM application.identities i "
                "JOIN application.external_identities e USING(user_id) "
                "WHERE e.provider='local' AND e.provider_instance=%s "
                "AND e.external_user_id=%s FOR UPDATE OF i",
                (OPERATOR_INSTANCE, owner),
            )
        ).fetchone()
        if row:
            user = decode(UserIdentity, row)
            if user.status != "active":
                raise RepositoryIntegrityError(
                    "The local operator is disabled; it was not reactivated."
                )
            return user.user_id
        user = UserIdentity(
            student_id="knowledge-operator-" + uuid4().hex,
            external_identities=[
                ExternalIdentity(
                    provider=IdentityProvider.LOCAL,
                    provider_instance=OPERATOR_INSTANCE,
                    external_user_id=owner,
                ),
            ],
        )
        await insert(connection, user)
        await insert(connection, UserContext(user_id=user.user_id))
        await connection.execute(
            "INSERT INTO application.external_identities VALUES ('local',%s,%s,%s)",
            (OPERATOR_INSTANCE, owner, user.user_id),
        )
        return user.user_id


def read_local_config(path: Path) -> tuple[dict, bytes | None]:
    if path.is_symlink():
        raise InfrastructureError("Operator configuration cannot be a symlink.")
    if not path.exists():
        return {}, None
    if not path.is_file():
        raise InfrastructureError("Operator configuration must be a regular JSON file.")
    secure_credentials_file(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise OperatorError("Operator configuration is too large.")
    original = path.read_bytes()
    try:
        value = json.loads(original)
    except (ValueError, UnicodeError):
        raise OperatorError(
            "Existing configuration is not valid JSON; it was not overwritten."
        ) from None
    if not isinstance(value, dict):
        raise OperatorError("Operator configuration must be a JSON object.")
    return value, original


def manager_ids(values: list, operator_id: UUID) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, str) and value.startswith("REPLACE_"):
            continue
        result.append(str(UUID(str(value))))
    return list(dict.fromkeys([*result, str(operator_id)]))


def write_local_config(path: Path, value: dict, original: bytes | None):
    """Protect before writing, detect competing edits and atomically publish JSON."""
    if path.is_symlink():
        raise InfrastructureError("Operator configuration cannot be a symlink.")
    current = path.read_bytes() if path.exists() else None
    if current != original:
        raise InfrastructureError(
            "Configuration changed during setup; rerun to preserve those edits."
        )
    if current is not None and json.loads(current) == value:
        secure_credentials_file(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".intake-config-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            secure_credentials_file(temporary_path)
            stream.write(json.dumps(value, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if (path.read_bytes() if path.exists() else None) != original:
            raise InfrastructureError(
                "Configuration changed during setup; rerun to preserve those edits."
            )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


async def owner_identity() -> str:
    return await asyncio.to_thread(local_owner)

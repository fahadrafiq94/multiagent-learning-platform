"""Protected local development credentials; production can supply connection URLs."""

from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr

from app.repositories.exceptions import RepositoryIntegrityError
from scripts.knowledge_infra import ROOT, secure_credentials_file

APPLICATION_KEYS = (
    "APPLICATION_MIGRATOR_PASSWORD",
    "APPLICATION_RUNTIME_PASSWORD",
    "APPLICATION_PILOT_AUTH_SECRET",
)


def ensure_credentials(root: Path = ROOT) -> bool:
    """Add missing application secrets atomically; never replace existing passwords."""
    path = root / "backend/.env.knowledge"
    if not path.is_file():
        raise RepositoryIntegrityError("Initialize local knowledge infrastructure first.")
    lock = root / "backend/.application-bootstrap.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    temporary = None
    try:
        secure_credentials_file(path)
        values = dotenv_values(path)
        missing = [name for name in APPLICATION_KEYS if not values.get(name)]
        if not missing:
            return False
        original = path.read_bytes()
        suffix = ("" if original.endswith(b"\n") else "\n") + "".join(
            f"{name}={secrets.token_hex(32)}\n" for name in missing
        )
        descriptor, name = tempfile.mkstemp(prefix=".application-credentials-", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            secure_credentials_file(temporary)
            stream.write(original + suffix.encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink()


def local_application_dsn(*, role: str = "runtime", database: str = "fredi_knowledge") -> SecretStr:
    if role not in {"runtime", "migrator"}:
        raise ValueError("Unknown application role.")
    values = dotenv_values(ROOT / "backend/.env.knowledge")
    password = values.get(f"APPLICATION_{role.upper()}_PASSWORD")
    if not password:
        raise RepositoryIntegrityError("Run application bootstrap to provision credentials.")
    return SecretStr(
        make_conninfo(
            host="127.0.0.1",
            port=values.get("KNOWLEDGE_POSTGRES_PORT") or "15432",
            user="fredi_app_" + role,
            password=password,
            dbname=database,
        )
    )


def configured_dsn(*, role: str = "runtime") -> SecretStr:
    key = "APPLICATION_DATABASE_URL" if role == "runtime" else "APPLICATION_MIGRATIONS_DATABASE_URL"
    return SecretStr(os.environ[key]) if os.environ.get(key) else local_application_dsn(role=role)

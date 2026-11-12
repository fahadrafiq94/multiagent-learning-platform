from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.persistence.database import ApplicationDatabase, ApplicationDatabaseConfig
from app.repositories.exceptions import RepositoryUnavailableError
from scripts.application_credentials import ensure_credentials
from scripts.knowledge_infra import InfrastructureError


def test_constructing_database_does_not_connect_or_expose_secret():
    config = ApplicationDatabaseConfig(dsn=SecretStr("private-marker"))
    database = ApplicationDatabase(config)
    assert database._pool is None and "private-marker" not in repr(config)


@pytest.mark.asyncio
async def test_invalid_dsn_is_sanitized_and_close_is_terminal():
    database = ApplicationDatabase(ApplicationDatabaseConfig(dsn=SecretStr("private-marker")))
    with pytest.raises(RepositoryUnavailableError) as error:
        await database.start()
    assert "private-marker" not in str(error.value)
    await database.close()
    await database.close()
    with pytest.raises(RepositoryUnavailableError):
        await database.start()


def test_credential_extension_keeps_original_file_on_permission_failure(tmp_path, monkeypatch):
    (tmp_path / "backend").mkdir()
    path = tmp_path / "backend/.env.knowledge"
    path.write_bytes(b"original-secret\n")

    def denied(path):
        raise InfrastructureError("Cannot protect credentials")

    monkeypatch.setattr("scripts.application_credentials.secure_credentials_file", denied)
    with pytest.raises(InfrastructureError):
        ensure_credentials(tmp_path)
    assert path.read_bytes() == b"original-secret\n"
    assert not (tmp_path / "backend/.application-bootstrap.lock").exists()


def test_credentials_are_extended_once_without_rotating_existing_values(tmp_path, monkeypatch):
    from dotenv import dotenv_values

    from scripts.application_credentials import APPLICATION_KEYS

    (tmp_path / "backend").mkdir()
    path = tmp_path / "backend/.env.knowledge"
    original = b"KNOWLEDGE_WRITER_PASSWORD=original-secret\n"
    path.write_bytes(original)
    protected = []
    monkeypatch.setattr(
        "scripts.application_credentials.secure_credentials_file", lambda p: protected.append(p)
    )
    assert ensure_credentials(tmp_path)
    content = path.read_bytes()
    assert content.startswith(original) and len(protected) == 2
    values = dotenv_values(path)
    assert all(len(values[key]) == 64 for key in APPLICATION_KEYS)
    assert not ensure_credentials(tmp_path)
    assert path.read_bytes() == content


def test_temporary_credential_protection_failure_preserves_original(tmp_path, monkeypatch):
    (tmp_path / "backend").mkdir()
    path = tmp_path / "backend/.env.knowledge"
    path.write_bytes(b"original-secret\n")

    def protect(candidate):
        if candidate != path:
            assert candidate.read_bytes() == b""
            raise InfrastructureError("Cannot protect temporary credentials")

    monkeypatch.setattr("scripts.application_credentials.secure_credentials_file", protect)
    with pytest.raises(InfrastructureError):
        ensure_credentials(tmp_path)
    assert path.read_bytes() == b"original-secret\n"
    assert not list(path.parent.glob(".application-credentials-*"))

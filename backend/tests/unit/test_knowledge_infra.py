from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.knowledge_infra import (
    PASSWORD_KEYS,
    InfrastructureError,
    KnowledgeInfrastructure,
    initialize_environment,
    secure_windows_credentials,
)


def test_credentials_are_random_unique_and_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / "backend").mkdir()
    assert initialize_environment(tmp_path)
    path = tmp_path / "backend/.env.knowledge"
    original = path.read_bytes()
    values = dict(
        line.split("=", 1)
        for line in original.decode().splitlines()
        if line and not line.startswith("#")
    )
    passwords = [values[key] for key in PASSWORD_KEYS]
    assert len(set(passwords)) == len(PASSWORD_KEYS)
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in passwords)
    assert not initialize_environment(tmp_path)
    assert path.read_bytes() == original
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "nt", reason="Real Windows owner/DACL contract")
def test_windows_credentials_have_private_verified_acl(tmp_path: Path) -> None:
    (tmp_path / "backend").mkdir()
    assert initialize_environment(tmp_path)
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            r"""
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$sections = [System.Security.AccessControl.AccessControlSections]::Access -bor
    [System.Security.AccessControl.AccessControlSections]::Owner
$acl = [System.IO.File]::GetAccessControl($env:FREDI_TEST_CREDENTIAL_PATH, $sections)
$entries = @($acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]))
@{ protected = $acl.AreAccessRulesProtected; current = $sid;
   owner = $acl.GetOwner([System.Security.Principal.SecurityIdentifier]).Value;
   entries = @($entries | ForEach-Object { @{sid=$_.IdentityReference.Value;
       inherited=$_.IsInherited; allow=($_.AccessControlType -eq 'Allow')} })
} | ConvertTo-Json -Depth 4 -Compress
""",
        ],
        env={**os.environ, "FREDI_TEST_CREDENTIAL_PATH": str(tmp_path / "backend/.env.knowledge")},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    acl = json.loads(result.stdout)
    assert acl["protected"] and acl["owner"] == acl["current"]
    assert {e["sid"] for e in acl["entries"]} == {acl["current"], "S-1-5-18", "S-1-5-32-544"}
    assert all(e["allow"] and not e["inherited"] for e in acl["entries"])


def test_permission_failure_never_writes_new_secrets(tmp_path, monkeypatch):
    (tmp_path / "backend").mkdir()
    observed = []

    def fail(path):
        observed.append(path.read_bytes())
        raise InfrastructureError("Simulated ACL failure")

    monkeypatch.setattr("scripts.knowledge_infra.secure_credentials_file", fail)
    with pytest.raises(InfrastructureError):
        initialize_environment(tmp_path)
    assert observed == [b""]
    assert not (tmp_path / "backend/.env.knowledge").exists()


def test_existing_secrets_survive_permission_failure(tmp_path, monkeypatch):
    (tmp_path / "backend").mkdir()
    path = tmp_path / "backend/.env.knowledge"
    path.write_bytes(b"existing-secret-must-remain")

    def fail(path):
        raise InfrastructureError("Simulated ACL failure")

    monkeypatch.setattr("scripts.knowledge_infra.secure_credentials_file", fail)
    with pytest.raises(InfrastructureError):
        initialize_environment(tmp_path)
    assert path.read_bytes() == b"existing-secret-must-remain"


def test_windows_acl_errors_are_sanitized(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        return subprocess.CompletedProcess([], 1, b"private-marker", b"private-marker")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(InfrastructureError) as error:
        secure_windows_credentials(tmp_path / "credentials")
    assert "private-marker" not in str(error.value)


def test_missing_configuration_fails_before_any_docker_access(tmp_path: Path) -> None:
    with pytest.raises(InfrastructureError, match="init"):
        KnowledgeInfrastructure(tmp_path)


@pytest.fixture
def infrastructure(tmp_path: Path) -> KnowledgeInfrastructure:
    (tmp_path / "backend").mkdir()
    initialize_environment(tmp_path)
    return KnowledgeInfrastructure(tmp_path)


@pytest.mark.parametrize(
    "database", ["postgres", "production", "fredi_smoke_123", "x; DROP ROLE x"]
)
def test_smoke_sql_rejects_unapproved_database_names(
    infrastructure: KnowledgeInfrastructure,
    database: str,
) -> None:
    with pytest.raises(ValueError, match="generated smoke database"):
        infrastructure.sql("SELECT 1;", database=database)


def test_smoke_sql_rejects_unknown_role(infrastructure: KnowledgeInfrastructure) -> None:
    with pytest.raises(ValueError, match="role"):
        infrastructure.sql("SELECT 1;", role="postgres")


def test_failed_docker_output_is_not_exposed(
    infrastructure: KnowledgeInfrastructure,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = "this-must-never-appear-in-error-output"

    def fail(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["docker"], 1, password, password)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(InfrastructureError) as exc:
        infrastructure.run("config", "--quiet")
    assert password not in str(exc.value)


def test_timeout_is_sanitized(
    infrastructure: KnowledgeInfrastructure,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["docker", "private-value"], 1, output="private-value")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(InfrastructureError) as exc:
        infrastructure.run("start")
    assert "private-value" not in str(exc.value)


def test_sql_password_expands_only_inside_container(
    infrastructure: KnowledgeInfrastructure,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def record(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs["input"] == "SELECT 1;"
        return subprocess.CompletedProcess(command, 0, "1\n", "")

    monkeypatch.setattr(subprocess, "run", record)
    assert infrastructure.sql("SELECT 1;", role="reader").stdout == "1\n"
    env = (infrastructure.root / "backend/.env.knowledge").read_text()
    for line in env.splitlines():
        if "PASSWORD=" in line:
            assert line.split("=", 1)[1] not in " ".join(commands[0])

"""Manage optional knowledge infrastructure without changing backend containers.

Uses the Docker CLI and Python's standard library; no backend dependency install
is needed for Checkpoint 1. Never prints resolved Compose configuration or secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PASSWORD_KEYS = (
    "KNOWLEDGE_POSTGRES_PASSWORD",
    "KNOWLEDGE_WRITER_PASSWORD",
    "KNOWLEDGE_READER_PASSWORD",
    "KNOWLEDGE_REDIS_PASSWORD",
)


class InfrastructureError(RuntimeError):
    """Sanitized infrastructure failure; command output may contain credentials."""


WINDOWS_CREDENTIAL_ACL = r"""
$ErrorActionPreference = 'Stop'
$path = $env:FREDI_KNOWLEDGE_CREDENTIAL_PATH
$acl = [System.Security.AccessControl.FileSecurity]::new()
$owner = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$acl.SetOwner($owner)
$allowed = @($owner.Value, 'S-1-5-18', 'S-1-5-32-544') | Sort-Object -Unique
$acl.SetAccessRuleProtection($true, $false)
foreach ($sid in $allowed) {
    $identity = [System.Security.Principal.SecurityIdentifier]::new($sid)
    $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $identity, 'FullControl', 'Allow')
    $acl.AddAccessRule($rule)
}
[System.IO.File]::SetAccessControl($path, $acl)
$sections = [System.Security.AccessControl.AccessControlSections]::Access -bor
    [System.Security.AccessControl.AccessControlSections]::Owner
$check = [System.IO.File]::GetAccessControl($path, $sections)
$actual = @($check.GetAccessRules($true, $true,
    [System.Security.Principal.SecurityIdentifier]))
if (-not $check.AreAccessRulesProtected -or $actual.Count -ne $allowed.Count -or
    $check.GetOwner([System.Security.Principal.SecurityIdentifier]).Value -ne $owner.Value) {
    throw 'Credential ACL verification failed'
}
foreach ($rule in $actual) {
    if ($rule.IsInherited -or $rule.AccessControlType -ne 'Allow' -or
        $rule.IdentityReference.Value -notin $allowed -or
        $rule.FileSystemRights -ne [System.Security.AccessControl.FileSystemRights]::FullControl) {
        throw 'Credential ACL verification failed'
    }
}
"""


def secure_windows_credentials(path: Path) -> None:
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_CREDENTIAL_ACL],
            env={**os.environ, "FREDI_KNOWLEDGE_CREDENTIAL_PATH": str(path.resolve())},
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise InfrastructureError("Windows credential permissions could not be applied.") from None
    if result.returncode:
        raise InfrastructureError(
            "Windows credential ACL protection failed; run init as the application user "
            "with ownership/ACL rights on the file."
        )


def secure_credentials_file(path: Path) -> None:
    if path.is_symlink():
        raise InfrastructureError("Credentials must be a regular local file, not a symlink.")
    if os.name == "nt":
        secure_windows_credentials(path)
    else:
        try:
            path.chmod(0o600)
        except OSError:
            raise InfrastructureError("Credential file permissions could not be applied.") from None


def initialize_environment(root: Path = ROOT) -> bool:
    """Preserve secrets; restrict new/existing files before writing any new secrets."""
    path = root / "backend/.env.knowledge"
    text = "# Local knowledge infrastructure credentials; do not commit.\n"
    text += "".join(f"{key}={secrets.token_hex(32)}\n" for key in PASSWORD_KEYS)
    text += "KNOWLEDGE_POSTGRES_PORT=15432\nKNOWLEDGE_REDIS_PORT=16379\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        secure_credentials_file(path)
        return False
    except OSError:
        raise InfrastructureError("Credential file could not be created.") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            # Protect the still-empty file before writing any new credentials.
            secure_credentials_file(path)
            stream.write(text)
    except BaseException:
        # Only our exclusively created file is removed, never an existing file.
        path.unlink(missing_ok=True)
        raise
    return True


class KnowledgeInfrastructure:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        env_file = root / "backend/.env.knowledge"
        if not env_file.is_file():
            raise InfrastructureError(
                "Run knowledge_infra.py init before using knowledge services."
            )
        self.command = ["docker", "compose", "--env-file", str(env_file)]
        for name in ("dev", "knowledge", "knowledge.dev"):
            self.command.extend(["-f", str(root / f"docker/compose/docker-compose.{name}.yml")])
        self.command.extend(["--profile", "knowledge"])

    def run(
        self, *args: str, stdin: str | None = None, timeout: int = 120, allow_failure: bool = False
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                [*self.command, *args],
                input=stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=self.root,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureError(f"Docker operation failed ({type(exc).__name__}).") from None
        if result.returncode and not allow_failure:
            raise InfrastructureError(
                f"Docker operation '{args[0]}' failed (exit {result.returncode}); "
                "inspect service health locally. Resolved configuration and logs are not printed."
            )
        return result

    def sql(
        self,
        statement: str,
        *,
        role: str = "admin",
        database: str = "fredi_knowledge",
        allow_failure: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if role not in {"admin", "writer", "reader"}:
            raise ValueError("Unknown database role.")
        if not re.fullmatch(r"fredi_knowledge|fredi_smoke_[0-9a-f]{16}", database):
            raise ValueError(
                "Only the knowledge database or a generated smoke database is allowed."
            )
        password = "POSTGRES_PASSWORD" if role == "admin" else f"KNOWLEDGE_{role.upper()}_PASSWORD"
        # Shell text is fixed application code; secrets expand inside the container.
        shell = (
            f'export PGPASSWORD="${password}"; '
            f'exec psql -h 127.0.0.1 -U fredi_knowledge_{role} -d "$1" '
            "--no-psqlrc --set ON_ERROR_STOP=1 --tuples-only --no-align"
        )
        return self.run(
            "exec",
            "-T",
            "postgres",
            "sh",
            "-ec",
            shell,
            "sh",
            database,
            stdin=statement,
            allow_failure=allow_failure,
        )

    def redis(self, *args: str, authenticate: bool = True) -> str:
        shell = (
            'export REDISCLI_AUTH="$REDIS_PASSWORD"; exec redis-cli --raw "$@"'
            if authenticate
            else 'unset REDISCLI_AUTH; exec redis-cli --raw "$@"'
        )
        return self.run("exec", "-T", "redis", "sh", "-ec", shell, "sh", *args).stdout.strip()

    def check_config(self) -> None:
        self.run("config", "--quiet")

    def start(self) -> None:
        self.run(
            "up",
            "-d",
            "--no-deps",
            "--wait",
            "--wait-timeout",
            "120",
            "postgres",
            "redis",
            timeout=600,
        )

    def smoke(self, *, restart: bool = False) -> dict[str, object]:
        """Check isolated data; optionally restart only the new PostgreSQL service."""
        identifier = secrets.token_hex(8)
        database = f"fredi_smoke_{identifier}"
        key = f"fredi:infra-smoke:{identifier}"
        marker = f".smoke-{identifier}"
        results: dict[str, object] = {}
        created = False
        try:
            results["pgvector_version"] = self.sql(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
            ).stdout.strip()
            if not results["pgvector_version"]:
                raise InfrastructureError("The vector extension is missing.")
            self.sql(f"CREATE DATABASE {database} OWNER fredi_knowledge_writer;")
            created = True
            self.sql("CREATE EXTENSION vector;", database=database)
            self.sql(
                "CREATE TABLE probe (id integer PRIMARY KEY, embedding vector(3));"
                "INSERT INTO probe VALUES (1, '[1,0,0]'), (2, '[0,1,0]');",
                role="writer",
                database=database,
            )
            ranking = self.sql(
                "SELECT id FROM probe ORDER BY embedding <=> '[1,0,0]'::vector, id;",
                role="writer",
                database=database,
            ).stdout.strip()
            if ranking != "1\n2":
                raise InfrastructureError("Known-vector cosine ranking failed.")
            results["cosine_ranking"] = "passed"
            # Exercise the actual main-database writer defaults using a scratch table.
            table = f"knowledge.smoke_{identifier}"
            self.sql(
                f"CREATE TABLE {table} (value integer); INSERT INTO {table} VALUES (7);",
                role="writer",
            )
            try:
                if self.sql(f"SELECT value FROM {table};", role="reader").stdout.strip() != "7":
                    raise InfrastructureError("Reader cannot read writer-owned knowledge tables.")
                denied = self.sql(
                    f"INSERT INTO {table} VALUES (8);", role="reader", allow_failure=True
                )
                if denied.returncode == 0 or "permission denied" not in denied.stderr.lower():
                    raise InfrastructureError(
                        "Read-only role did not reject the write as expected."
                    )
                results["database_roles"] = "passed"
                if restart:
                    self.run("restart", "postgres")
                    for _ in range(30):
                        try:
                            persisted = self.sql(f"SELECT value FROM {table};", role="reader")
                        except InfrastructureError:
                            time.sleep(1)
                        else:
                            if persisted.stdout.strip() != "7":
                                raise InfrastructureError("Database persistence check failed.")
                            results["postgres_restart_persistence"] = "passed"
                            break
                    else:
                        raise InfrastructureError("PostgreSQL did not become ready after restart.")
            finally:
                self.sql(f"DROP TABLE {table};", role="writer")
            if self.redis("PING") != "PONG":
                raise InfrastructureError("Authenticated Redis PING failed.")
            if "NOAUTH" not in self.redis("PING", authenticate=False):
                raise InfrastructureError("Redis unexpectedly permits unauthenticated access.")
            if self.redis("SET", key, identifier, "EX", "2") != "OK":
                raise InfrastructureError("Redis expiring SET failed.")
            if self.redis("GET", key) != identifier:
                raise InfrastructureError("Redis GET failed.")
            time.sleep(2.1)
            if self.redis("EXISTS", key) != "0":
                raise InfrastructureError("Redis TTL expiration failed.")
            results["redis_auth_and_expiry"] = "passed"
            self.run(
                "exec",
                "-T",
                "backend",
                "python",
                "-c",
                "import socket; "
                "socket.create_connection(('postgres', 5432), timeout=3).close(); "
                "socket.create_connection(('redis', 6379), timeout=3).close()",
            )
            results["backend_network_connectivity"] = "passed"
            # Two independent one-off containers prove source-volume persistence.
            self.run(
                "run",
                "--rm",
                "--no-deps",
                "-T",
                "knowledge-volume-check",
                f"umask 077; printf '%s' '{identifier}' > '/sources/{marker}'",
            )
            try:
                observed = self.run(
                    "run",
                    "--rm",
                    "--no-deps",
                    "-T",
                    "knowledge-volume-check",
                    f"cat '/sources/{marker}'",
                ).stdout.strip()
                if observed != identifier:
                    raise InfrastructureError("Source-volume persistence check failed.")
                results["source_volume_persistence"] = "passed"
            finally:
                self.run(
                    "run",
                    "--rm",
                    "--no-deps",
                    "-T",
                    "knowledge-volume-check",
                    f"rm -f '/sources/{marker}'",
                )
            return results
        finally:
            if created:
                # Identifier is locally generated, tightly validated, and never a user database.
                self.sql(f"DROP DATABASE {database};")
            self.redis("DEL", key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "config", "start", "status", "smoke", "import"))
    parser.add_argument(
        "--restart-postgres",
        action="store_true",
        help="During smoke, restart only the knowledge PostgreSQL container.",
    )
    args = parser.parse_args()
    if args.restart_postgres and args.action != "smoke":
        parser.error("--restart-postgres is valid only with smoke")
    try:
        if args.action == "init":
            created = initialize_environment()
            print(
                "Created and protected backend/.env.knowledge."
                if created
                else "Existing credentials preserved; file permissions protected."
            )
            return 0
        infra = KnowledgeInfrastructure()
        if args.action == "config":
            infra.check_config()
            print("Knowledge Compose configuration is valid.")
        elif args.action == "start":
            infra.check_config()
            infra.start()
            print("PostgreSQL/pgvector and Redis are healthy; backend/Ollama were not recreated.")
        elif args.action == "status":
            print(infra.run("ps", "postgres", "redis").stdout, end="")
        elif args.action == "import":
            result = infra.run(
                "run",
                "--rm",
                "--no-deps",
                "--build",
                "knowledge-import",
                "--registration",
                "/config/ap40-sources.json",
                "--source-root",
                "/input",
                "--source-store",
                "/sources",
                timeout=600,
                allow_failure=True,
            )
            # Build/Compose stderr is not printed: resolved configuration can
            # contain credentials. The importer emits one bounded JSON outcome.
            lines = result.stdout.strip().splitlines()
            if lines:
                try:
                    report = json.loads(lines[-1])
                    if set(report) == {
                        "revision_id",
                        "reused",
                        "documents",
                        "sections",
                        "diagnostics",
                        "failed_documents",
                    }:
                        print(json.dumps(report))
                except ValueError:
                    pass
            if result.returncode:
                raise InfrastructureError(
                    "Registered import failed; inspect inputs and extraction diagnostic codes."
                )
        else:
            print(json.dumps(infra.smoke(restart=args.restart_postgres), indent=2))
        return 0
    except InfrastructureError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

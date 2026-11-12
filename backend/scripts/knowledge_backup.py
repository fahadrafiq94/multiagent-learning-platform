"""Coordinated local backup and scratch-only restore verification; no in-place restore."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from app.knowledge.errors import KnowledgeError
from app.knowledge.source_store import FileSourceStore
from scripts.knowledge_infra import ROOT, InfrastructureError, KnowledgeInfrastructure
from scripts.migrate_knowledge import local_dsn

DIRECTORY_ACL = r"""
$ErrorActionPreference = 'Stop'
$path = $env:FREDI_PRIVATE_DIRECTORY
$security = [System.Security.AccessControl.DirectorySecurity]::new()
$owner = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$security.SetOwner($owner)
$security.SetAccessRuleProtection($true, $false)
$allowed = @($owner.Value, 'S-1-5-18', 'S-1-5-32-544') | Sort-Object -Unique
foreach ($sid in $allowed) {
  $identity = [System.Security.Principal.SecurityIdentifier]::new($sid)
  $rule = [System.Security.AccessControl.FileSystemAccessRule]::new($identity,
    'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
  $security.AddAccessRule($rule)
}
[System.IO.Directory]::SetAccessControl($path, $security)
$check = [System.IO.Directory]::GetAccessControl($path)
if (-not $check.AreAccessRulesProtected -or
    $check.GetOwner([System.Security.Principal.SecurityIdentifier]).Value -ne $owner.Value) {
  throw 'Private directory verification failed'
}
$entries = @($check.GetAccessRules($true,$true,[System.Security.Principal.SecurityIdentifier]))
if ($entries.Count -ne $allowed.Count) { throw 'Unexpected private directory permissions' }
foreach ($entry in $entries) {
  if ($entry.IsInherited -or $entry.IdentityReference.Value -notin $allowed -or
      $entry.AccessControlType -ne 'Allow' -or
      $entry.FileSystemRights -ne [System.Security.AccessControl.FileSystemRights]::FullControl) {
    throw 'Unexpected private directory grant'
  }
}
"""


def private_directory(path: Path):
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("Choose a new output directory; existing content is never overwritten.")
    path.mkdir(parents=True, mode=0o700)
    if os.name == "nt":
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", DIRECTORY_ACL],
            env={**os.environ, "FREDI_PRIVATE_DIRECTORY": str(path)},
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            raise InfrastructureError("Private output permissions could not be applied.")
    else:
        path.chmod(0o700)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def validate_database(name):
    if not re.fullmatch(r"fredi_knowledge|fredi_(?:smoke|app_test)_[0-9a-f]{16}", name):
        raise ValueError(
            "Backup supports the local database or explicitly generated scratch databases."
        )


async def table_fingerprints(connection):
    rows = await (
        await connection.execute(
            "SELECT schemaname,tablename FROM pg_tables "
            "WHERE schemaname IN ('knowledge','application') "
            "ORDER BY schemaname,tablename"
        )
    ).fetchall()
    result = {}
    for row in rows:
        schema, table = row["schemaname"], row["tablename"]
        values = await (
            await connection.execute(
                sql.SQL(
                    "SELECT count(*) AS count, "
                    "md5(COALESCE(string_agg(body,'' ORDER BY body),'')) AS checksum "
                    "FROM (SELECT to_jsonb(t)::text AS body FROM {}.{} t) records"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
        ).fetchone()
        result[f"{schema}.{table}"] = values
    return result


def docker_command(*args, stdin=None, stdout=None):
    command = [*KnowledgeInfrastructure().command, *args]
    result = subprocess.run(
        command,
        input=stdin,
        stdout=stdout or subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    if result.returncode:
        raise InfrastructureError(
            "Backup/restore container operation failed; sensitive output was suppressed."
        )
    return result.stdout


def volume_source(digest):
    # Static Python code and a validated hash argument; never interpolate PDF paths.
    return docker_command(
        "--profile",
        "knowledge-tools",
        "run",
        "--rm",
        "--no-deps",
        "-T",
        "--entrypoint",
        "python",
        "knowledge-import",
        "-c",
        "import sys; from pathlib import Path; "
        "from app.knowledge.source_store import FileSourceStore; "
        "sys.stdout.buffer.write(FileSourceStore(Path('/sources')).get(sys.argv[1]))",
        digest,
    )


async def backup(
    output: Path, *, database="fredi_knowledge", source_roots=(), volume_reader=volume_source
):
    validate_database(database)
    await asyncio.to_thread(private_directory, output)
    (output / "sources").mkdir(mode=0o700)
    destination = FileSourceStore(output / "sources")
    async with await psycopg.AsyncConnection.connect(
        local_dsn(role="admin", database=database).get_secret_value(), row_factory=dict_row
    ) as connection:
        await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        snapshot_row = await (
            await connection.execute("SELECT pg_export_snapshot() AS snapshot")
        ).fetchone()
        if snapshot_row is None:
            raise ValueError("Backup snapshot is unavailable.")
        snapshot = snapshot_row["snapshot"]
        tables = await table_fingerprints(connection)
        hashes = {
            row["hash"]
            for row in await (
                await connection.execute(
                    "SELECT source_hash AS hash FROM knowledge.knowledge_documents "
                    "UNION SELECT value FROM knowledge.knowledge_revisions r, "
                    "LATERAL jsonb_each_text(r.payload->'manifest') "
                    "UNION SELECT payload->>'source_hash' FROM knowledge.intake_jobs"
                )
            ).fetchall()
        }
        manifest = {
            "version": 1,
            "complete": False,
            "database": database,
            "tables": tables,
            "sources": [],
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for digest in sorted(hashes):
            if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError("Stored source identity is invalid.")
            content = None
            for root in source_roots:
                path = FileSourceStore(root)._path(digest)
                if path.exists():
                    content = await asyncio.to_thread(FileSourceStore(root).get, digest)
                    break
            if content is None:
                content = await asyncio.to_thread(volume_reader, digest)
            if not isinstance(content, bytes):
                raise ValueError("Referenced source bytes are unavailable.")
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Backup source failed hash verification.")
            ref = await asyncio.to_thread(destination.put, content)
            manifest["sources"].append(ref.model_dump())
        dump = output / "database.dump"
        # Exported snapshot keeps dump + manifest + source references consistent
        # while content-addressed blobs remain immutable in their durable stores.
        with dump.open("xb") as stream:
            await asyncio.to_thread(
                docker_command,
                "exec",
                "-T",
                "postgres",
                "sh",
                "-ec",
                'export PGPASSWORD="$POSTGRES_PASSWORD"; exec pg_dump -h 127.0.0.1 '
                '-U "$POSTGRES_USER" -d "$1" --snapshot "$2" --format=custom',
                "sh",
                database,
                snapshot,
                stdout=stream,
            )
        manifest.update(complete=True, dump_sha256=sha_file(dump))
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"complete": True, "source_count": len(hashes), "table_count": len(tables)}


def verified_manifest(root: Path):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Backup must be a regular trusted local directory.")
    path = root / "manifest.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Backup manifest is invalid.")
    manifest = json.loads(path.read_bytes())
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or manifest.get("complete") is not True
        or not isinstance(manifest.get("tables"), dict)
        or not isinstance(manifest.get("sources"), list)
        or not isinstance(manifest.get("dump_sha256"), str)
    ):
        raise ValueError("Backup is incomplete or unsupported.")
    dump = root / "database.dump"
    if dump.is_symlink() or not dump.is_file() or sha_file(dump) != manifest["dump_sha256"]:
        raise ValueError("Backup database checksum does not match.")
    sources = root / "sources"
    if sources.is_symlink() or not sources.is_dir():
        raise ValueError("Backup source directory must be local to the bundle.")
    hashes = []
    for source in manifest["sources"]:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("source_hash"), str)
            or type(source.get("size_bytes")) is not int
            or source["size_bytes"] <= 0
        ):
            raise ValueError("Backup source record is invalid.")
        content = FileSourceStore(root / "sources").get(source["source_hash"])
        if len(content) != source["size_bytes"]:
            raise ValueError("Backup source size does not match.")
        hashes.append(source["source_hash"])
    if len(hashes) != len(set(hashes)):
        raise ValueError("Backup has duplicate source records.")
    return manifest


async def verify_restore(root: Path, *, recover=False):
    manifest = await asyncio.to_thread(verified_manifest, root)
    # No caller-supplied target: restore can only create its own fresh scratch DB.
    name = "fredi_restore_" + uuid4().hex[:16]
    admin = await psycopg.AsyncConnection.connect(
        local_dsn(role="admin").get_secret_value(), autocommit=True
    )
    created = False
    try:
        await admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER fredi_knowledge_admin").format(sql.Identifier(name))
        )
        created = True

        # pg_restore reads stdin; avoid loading the full dump into host memory.
        def restore():
            with (root / "database.dump").open("rb") as stream:
                command = [
                    *KnowledgeInfrastructure().command,
                    "exec",
                    "-T",
                    "postgres",
                    "sh",
                    "-ec",
                    'export PGPASSWORD="$POSTGRES_PASSWORD"; exec pg_restore -h 127.0.0.1 '
                    '-U "$POSTGRES_USER" --dbname "$1" --exit-on-error --single-transaction',
                    "sh",
                    name,
                ]
                result = subprocess.run(
                    command,
                    stdin=stream,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=300,
                    check=False,
                )
                if result.returncode:
                    raise InfrastructureError(
                        "Scratch restore failed; existing databases were not modified."
                    )

        await asyncio.to_thread(restore)
        async with await psycopg.AsyncConnection.connect(
            local_dsn(role="admin", database=name).get_secret_value(), row_factory=dict_row
        ) as connection:
            await connection.execute("SET TRANSACTION READ ONLY")
            actual = await table_fingerprints(connection)
            if actual != manifest["tables"]:
                raise ValueError("Restored table fingerprints do not match the backup snapshot.")
            referenced = {
                r["hash"]
                for r in await (
                    await connection.execute(
                        "SELECT source_hash AS hash FROM knowledge.knowledge_documents "
                        "UNION SELECT value FROM knowledge.knowledge_revisions r, "
                        "LATERAL jsonb_each_text(r.payload->'manifest') "
                        "UNION SELECT payload->>'source_hash' FROM knowledge.intake_jobs"
                    )
                ).fetchall()
            }
            if referenced != {s["source_hash"] for s in manifest["sources"]}:
                raise ValueError("Restored source manifest is incomplete.")
        # Re-open as the actual runtime roles to prove ownership/ACL restoration.
        from app.persistence.database import ApplicationDatabase, ApplicationDatabaseConfig
        from app.persistence.worker import ApplicationWorker
        from scripts.application_credentials import local_application_dsn

        async with await psycopg.AsyncConnection.connect(
            local_dsn(role="reader", database=name).get_secret_value()
        ) as reader:
            access = await (
                await reader.execute(
                    "SELECT has_schema_privilege(current_user,'application','USAGE')"
                )
            ).fetchone()
            if access is None or access[0] is not False:
                raise ValueError("Restored reader role has unexpected application access.")
            await reader.execute("SELECT count(*) FROM knowledge.knowledge_chunks")
        recovered = None
        async with await psycopg.AsyncConnection.connect(
            local_application_dsn(database=name).get_secret_value()
        ) as runtime_reader:
            access = await (
                await runtime_reader.execute(
                    "SELECT has_schema_privilege(current_user,'knowledge','USAGE')"
                )
            ).fetchone()
            if access is None or access[0] is not False:
                raise ValueError("Restored application role has unexpected knowledge access.")
            await runtime_reader.execute("SELECT count(*) FROM application.conversation_messages")
        if recover:
            runtime = ApplicationDatabase(
                ApplicationDatabaseConfig(dsn=local_application_dsn(database=name))
            )
            worker = ApplicationWorker(runtime)
            try:
                await worker.start()
                recovered = worker.recovered_turns
            finally:
                await worker.close()
                await runtime.close()
        return {
            "verified": True,
            "table_count": len(actual),
            "source_count": len(referenced),
            "recovered_turns": recovered,
            "scratch_database_removed": True,
        }
    finally:
        if created:
            if not re.fullmatch("fredi_restore_[0-9a-f]{16}", name):
                raise ValueError("Unsafe scratch cleanup identity.")
            await admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
        await admin.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create")
    make.add_argument("--output", required=True, type=Path)
    make.add_argument("--database", default="fredi_knowledge")
    make.add_argument(
        "--source-store", action="append", type=Path, default=[ROOT / "backend/.knowledge-sources"]
    )
    check = commands.add_parser("verify")
    check.add_argument("--backup", required=True, type=Path)
    check.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    try:
        result = asyncio.run(
            backup(args.output, database=args.database, source_roots=args.source_store)
            if args.command == "create"
            else verify_restore(args.backup, recover=args.recover),
            loop_factory=asyncio.SelectorEventLoop,
        )
        print(json.dumps(result))
        return 0
    except (InfrastructureError, KnowledgeError, psycopg.Error, ValueError, OSError, KeyError):
        print(
            "Backup/verification failed. Keep partial output private; "
            "no existing database was restored over.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

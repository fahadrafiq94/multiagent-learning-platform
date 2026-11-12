"""Freeze/check the candidate and audit the read-only course/database baseline."""

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psycopg
from psycopg.rows import dict_row

from app.config.settings import settings
from scripts.evaluate_student_acceptance import verify_snapshot
from scripts.knowledge_backup import private_directory, table_fingerprints
from scripts.knowledge_infra import ROOT
from scripts.migrate_knowledge import local_dsn
from scripts.run_student_simulation import save_json


def frozen_files():
    files = []
    for folder in ("backend/app", "backend/scripts", "backend/tests"):
        files.extend((ROOT / folder).rglob("*.py"))
    files.extend((ROOT / "docs/knowledge").glob("*.json"))
    files.extend((ROOT / "docs/knowledge").glob("*.md"))
    files.extend((ROOT / "docker").rglob("*.yml"))
    files.extend(ROOT / name for name in ("backend/pyproject.toml", "backend/uv.lock"))
    # Consumption state is deliberately mutable; the reserved case bytes are
    # checked against their frozen manifest digest by the acceptance runner.
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(files))
        if path.name != "student-quality-reserved-v1-manifest.json"
    }


async def database_audit():
    async with await psycopg.AsyncConnection.connect(
        local_dsn(role="admin").get_secret_value(), row_factory=dict_row
    ) as connection:
        tables = await table_fingerprints(connection)
        published = await (
            await connection.execute(
                "SELECT c.collection_id,c.active_revision_id,c.access_domain,r.indexing_status "
                "FROM knowledge.knowledge_collections c JOIN knowledge.knowledge_revisions r "
                "ON r.revision_id=c.active_revision_id ORDER BY c.collection_id"
            )
        ).fetchall()
        scratch = await (
            await connection.execute(
                "SELECT datname FROM pg_database WHERE datname LIKE 'fredi_app_test_%' "
                "OR datname LIKE 'fredi_eval_%' OR datname LIKE 'fredi_restore_%'"
            )
        ).fetchall()
        sources = await (
            await connection.execute(
                "SELECT revision_id,document_id,source_hash FROM knowledge.knowledge_documents "
                "ORDER BY revision_id,document_id"
            )
        ).fetchall()
    blobs = {
        str(path.relative_to(ROOT / "backend/.knowledge-sources")): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((ROOT / "backend/.knowledge-sources").rglob("*"))
        if path.is_file()
    }
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(settings.ollama_base_url.rstrip("/") + "/api/tags")
        response.raise_for_status()
        models = {
            item["name"]: item["digest"]
            for item in response.json()["models"]
            if item["name"] in {"qwen3.5:4b", "qwen3:1.7b", "nomic-embed-text-v2-moe:latest"}
        }
    return json.loads(
        json.dumps(
            {
                "tables": tables,
                "published": published,
                "sources": sources,
                "source_blobs": blobs,
                "scratch_databases": scratch,
                "serving_models": models,
                "runtime_flags": {
                    "rag": settings.rag_enabled,
                    "cache": settings.knowledge_cache_enabled,
                    "intake": settings.knowledge_intake_enabled,
                },
            },
            default=str,
        )
    )


async def execute(args):
    if args.command == "freeze":
        private_directory(args.output)
        audit = await database_audit()
        source = json.loads(
            (ROOT / "docs/knowledge/student-journey-step6-source-v1.json").read_bytes()
        )
        if not any(
            row["collection_id"] == source["collection_id"]
            and row["active_revision_id"] == source["revision_id"]
            and row["access_domain"] == source["access_domain"]
            and row["indexing_status"] == "ready"
            for row in audit["published"]
        ):
            raise ValueError("Intended document revision is not published and ready.")
        snapshot = {
            "frozen_at": datetime.now(UTC).isoformat(),
            "files": frozen_files(),
            "source": source,
            "baseline": audit,
            "profile": {
                "model": "qwen3.5:4b",
                "planner": "qwen3:1.7b",
                "window": 8192,
                "thinking": False,
                "retries": 0,
                "verification_seconds": 30,
                "verification_tokens": 512,
            },
            "targets": {
                "p50_seconds": 90,
                "p95_seconds": 150,
                "p50_method": "median",
                "p95_method": "nearest_rank",
            },
        }
        save_json(args.output / "snapshot.json", snapshot)
        print(
            json.dumps({"frozen_files": len(snapshot["files"]), "revision": source["revision_id"]})
        )
    else:
        snapshot = json.loads(args.snapshot.read_bytes())
        verify_snapshot(snapshot)
        current = await database_audit()
        checks = {
            key + "_unchanged": current[key] == snapshot["baseline"][key]
            for key in (
                "tables",
                "published",
                "sources",
                "source_blobs",
                "serving_models",
                "runtime_flags",
            )
        }
        checks["no_scratch_databases"] = not current["scratch_databases"]
        checks["frozen_file_set_unchanged"] = frozen_files() == snapshot["files"]
        save_json(
            args.snapshot.parent / "final-audit.json",
            {
                "checked_at": datetime.now(UTC).isoformat(),
                "checks": checks,
                "current": current,
            },
        )
        print(json.dumps(checks))
        if not all(checks.values()):
            raise ValueError("Candidate/data audit failed; inspect the protected report.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze").add_argument("--output", type=Path, required=True)
    commands.add_parser("check").add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(execute(args))


if __name__ == "__main__":
    main()

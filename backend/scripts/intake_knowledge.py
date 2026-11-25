"""Manage durable intake or run the bounded polling worker; output IDs/status only."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

from app.config.settings import settings
from app.knowledge.errors import KnowledgeError
from app.knowledge.extraction import MAX_SOURCE_BYTES
from app.knowledge.intake_runtime import create_intake_runtime
from app.services.model_service.exceptions import ModelServiceError
from app.services.model_service.service import close_model_services
from scripts.import_knowledge import import_dsn


async def execute(args):
    profile = settings.model_copy(
        update={
            "knowledge_intake_config_file": args.config,
            "knowledge_source_root": args.source_store,
            "knowledge_ingest_database_url": import_dsn(),
        }
    )
    service, database = create_intake_runtime(profile, require_auth=False)
    try:
        if args.command == "worker":
            while True:
                processed = 0
                for collection_id in service.bindings:
                    try:
                        result = await service.run_once(collection_id)
                        if result:
                            print(json.dumps(result.public()), flush=True)
                            processed += 1
                    except KnowledgeError:
                        print(
                            "Intake worker could not process a configured collection.",
                            file=sys.stderr,
                            flush=True,
                        )
                        if args.once:
                            return 1
                if args.once:
                    return 0
                if not processed:
                    await asyncio.sleep(2)
        binding = service.binding(args.collection)
        if args.command == "submit":
            with args.pdf.open("rb") as stream:
                content = stream.read(MAX_SOURCE_BYTES + 1)
            result = await service.submit(
                args.collection,
                content,
                title=args.title,
                replace_document_id=args.replace_document,
            )
        elif args.command == "retry":
            result = await service.jobs.retry(binding, args.job)
        else:
            result = await service.jobs.get(binding, args.job)
        print(json.dumps(result.public()))
        return 0
    finally:
        try:
            await close_model_services(service)
        finally:
            await database.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-store", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--collection", type=UUID, required=True)
    submit.add_argument("--pdf", type=Path, required=True)
    submit.add_argument("--title", required=True)
    submit.add_argument("--replace-document", type=UUID)
    for name in ("status", "retry"):
        command = commands.add_parser(name)
        command.add_argument("--collection", type=UUID, required=True)
        command.add_argument("--job", type=UUID, required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        return asyncio.run(execute(args), loop_factory=asyncio.SelectorEventLoop)
    except (KnowledgeError, ModelServiceError, OSError, ValueError, KeyError):
        print(
            "Intake failed; check configuration, permissions, source and storage.", file=sys.stderr
        )
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

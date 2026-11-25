"""Interactive local knowledge management. No student-facing endpoint is enabled."""

import argparse
import asyncio
import copy
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import UUID

from app.config.settings import settings
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgeError
from app.knowledge.extraction import PdfExtractor
from app.knowledge.indexer import IndexConfig
from app.knowledge.intake import IntakeService
from app.knowledge.intake_repository import IntakeRepository
from app.knowledge.intake_schemas import IntakeConfig
from app.knowledge.metadata import MetadataConfig
from app.knowledge.migrations import migrate
from app.knowledge.postgres import decode
from app.knowledge.schemas import KnowledgeCollection, KnowledgeWriteScope
from app.knowledge.source_store import FileSourceStore
from app.persistence.database import ApplicationDatabase, ApplicationDatabaseConfig
from app.repositories.exceptions import RepositoryError
from app.services.model_service import create_model_service
from app.services.model_service.exceptions import ModelServiceError
from app.services.model_service.service import close_model_services
from scripts.application_credentials import configured_dsn
from scripts.import_knowledge import import_dsn
from scripts.knowledge_infra import ROOT, InfrastructureError
from scripts.knowledge_operator_setup import (
    OperatorError,
    manager_ids,
    owner_identity,
    provision_operator,
    read_local_config,
    write_local_config,
)
from scripts.knowledge_operator_ui import Console, jobs_flow, upload_flow


def presets(root: Path):
    result = []
    for path in sorted((root / "docs/knowledge").glob("indexing*.json")):
        if path.stat().st_size <= 1024 * 1024:
            result.append(IndexConfig.model_validate_json(path.read_bytes()))
    return result


async def complete_config(raw, jobs, operator_id, console, *, root=ROOT):
    """Fill missing deployment data from scoped DB records; preserve valid choices."""
    result = copy.deepcopy(raw)
    async with jobs.database.connection() as connection:
        rows = await (
            await connection.execute(
                "SELECT * FROM knowledge.knowledge_collections ORDER BY collection_id"
            )
        ).fetchall()
    catalog = {row["collection_id"]: row for row in rows}
    if not catalog:
        raise ValueError(
            "No knowledge collection exists. Register a collection before using intake."
        )
    bindings = result.setdefault("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("Configuration bindings must be a list.")
    if not bindings:
        collection = (
            rows[0]
            if len(rows) == 1
            else console.choose(
                "Select the collection to manage:",
                rows,
                lambda r: f"{r['payload'].get('name', '')} ({r['collection_id']})",
            )
        )
        if collection is None:
            raise ValueError("Setup cancelled.")
        bindings.append({"collection_id": str(collection["collection_id"])})
    for value in bindings:
        if not isinstance(value, dict):
            raise ValueError("Every binding must be a JSON object.")
        if "collection_id" not in value and len(rows) == 1:
            value["collection_id"] = str(rows[0]["collection_id"])
        row = catalog.get(UUID(value["collection_id"]))
        if row is None:
            raise ValueError("Configured collection was not found.")
        collection = decode(KnowledgeCollection, "knowledge_collections", row)
        scope = value.setdefault("scope", {})
        scope.setdefault("access_domain", collection.access_domain)
        scope.setdefault("can_manage_knowledge", True)
        if collection.owner_user_id:
            scope.setdefault("user_id", str(collection.owner_user_id))
        grant = KnowledgeWriteScope.model_validate(scope)
        if not grant.can_manage_knowledge or not collection.permits(grant):
            raise ValueError(
                "Existing scope does not authorize this collection; it was not broadened."
            )
        baseline_id = row["active_revision_id"] or value.get("initial_revision_id")
        if baseline_id is None:
            async with jobs.database.connection() as connection:
                ready = await (
                    await connection.execute(
                        "SELECT revision_id FROM knowledge.knowledge_revisions "
                        "WHERE collection_id=%s "
                        "AND indexing_status='ready' ORDER BY revision_id",
                        (collection.collection_id,),
                    )
                ).fetchall()
            if ready:
                chosen = (
                    ready[0]
                    if len(ready) == 1
                    else console.choose(
                        "Select the ready revision to preserve as the starting document set:",
                        ready,
                        lambda r: str(r["revision_id"]),
                    )
                )
                if chosen is None:
                    raise ValueError("Setup cancelled.")
                baseline_id = chosen["revision_id"]
                value["initial_revision_id"] = str(baseline_id)
        baseline = (
            await jobs.knowledge.get_revision(UUID(str(baseline_id)), grant)
            if baseline_id
            else None
        )
        if baseline_id and (
            baseline is None
            or baseline.collection_id != collection.collection_id
            or baseline.indexing_status != "ready"
        ):
            raise ValueError("The configured initial revision is not ready in this collection.")
        index = value.setdefault("index", {})
        if not index:
            choices = [
                p for p in presets(root) if baseline is None or p.embedding == baseline.embedding
            ]
            if len(choices) == 1:
                index.update(choices[0].model_dump(mode="json"))
            elif baseline is not None:
                data = await jobs.knowledge.get_revision_data(baseline.revision_id, grant)
                namespaces = {s.namespace for s in data.steps} if data else set()
                default = IndexConfig(
                    embedding=baseline.embedding,
                    metadata=MetadataConfig(
                        namespace=next(iter(namespaces)) if len(namespaces) == 1 else None
                    ),
                )
                index.update(default.model_dump(mode="json"))
            elif choices:
                chosen = console.choose(
                    "Select a trusted embedding/index preset:",
                    choices,
                    lambda c: f"{c.embedding.model}, {c.embedding.dimension} dimensions",
                )
                if chosen is None:
                    raise ValueError("Setup cancelled.")
                index.update(chosen.model_dump(mode="json"))
            else:
                raise ValueError(
                    "An empty collection needs a trusted index preset in its configuration."
                )
        elif baseline is not None:
            embedding = index.setdefault("embedding", {})
            for key, item in baseline.embedding.model_dump(mode="json").items():
                if key not in embedding or (
                    isinstance(embedding[key], str) and embedding[key].startswith("REPLACE_")
                ):
                    embedding[key] = item
        value["manager_user_ids"] = manager_ids(value.get("manager_user_ids", []), operator_id)
    return IntakeConfig.model_validate(result)


async def verified_model(config, model_url=None):
    candidates = (
        [model_url]
        if model_url
        else [settings.embedding_base_url]
        if settings.embedding_base_url
        else [settings.ollama_base_url, "http://127.0.0.1:11434"]
    )
    for endpoint in dict.fromkeys(e for e in candidates if e):
        profile = settings.model_copy(
            update={"model_max_retries": 0, "model_request_timeout_seconds": 10}
        )
        model = create_model_service(profile, role="embedding", base_url=endpoint)
        try:
            for binding in config.bindings:
                expected = binding.index.embedding
                actual = await model.inspect_embedding_model(expected.model)
                if (actual.provider, actual.revision, actual.dimension) != (
                    expected.provider,
                    expected.revision,
                    expected.dimension,
                ):
                    raise OperatorError(
                        "Installed embeddings do not match the pinned profile; "
                        "settings were preserved."
                    )
            # Real indexing may take longer than the quick setup probe.
            return create_model_service(
                profile.model_copy(update={"model_request_timeout_seconds": 180}),
                role="embedding",
                base_url=endpoint,
            )
        except ModelServiceError:
            continue
        finally:
            await close_model_services(model)
    raise OperatorError(
        "Embedding service unavailable. Start Ollama or supply --model-url; "
        "no model was downloaded."
    )


async def execute(args, console=None):
    console = console or Console()
    raw, original = await asyncio.to_thread(read_local_config, args.config)
    owner = await owner_identity()
    async with (
        AsyncExitStack() as resources,
        KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=import_dsn())) as database,
    ):
        jobs = IntakeRepository(database)
        async with ApplicationDatabase(
            ApplicationDatabaseConfig(dsn=configured_dsn())
        ) as application:
            operator_id = await provision_operator(application, owner)
        config = await complete_config(raw, jobs, operator_id, console)
        model = await verified_model(config, args.model_url)
        resources.push_async_callback(close_model_services, model)
        dimensions = {b.index.embedding.dimension for b in config.bindings}
        if len(dimensions) != 1:
            raise ValueError(
                "All configured profiles must match this database's embedding dimension."
            )
        await migrate(database, embedding_dimension=next(iter(dimensions)))
        for binding in config.bindings:
            await jobs.head(binding)
        await asyncio.to_thread(
            write_local_config, args.config, config.model_dump(mode="json"), original
        )
        args.inbox.mkdir(parents=True, exist_ok=True)
        service = IntakeService(
            jobs, FileSourceStore(args.source_store), PdfExtractor(), model, config
        )
        console.say(f"Setup ready. Local management user: {operator_id}")
        console.say(f"Protected configuration: {args.config}")
        console.say(
            "Embedding profile verified. Student RAG and HTTP intake settings were not changed."
        )
        if args.setup_only:
            return 0
        while True:
            action = console.ask(
                "\n1. Add new PDFs\n2. Replace existing documents"
                "\n3. View/retry processing jobs\n0. Exit\nChoose: "
            )
            if action == "0":
                return 0
            if action not in {"1", "2", "3"}:
                console.say("Enter 1, 2, 3 or 0.")
                continue
            bindings = list(service.bindings.values())
            binding = (
                bindings[0]
                if len(bindings) == 1
                else console.choose("Select collection:", bindings, lambda b: str(b.collection_id))
            )
            if binding is None:
                continue
            try:
                if action == "3":
                    await jobs_flow(service, binding, console)
                else:
                    await upload_flow(
                        service, binding, operator_id, args.inbox, console, replace=action == "2"
                    )
            except OperatorError as exc:
                console.say(str(exc))
            except (KnowledgeError, ValueError, OSError):
                console.say(
                    "Operation could not complete. Check PDF paths, current scope "
                    "and job status (option 3). Saved jobs remain durable."
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / ".knowledge-intake.json")
    parser.add_argument("--source-store", type=Path, default=ROOT / "backend/.knowledge-sources")
    parser.add_argument("--inbox", type=Path, default=ROOT / ".knowledge-inbox")
    parser.add_argument("--model-url")
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()
    try:
        return asyncio.run(execute(args), loop_factory=asyncio.SelectorEventLoop)
    except OperatorError as exc:
        print(str(exc))
        return 1
    except (
        KnowledgeError,
        RepositoryError,
        InfrastructureError,
        ModelServiceError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ):
        print(
            "Setup failed. Check local credential access, PostgreSQL/Ollama and JSON settings. "
            "Existing files were not replaced with defaults."
        )
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Submitted jobs remain saved for retry/recovery.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

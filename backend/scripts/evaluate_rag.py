"""Frozen read-only course evaluation plus a disposable generic-PDF corpus."""

import argparse
import asyncio
import hashlib
import io
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.agents.ap_plus_navigator import APPlusNavigatorAgent
from app.agents.process_coach import ProcessCoachAgent
from app.agents.scenario import ScenarioAgent
from app.agents.schemas import AgentRequest
from app.config.settings import settings
from app.context.agent_views import AgentContextView, ContextTarget, RuntimeSessionContext
from app.context.grounding import GroundingContext
from app.domain import GuidanceLevel
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgeError
from app.knowledge.evaluation import EvaluationSuite, retrieval_score, summarize
from app.knowledge.extraction import PdfExtractor
from app.knowledge.indexer import IndexConfig
from app.knowledge.intake import IntakeService
from app.knowledge.intake_repository import IntakeRepository
from app.knowledge.intake_schemas import IntakeBinding, IntakeConfig
from app.knowledge.integration import project_evidence
from app.knowledge.migrations import migrate
from app.knowledge.planner import HybridRetrievalService
from app.knowledge.planner_schemas import EvidenceReceipt, PlannerConfig, PlannerRequest
from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.retrieval_schemas import MetadataRequest, RetrievalRequest
from app.knowledge.schemas import AccessScope, KnowledgeCollection, KnowledgeWriteScope
from app.knowledge.search_repository import PostgresSearchRepository
from app.knowledge.source_store import FileSourceStore
from app.services.model_service import create_model_service
from app.services.model_service.exceptions import ModelServiceError
from scripts.knowledge_backup import private_directory
from scripts.knowledge_infra import ROOT, InfrastructureError
from scripts.migrate_knowledge import local_dsn
from scripts.retrieve_knowledge import retrieval_dsn


def generic_pdf(text):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    escaped = [
        line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        for line in text.splitlines()
    ]
    stream = DecodedStreamObject()
    stream.set_data(
        (
            "BT /F1 10 Tf 20 750 Td 15 TL " + " ".join(f"({s}) Tj T*" for s in escaped) + " ET"
        ).encode()
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class ProfileModel:
    def __init__(self, model, *, capture_output=False):
        self.model = model
        self.responses = []
        self.calls = []
        self.scope = {}
        self.capture_output = capture_output

    async def generate(self, request):
        request = request.model_copy(
            update={"context_window_tokens": 8192, "reasoning_enabled": False}
        )
        started = time.perf_counter()
        row = {
            **self.scope,
            "task": request.metadata.get("task"),
            "requested_model": request.model,
            "context_window": request.context_window_tokens,
            "reasoning": request.reasoning_enabled,
            "max_output_tokens": request.max_tokens,
            "message_characters": sum(len(m.content) for m in request.messages),
            "schema_characters": len(json.dumps(request.response_schema)),
        }
        try:
            response = await self.model.generate(request)
            self.responses.append(response)
            row.update(
                outcome="completed",
                model=response.model,
                usage=response.usage.model_dump() if response.usage else None,
                response_characters=len(response.content),
            )
            raw = response.raw or {}
            for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
                value = raw.get(key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and (math.isfinite(value) and value >= 0)
                ):
                    row[key + "_ms"] = round(value / 1_000_000, 2)
            for key in ("prompt_eval_count", "eval_count"):
                if type(raw.get(key)) is int and raw[key] >= 0:
                    row[key] = raw[key]
            if raw.get("done_reason") in {"stop", "length", "load", "unload"}:
                row["done_reason"] = raw["done_reason"]
            if row.get("eval_duration_ms", 0) > 0 and "eval_count" in row:
                row["output_tokens_per_second"] = round(
                    row["eval_count"] * 1000 / row["eval_duration_ms"], 2
                )
            if self.capture_output:
                row["private_model_output"] = response.content[:16000]
                row["output_truncated"] = len(response.content) > 16000
            return response
        except BaseException as exc:
            row.update(
                outcome="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                failure_type=type(exc).__name__,
            )
            if isinstance(exc, ModelServiceError):
                row["failure_code"] = exc.code
            raise
        finally:
            row["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.calls.append(row)

    async def embed(self, request):
        return await self.model.embed(request)

    async def inspect_embedding_model(self, model):
        return await self.model.inspect_embedding_model(model)


def select_cases(suite, args):
    cases = [c for c in suite.cases if c.split == args.split]
    selected = set(args.case_id or [])
    if selected - {c.case_id for c in cases}:
        raise ValueError("Selected cases must belong to the selected split.")
    if args.probe_teaching and args.split != "development":
        raise ValueError("Extra teaching probes are development-only.")
    if len(set(args.guidance)) != len(args.guidance):
        raise ValueError("Guidance levels must be unique.")
    return [c for c in cases if not selected or c.case_id in selected]


async def execute(args):
    suite, suite_hash = EvaluationSuite.read(args.suite)
    cases = select_cases(suite, args)
    private_directory(args.output)
    profile = settings.model_copy(
        update={
            "model_provider": "ollama",
            "ollama_base_url": args.model_url,
            "ollama_chat_model": args.model,
            "ollama_embedding_model": "nomic-embed-text-v2-moe:latest",
            "model_max_retries": 0,
            "model_request_timeout_seconds": args.timeout,
        }
    )
    real = create_model_service(profile)
    health = await real.health()
    identities = [m for m in health.details.get("models", []) if m.get("name") == args.model]
    metadata = {
        "suite_sha256": suite_hash,
        "suite_version": suite.version,
        "split": args.split,
        "model": args.model,
        "model_digest": identities[0].get("digest") if identities else None,
        "context_window": 8192,
        "reasoning": False,
        "max_output_tokens": 1024,
        "os": platform.system(),
        "python": platform.python_version(),
        "started_at": datetime.now(UTC).isoformat(),
        "stages": args.stages,
        "retrieval_config": suite.retrieval_config.model_dump(mode="json"),
        "selected_cases": [c.case_id for c in cases],
        "guidance_levels": args.guidance,
        "extra_development_teaching": args.probe_teaching,
        "private_output_capture": args.diagnostics,
        "provider_timeout_seconds": args.timeout,
        "grounding_code_sha256": hashlib.sha256(
            (ROOT / "backend/app/agents/grounding.py").read_bytes()
        ).hexdigest(),
        "planner_prompt_sha256": hashlib.sha256(
            (ROOT / "backend/app/knowledge/planner_prompt.py").read_bytes()
        ).hexdigest(),
        "planner_config": PlannerConfig(
            model=args.model, timeout_seconds=args.timeout
        ).model_dump(),
    }
    model = ProfileModel(real)
    records = []
    admin = await psycopg.AsyncConnection.connect(
        local_dsn(role="admin").get_secret_value(), autocommit=True
    )
    generic_name = "fredi_eval_" + uuid4().hex[:16]
    created = False
    generic_database = None
    try:
        async with KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=retrieval_dsn())) as database:
            main = PostgresSearchRepository(database)
            owner = uuid4()
            scope = AccessScope(access_domain=suite.access_domain, user_id=owner)
            pin = await main.admit(suite.collection_id, suite.revision_id, scope)
            if pin is None:
                raise ValueError("Frozen source revision is unavailable.")
            profile = profile.model_copy(
                update={"ollama_embedding_model": pin.revision.embedding.model}
            )
            model = ProfileModel(create_model_service(profile), capture_output=args.diagnostics)
            metadata["source_revision"] = str(pin.revision.revision_id)
            metadata["embedding_profile"] = pin.revision.embedding.model_dump(mode="json")
            await admin.execute(
                sql.SQL("CREATE DATABASE {} OWNER fredi_knowledge_writer").format(
                    sql.Identifier(generic_name)
                )
            )
            created = True
            async with await psycopg.AsyncConnection.connect(
                local_dsn(role="admin", database=generic_name).get_secret_value()
            ) as bootstrap:
                await bootstrap.execute(
                    "CREATE EXTENSION vector; "
                    "CREATE SCHEMA knowledge AUTHORIZATION fredi_knowledge_writer; "
                    "GRANT USAGE ON SCHEMA knowledge TO fredi_knowledge_reader; "
                    "ALTER DEFAULT PRIVILEGES FOR ROLE fredi_knowledge_writer IN SCHEMA knowledge "
                    "GRANT SELECT ON TABLES TO fredi_knowledge_reader"
                )
            generic_database = KnowledgeDatabase(
                KnowledgeDatabaseConfig(dsn=local_dsn(database=generic_name))
            )
            await migrate(generic_database, embedding_dimension=pin.revision.embedding.dimension)
            collection = KnowledgeCollection(
                name="Frozen generic evaluation", access_domain="evaluation:" + generic_name
            )
            binding = IntakeBinding(
                collection_id=collection.collection_id,
                scope=KnowledgeWriteScope(
                    access_domain=collection.access_domain, can_manage_knowledge=True
                ),
                manager_user_ids=[owner],
                index=IndexConfig(embedding=pin.revision.embedding),
            )
            jobs = IntakeRepository(generic_database)
            await jobs.knowledge.create_collection(collection, scope=binding.scope)
            sources = FileSourceStore(args.output / "generic-sources")
            service = IntakeService(
                jobs, sources, PdfExtractor(), model, IntakeConfig(bindings=[binding])
            )
            job = await service.submit(
                collection.collection_id,
                generic_pdf(suite.generic_text),
                title="Generic evaluation archive",
            )
            # Stable gold document ID without changing the source-independent intake algorithm.
            suite_generic_id = job.payload.document_id
            completed = await service.run_once(collection.collection_id)
            if completed.status != "succeeded":
                raise ValueError("Generic evaluation intake failed.")
            generic = PostgresSearchRepository(generic_database)
            generic_scope = AccessScope(access_domain=collection.access_domain, user_id=owner)
            generic_pin = await generic.admit(
                collection.collection_id, completed.result_revision_id, generic_scope
            )
            metadata["generic_document_id"] = str(suite_generic_id)
            for case in cases:
                repository, admitted, access = (
                    (generic, generic_pin, generic_scope)
                    if case.corpus == "generic"
                    else (main, pin, scope)
                )
                # Gold fixture IDs are resolved to the newly generated fixture's ID only
                # in scoring, never added to any query or model prompt.
                scored_case = (
                    case.model_copy(update={"expected_documents": [suite_generic_id]})
                    if case.corpus == "generic" and case.answerable
                    else case
                )
                retrieval = KnowledgeRetrievalService(repository, model, suite.retrieval_config)
                query = RetrievalRequest(
                    collection_id=admitted.revision.collection_id,
                    revision_id=admitted.revision.revision_id,
                    query=case.question,
                    purpose=case.specialist,
                    current_step_id=case.current_step_id,
                    namespace=suite.namespace if case.corpus == "course" else None,
                    top_k=4,
                )
                result = None
                if "retrieval" in args.stages and case.retrieval:
                    start = time.perf_counter()
                    result = await retrieval.search(query, scope=access)
                    records.append(
                        {
                            "stage": "retrieval",
                            "case_id": case.case_id,
                            "status": result.status,
                            "score": retrieval_score(scored_case, result),
                            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                            "references": sorted(
                                {i.reference_id for i in result.items if i.reference_id}
                            ),
                            "returned": len(result.items),
                        }
                    )
                if "planner" in args.stages:
                    model.scope = {"stage": "planner", "case_id": case.case_id}
                    request = PlannerRequest(
                        user_id=owner,
                        session_id=uuid4(),
                        specialist=case.specialist,
                        collection_id=admitted.revision.collection_id,
                        revision_id=admitted.revision.revision_id,
                        namespace=query.namespace,
                        question=case.question,
                        current_step_id=case.current_step_id,
                    )
                    seed = None
                    if case.seed_step:
                        seed = await retrieval.lookup(
                            MetadataRequest(
                                collection_id=query.collection_id,
                                revision_id=query.revision_id,
                                namespace=query.namespace,
                                step_id=case.seed_step,
                                fields=["next_step_id"],
                            ),
                            scope=access,
                        )
                    elif case.seed_query:
                        seed = await retrieval.search(
                            query.model_copy(update={"query": case.seed_query}), scope=access
                        )
                    if seed and seed.items:
                        request = request.model_copy(
                            update={
                                "available": [
                                    EvidenceReceipt(
                                        user_id=owner,
                                        session_id=request.session_id,
                                        scope=access,
                                        collection_id=query.collection_id,
                                        revision_id=admitted.revision.revision_id,
                                        created_at=datetime.now(UTC),
                                        items=seed.items,
                                    )
                                ]
                            }
                        )
                    planned = await HybridRetrievalService(
                        retrieval,
                        model,
                        PlannerConfig(model=args.model, timeout_seconds=args.timeout),
                    ).execute(request, scope=access)
                    actions = [a.action for a in planned.plan.actions] if planned.plan else []
                    refs = {i.reference_id for i in planned.items if i.reference_id}
                    passed = (
                        planned.origin == "model"
                        and bool(set(actions) & set(case.allowed_actions))
                        and planned.status in case.expected_statuses
                    )
                    if case.answerable:
                        passed = passed and (
                            not case.expected_references
                            or set(case.expected_references).issubset(refs)
                        )
                        if scored_case.expected_documents:
                            documents = {c.document_id for i in planned.items for c in i.citations}
                            passed = passed and set(scored_case.expected_documents).issubset(
                                documents
                            )
                    records.append(
                        {
                            "stage": "planner",
                            "case_id": case.case_id,
                            "passed": passed,
                            "has_plan": planned.plan is not None,
                            "origin": planned.origin,
                            "actions": actions,
                            "status": planned.status,
                            "latency_ms": planned.elapsed_ms,
                            "planner_latency_ms": planned.planner_latency_ms,
                            "failure": planned.planner_failure,
                            "failure_detail": planned.planner_failure_detail,
                        }
                    )
                if "teacher" in args.stages and (case.guidance_probe or args.probe_teaching):
                    result = result or await retrieval.search(query, scope=access)
                    for level in (GuidanceLevel(value) for value in args.guidance):
                        model.scope = {
                            "stage": "teacher",
                            "case_id": case.case_id,
                            "guidance": level.value,
                        }
                        target = ContextTarget(case.specialist)
                        view = AgentContextView(
                            target=target,
                            user_id=owner,
                            session_id=uuid4(),
                            current_message=case.question,
                            guidance_level=level,
                            policy_version="frozen-evaluation-v1",
                            runtime_session=RuntimeSessionContext(
                                current_step_id=case.current_step_id
                            ),
                        )
                        view.grounding = GroundingContext(
                            status=result.status
                            if result.status in {"found", "no_evidence", "error", "unavailable"}
                            else "error",
                            items=[project_evidence(i) for i in result.items],
                            reasoning_enabled=False,
                            reserved_output_tokens=1024,
                            max_context_tokens=8192,
                        )
                        agent = {
                            "scenario": ScenarioAgent,
                            "process_coach": ProcessCoachAgent,
                            "ap_plus_navigator": APPlusNavigatorAgent,
                        }[case.specialist](model)
                        start = time.perf_counter()
                        try:
                            response = await agent.execute(
                                AgentRequest(
                                    session_id=str(view.session_id),
                                    user_message=case.question,
                                    context=view,
                                    guidance_level=level,
                                )
                            )
                            row = {
                                "stage": "teacher",
                                "case_id": case.case_id,
                                "guidance": level.value,
                                "valid": True,
                                "answer": response.content,
                                "citation_count": len(response.source_references),
                                "grounding": response.metadata.get("grounding"),
                                "review": "pending",
                            }
                        except Exception as exc:
                            row = {
                                "stage": "teacher",
                                "case_id": case.case_id,
                                "guidance": level.value,
                                "valid": False,
                                "review": "invalid output/provider failure",
                                "failure_type": type(exc).__name__,
                                "cause_type": type(exc.__cause__).__name__
                                if exc.__cause__
                                else None,
                            }
                        row["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
                        row["provided_evidence"] = [i.model_dump(mode="json") for i in result.items]
                        records.append(row)
                        save(args.output, suite, metadata, records, args.split, calls=model.calls)
                save(args.output, suite, metadata, records, args.split, calls=model.calls)
                print(
                    json.dumps({"case_id": case.case_id, "completed_stages": args.stages}),
                    flush=True,
                )
            # Fresh scope denial is separate from semantic relevance metrics.
            denied = await main.admit(
                suite.collection_id,
                suite.revision_id,
                AccessScope(access_domain="evaluation-denied", user_id=owner),
            )
            metadata["unauthorized_admission"] = denied is not None
            metadata["completed_at"] = datetime.now(UTC).isoformat()
            save(args.output, suite, metadata, records, args.split, calls=model.calls)
    finally:
        if generic_database:
            await generic_database.close()
        if created:
            await admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(generic_name)))
        await admin.close()
    return 0


def save(output, suite, metadata, records, split, *, calls=None):
    payload = {
        "metadata": metadata,
        "summary": summarize(suite, records, split=split),
        "records": records,
    }
    if calls is not None:
        payload["model_calls"] = calls
    path = output / "results.json"
    temporary = output / ".results.tmp"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "docs/knowledge/rag-evaluation-v1.json"
    )
    parser.add_argument("--split", choices=["development", "heldout"], required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["retrieval", "planner", "teacher"],
        default=["retrieval", "planner", "teacher"],
    )
    parser.add_argument("--model-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--case-id", action="append", help="Repeat to select diagnostic cases.")
    parser.add_argument(
        "--probe-teaching",
        action="store_true",
        help="Probe selected development questions at the requested guidance levels.",
    )
    parser.add_argument(
        "--guidance",
        nargs="+",
        choices=[g.value for g in GuidanceLevel],
        default=[g.value for g in GuidanceLevel],
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Retain bounded raw model outputs only in the private result artifact.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return asyncio.run(execute(args), loop_factory=asyncio.SelectorEventLoop)
    except (
        KnowledgeError,
        ModelServiceError,
        InfrastructureError,
        psycopg.Error,
        OSError,
        ValueError,
    ):
        print(
            "Evaluation failed; inspect private partial results and source/serving configuration."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

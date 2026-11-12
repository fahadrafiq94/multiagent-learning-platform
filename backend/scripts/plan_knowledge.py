"""Trusted read-only operator for hybrid decisions; does not invoke a student chat turn."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from app.config.settings import settings
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgeError
from app.knowledge.planner import HybridRetrievalService
from app.knowledge.planner_schemas import EvidenceReceipt, PlannerConfig, PlannerRequest
from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.retrieval_schemas import RetrievalRequest
from app.knowledge.schemas import AccessScope
from app.knowledge.search_repository import PostgresSearchRepository
from app.services.model_service import create_model_service
from app.services.model_service.exceptions import ModelServiceError
from scripts.retrieve_knowledge import retrieval_dsn


async def execute(args) -> int:
    cases = [
        {
            "id": "operator",
            "question": args.question,
            "specialist": args.specialist,
            "current_step_id": args.current_step,
            "namespace": args.namespace,
            "phase": args.phase,
            "objective": args.objective,
            "clear_course_query": args.clear_course_query,
            "seed_query": args.seed_query,
        }
    ]
    if args.cases:
        if args.cases.stat().st_size > 65536:
            raise ValueError("Case file exceeds its bound.")
        cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
        if not 1 <= len(cases) <= 30:
            raise ValueError("Expected one to thirty operator cases.")
    job_settings = settings.model_copy(
        update={
            "model_provider": "ollama",
            "ollama_base_url": args.model_url or settings.ollama_base_url,
            "ollama_chat_model": args.planner_model,
            "ollama_embedding_model": settings.ollama_embedding_model or "unused-until-admission",
            "model_max_retries": 0,
            "model_request_timeout_seconds": args.timeout,
        }
    )
    planner_model = create_model_service(job_settings)
    config = PlannerConfig(model=args.planner_model, timeout_seconds=args.timeout)
    failed = 0
    async with KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=retrieval_dsn())) as database:
        repository = PostgresSearchRepository(database)
        scope = AccessScope(access_domain=args.scope_domain, user_id=args.owner_user_id)
        pin = await repository.admit(args.collection, args.revision, scope)
        # Embedding configuration comes from the admitted index, not from planner output.
        embedding_settings = job_settings.model_copy(
            update={
                "ollama_embedding_model": pin.revision.embedding.model if pin else "unavailable",
            }
        )
        retrieval = KnowledgeRetrievalService(repository, create_model_service(embedding_settings))
        service = HybridRetrievalService(retrieval, planner_model, config)
        for case in cases:
            values = dict(case)
            case_id = values.pop("id")
            expected_actions = values.pop("expected_actions", [])
            expected_all_actions = values.pop("expected_all_actions", [])
            expected_status = values.pop("expected_status", None)
            expected_refs = values.pop("expected_any_reference", [])
            expected_all_refs = values.pop("expected_all_references", [])
            seed_query = values.pop("seed_query", None)
            req = PlannerRequest(
                user_id=args.owner_user_id or uuid4(),
                session_id=uuid4(),
                collection_id=args.collection,
                revision_id=args.revision,
                max_chars=args.max_chars,
                **values,
            )
            if seed_query:
                seed = await retrieval.search(
                    RetrievalRequest(
                        collection_id=args.collection,
                        revision_id=args.revision,
                        query=seed_query,
                    ),
                    scope=scope,
                )
                if seed.status == "found" and seed.revision_id:
                    req = req.model_copy(
                        update={
                            "available": [
                                EvidenceReceipt(
                                    user_id=req.user_id,
                                    session_id=req.session_id,
                                    scope=scope,
                                    collection_id=args.collection,
                                    revision_id=seed.revision_id,
                                    created_at=datetime.now(UTC),
                                    items=seed.items,
                                )
                            ]
                        }
                    )
            result = await service.execute(req, scope=scope)
            action_names = [action.action for action in result.plan.actions] if result.plan else []
            refs = sorted({item.reference_id for item in result.items if item.reference_id})
            passed = (
                result.origin == "model"
                and (not expected_actions or any(name in expected_actions for name in action_names))
                and (not expected_status or result.status == expected_status)
                and (not expected_refs or bool(set(refs).intersection(expected_refs)))
                and set(expected_all_actions).issubset(action_names)
                and set(expected_all_refs).issubset(refs)
            )
            payload = result.model_dump(mode="json")
            if not args.include_evidence:
                for item in payload["items"]:
                    item.pop("content")
                    item.pop("context_prefix")
            print(
                json.dumps(
                    {"case_id": case_id, "passed": passed, "references": refs, "result": payload}
                ),
                flush=True,
            )
            failed += not passed
    return int(bool(failed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=UUID)
    parser.add_argument("--revision", type=UUID)
    parser.add_argument("--scope-domain", required=True)
    parser.add_argument("--owner-user-id", type=UUID)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--question")
    inputs.add_argument("--cases", type=Path)
    parser.add_argument(
        "--specialist",
        choices=["scenario", "process_coach", "ap_plus_navigator"],
        default="process_coach",
    )
    parser.add_argument("--planner-model", required=True)
    parser.add_argument("--current-step")
    parser.add_argument("--namespace")
    parser.add_argument("--phase")
    parser.add_argument("--objective", default="")
    parser.add_argument(
        "--clear-course-query",
        action="store_true",
        help="Trusted assertion permitting search fallback on planner failure.",
    )
    parser.add_argument(
        "--seed-query", help="Fetch fresh evidence before planning, for reuse testing."
    )
    parser.add_argument("--model-url")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--include-evidence", action="store_true")
    args = parser.parse_args()
    try:
        PlannerConfig(model=args.planner_model, timeout_seconds=args.timeout)
        return asyncio.run(execute(args), loop_factory=asyncio.SelectorEventLoop)
    except (KnowledgeError, ModelServiceError, ValueError, OSError, KeyError, TypeError):
        print(
            "Planning failed; verify trusted inputs and read/model configuration.", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

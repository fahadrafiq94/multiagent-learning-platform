"""Real supervisor/reply-node development probes; no teacher/DB/held-out execution."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from app.agents.orchestrator import OrchestratorAgent
from app.config.settings import settings
from app.context.schemas import ScenarioDraft, SessionContextUpdate
from app.identity import UserIdentity
from app.orchestration.factory import build_orchestration_runtime
from app.orchestration.nodes import fallback_response_node, route_request_node
from app.services.model_service import ModelService, create_model_service
from scripts.knowledge_backup import private_directory
from scripts.knowledge_infra import ROOT
from scripts.run_student_simulation import RecordedModel, save_json
from scripts.student_quality import QualitySuite, project_turn, score_case


async def execute(args: argparse.Namespace) -> dict[str, Any]:
    suite, digest = QualitySuite.read(args.suite)
    if suite.split != "development":
        raise ValueError("Only development cases are supported.")
    cases = [
        case
        for case in suite.cases
        if case.finding
        in {
            "routing",
            "acknowledgement",
            "retrieval_need",
        }
        or case.case_id == "question-original"
    ]
    if args.case_id:
        unknown = set(args.case_id) - {case.case_id for case in cases}
        if unknown:
            raise ValueError("Unknown case selection.")
        cases = [case for case in cases if case.case_id in args.case_id]
    archive_bytes = args.archive.read_bytes()
    archive = json.loads(archive_bytes)
    if archive.get("synthetic") is not True or archive.get("run_id") != suite.baseline_run_id:
        raise ValueError("Historical context must come from the declared synthetic run.")
    turns = archive["turns"]
    private_directory(args.output)
    profile = settings.model_copy(
        update={
            "model_provider": "ollama",
            "ollama_base_url": args.model_url,
            "ollama_chat_model": args.model,
            "model_max_retries": 0,
            "model_request_timeout_seconds": 150,
        }
    )
    model = RecordedModel(create_model_service(profile), args.output / "model-calls.jsonl")
    # This component path uses generate/embed, which the instrumentation forwards.
    # It never invokes ModelService health/stream operations on the wrapper.
    runtime_model = cast(ModelService, model)
    report: dict[str, Any] = {
        "suite": str(args.suite),
        "suite_sha256": digest,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "model": args.model,
        "context_window": 8192,
        "reasoning": False,
        "scope": (
            "Real supervisor plus actual route/reply nodes; no specialist, HTTP or source search."
        ),
        "source_code_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in [
                "backend/app/agents/orchestrator.py",
                "backend/app/agents/routing.py",
                "backend/app/context/policies.py",
                "backend/app/services/conversation_reply.py",
                "backend/app/orchestration/nodes.py",
            ]
        },
        "rows": [],
        "semantic_reviews_automatically_passed": False,
    }
    save_json(args.output / "results.json", report)
    for case in cases:
        model.scope = {"case": case.case_id}
        start_calls = len(model.calls)
        runtime = build_orchestration_runtime(model_service=runtime_model)
        user = await runtime.repositories.identities.create(UserIdentity(student_id=uuid4().hex))
        session = await runtime.conversation_service.create_session(user_id=user.user_id)
        card_data: Any = case.context.get("scenario") or {}
        history: Any = case.context.get("history") or []
        step: Any = case.context.get("current_step_id")
        last_agent = "scenario" if case.case_id == "hint-route-original" else None
        if case.baseline_turn:
            index = next(i for i, t in enumerate(turns) if t["id"] == case.baseline_turn)
            if index:
                prior = turns[index - 1]
                card_data = prior.get("user_context", {}).get("scenario_card") or {}
                history = prior.get("history", [])
                step = prior.get("session_context", {}).get("current_step_id")
                last_agent = prior.get("session_context", {}).get("last_selected_agent")
        await runtime.scenario_card_service.confirm_draft(
            user_id=user.user_id,
            draft=ScenarioDraft(
                user_id=user.user_id,
                session_id=session.session_id,
                company_context=card_data.get("company_context")
                or card_data.get("company")
                or "A repair workshop.",
                learner_role=card_data.get("learner_role") or card_data.get("role") or "buyer",
                business_problem=card_data.get("business_problem") or "Missing parts delay work.",
                business_goals=card_data.get("business_goals")
                or card_data.get("goals")
                or ["Reduce delays."],
            ),
        )
        await runtime.session_context_service.update(
            user_id=user.user_id,
            session_id=session.session_id,
            changes=SessionContextUpdate(current_step_id=step, last_selected_agent=last_agent),
        )
        for message in history:
            method = (
                runtime.conversation_service.append_user_message
                if message["role"] == "user"
                else runtime.conversation_service.append_assistant_message
            )
            await method(
                user_id=user.user_id, session_id=session.session_id, content=message["content"]
            )
        current = await runtime.conversation_service.append_user_message(
            user_id=user.user_id, session_id=session.session_id, content=case.message
        )
        context = await runtime.context_service.build(
            user_id=user.user_id,
            session_id=session.session_id,
            current_message=case.message,
            current_message_id=current.message_id,
        )
        row: dict[str, Any] = {"case_id": case.case_id}
        try:
            state = await route_request_node(
                {
                    "user_id": user.user_id,
                    "session_uuid": session.session_id,
                    "student_id": user.student_id,
                    "session_id": str(session.session_id),
                    "user_message": case.message,
                    "base_context": context,
                    "current_message_id": current.message_id,
                    "metadata": {},
                },
                orchestrator=OrchestratorAgent(runtime_model),
                context_policy_service=runtime.context_policy_service,
                session_state_update_service=runtime.session_state_update_service,
                context_service=runtime.context_service,
            )
            if state.get("metadata", {}).get("context_reply_handled"):
                state = fallback_response_node(state)
            metadata = state.get("metadata", {})
            saved_state = await runtime.session_context_service.get_required(
                user_id=user.user_id, session_id=session.session_id
            )
            observed = project_turn(
                {
                    "response": {"route": state.get("route"), "metadata": metadata},
                    "session_context": saved_state.model_dump(mode="json"),
                }
            )
            row.update(
                score=score_case(case, observed),
                interaction=metadata.get("interaction_kind"),
                direct_reply=metadata.get("context_reply_handled", False),
                reply=state.get("agent_response"),
                keyword_fallback_used=metadata.get("keyword_fallback_used"),
                semantic_routing_failed=metadata.get("semantic_routing_failed"),
                saved_step=saved_state.current_step_id,
                student_step_status=metadata.get("student_step_status"),
            )
        except Exception as exc:
            row.update(error_type=type(exc).__name__, score=score_case(case, None))
        row["model_calls"] = model.calls[start_calls:]
        report["rows"].append(row)
        save_json(args.output / "results.json", report)
        print(case.case_id, row["score"]["status"], row.get("interaction"), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "docs/knowledge/student-quality-development-v1.json",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "backend/reports/student-journey-20261106-presentation-2/run.json",
    )
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--model-url", default="http://127.0.0.1:11434")
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(execute(args))


if __name__ == "__main__":
    main()

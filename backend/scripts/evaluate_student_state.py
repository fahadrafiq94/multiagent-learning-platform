"""Bounded real-model development probes for Step 2; no held-out cases or main DB writes."""

import argparse
import asyncio
import hashlib
from pathlib import Path
from uuid import uuid4

from app.agents.orchestrator import OrchestratorAgent
from app.agents.routing import OrchestratorRequest
from app.agents.scenario import ScenarioAgent
from app.config.settings import settings
from app.context.agent_views import ContextTarget
from app.context.schemas import ScenarioDraft, SessionContextUpdate
from app.identity import UserIdentity
from app.orchestration.factory import build_orchestration_runtime
from app.services.model_service import create_model_service
from scripts.knowledge_backup import private_directory
from scripts.knowledge_infra import ROOT
from scripts.run_student_simulation import RecordedModel, save_json
from scripts.student_quality import QualitySuite, score_case


async def execute(args):
    suite, digest = QualitySuite.read(ROOT / "docs/knowledge/student-quality-development-v1.json")
    if suite.split != "development":
        raise ValueError("Only development cases may be executed by this tool.")
    cases = [case for case in suite.cases if case.finding in {"extraction", "saved_state"}]
    if args.finding:
        cases = [case for case in cases if case.finding in args.finding]
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
    report = {
        "suite_sha256": digest,
        "model": args.model,
        "reasoning": False,
        "context_window": 8192,
        "max_output_tokens": 1024,
        "execution": "Real model; isolated in-memory repositories; component probes, not HTTP.",
        "rows": [],
        "semantic_reviews_automatically_passed": False,
        "source_code_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "backend/app/agents/scenario.py",
                "backend/app/agents/orchestrator.py",
                "backend/app/agents/routing.py",
                "backend/app/services/session_state_update_service.py",
            )
        },
    }
    save_json(args.output / "results.json", report)
    for case in cases:
        model.scope = {"case": case.case_id}
        runtime = build_orchestration_runtime(model_service=model)
        user = await runtime.repositories.identities.create(UserIdentity(student_id=uuid4().hex))
        session = await runtime.conversation_service.create_session(user_id=user.user_id)
        message = await runtime.conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=case.message,
        )
        before_calls = len(model.calls)
        row = {"case_id": case.case_id}
        try:
            if case.finding == "extraction":
                await runtime.scenario_onboarding_service.start(
                    user_id=user.user_id, session_id=session.session_id
                )
                if case.context.get("draft"):
                    data = case.context["draft"]
                    await runtime.scenario_onboarding_service.save_draft(
                        user_id=user.user_id,
                        session_id=session.session_id,
                        draft=ScenarioDraft(
                            user_id=user.user_id,
                            session_id=session.session_id,
                            company_context=data.get("company"),
                            learner_role=data.get("learner_role"),
                        ),
                    )
                context = await runtime.context_service.build(
                    user_id=user.user_id,
                    session_id=session.session_id,
                    current_message=case.message,
                    current_message_id=message.message_id,
                )
                view = runtime.context_policy_service.select(
                    target=ContextTarget.SCENARIO, context=context
                )
                draft = await ScenarioAgent(model).extract_draft(
                    context=view, source_message_id=message.message_id
                )
                await runtime.scenario_onboarding_service.save_draft(
                    user_id=user.user_id, session_id=session.session_id, draft=draft
                )
                stored = await runtime.scenario_onboarding_service.get_draft(
                    user_id=user.user_id, session_id=session.session_id
                )
                observed = {"draft.learner_role": stored.learner_role}
                row["draft"] = stored.model_dump(mode="json")
                row["source_recorded"] = message.message_id in stored.source_message_ids
            else:
                await runtime.scenario_card_service.confirm_draft(
                    user_id=user.user_id,
                    draft=ScenarioDraft(
                        user_id=user.user_id,
                        session_id=session.session_id,
                        company_context="A synthetic repair workshop.",
                        learner_role="buyer",
                        business_problem="Parts arrive late.",
                        business_goals=["Reduce delays."],
                    ),
                )
                current = await runtime.session_context_service.get_required(
                    user_id=user.user_id, session_id=session.session_id
                )
                if case.context.get("current_step_id"):
                    await runtime.session_context_service.update(
                        user_id=user.user_id,
                        session_id=session.session_id,
                        changes=SessionContextUpdate(
                            current_step_id=case.context["current_step_id"]
                        ),
                        expected_version=current.version,
                    )
                context = await runtime.context_service.build(
                    user_id=user.user_id,
                    session_id=session.session_id,
                    current_message=case.message,
                    current_message_id=message.message_id,
                )
                view = runtime.context_policy_service.select(
                    target=ContextTarget.ORCHESTRATOR, context=context
                )
                routed = await OrchestratorAgent(model).route(
                    OrchestratorRequest(
                        session_id=str(session.session_id), user_message=case.message, context=view
                    )
                )
                if routed.decision.current_step is None:
                    status = (
                        "rejected"
                        if "current_step" in routed.decision.model_fields_set
                        else "omitted"
                    )
                else:
                    status, _ = await runtime.session_state_update_service.apply_student_step(
                        user_id=user.user_id,
                        session_id=session.session_id,
                        message_id=message.message_id,
                        current_message=case.message,
                        decision=routed.decision.current_step,
                        expected_version=context.session_context.version,
                    )
                stored = await runtime.session_context_service.get_required(
                    user_id=user.user_id, session_id=session.session_id
                )
                user_context = await runtime.user_context_service.get_or_create(
                    user_id=user.user_id
                )
                observed = {
                    "state.current_step_id": stored.current_step_id,
                    "progress.completed_steps": sorted(
                        step.step_id
                        for step in user_context.progress.steps
                        if step.status.value == "completed"
                    ),
                }
                row.update(
                    decision=routed.decision.model_dump(mode="json"), persistence_status=status
                )
            row["score"] = score_case(case, observed)
        except Exception as exc:
            row.update(error_type=type(exc).__name__, score=score_case(case, None))
        row["model_calls"] = model.calls[before_calls:]
        report["rows"].append(row)
        save_json(args.output / "results.json", report)
        print(case.case_id, row["score"]["status"], flush=True)
    report["all_observed_contracts_passed"] = all(
        row["score"]["checks"]
        and all(check["status"] == "pass" for check in row["score"]["checks"])
        and row.get("persistence_status", "unchanged") in {"persisted", "unchanged"}
        for row in report["rows"]
    )
    save_json(args.output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--model-url", default="http://127.0.0.1:11434")
    parser.add_argument("--finding", choices=["extraction", "saved_state"], action="append")
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(execute(args))


if __name__ == "__main__":
    main()

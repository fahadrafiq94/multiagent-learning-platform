"""Real-model development teaching/assessment probes; no HTTP/DB/search or held-out use."""

import argparse
import asyncio
import hashlib
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from app.agents.factory import create_agent_registry
from app.config.settings import Settings
from app.context.schemas import ScenarioDraft
from app.domain.guidance import GuidanceLevel
from app.identity import UserIdentity
from app.knowledge.integration import GroundingAdmission
from app.orchestration.factory import build_orchestration_runtime
from app.orchestration.nodes import execute_agent_node, record_response_signal_node
from app.services.model_service import ModelService, create_model_service
from scripts.evaluate_citation_support import ProbeModel, agent_request
from scripts.knowledge_backup import private_directory
from scripts.knowledge_infra import ROOT
from scripts.run_student_simulation import save_json
from scripts.student_quality import QualityCase, QualitySuite, score_case


class SyntheticGrounding:
    def __init__(self, texts: list[str]):
        self.context = agent_request(texts).context.grounding

    async def prepare(self, context, specialist, *, current_message_id):
        return GroundingAdmission(self.context, {"status": "found", "planner_calls": 0})

    async def authorize(self, admission):
        # These are declared synthetic component fixtures, not a real ACL test.
        return None


async def execute(args: argparse.Namespace) -> dict[str, Any]:
    paths = [
        ROOT / "docs/knowledge/student-quality-development-v1.json",
        ROOT / "docs/knowledge/student-assessment-development-v1.json",
    ]
    cases: list[QualityCase] = []
    digests = {}
    for path in paths:
        suite, digest = QualitySuite.read(path)
        if suite.split != "development":
            raise ValueError("Development probes only.")
        digests[str(path.relative_to(ROOT))] = digest
        cases.extend(
            case
            for case in suite.cases
            if case.stage == "controlled_teaching" or case.finding == "assessment"
        )
    if args.case_id:
        if set(args.case_id) - {c.case_id for c in cases}:
            raise ValueError("Unknown case selection.")
        cases = [case for case in cases if case.case_id in args.case_id]
    cases.sort(key=lambda case: case.finding != "assessment")
    private_directory(args.output)
    model = ProbeModel(
        create_model_service(
            Settings(
                _env_file=None,
                model_provider="ollama",
                ollama_base_url=args.model_url,
                ollama_chat_model=args.model,
                ollama_embedding_model="nomic-embed-text",
                model_max_retries=0,
                model_request_timeout_seconds=150,
                model_max_tokens=1024,
                model_temperature=0,
            )
        ),
        args.output / "model-calls.jsonl",
    )
    runtime_model = cast(ModelService, model)
    report: dict[str, Any] = {
        "scope": (
            "Real specialists/shared source-teaching verifier and in-memory signal persistence; "
            "no routing, HTTP, SQL, source search or natural adaptation."
        ),
        "split": "development",
        "synthetic": True,
        "model": args.model,
        "context_window": 8192,
        "generation_tokens": 1024,
        "reasoning": False,
        "verification_tokens": 512,
        "verification_timeout_seconds": 30,
        "temperature": 0,
        "assessment_level_override": getattr(args, "assessment_level", None),
        "provider_retries": 0,
        "fixture_hashes": digests,
        "code_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "backend/app/agents/assessment_guard.py",
                "backend/app/agents/schemas.py",
                "backend/app/agents/policies.py",
                "backend/app/agents/grounding.py",
                "backend/app/agents/citation_support.py",
                "backend/app/domain/guidance_policy.py",
                "backend/app/orchestration/nodes.py",
                "backend/scripts/evaluate_student_teaching.py",
            ]
        },
        "rows": [],
        "semantic_review_status": "review_pending",
    }
    save_json(args.output / "results.json", report)
    for case in cases:
        model.case_id = case.case_id
        begin = len(model.calls)
        runtime = build_orchestration_runtime(model_service=runtime_model)
        user = await runtime.repositories.identities.create(UserIdentity(student_id=uuid4().hex))
        session = await runtime.conversation_service.create_session(user_id=user.user_id)
        scenario = cast(dict[str, Any], case.context.get("scenario") or {})
        await runtime.scenario_card_service.confirm_draft(
            user_id=user.user_id,
            draft=ScenarioDraft(
                user_id=user.user_id,
                session_id=session.session_id,
                company_context=scenario.get("company", "Workshop training team"),
                learner_role=scenario.get("role", "Logistics coordinator"),
                business_problem="Untraceable equipment handovers.",
                business_goals=scenario.get("goals", ["Reduce untraceable handovers."]),
            ),
        )
        context = await runtime.context_service.build(
            user_id=user.user_id, session_id=session.session_id, current_message=case.message
        )
        level = GuidanceLevel(cast(str, case.context.get("guidance", "minimal")))
        if case.finding == "assessment" and getattr(args, "assessment_level", None):
            level = GuidanceLevel(args.assessment_level)
        context.guidance_level = level  # Only declared controlled probes, never runtime defaults.
        specialist = cast(
            Any,
            case.context.get(
                "specialist",
                "process_coach"
                if case.case_id not in {"question-original", "question-paraphrase"}
                else "scenario",
            ),
        )
        texts = [
            cast(dict[str, Any], item)["content"]
            for item in cast(list, case.context.get("evidence", []))
        ]
        if not texts:
            texts = [
                "The workshop exercise concerns equipment handovers; "
                "it specifies no company-specific goals here."
            ]
        before = await runtime.user_context_service.get_required(user_id=user.user_id)
        try:
            state = await execute_agent_node(
                {
                    "user_id": user.user_id,
                    "session_uuid": session.session_id,
                    "session_id": str(session.session_id),
                    "student_id": user.student_id,
                    "user_message": case.message,
                    "base_context": context,
                    "metadata": {},
                },
                agent_registry=create_agent_registry(model_service=runtime_model),
                agent_name=specialist,
                context_policy_service=runtime.context_policy_service,
                grounding_service=cast(Any, SyntheticGrounding(texts)),
            )
            state = await record_response_signal_node(
                state, user_context_service=runtime.user_context_service
            )
            after = await runtime.user_context_service.get_required(user_id=user.user_id)
            assessment = state.get("response_assessment")
            observed = (
                {}
                if state.get("error")
                else {
                    "guidance.used": state.get("metadata", {}).get("guidance_level_used"),
                    "assessment.understanding": assessment.understanding.value
                    if assessment is not None
                    else None,
                    "assessment.signal_recorded": state.get("metadata", {}).get(
                        "response_signal_recorded"
                    ),
                }
            )
            if state.get("metadata", {}).get("rag", {}).get("learning_signal_suppressed"):
                observed.pop("assessment.understanding", None)
            row = {
                "case_id": case.case_id,
                "agent": specialist,
                "level": level.value,
                "content": state.get("agent_response"),
                "error": state.get("error"),
                "metadata": state.get("metadata"),
                "before": before.hint_state.model_dump(mode="json"),
                "after": after.hint_state.model_dump(mode="json"),
                "observed": observed,
                "score": score_case(case, observed),
            }
        except Exception as exc:
            row = {
                "case_id": case.case_id,
                "failure_type": type(exc).__name__,
                "score": score_case(case, None),
            }
        row["calls"] = len(model.calls) - begin
        report["rows"].append(row)
        save_json(args.output / "results.json", report)
        print(case.case_id, row["score"]["status"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--model-url", default="http://127.0.0.1:11434")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--assessment-level", choices=[level.value for level in GuidanceLevel])
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(execute(args))


if __name__ == "__main__":
    main()

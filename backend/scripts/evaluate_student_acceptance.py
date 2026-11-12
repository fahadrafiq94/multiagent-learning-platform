"""One-time execution of the reserved synthetic pack on a hash-frozen candidate.

Uses real extraction, routing/reply, specialist, verification and signal nodes.
Component probes use isolated in-memory storage and declared synthetic passages;
the separate signed HTTP journey covers PostgreSQL, real search and persistence.
Gold checks/reviews are used only after execution and never enter model requests.
"""

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from app.agents.factory import create_agent_registry
from app.agents.orchestrator import OrchestratorAgent
from app.agents.scenario import ScenarioAgent
from app.config.settings import Settings
from app.context.agent_views import ContextTarget
from app.context.schemas import ScenarioDraft, SessionContextUpdate
from app.domain.guidance import GuidanceLevel
from app.identity import UserIdentity
from app.orchestration.factory import build_orchestration_runtime
from app.orchestration.nodes import (
    execute_agent_node,
    fallback_response_node,
    record_response_signal_node,
    route_request_node,
)
from app.services.model_service import ModelService, create_model_service
from app.services.model_service.schemas import ModelChatRequest
from scripts.evaluate_citation_support import ProbeModel
from scripts.evaluate_student_teaching import SyntheticGrounding
from scripts.knowledge_backup import private_directory
from scripts.knowledge_infra import ROOT
from scripts.run_student_simulation import save_json
from scripts.student_quality import QualityCase, QualitySuite, project_turn, score_case

MANIFEST = ROOT / "docs/knowledge/student-quality-reserved-v1-manifest.json"


class AcceptanceModel(ProbeModel):
    """Apply the frozen serving controls to every component, including onboarding."""

    async def generate(self, request: ModelChatRequest):
        return await super().generate(
            request.model_copy(
                update={
                    "context_window_tokens": 8192,
                    "reasoning_enabled": False,
                    "max_tokens": min(request.max_tokens or 1024, 1024),
                }
            )
        )


def execution_split(manifest_status: str, *, consumed_regression: bool) -> str:
    if consumed_regression:
        if manifest_status != "consumed":
            raise ValueError("Regression replay requires an already consumed pack.")
        return "consumed_regression"
    if manifest_status != "reserved_unexecuted":
        raise ValueError("Reserved pack already consumed; preserve its original acceptance run.")
    return "heldout"


def verify_snapshot(snapshot: dict[str, Any]) -> None:
    mismatches = [
        name
        for name, digest in snapshot["files"].items()
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    if mismatches:
        raise ValueError("Frozen candidate changed: " + ", ".join(mismatches))


async def observe(case: QualityCase, model: ModelService) -> dict[str, Any]:
    runtime = build_orchestration_runtime(model_service=model)
    user = await runtime.repositories.identities.create(UserIdentity(student_id=uuid4().hex))
    session = await runtime.conversation_service.create_session(user_id=user.user_id)
    uid, sid = user.user_id, session.session_id
    values: Any = case.context
    if case.stage == "onboarding":
        await runtime.scenario_onboarding_service.start(user_id=uid, session_id=sid)
        prior = values.get("draft", {})
        if prior:
            await runtime.scenario_onboarding_service.save_draft(
                user_id=uid,
                session_id=sid,
                draft=ScenarioDraft(
                    user_id=uid,
                    session_id=sid,
                    company_context=prior.get("company"),
                    learner_role=prior.get("learner_role"),
                ),
            )
    else:
        scenario = values.get("scenario", {})
        await runtime.scenario_card_service.confirm_draft(
            user_id=uid,
            draft=ScenarioDraft(
                user_id=uid,
                session_id=sid,
                company_context=scenario.get("company", "Synthetic workshop"),
                learner_role=scenario.get("role", "Quality inspector"),
                business_problem=scenario.get(
                    "business_problem", "Inspection rework delays dispatch."
                ),
                business_goals=scenario.get("goals", ["Reduce inspection rework."]),
            ),
        )
        await runtime.session_context_service.update(
            user_id=uid,
            session_id=sid,
            changes=SessionContextUpdate(current_step_id=values.get("current_step_id")),
        )
    for item in values.get("history", []):
        append = (
            runtime.conversation_service.append_user_message
            if item["role"] == "user"
            else (runtime.conversation_service.append_assistant_message)
        )
        await append(user_id=uid, session_id=sid, content=item["content"])
    message = await runtime.conversation_service.append_user_message(
        user_id=uid, session_id=sid, content=case.message
    )
    context = await runtime.context_service.build(
        user_id=uid,
        session_id=sid,
        current_message=case.message,
        current_message_id=message.message_id,
    )
    if case.stage == "onboarding":
        view = runtime.context_policy_service.select(target=ContextTarget.SCENARIO, context=context)
        draft = await ScenarioAgent(model).extract_draft(
            context=view, source_message_id=message.message_id
        )
        await runtime.scenario_onboarding_service.save_draft(
            user_id=uid, session_id=sid, draft=draft
        )
        stored = await runtime.scenario_onboarding_service.get_draft(user_id=uid, session_id=sid)
        if stored is None:
            raise ValueError("Extraction did not persist a draft.")
        return {
            "observed": {"draft.learner_role": stored.learner_role},
            "draft": stored.model_dump(mode="json"),
        }
    before = await runtime.user_context_service.get_required(user_id=uid)
    state: Any = {
        "user_id": uid,
        "session_uuid": sid,
        "session_id": str(sid),
        "student_id": user.student_id,
        "user_message": case.message,
        "base_context": context,
        "current_message_id": message.message_id,
        "metadata": {},
    }
    teacher = case.finding in {"assessment", "guidance", "citation_support"}
    if teacher:
        # Defined before execution: unspecified assessment level is minimal;
        # citation probes use explicit to exercise substantive supported claims.
        level = GuidanceLevel(
            values.get("guidance", "explicit" if case.finding == "citation_support" else "minimal")
        )
        context.guidance_level = level
        texts = [item["content"] for item in values.get("evidence", [])]
        if not texts:
            texts = ["This synthetic workshop exercise provides no additional process facts."]
        specialist = values.get(
            "specialist",
            "ap_plus_navigator" if case.finding == "citation_support" else "process_coach",
        )
        state = await execute_agent_node(
            state,
            agent_registry=create_agent_registry(model_service=model),
            agent_name=specialist,
            context_policy_service=runtime.context_policy_service,
            grounding_service=cast(Any, SyntheticGrounding(texts)),
        )
        state = await record_response_signal_node(
            state, user_context_service=runtime.user_context_service
        )
    else:
        state = await route_request_node(
            state,
            orchestrator=OrchestratorAgent(model),
            context_policy_service=runtime.context_policy_service,
            session_state_update_service=runtime.session_state_update_service,
            context_service=runtime.context_service,
        )
        if state.get("metadata", {}).get("context_reply_handled"):
            state = fallback_response_node(state)
    saved = await runtime.session_context_service.get_required(user_id=uid, session_id=sid)
    after = await runtime.user_context_service.get_required(user_id=uid)
    metadata = state.get("metadata", {})
    refs = state.get("source_references", [])
    observed = project_turn(
        {
            "response": {"route": state.get("route"), "metadata": metadata},
            "session_context": saved.model_dump(mode="json"),
            "user_context": after.model_dump(mode="json"),
        }
    )
    assessment = state.get("response_assessment")
    if teacher:
        observed["assessment.understanding"] = (
            assessment.understanding.value if assessment else None
        )
        observed["assessment.signal_recorded"] = metadata.get("response_signal_recorded", False)
        observed["guidance.used"] = metadata.get("guidance_level_used")
        allowed = {f"K{i}" for i in range(1, len(values.get("evidence", [])) + 1)}
        observed["citations.identity_valid"] = bool(refs) and all(
            ref.citation_label in allowed for ref in refs
        )
    return {
        "observed": observed,
        "content": state.get("agent_response"),
        "error": state.get("error"),
        "metadata": metadata,
        "before": before.hint_state.model_dump(mode="json"),
        "after": after.hint_state.model_dump(mode="json"),
        "source_references": [ref.model_dump(mode="json") for ref in refs],
    }


async def execute(args):
    args.output = args.output.absolute()
    snapshot = json.loads(args.snapshot.read_bytes())
    verify_snapshot(snapshot)
    manifest = json.loads(MANIFEST.read_bytes())
    suite, digest = QualitySuite.read(ROOT / manifest["private_path"])
    if digest != manifest["sha256"] or suite.split != "heldout":
        raise ValueError("Reserved fixture identity mismatch.")
    split = execution_split(
        manifest["status"], consumed_regression=getattr(args, "consumed_regression", False)
    )
    private_directory(args.output)
    model = AcceptanceModel(
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
    report = {
        "split": split,
        "synthetic": True,
        "fixture_sha256": digest,
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "model": args.model,
        "context_window": 8192,
        "reasoning": False,
        "generation_tokens": 1024,
        "verification_tokens": 512,
        "provider_retries": 0,
        "started_at": datetime.now(UTC).isoformat(),
        "scope": "Real component nodes; in-memory storage; synthetic evidence; not HTTP/search.",
        "reserved_pack_review": "Prior Phase 0 pedagogical review retained; labels unchanged.",
        "status": "running",
        "rows": [],
        "semantic_review_status": "review_pending",
    }
    if split == "heldout":
        manifest.update(
            status="consumed_execution_started",
            acceptance_output=str(args.output.relative_to(ROOT)),
            model_calls=0,
        )
        save_json(MANIFEST, manifest)  # Interruption also consumes a previously fresh pack.
    save_json(args.output / "results.json", report)
    try:
        for case in suite.cases:
            model.case_id = case.case_id
            begin = len(model.calls)
            try:
                row = await observe(case, cast(ModelService, model))
                row["score"] = score_case(case, None if row.get("error") else row["observed"])
            except Exception as exc:
                row = {"failure_type": type(exc).__name__, "score": score_case(case, None)}
            row.update(case_id=case.case_id, calls=len(model.calls) - begin)
            report["rows"].append(row)
            save_json(args.output / "results.json", report)
            print(case.case_id, row["score"]["status"], flush=True)
        verify_snapshot(snapshot)
        report["status"] = "completed_review_pending"
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["model_calls"] = len(model.calls)
        if split == "heldout":
            manifest.update(
                status="consumed", model_calls=len(model.calls), acceptance_status=report["status"]
            )
            save_json(MANIFEST, manifest)
        save_json(args.output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--model-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--consumed-regression",
        action="store_true",
        help="Replay consumed cases as regression evidence; preserves the original manifest/run.",
    )
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(execute(args))


if __name__ == "__main__":
    main()

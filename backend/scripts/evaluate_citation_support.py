"""Real-model development probes for Step 5; no DB, HTTP, retrieval or held-out claims."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

from app.agents.ap_plus_navigator import APPlusNavigatorAgent
from app.agents.grounding import build_specialist_request
from app.agents.process_coach import ProcessCoachAgent
from app.agents.scenario import ScenarioAgent
from app.agents.schemas import AgentRequest
from app.config.settings import Settings
from app.context.agent_views import AgentContextView, ContextTarget
from app.context.grounding import EvidenceContext, GroundingContext
from app.context.schemas import MessageSourceReference
from app.domain.guidance import GuidanceLevel
from app.services.model_service import ModelService, create_model_service
from app.services.model_service.schemas import ModelChatRequest, ModelChatResponse
from scripts.knowledge_backup import private_directory
from scripts.knowledge_infra import ROOT
from scripts.run_student_simulation import save_json

FIXTURE = ROOT / "docs/knowledge/student-citation-support-development-v1.json"


def read_suite(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_bytes())
    if (
        data.get("split") != "development"
        or data.get("synthetic") is not True
        or data.get("version") != "student-citation-support-development-v1"
    ):
        raise ValueError("Only the declared synthetic development suite is supported.")
    return cast(dict[str, Any], data)


class ProbeModel:
    def __init__(self, service: ModelService, output: Path):
        self.service = service
        self.output = output
        self.calls: list[dict[str, Any]] = []
        self.case_id = ""

    async def generate(self, request: ModelChatRequest) -> ModelChatResponse:
        started = time.perf_counter()
        row: dict[str, Any] = {
            "case_id": self.case_id,
            "request": request.model_dump(mode="json"),
        }
        try:
            response = await self.service.generate(request)
            row["response"] = response.model_dump(mode="json")
            return response
        except BaseException as exc:
            row["failure_type"] = type(exc).__name__
            raise
        finally:
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.calls.append(row)
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def agent_request(texts: list[str], *, agent="process_coach", question="Check the answer."):
    view = AgentContextView(
        user_id=uuid5(NAMESPACE_URL, "step5-probe-user"),
        session_id=uuid5(NAMESPACE_URL, "step5-probe-session"),
        target=ContextTarget(agent),
        current_message=question,
        policy_version="step5-controlled-development",
    )
    view.grounding = GroundingContext(
        status="found",
        max_context_tokens=8192,
        reserved_output_tokens=1024,
        reasoning_enabled=False,
        items=[
            EvidenceContext(
                evidence_id=f"synthetic-{index}",
                content=text,
                references=[
                    MessageSourceReference(
                        document_id="synthetic-support-doc",
                        revision_id="synthetic-v1",
                        chunk_id=f"synthetic-{index}",
                        evidence_id=f"synthetic-{index}",
                        title="Synthetic citation support development passage",
                        page_start=index,
                        page_end=index,
                    )
                ],
            )
            for index, text in enumerate(texts, 1)
        ],
    )
    return AgentRequest(session_id=str(view.session_id), user_message=question, context=view)


async def execute(args: argparse.Namespace) -> dict[str, Any]:
    suite = read_suite(args.suite)
    cases = suite["cases"]
    if args.case_id:
        if set(args.case_id) - {case["case_id"] for case in cases}:
            raise ValueError("Unknown development case selection.")
        cases = [case for case in cases if case["case_id"] in args.case_id]
    private_directory(args.output)
    service = create_model_service(
        Settings(
            _env_file=None,
            model_provider="ollama",
            ollama_base_url=args.model_url,
            ollama_chat_model=args.model,
            ollama_embedding_model="nomic-embed-text",  # Factory requires a name; never invoked.
            model_max_retries=0,
            model_request_timeout_seconds=150,
            model_max_tokens=1024,
            model_temperature=0,
        )
    )
    model = ProbeModel(service, args.output / "model-calls.jsonl")
    runtime_model = cast(ModelService, model)
    report: dict[str, Any] = {
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "split": "development",
        "synthetic": True,
        "scope": (
            "Admitted synthetic passages; real verifier and three specialists. No DB/HTTP/search."
        ),
        "model": args.model,
        "model_url": args.model_url,
        "context_window": 8192,
        "reasoning": False,
        "generation_temperature": 0,
        "selected_case_ids": [case["case_id"] for case in cases],
        "skip_generation": args.skip_generation,
        "verifier_timeout_seconds": 30,
        "verifier_max_tokens": 512,
        "provider_retries": 0,
        "controlled_answer_guidance": "explicit",
        "controlled_answer_specialist": "ap_plus_navigator",
        "semantic_review_status": "review_pending",
        "source_code_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "backend/app/agents/citation_support.py",
                "backend/app/agents/grounding.py",
                "backend/app/agents/process_coach.py",
                "backend/app/agents/ap_plus_navigator.py",
                "backend/app/agents/scenario.py",
                "backend/scripts/evaluate_citation_support.py",
            ]
        },
        "cases": [],
        "generation_probes": [],
    }
    save_json(args.output / "results.json", report)
    for case in cases:
        model.case_id = case["case_id"]
        # These are fixed factual source-support tests, not minimal teaching
        # probes. Keep a format rejection from bypassing the semantic checker.
        controlled_request = agent_request(case["evidence"], agent="ap_plus_navigator")
        controlled_request.guidance_level = GuidanceLevel.EXPLICIT
        controlled_request.context.guidance_level = GuidanceLevel.EXPLICIT
        prepared = build_specialist_request(
            controlled_request,
            "Check supplied evidence only.",
            model=args.model,
            task="controlled-answer",
            agent="ap_plus_navigator",
        )
        output = prepared.parse(
            json.dumps(
                {
                    "content": case["answer"],
                    "claims": case["claims"],
                    "citation_ids": case["citation_ids"],
                }
            )
        )
        verified = await prepared.verify(output, runtime_model)
        trace = prepared.support_trace
        # Infrastructure failure/uncertainty cannot count as semantic detection.
        observed = trace["support_status"] in {"passed", "unsupported"}
        matches = observed and (
            trace["support_status"] == "passed"
            if case["expected"] == "passed"
            else trace["support_status"] == "unsupported"
        )
        report["cases"].append(
            {
                "case_id": case["case_id"],
                "expected": case["expected"],
                "observed_comparable_verdict": observed,
                "contract_match": matches,
                "trace": trace,
                "final_content": verified.content,
                "reference_count": len(prepared.references(verified)),
            }
        )
        save_json(args.output / "results.json", report)
        print(case["case_id"], trace["support_status"], flush=True)
    agents: dict[
        str, type[ScenarioAgent] | type[ProcessCoachAgent] | type[APPlusNavigatorAgent]
    ] = {
        "scenario": ScenarioAgent,
        "process_coach": ProcessCoachAgent,
        "ap_plus_navigator": APPlusNavigatorAgent,
    }
    for probe in [] if args.skip_generation else suite["generation_probes"]:
        model.case_id = "generation-" + probe["agent"]
        req = agent_request(probe["evidence"], agent=probe["agent"], question=probe["question"])
        req.guidance_level = GuidanceLevel(probe["guidance"])
        before = len(model.calls)
        try:
            result = await agents[probe["agent"]](runtime_model, model=args.model).execute(req)
            row = {
                "agent": probe["agent"],
                "response": result.model_dump(mode="json"),
                "semantic_review_status": "review_pending",
            }
        except Exception as exc:
            row = {
                "agent": probe["agent"],
                "failure_type": type(exc).__name__,
                "semantic_review_status": "not_observed",
            }
        row["calls"] = len(model.calls) - before
        report["generation_probes"].append(row)
        save_json(args.output / "results.json", report)
        print(model.case_id, row.get("failure_type", "completed"), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--model-url", default="http://127.0.0.1:11434")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(execute(args))


if __name__ == "__main__":
    main()

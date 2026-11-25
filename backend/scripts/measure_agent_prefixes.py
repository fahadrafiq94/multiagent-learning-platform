"""Probe real agent calls with two synthetic students and disposable in-memory repositories."""

import asyncio
import json
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from os.path import commonprefix
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx

from app.context.schemas import ScenarioDraft
from app.identity import UserIdentity
from app.orchestration.factory import build_orchestration_runtime
from app.orchestration.schemas import OrchestrationRequest
from app.services.model_service.factory import create_model_service
from app.services.model_service.vllm_provider import VLLMModelProvider
from scripts.measure_prefix_baseline import counters

TASK_AGENT = {
    "semantic_routing": "Supervisor",
    "scenario_draft_extraction": "Scenario Agent",
    "process_coach_response": "Process Coach",
    "ap_plus_navigator_response": "APplus Navigator",
    "scenario_agent_response": "Scenario Agent",
}
QUESTIONS = {
    "explore": [
        "Suggest two companies for my manufacturing exercise.",
        "Can you suggest two company ideas for practicing ERP?",
    ],
    "process": [
        "Why must I create a supplier before placing a purchase order?",
        "Why does the supplier record have to exist before I enter a purchase order?",
    ],
    "navigation": [
        "Where in the APplus interface do I find the supplier address field?",
        "Which APplus screen or tab should I open to find a supplier address?",
    ],
}


def specialist_difference(first: str, second: str) -> str:
    """Locate the first changed context field without retaining its value."""
    if first == second:
        return "after system message"
    offset = len(commonprefix([first, second]))
    marker = "\n\nFILTERED RUNTIME CONTEXT:\n"
    context = second.find(marker)
    if context < 0 or offset < context:
        return "before runtime context"
    opening = second.find("\n\n{", context)
    if opening < 0:
        return "runtime context header"
    decoder = json.JSONDecoder()
    start = opening + 2
    obj, end = decoder.raw_decode(second, start)
    if offset >= end:
        return "after runtime context"
    # The renderer writes indented JSON, one field per line. Record only schema keys.
    path: dict[int, str] = {}
    for line in second[start : second.find("\n", offset)].splitlines():
        stripped = line.lstrip()
        if not stripped.startswith('"'):
            continue
        name, pos = decoder.raw_decode(stripped)
        if not stripped[pos:].startswith(":"):
            continue
        depth = len(line) - len(stripped)
        path = {k: v for k, v in path.items() if k < depth}
        path[depth] = name
    return (
        ".".join(path.values()) if path and next(iter(path.values())) in obj else "runtime context"
    )


async def main():
    service = create_model_service()
    runtime = build_orchestration_runtime(model_service=service)
    provider_generate = VLLMModelProvider.generate
    provider_json = VLLMModelProvider._json
    call = ContextVar("call")
    baseline = {}
    systems = {}
    salts = {}
    rows = []
    turns = []
    directory = Path(__file__).resolve().parents[1] / "reports"
    directory.mkdir(exist_ok=True)
    report = directory / f"agent-cache-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"

    def save():
        report.write_text(
            json.dumps(
                {
                    "workload": "Serial orchestration, synthetic in-memory data, RAG disabled. "
                    "Fresh cache salt per task pair. No prompts or token IDs retained.",
                    "calls": rows,
                    "turns": turns,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    async def generate(provider, request):
        task = request.metadata.get("task", "unknown")
        marker = call.set({"task": task, "policy": request.metadata.get("policy_version")})
        try:
            return await provider_generate(provider, request)
        finally:
            call.reset(marker)

    async def observe(provider, method, url, payload=None, **kwargs):
        if not url.endswith("/chat/completions"):
            data = await provider_json(provider, method, url, payload, **kwargs)
            if url.endswith("/tokenize"):
                call.get()["tokens"] = data["tokens"]
            return data
        state = call.get()
        model = payload["model"]
        task = state["task"]
        index = sum(
            r["student"] == student and r["phase"] == phase and r["task"] == task for r in rows
        )
        key = (phase, task, index, provider.server_url, model)
        tokens = state["tokens"]
        system = payload["messages"][0]["content"]
        difference = None
        reference = baseline.get(key)
        common = None
        if student == "A":
            baseline[key] = tokens
            systems[key] = system
        elif reference is not None:
            if task in {"process_coach_response", "ap_plus_navigator_response"}:
                difference = specialist_difference(systems[key], system)
            common = next(
                (i for i, (a, b) in enumerate(zip(tokens, reference, strict=False)) if a != b),
                min(len(tokens), len(reference)),
            )

        with httpx.Client(base_url=provider.server_url, timeout=15, trust_env=False) as metrics:
            before = counters(metrics, model)
            start = time.perf_counter()
            data = await provider_json(
                provider,
                method,
                url,
                {**payload, "cache_salt": salts.setdefault(key, uuid4().hex)},
                **kwargs,
            )
            duration = round((time.perf_counter() - start) * 1000)
            for attempt in range(16):
                after = counters(metrics, model)
                delta = {k: after[k] - v for k, v in before.items() if k in after}
                if delta.get("request_success_total", 0) >= 1:
                    break
                if attempt < 15:
                    await asyncio.sleep(1)
        usage = data["usage"]
        reported = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        clean = (
            delta.get("request_success_total") == 1
            and delta.get("prefix_cache_queries_total") == usage["prompt_tokens"]
        )
        rows.append(
            dict(
                student=student,
                phase=phase,
                task=task,
                agent=TASK_AGENT.get(task, task),
                model=model,
                policy=state["policy"],
                prompt_tokens=usage["prompt_tokens"],
                tokenizer_matches_usage=len(tokens) == usage["prompt_tokens"],
                common_prefix_with_A=common,
                first_difference=difference,
                output_tokens=usage["completion_tokens"],
                cached_tokens=reported
                if reported is not None
                else delta.get("prefix_cache_hits_total")
                if clean
                else None,
                cached_source="response"
                if reported is not None
                else "isolated counters"
                if clean
                else "ambiguous",
                elapsed_ms=duration,
                server_delta=delta,
            )
        )
        save()
        return data

    try:
        with (
            patch.object(VLLMModelProvider, "generate", generate),
            patch.object(VLLMModelProvider, "_json", observe),
        ):
            for n, student in enumerate(("A", "B")):
                user = await runtime.repositories.identities.create(
                    UserIdentity(student_id=f"probe-{student}")
                )
                session = await runtime.conversation_service.create_session(user_id=user.user_id)
                for phase, questions in QUESTIONS.items():
                    if phase == "process":
                        session = await runtime.conversation_service.create_session(
                            user_id=user.user_id
                        )
                        await runtime.scenario_card_service.confirm_draft(
                            user_id=user.user_id,
                            draft=ScenarioDraft(
                                user_id=user.user_id,
                                session_id=session.session_id,
                                company_context=[
                                    "A bicycle manufacturer.",
                                    "A furniture manufacturer.",
                                ][n],
                                learner_role="Production planner",
                                business_problem="Orders arrive late.",
                                business_goals=["Improve delivery reliability."],
                            ),
                        )
                    result = await runtime.orchestration_service.execute(
                        OrchestrationRequest(
                            student_id=user.student_id,
                            session_id=str(session.session_id),
                            request_id=uuid4(),
                            user_message=questions[n],
                        ),
                        trusted_user_id=user.user_id,
                    )
                    turns.append(
                        dict(
                            student=student,
                            phase=phase,
                            route=result.route,
                            failed=bool(result.error),
                        )
                    )
                    save()
    finally:
        await service.aclose()
        print(report)


if __name__ == "__main__":
    asyncio.run(main())

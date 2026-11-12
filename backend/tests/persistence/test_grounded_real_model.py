"""Explicit opt-in: retained real corpus/models, disposable application database only."""

import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.config.settings import settings
from app.domain.guidance import GuidanceLevel
from app.knowledge.integration import CourseBinding, RagRuntimeConfig
from app.knowledge.planner_schemas import PlannerConfig
from app.main import app
from app.runtime import runtime_lifespan
from app.services.model_service import create_model_service
from app.services.model_service.exceptions import ModelServiceError
from app.services.model_service.schemas import ModelChatResponse
from scripts.migrate_knowledge import local_dsn
from tests.persistence.test_grounded_runtime import body, create_student
from tests.persistence.test_http_runtime import SECRET, headers

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("FREDI_RAG_REAL_MODEL") != "1",
        reason="Opt-in retained corpus and real local model.",
    ),
]


async def test_real_source_planner_and_specialist_through_signed_http(pg, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[3]
    corpus = json.loads((root / "docs/knowledge/retrieval-smoke.json").read_text())
    endpoint = os.environ.get("FREDI_RAG_MODEL_URL", "http://127.0.0.1:11434")
    model_name = os.environ.get("FREDI_RAG_MODEL", "qwen3.5:4b")
    uid = uuid4()
    cfg = RagRuntimeConfig(
        planner=PlannerConfig(model=model_name, timeout_seconds=90),
        bindings=[
            CourseBinding(
                user_id=uid,
                collection_id=corpus["collection_id"],
                revision_id=corpus["revision_id"],
                access_domain=corpus["access_domain"],
            )
        ],
        planner_url=endpoint,
        embedding_url=endpoint,
    )
    cfg.prompt.reasoning_enabled = False
    cfg.prompt.reserved_output_tokens = 1024
    path = tmp_path / "trusted-real-rag.json"
    path.write_text(cfg.model_dump_json(), encoding="utf-8")
    profile = settings.model_copy(
        update={
            "ollama_base_url": endpoint,
            "ollama_chat_model": model_name,
            "model_request_timeout_seconds": 120,
            "model_max_retries": 0,
        }
    )
    real = create_model_service(profile)
    teacher_requests, errors, responses = [], [], []

    class RoutedModel:
        async def generate(self, request):
            if request.metadata.get("task") == "semantic_routing":
                return ModelChatResponse(
                    content=json.dumps(
                        {"route": "process_coach", "reason": "Bounded integration smoke."}
                    ),
                    model="deterministic-test-router",
                    provider="ollama",
                    latency_ms=0,
                )
            teacher_requests.append(request)
            try:
                result = await real.generate(request)
                responses.append(result)
                return result
            except ModelServiceError as exc:
                errors.append(
                    {
                        "code": exc.code,
                        "status": exc.details.get("status_code"),
                        "grammar_error": "grammar" in exc.details.get("response_text", ""),
                    }
                )
                raise

    monkeypatch.setattr(settings, "application_storage", "postgres")
    monkeypatch.setattr(settings, "application_database_url", pg.database.config.dsn)
    monkeypatch.setattr(settings, "application_pilot_auth_secret", SECRET)
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "rag_config_file", path)
    monkeypatch.setattr(settings, "rag_database_url", local_dsn(role="reader"))
    monkeypatch.setattr(settings, "ollama_base_url", endpoint)
    monkeypatch.setattr(settings, "ollama_chat_model", model_name)
    monkeypatch.setattr(settings, "process_coach_agent_model", None)
    monkeypatch.setattr("app.runtime.create_model_service", lambda: RoutedModel())
    async with runtime_lifespan(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            runtime = app.state.runtime
            user, session = await create_student(client, runtime, uid)
            context = await runtime.user_context_service.get_or_create(user_id=uid)
            context.hint_state.guidance_level = GuidanceLevel.EXPLICIT
            context.hint_state.manually_overridden = True
            await runtime.user_context_service.save(
                user_id=uid, context=context, expected_version=context.version
            )
            payload = {
                **body(user, session),
                "user_message": (
                    "In the AP40 course, what business object is handled in AP40-02? "
                    "Give a concise answer with a citation to the provided course source. "
                    "Do not update my progress."
                ),
            }
            reply = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
            data = reply.json()
            report = {
                "http_status": reply.status_code,
                "model": model_name,
                "rag": data.get("metadata", {}).get("rag"),
                "errors": errors,
                "source_reference_count": len(data.get("source_references", [])),
                "teacher_calls": len(teacher_requests),
                "teacher_window": teacher_requests[-1].context_window_tokens
                if teacher_requests
                else None,
                "teacher_prompt_tokens": responses[-1].usage.prompt_tokens
                if responses and responses[-1].usage
                else None,
            }
            artifact = os.environ.get("FREDI_RAG_REAL_REPORT")
            if artifact:
                Path(artifact).write_text(json.dumps(report, indent=2), encoding="utf-8")
            assert reply.status_code == 200, report
            assert data["source_references"], report
            assert data["metadata"]["rag"]["provided_count"] > 0
            assert (
                data["metadata"]["estimated_input_tokens"] <= data["metadata"]["input_token_budget"]
            )
            assert str(uid) not in teacher_requests[-1].messages[0].content
            assert teacher_requests[-1].messages[1].content == payload["user_message"]
            stored = (
                await client.get(
                    f"/conversations/{session.session_id}/messages", headers=headers(user)
                )
            ).json()["items"]
            assert len(stored) == 2 and stored[1]["source_references"] == data["source_references"]
            again = await client.post("/orchestration/invoke", headers=headers(user), json=payload)
            assert again.json() == data and len(teacher_requests) == 1
            session_state = await runtime.session_context_service.get_required(
                user_id=uid,
                session_id=session.session_id,
            )
            assert session_state.current_step_id is None

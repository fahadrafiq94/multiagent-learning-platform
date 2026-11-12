"""Real synthetic onboarding/journey with incremental, private diagnostics.

Run from backend: .venv/Scripts/python.exe -m scripts.run_student_simulation
Only a generated scratch application database is written/dropped. Course access
is read-only. No live server flags, credentials, or student records are changed.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import psycopg
from dotenv import dotenv_values
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import SecretStr

from app.config.settings import settings
from app.context.schemas import ScenarioDraft, SessionContextUpdate
from app.identity import UserIdentity
from app.identity.pilot_tokens import issue_token
from app.knowledge.integration import CourseBinding, RagRuntimeConfig
from app.knowledge.planner_schemas import PlannerConfig
from app.logging.config import configure_logging
from app.main import app
from app.persistence.bootstrap import bootstrap
from app.persistence.database import ApplicationDatabase, ApplicationDatabaseConfig
from app.persistence.migrations import migrate
from app.runtime import runtime_lifespan
from app.services.model_service import create_model_service
from app.services.model_service.exceptions import ModelServiceError
from scripts.application_credentials import local_application_dsn
from scripts.evaluate_rag import ProfileModel
from scripts.knowledge_backup import private_directory, table_fingerprints
from scripts.knowledge_infra import ROOT
from scripts.migrate_knowledge import local_dsn
from scripts.student_simulation_report import render_report


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class RecordedModel(ProfileModel):
    """Instrumentation of actual providers; never replaces model responses."""

    def __init__(self, model, archive):
        super().__init__(model, capture_output=True)
        self.archive = archive

    async def generate(self, request):
        request = request.model_copy(update={"max_tokens": min(request.max_tokens or 1024, 1024)})
        failure = None
        try:
            return await super().generate(request)
        except ModelServiceError as exc:
            failure = {
                "status_code": exc.details.get("status_code"),
                "response_text": str(exc.details.get("response_text", ""))[:4000],
            }
            raise
        finally:
            call = self.calls[-1]
            call["private_messages"] = [m.model_dump(mode="json") for m in request.messages]
            call["private_response_schema"] = request.response_schema
            if failure is not None:
                call["private_provider_failure"] = failure
            with self.archive.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(call, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())


def technical_checks(turn, expected_event_count, events, confirmation=None):
    data = turn["response"]
    metadata = data.get("metadata", {})
    workflow_ok = (
        turn["http_status"] == 200
        and not data.get("error")
        and "detail" not in data
        and isinstance(data.get("final_response"), str)
        and bool(data["final_response"].strip())
        and isinstance(data.get("source_references"), list)
    )
    history = turn.get("history", [])
    checks = {
        "http_ok": turn["http_status"] == 200,
        "workflow_ok": workflow_ok,
        "assistant_persisted": metadata.get("assistant_message_persisted") is True,
        "turn_status_completed": turn.get("turn_status") == "completed",
        "one_request_event": len(events) == expected_event_count,
        "event_request_correlated": bool(events)
        and events[-1].get("request_id") == turn["request_id"],
        "history_pair_added": len(history) == expected_event_count * 2,
        "citations_persisted": workflow_ok
        and bool(history)
        and history[-1].get("role") == "assistant"
        and history[-1].get("content") == data["final_response"]
        and history[-1].get("source_references", []) == data["source_references"],
    }
    if metadata.get("trace_id") is not None:
        checks["event_trace_correlated"] = (
            bool(events) and events[-1].get("trace_id") == metadata["trace_id"]
        )
    estimate, budget = metadata.get("estimated_input_tokens"), metadata.get("input_token_budget")
    if estimate is not None and budget is not None:
        checks["context_within_budget"] = estimate <= budget
    if turn["id"] == "confirmation":
        card = turn["user_context"].get("scenario_card") or {}
        # Fixture-declared values; defaults preserve the original v1 journey check.
        confirmation = confirmation or {}
        checks["onboarding_confirmed"] = turn["onboarding_status"] == "confirmed"
        checks["corrected_role_saved"] = (
            confirmation.get("learner_role", "purchasing coordinator").lower()
            in (card.get("learner_role") or "").lower()
        )
        checks["company_saved"] = (
            confirmation.get("company_context", "urbancycle").lower()
            in (card.get("company_context") or "").lower()
        )
    return checks


async def capacity_probe(client, runtime, source, secret, models):
    """Two synthetic, pre-confirmed students; no changes to the live binding file."""
    students = []
    for _ in range(2):
        user = await runtime.repositories.identities.create(
            UserIdentity(student_id="capacity-" + uuid4().hex[:12])
        )
        auth = {"Authorization": "Bearer " + issue_token(user.user_id, secret)}
        created = await client.post(
            "/conversations", headers=auth, json={"title": "Capacity probe"}
        )
        created.raise_for_status()
        sid = UUID(created.json()["session_id"])
        await runtime.scenario_card_service.confirm_draft(
            user_id=user.user_id,
            draft=ScenarioDraft(
                user_id=user.user_id,
                session_id=sid,
                company_context="Synthetic receiving team",
                learner_role="Receiving supervisor",
                business_problem="Receiving errors delay dispatch.",
                business_goals=["Reduce receiving errors."],
            ),
        )
        await runtime.session_context_service.update(
            user_id=user.user_id,
            session_id=sid,
            changes=SessionContextUpdate(current_step_id="AP40-32"),
        )
        # Trusted test setup grants only the already-authorized shared corpus in
        # this disposable runtime. Student payloads cannot supply these bindings.
        runtime.grounding_service.bindings[user.user_id] = CourseBinding(
            user_id=user.user_id,
            collection_id=source["collection_id"],
            revision_id=source["revision_id"],
            access_domain=source["access_domain"],
            namespace=source.get("namespace", "ap40"),
        )
        students.append((user, sid, auth))
    for wrapped in models:
        wrapped.scope = {"phase": "capacity", "concurrency": 2}

    async def invoke(student):
        user, sid, auth = student
        payload = {
            "student_id": user.student_id,
            "session_id": str(sid),
            "request_id": str(uuid4()),
            "user_message": "What prerequisite does the course list for AP40-32?",
        }
        started = time.perf_counter()
        try:
            async with asyncio.timeout(240):
                response = await client.post("/orchestration/invoke", headers=auth, json=payload)
            return {
                "request_id": payload["request_id"],
                "http_status": response.status_code,
                "duration_ms": (time.perf_counter() - started) * 1000,
                "response": response.json(),
            }
        except Exception as exc:
            return {
                "request_id": payload["request_id"],
                "failure_type": type(exc).__name__,
                "duration_ms": (time.perf_counter() - started) * 1000,
            }

    started = time.perf_counter()
    rows = await asyncio.gather(*(invoke(student) for student in students))
    denied = await client.get(f"/conversations/{students[0][1]}", headers=students[1][2])
    return {
        "scope": "Two signed HTTP requests; pre-confirmed students; real SQL, search and models.",
        "concurrency": 2,
        "outer_deadline_seconds": 240,
        "cache_enabled": False,
        "batch_duration_ms": (time.perf_counter() - started) * 1000,
        "requests": rows,
        "cross_student_history_denied": denied.status_code == 404,
        "sla": None,
    }


async def execute(args):
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    output = (args.output or ROOT / "backend/reports" / ("student-journey-" + run_id)).absolute()
    presentation = (
        args.presentation or ROOT / "docs/demos" / ("student-journey-" + run_id)
    ).absolute()
    if presentation.exists():
        raise ValueError("Choose a new presentation directory to preserve previous results.")
    private_directory(output)
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    source_path = getattr(args, "source_config", None) or (
        ROOT / "docs/knowledge/retrieval-smoke.json"
    )
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    uid, secret = uuid4(), SecretStr(secrets.token_hex(32))
    dbname = "fredi_app_test_" + uuid4().hex[:16]
    run = {
        "synthetic": True,
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "model": args.model,
        "planner_model": args.planner_model,
        "fixture_version": fixture["version"],
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "scratch_database": dbname,
        "user_id": str(uid),
        "source": source,
        "source_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "context_window": 8192,
        "reasoning": False,
        "provider_timeout_seconds": args.timeout,
        "cache_enabled": False,
        "transport": "signed HTTP via native ASGI; not external TCP",
        "turns": [],
        "checks": {},
        "source_code_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "backend/app/agents/scenario.py",
                "backend/app/agents/grounding.py",
                "backend/app/knowledge/planner_prompt.py",
                "backend/app/observability/events.py",
                "backend/app/orchestration/service.py",
                "backend/scripts/run_student_simulation.py",
            )
        },
    }
    models = []

    def persist():
        save_json(output / "run.json", run)
        render_report(run, presentation)

    persist()
    print(f"Private archive: {output}\nPresentation: {presentation}", flush=True)
    admin = None
    created = False
    profile = settings.model_copy(
        update={
            "model_provider": "ollama",
            "ollama_base_url": args.model_url,
            "ollama_chat_model": args.model,
            "model_max_retries": 0,
            "model_request_timeout_seconds": args.timeout,
        }
    )
    real = create_model_service(profile)
    model = RecordedModel(real, output / "model-calls.jsonl")
    models.append(model)
    cfg = RagRuntimeConfig(
        planner=PlannerConfig(model=args.planner_model, timeout_seconds=args.timeout),
        bindings=[
            CourseBinding(
                user_id=uid,
                collection_id=source["collection_id"],
                revision_id=source["revision_id"],
                access_domain=source["access_domain"],
                namespace=source.get("namespace", "ap40"),
            )
        ],
        planner_url=args.model_url,
        embedding_url=args.model_url,
    )
    cfg.prompt.reasoning_enabled = False
    cfg.prompt.reserved_output_tokens = 1024
    config_path = output / "trusted-rag.json"
    config_path.write_text(cfg.model_dump_json(), encoding="utf-8")

    def instrumented_provider(provider_settings):
        wrapped = RecordedModel(
            create_model_service(provider_settings), output / "model-calls.jsonl"
        )
        models.append(wrapped)
        return wrapped

    try:
        health = await real.health()
        run["installed_models"] = [
            {"name": item.get("name"), "digest": item.get("digest")}
            for item in health.details.get("models", [])
            if item.get("name")
            in {args.model, args.planner_model, "nomic-embed-text-v2-moe:latest"}
        ]
        admin = await psycopg.AsyncConnection.connect(
            local_dsn(role="admin").get_secret_value(), autocommit=True, row_factory=dict_row
        )
        run["main_before"] = await table_fingerprints(admin)
        await admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER fredi_app_migrator").format(sql.Identifier(dbname))
        )
        created = True
        values = dotenv_values(ROOT / "backend/.env.knowledge")
        async with await psycopg.AsyncConnection.connect(
            local_dsn(role="admin", database=dbname).get_secret_value(), row_factory=dict_row
        ) as connection:
            await bootstrap(
                connection,
                {
                    "migrator": values.get("APPLICATION_MIGRATOR_PASSWORD") or "",
                    "runtime": values.get("APPLICATION_RUNTIME_PASSWORD") or "",
                },
            )
        async with ApplicationDatabase(
            ApplicationDatabaseConfig(dsn=local_application_dsn(role="migrator", database=dbname))
        ) as migrator:
            await migrate(migrator)
        changes = {
            "application_storage": "postgres",
            "application_database_url": local_application_dsn(database=dbname),
            "application_pilot_auth_secret": secret,
            "rag_enabled": True,
            "rag_config_file": config_path,
            "rag_database_url": local_dsn(role="reader"),
            "knowledge_cache_enabled": False,
            "knowledge_intake_enabled": False,
            "observability_log_file": str(output / "request-events.jsonl"),
            "observability_log_max_bytes": 10485760,
            "observability_log_backup_count": 5,
            "ollama_base_url": args.model_url,
            "ollama_chat_model": args.model,
            "ollama_embedding_model": "nomic-embed-text-v2-moe:latest",
            "model_request_timeout_seconds": args.timeout,
            "model_max_retries": 0,
            "orchestrator_model": None,
            "scenario_agent_model": None,
            "process_coach_agent_model": None,
            "ap_plus_navigator_model": None,
            "summary_model": None,
        }
        with ExitStack() as stack:
            for name, value in changes.items():
                stack.enter_context(patch.object(settings, name, value))
            # Only provider instrumentation is substituted; all generated responses,
            # graphs, repository methods, routes, authorization and decisions are real.
            stack.enter_context(patch("app.runtime.create_model_service", lambda: model))
            stack.enter_context(
                patch("app.knowledge.runtime.create_model_service", instrumented_provider)
            )
            configure_logging()
            logging.getLogger("fredi.request_events").setLevel(logging.INFO)
            auth = {"Authorization": "Bearer " + issue_token(uid, secret, lifetime_seconds=86400)}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://simulation"
            ) as client:
                async with runtime_lifespan(app):
                    runtime = app.state.runtime
                    user = await runtime.repositories.identities.create(
                        UserIdentity(user_id=uid, student_id="synthetic-" + uuid4().hex[:12])
                    )
                    response = await client.post(
                        "/conversations",
                        headers=auth,
                        json={"title": "Synthetic UrbanCycle journey"},
                    )
                    response.raise_for_status()
                    session_id = response.json()["session_id"]
                    run["session_id"] = session_id
                    save_json(output / "session-created.json", response.json())
                    persist()
                    completed_payload = None
                    completed_response = None
                    for index, case in enumerate(fixture["turns"], 1):
                        print(
                            f"Turn {index}/{len(fixture['turns'])}: {case['id']} started",
                            flush=True,
                        )
                        for wrapped in models:
                            wrapped.scope = {"turn": index, "case": case["id"]}
                        prior_calls = [len(w.calls) for w in models]
                        payload = {
                            "student_id": user.student_id,
                            "session_id": session_id,
                            "request_id": str(uuid4()),
                            "user_message": case["message"],
                        }
                        started = time.perf_counter()
                        response = await client.post(
                            "/orchestration/invoke", headers=auth, json=payload
                        )
                        elapsed = (time.perf_counter() - started) * 1000
                        turn = {
                            **case,
                            "request_id": payload["request_id"],
                            "http_status": response.status_code,
                            "duration_ms": elapsed,
                            "response": response.json(),
                        }
                        run["turns"].append(turn)
                        persist()  # Preserve the response before any diagnostic query can fail.
                        session = await client.get(f"/conversations/{session_id}", headers=auth)
                        session.raise_for_status()
                        context = await runtime.user_context_service.get_or_create(user_id=uid)
                        history = await client.get(
                            f"/conversations/{session_id}/messages?limit=100", headers=auth
                        )
                        history.raise_for_status()
                        turn_view = await client.get(
                            f"/conversations/{session_id}/turns/{payload['request_id']}",
                            headers=auth,
                        )
                        turn.update(
                            session_context=session.json()["context"],
                            user_context=context.model_dump(mode="json"),
                            guidance=context.hint_state.guidance_level.value,
                            onboarding_status=session.json()["context"]["scenario_setup_status"],
                            history=history.json()["items"],
                            turn_status=turn_view.json().get("status"),
                            turn_error_code=turn_view.json().get("error_code"),
                            model_calls=[
                                call
                                for wrapped, offset in zip(models, prior_calls, strict=True)
                                for call in wrapped.calls[offset:]
                            ],
                        )
                        events_path = output / "request-events.jsonl"
                        events = (
                            [
                                json.loads(line)
                                for line in events_path.read_text(encoding="utf-8").splitlines()
                            ]
                            if events_path.exists()
                            else []
                        )
                        turn["checks"] = technical_checks(
                            turn, index, events, fixture.get("confirmation")
                        )
                        if response.status_code == 200 and turn["turn_status"] == "completed":
                            completed_payload = payload.copy()
                            completed_response = response.json()
                        persist()
                        print(
                            f"Turn {index}: HTTP {response.status_code}, {elapsed / 1000:.2f}s, "
                            f"route={turn['response'].get('route')}, "
                            f"onboarding={turn['onboarding_status']}, "
                            f"checks={all(turn['checks'].values())}",
                            flush=True,
                        )
                        if not all(turn["checks"].values()):
                            run["status"] = "stopped_on_technical_failure"
                            break
                    calls_before = sum(len(w.calls) for w in models)
                    # A failed terminal turn is not a completed-response replay.
                    # Its retry must conflict without invoking the provider again.
                    if run["turns"][-1]["turn_status"] == "failed":
                        failed_retry = await client.post(
                            "/orchestration/invoke", headers=auth, json=payload
                        )
                        run["checks"]["failed_turn_retry_conflicts"] = (
                            failed_retry.status_code == 409
                        )
                    if completed_payload is not None:
                        replay = await client.post(
                            "/orchestration/invoke", headers=auth, json=completed_payload
                        )
                        run["checks"]["replay_response_identical"] = (
                            replay.status_code == 200 and replay.json() == completed_response
                        )
                    else:
                        run["checks"]["replay_response_identical"] = False
                    final_history = (
                        await client.get(
                            f"/conversations/{session_id}/messages?limit=100", headers=auth
                        )
                    ).json()
                    run["checks"]["replay_no_new_model_calls"] = (
                        sum(len(w.calls) for w in models) == calls_before
                    )
                    run["checks"]["replay_no_duplicate_messages"] = (
                        final_history["items"] == run["turns"][-1]["history"]
                    )
                    persisted_user = (
                        await runtime.user_context_service.get_or_create(user_id=uid)
                    ).model_dump(mode="json")
                    if getattr(args, "capacity_probe", False):
                        run["capacity_probe"] = await capacity_probe(
                            client, runtime, source, secret, models
                        )
                        persist()
                # Rebuild the complete runtime to verify persisted history/context.
                async with runtime_lifespan(app):
                    reopened = await client.get(f"/conversations/{session_id}", headers=auth)
                    restored_history = await client.get(
                        f"/conversations/{session_id}/messages?limit=100", headers=auth
                    )
                    restored_user = await app.state.runtime.user_context_service.get_or_create(
                        user_id=uid
                    )
                    run["checks"].update(
                        restart_reopen_ok=reopened.status_code == 200,
                        restart_session_context_identical=(
                            reopened.json().get("context") == run["turns"][-1]["session_context"]
                        ),
                        restart_history_identical=restored_history.json() == final_history,
                        restart_user_context_identical=restored_user.model_dump(mode="json")
                        == persisted_user,
                    )
                    save_json(output / "restored-history.json", restored_history.json())
                    save_json(output / "restored-context.json", reopened.json())
            for handler in logging.getLogger("fredi.request_events").handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logging.getLogger("fredi.request_events").removeHandler(handler)
        if run["status"] == "running":
            run["status"] = (
                "journey_completed_semantic_review_pending"
                if all(run["checks"].values())
                else "journey_completed_with_technical_failures"
            )
    except BaseException as exc:
        run["status"] = (
            "interrupted"
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
            else "failed"
        )
        run["failure_type"] = type(exc).__name__
        raise
    finally:
        try:
            if admin is not None:
                run["main_after"] = await table_fingerprints(admin)
                run["checks"]["main_database_unchanged"] = (
                    run.get("main_before") == run["main_after"]
                )
                persist()  # Durable archive before scratch database deletion.
                if created:
                    await admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(dbname)))
                    run["checks"]["scratch_database_removed"] = True
        finally:
            if admin is not None:
                await admin.close()
            run["finished_at"] = datetime.now(UTC).isoformat()
            if run["status"] == "journey_completed_semantic_review_pending" and not all(
                run["checks"].values()
            ):
                run["status"] = "journey_completed_with_technical_failures"
            persist()
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=ROOT / "docs/knowledge/student-journey-v2.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-config",
        type=Path,
        help="Trusted collection/revision/access-domain binding; defaults to the frozen v1 smoke.",
    )
    parser.add_argument("--presentation", type=Path)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--planner-model", default="qwen3:1.7b")
    parser.add_argument("--model-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=150)
    parser.add_argument(
        "--capacity-probe",
        action="store_true",
        help="Measure two synthetic students concurrently after the sequential journey.",
    )
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        result = runner.run(execute(args))
    print(f"Finished: {result['status']}", flush=True)
    return (
        0
        if result["status"] == "journey_completed_semantic_review_pending"
        and all(result["checks"].values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

import json
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from app.agents.orchestrator import routing_generation_schema
from app.agents.routing import RoutingDecision, StudentStepDecision
from app.context.grounding import GroundingContext
from app.context.schemas import SessionContextUpdate
from app.knowledge.integration import GroundingAdmission
from app.orchestration.factory import build_orchestration_runtime
from app.orchestration.schemas import OrchestrationRequest
from app.repositories.exceptions import RepositoryNotFoundError, VersionConflictError
from app.services.model_service.schemas import ModelChatResponse
from tests.orchestration.test_graph import (
    confirm_scenario,
    create_model_service,
    create_user_and_session,
)

pytestmark = pytest.mark.asyncio


def decision(action="set", step="FLOW-B7", quote="I am now working on FLOW-B7."):
    return StudentStepDecision(action=action, step_id=step, source_quote=quote)


async def setup_runtime(*, step=None, omitted=False, route="scenario", grounding=None):
    model, generate, requests = create_model_service()
    original = generate.side_effect

    async def dispatch(request):
        if request.metadata.get("task") == "semantic_routing":
            payload = {"route": route, "reason": "Controlled student-state regression."}
            if not omitted:
                payload["current_step"] = (
                    step.model_dump()
                    if step
                    else {
                        "action": "unchanged",
                        "step_id": None,
                        "source_quote": None,
                    }
                )
            return ModelChatResponse(
                content=json.dumps(payload), model="controlled", provider="ollama", latency_ms=0
            )
        return await original(request)

    generate.side_effect = dispatch
    runtime = build_orchestration_runtime(model_service=model, grounding_service=grounding)
    user, session = await create_user_and_session(runtime)
    await confirm_scenario(runtime, user=user, session=session)
    initial = await runtime.session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )
    await runtime.session_context_service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(current_step_id="FLOW-A2"),
        expected_version=initial.version,
    )
    return runtime, user, session, generate, requests


class GroundingSpy:
    def __init__(self):
        self.steps = []
        self.authorize = AsyncMock()

    async def prepare(self, context, specialist, **kwargs):
        self.steps.append(context.session_context.current_step_id)
        return GroundingAdmission(GroundingContext(status="skipped"), {"status": "skipped"})


async def invoke(runtime, user, session, message, metadata=None):
    return await runtime.orchestration_service.execute(
        OrchestrationRequest(
            student_id=user.student_id,
            session_id=str(session.session_id),
            user_message=message,
            metadata=metadata or {},
        )
    )


async def saved(runtime, user, session):
    return await runtime.session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )


async def test_current_step_is_saved_before_grounding_even_when_scenario_is_selected():
    spy = GroundingSpy()
    runtime, user, session, _, requests = await setup_runtime(step=decision(), grounding=spy)
    response = await invoke(runtime, user, session, "I am now working on FLOW-B7.")
    assert response.error is None
    assert response.route == "scenario"
    assert (await saved(runtime, user, session)).current_step_id == "FLOW-B7"
    assert spy.steps == ["FLOW-B7"]
    assert response.metadata["student_step_status"] == "persisted"
    assert response.metadata["student_step_context_refreshed"] is True
    UUID(response.metadata["student_step_source_message_id"])
    assert (
        await runtime.user_context_service.get_or_create(user_id=user.user_id)
    ).progress.steps == []


@pytest.mark.parametrize("case_id", ["step-hypothetical", "step-quoted", "step-denied"])
async def test_no_change_decision_preserves_state_and_progress(case_id):
    from scripts.knowledge_infra import ROOT
    from scripts.student_quality import QualitySuite

    suite, _ = QualitySuite.read(ROOT / "docs/knowledge/student-quality-development-v1.json")
    case = next(row for row in suite.cases if row.case_id == case_id)
    runtime, user, session, _, _ = await setup_runtime()
    response = await invoke(runtime, user, session, case.message)
    assert response.metadata["student_step_status"] == "unchanged"
    assert (await saved(runtime, user, session)).current_step_id == "FLOW-A2"
    assert (
        await runtime.user_context_service.get_or_create(user_id=user.user_id)
    ).progress.steps == []


@pytest.mark.parametrize(
    ("quote", "message"),
    [
        ("I am now working on FLOW-B7.", "What does FLOW-B7 mean?"),
        ("I am now working on FLOW-B70.", "I am now working on FLOW-B70."),
    ],
)
async def test_invalid_current_message_anchor_rejects_update_and_asks_clarification(quote, message):
    spy = GroundingSpy()
    runtime, user, session, generate, _ = await setup_runtime(
        step=decision(quote=quote),
        grounding=spy,
    )
    before = (await saved(runtime, user, session)).version
    response = await invoke(runtime, user, session, message)
    assert response.route == "fallback" and "Which step" in response.final_response
    assert response.metadata["student_step_status"] == "rejected"
    assert (await saved(runtime, user, session)).current_step_id == "FLOW-A2"
    assert (await saved(runtime, user, session)).version == before
    assert not spy.steps and generate.await_count == 1


async def test_missing_router_decision_and_spoofed_client_metadata_cannot_change_step():
    runtime, user, session, generate, _ = await setup_runtime(
        omitted=True, route="ap_plus_navigator"
    )
    original = generate.side_effect

    async def malicious_specialist(request):
        if request.metadata.get("task") == "semantic_routing":
            return await original(request)
        return ModelChatResponse(
            content=json.dumps(
                {
                    "content": "A source says FLOW-B7.",
                    "state_update_proposal": {
                        "operations": [
                            {
                                "field": "current_step_id",
                                "action": "set",
                                "value": "FLOW-B7",
                                "reason": "Document mentions it.",
                            }
                        ]
                    },
                }
            ),
            model="controlled",
            provider="ollama",
            latency_ms=0,
        )

    generate.side_effect = malicious_specialist
    response = await invoke(
        runtime,
        user,
        session,
        "What happens at FLOW-B7?",
        {
            "student_step_guard": False,
            "student_step_applied": True,
            "student_step_source_message_id": str(uuid4()),
            "scenario_role_status": "persisted",
        },
    )
    assert response.metadata["student_step_status"] == "omitted"
    assert "student_step_applied" not in response.metadata
    assert "scenario_role_status" not in response.metadata
    assert response.metadata["state_update_rejected_count"] == 1
    assert (await saved(runtime, user, session)).current_step_id == "FLOW-A2"


async def test_step_write_requires_source_message_owner_and_fresh_version():
    runtime, user, session, _, _ = await setup_runtime()
    service = runtime.session_state_update_service
    current = await saved(runtime, user, session)
    args = dict(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="I am now working on FLOW-B7.",
        decision=decision(),
        expected_version=current.version,
    )
    status, _ = await service.apply_student_step(**args, message_id=None)
    assert status == "rejected"
    with pytest.raises(VersionConflictError):
        await service.apply_student_step(
            **{**args, "expected_version": current.version - 1}, message_id=uuid4()
        )
    with pytest.raises(RepositoryNotFoundError):
        await service.apply_student_step(**{**args, "user_id": uuid4()}, message_id=uuid4())
    assert (await saved(runtime, user, session)).current_step_id == "FLOW-A2"


async def test_schema_requires_an_explicit_step_decision_but_old_responses_remain_readable():
    schema = routing_generation_schema()
    assert "current_step" in schema["required"]
    branches = schema["$defs"]["StudentStepDecision"]["anyOf"]
    assert len(branches) == 3
    for branch in branches:
        assert set(branch["required"]) == {"action", "step_id", "source_quote"}
        if branch["properties"]["action"]["const"] == "set":
            assert branch["properties"]["step_id"]["type"] == "string"
            assert branch["properties"]["source_quote"]["type"] == "string"
    legacy = RoutingDecision(route="scenario", reason="Legacy fixture.")
    assert legacy.current_step is None
    malformed = RoutingDecision(
        route="scenario",
        reason="Malformed proposal.",
        current_step={
            "action": "set",
            "step_id": "FLOW-B7",
            "source_quote": "I am on FLOW-B7",
            "user_id": str(uuid4()),
        },
    )
    assert malformed.current_step is None


async def test_persistence_conflict_stops_specialist_work_without_changing_state():
    runtime, user, session, generate, _ = await setup_runtime(step=decision())
    runtime.session_state_update_service.apply_student_step = AsyncMock(
        side_effect=VersionConflictError(
            entity_name="SessionContext",
            entity_id=str(session.session_id),
            expected_version=1,
            actual_version=2,
        )
    )
    response = await invoke(runtime, user, session, "I am now working on FLOW-B7.")
    assert response.error
    assert response.metadata["student_step_status"] == "failed"
    assert generate.await_count == 1
    assert (await saved(runtime, user, session)).current_step_id == "FLOW-A2"


async def test_context_refresh_failure_does_not_misreport_a_committed_update():
    runtime, user, session, generate, _ = await setup_runtime(step=decision())
    build = runtime.context_service.build
    calls = 0

    async def refresh(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("Controlled rendering failure.")
        return await build(**kwargs)

    runtime.context_service.build = refresh
    response = await invoke(runtime, user, session, "I am now working on FLOW-B7.")
    assert response.error and generate.await_count == 1
    assert response.metadata["student_step_status"] == "persisted"
    assert response.metadata["student_step_applied"] is True
    assert (await saved(runtime, user, session)).current_step_id == "FLOW-B7"


async def test_later_specialist_write_failure_does_not_hide_committed_student_step():
    runtime, user, session, _, _ = await setup_runtime(step=decision())
    runtime.session_state_update_service.apply = AsyncMock(
        side_effect=ValueError("Controlled error.")
    )
    response = await invoke(runtime, user, session, "I am now working on FLOW-B7.")
    assert response.error is None
    assert response.metadata["student_step_status"] == "persisted"
    assert response.metadata["state_update_applied"] is True
    assert (await saved(runtime, user, session)).current_step_id == "FLOW-B7"

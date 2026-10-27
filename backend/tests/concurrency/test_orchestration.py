"""
Sprint 4B.11 layer B: shared-runtime orchestration concurrency.

One shared OrchestrationRuntime (one service, one graph, one agent
instance per role) serves every concurrent user. Interleaving is
forced with exact barriers; leakage is detected with unique canary
strings per student/session. Assertions cover counts, versions, and
canary absence — never timing or completion order.
"""

from __future__ import annotations

import asyncio
import json
from typing import (
    cast,
)
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import pytest

from app.agents.factory import (
    create_agent_registry,
)
from app.agents.schemas import (
    AgentName,
)
from app.context import (
    AgentContext,
    ConversationSession,
    ScenarioDraft,
)
from app.identity import (
    UserIdentity,
)
from app.orchestration.factory import (
    OrchestrationRuntime,
    build_orchestration_runtime,
)
from app.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
)
from app.services.model_service import (
    ModelService,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)


class Barrier:
    """Releases exactly when all workers have arrived."""

    def __init__(self, count: int) -> None:
        self._arrivals = asyncio.Event()

        self._count = count

        self._seen = 0

    async def wait(self) -> None:
        self._seen += 1

        if self._seen == self._count:
            self._arrivals.set()

        await self._arrivals.wait()


def last_user_content(
    request: ModelChatRequest,
) -> str:
    return next(
        (message.content for message in reversed(request.messages) if message.role == "user"),
        "",
    )


def state_operations_for(
    user_text: str,
) -> list[dict[str, object]]:
    for marker in ("blocker-alpha", "blocker-beta"):
        if marker in user_text:
            return [
                {
                    "field": "active_blocker",
                    "action": "set",
                    "value": f"{marker}-resolved-value",
                    "reason": "Concurrent state-update test proposal.",
                },
            ]

    return []


def create_shared_model_service() -> tuple[ModelService, list[ModelChatRequest]]:
    """
    Fake model service shared by one whole runtime.

    Routing: "ap+ field" in the student message -> ap_plus_navigator,
    otherwise process_coach. Specialists echo the student message back
    as their content, so every response carries its own canary. Every
    specialist answer also assesses the student (4B.9 signal writes).
    """

    model_service = cast(
        ModelService,
        MagicMock(spec=ModelService),
    )

    requests: list[ModelChatRequest] = []

    async def generate(
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        requests.append(request)

        user_text = last_user_content(request)

        if request.metadata.get("task") == "semantic_routing":
            route = "ap_plus_navigator" if "ap+ field" in user_text.lower() else "process_coach"

            content = json.dumps(
                {
                    "route": (route),
                    "reason": ("Deterministic concurrency routing."),
                }
            )
        else:
            content = json.dumps(
                {
                    "content": (f"Specialist reply about: {user_text}"),
                    "state_update_proposal": {
                        "operations": (state_operations_for(user_text)),
                    },
                    "response_assessment": {
                        "understanding": "incorrect",
                        "misconception_detected": False,
                        "reason": ("Concurrent concurrency test assessment."),
                    },
                }
            )

        return ModelChatResponse(
            content=content,
            model="fake-model",
            provider="ollama",
            latency_ms=5.0,
        )

    generate_mock = AsyncMock(side_effect=generate)

    model_service.generate = generate_mock  # type: ignore[method-assign]

    return (
        model_service,
        requests,
    )


async def confirm_scenario(
    runtime: OrchestrationRuntime,
    *,
    user: UserIdentity,
    session: ConversationSession,
) -> None:
    """
    Complete scenario onboarding deterministically (no LLM turns) so
    requests exercise specialist routing instead of onboarding.
    """

    await runtime.scenario_card_service.confirm_draft(
        user_id=user.user_id,
        draft=ScenarioDraft(
            user_id=user.user_id,
            session_id=(session.session_id),
            company_context=("UrbanCycle manufactures bicycles."),
            learner_role=("Production Planner"),
            business_problem=("Customer orders are delayed."),
            business_goals=[("Improve delivery reliability.")],
        ),
    )


async def seed_user_session(
    runtime: OrchestrationRuntime,
    *,
    student_id: str,
) -> tuple[UserIdentity, ConversationSession]:
    user = await runtime.repositories.identities.create(UserIdentity(student_id=student_id))

    session = await runtime.conversation_service.create_session(user_id=user.user_id)

    await confirm_scenario(runtime, user=user, session=session)

    return (
        user,
        session,
    )


@pytest.mark.asyncio
async def test_multi_user_multi_request_on_shared_service_all_complete() -> None:
    model_service, _ = create_shared_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    plans: list[tuple[str, str]] = [
        ("K-CONC-1", "Walk me through the purchasing process."),
        ("K-CONC-2", "Help me reason about invoice approval."),
        ("K-CONC-3", "Which AP+ field captures the supplier number?"),
        ("K-CONC-4", "Which AP+ field holds the payment term?"),
    ]

    seeded = [await seed_user_session(runtime, student_id=(student_id)) for student_id, _ in plans]

    barrier = Barrier(len(plans) * 2)

    async def one(
        student_id: str,
        session: ConversationSession,
        message: str,
    ) -> OrchestrationResponse:
        await barrier.wait()

        return await runtime.orchestration_service.execute(
            OrchestrationRequest(
                session_id=str(session.session_id),
                student_id=student_id,
                user_message=message,
            )
        )

    task_specs = [
        (student_id, message, session)
        for (student_id, message), (_, session) in zip(plans, seeded, strict=True)
        for _ in range(2)
    ]

    results = await asyncio.gather(
        *(one(student_id, session, message) for student_id, message, session in task_specs)
    )

    assert len(results) == 8

    for (_, message, _), result in zip(task_specs, results, strict=True):
        assert result.error is None

        expected_route = "ap_plus_navigator" if "ap+ field" in message.lower() else "process_coach"

        assert result.route == expected_route

        assert result.final_response


@pytest.mark.asyncio
async def test_no_cross_user_leakage_in_responses_prompts_and_archive() -> None:
    model_service, captured = create_shared_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    students: list[tuple[str, str]] = [
        ("K-LEAK-A", "canary-alpha-7f3a"),
        ("K-LEAK-B", "canary-bravo-8b4c"),
        ("K-LEAK-C", "canary-charlie-9d5e"),
        ("K-LEAK-D", "canary-delta-af6f"),
    ]

    seeded = [
        await seed_user_session(runtime, student_id=(student_id)) for student_id, _ in students
    ]

    all_canaries = [canary for _, canary in students]

    barrier = Barrier(len(students) * 2)

    async def one(
        student_id: str,
        canary: str,
        session: ConversationSession,
    ) -> OrchestrationResponse:
        await barrier.wait()

        return await runtime.orchestration_service.execute(
            OrchestrationRequest(
                session_id=str(session.session_id),
                student_id=student_id,
                user_message=(f"Help me reason about {canary} in the AP process."),
            )
        )

    task_specs = [
        (student_id, canary, session)
        for (student_id, canary), (_, session) in zip(students, seeded, strict=True)
        for _ in range(2)
    ]

    results = await asyncio.gather(
        *(one(student_id, canary, session) for student_id, canary, session in task_specs)
    )

    # (a) Responses mention only their own student's canary.
    for (_, canary, _), result in zip(task_specs, results, strict=True):
        for foreign in all_canaries:
            if foreign != canary:
                assert foreign not in result.final_response

    # (b) No model request ever mixes two students' canaries.
    for request in captured:
        joined = " ".join(message.content for message in request.messages)

        present = {canary for canary in all_canaries if canary in joined}

        assert len(present) <= 1, f"mixed canaries in one model request: {present}"

    # (c) Each session's archive holds only its own student's canaries.
    for (_, canary), (user, session) in zip(students, seeded, strict=True):
        messages = await runtime.repositories.conversations.list_for_session(
            user_id=user.user_id,
            session_id=session.session_id,
        )

        assert messages

        for message in messages:
            for foreign in all_canaries:
                if foreign != canary:
                    assert foreign not in message.content


@pytest.mark.asyncio
async def test_one_user_two_concurrent_sessions_no_cross_session_bleed() -> None:
    model_service, _ = create_shared_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    user = await runtime.repositories.identities.create(UserIdentity(student_id="K-TWO-SESSIONS"))

    session_a = await runtime.conversation_service.create_session(user_id=user.user_id)

    session_b = await runtime.conversation_service.create_session(user_id=user.user_id)

    await confirm_scenario(runtime, user=user, session=session_a)

    canary_a = "canary-session-a-1a2b"

    canary_b = "canary-session-b-3c4d"

    barrier = Barrier(2)

    async def one(
        session: ConversationSession,
        canary: str,
    ) -> OrchestrationResponse:
        await barrier.wait()

        return await runtime.orchestration_service.execute(
            OrchestrationRequest(
                session_id=str(session.session_id),
                student_id="K-TWO-SESSIONS",
                user_message=(f"Tell me about {canary} in the process."),
            )
        )

    result_a, result_b = await asyncio.gather(
        one(session_a, canary_a),
        one(session_b, canary_b),
    )

    assert result_a.error is None

    assert result_b.error is None

    assert canary_b not in result_a.final_response

    assert canary_a not in result_b.final_response

    messages_a = await runtime.repositories.conversations.list_for_session(
        user_id=user.user_id,
        session_id=session_a.session_id,
    )

    messages_b = await runtime.repositories.conversations.list_for_session(
        user_id=user.user_id,
        session_id=session_b.session_id,
    )

    assert messages_a

    assert messages_b

    for message in messages_a:
        assert canary_b not in message.content

    for message in messages_b:
        assert canary_a not in message.content


@pytest.mark.asyncio
async def test_simultaneous_state_updates_on_one_session_stay_coherent() -> None:
    model_service, _ = create_shared_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    user = await runtime.repositories.identities.create(UserIdentity(student_id="K-STATE-RACE"))

    session = await runtime.conversation_service.create_session(user_id=user.user_id)

    await confirm_scenario(runtime, user=user, session=session)

    initial = await runtime.repositories.session_contexts.get(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert initial is not None

    barrier = Barrier(2)

    async def one(
        marker: str,
    ) -> OrchestrationResponse:
        await barrier.wait()

        return await runtime.orchestration_service.execute(
            OrchestrationRequest(
                session_id=str(session.session_id),
                student_id="K-STATE-RACE",
                user_message=(f"I am blocked: {marker} in the process."),
            )
        )

    results = await asyncio.gather(one("blocker-alpha"), one("blocker-beta"))

    # Both turns complete; the conflict, when it happens, degrades a
    # state write — never the student's turn.
    for result in results:
        assert result.error is None

    applied = [r for r in results if r.metadata.get("state_update_applied") is True]

    degraded = [r for r in results if r.metadata.get("state_update_applied") is False]

    # Every write resolved one way or the other; losers are recorded,
    # never silently dropped.
    assert len(applied) + len(degraded) == 2

    for result in degraded:
        assert result.metadata.get("state_update_error")

    final = await runtime.repositories.session_contexts.get(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert final is not None

    # The version advanced exactly by the number of applied writes —
    # no lost update, no phantom write.
    assert final.version == initial.version + len(applied)

    # One coherent blocker value, never a mix of both proposals.
    assert final.active_blocker in (
        "blocker-alpha-resolved-value",
        "blocker-beta-resolved-value",
    )


@pytest.mark.asyncio
async def test_concurrent_context_builds_stay_isolated() -> None:
    model_service, _ = create_shared_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    users = [
        await runtime.repositories.identities.create(UserIdentity(student_id=(f"K-BUILD-{index}")))
        for index in range(3)
    ]

    sessions = [
        await runtime.conversation_service.create_session(user_id=(user.user_id)) for user in users
    ]

    canaries = [f"objective-canary-{index}-abcd" for index in range(3)]

    for user, session, canary in zip(users, sessions, canaries, strict=True):
        context = await runtime.repositories.session_contexts.get(
            user_id=user.user_id,
            session_id=session.session_id,
        )

        assert context is not None

        candidate = context.model_copy(deep=True)

        candidate.current_objective = canary

        await runtime.repositories.session_contexts.save(
            candidate,
            expected_version=(context.version),
        )

    barrier = Barrier(len(users))

    async def one(
        user: UserIdentity,
        session: ConversationSession,
    ) -> AgentContext:
        await barrier.wait()

        return await runtime.context_service.build(
            user_id=user.user_id,
            session_id=session.session_id,
            current_message="continue the current objective.",
        )

    pairs = list(zip(users, sessions, strict=True))

    contexts = await asyncio.gather(*(one(user, session) for user, session in pairs))

    triplets = list(zip(users, sessions, canaries, strict=True))

    for agent_context, (user, session, canary) in zip(contexts, triplets, strict=True):
        assert agent_context.user_id == user.user_id

        assert agent_context.session_id == session.session_id

        assert agent_context.session_context is not None

        # Each AgentContext carries exactly its own session state.
        assert agent_context.session_context.current_objective == canary


def test_shared_agent_registry_serves_single_stateless_instances() -> None:
    model_service, _ = create_shared_model_service()

    registry = create_agent_registry(model_service)

    names: tuple[AgentName, ...] = ("scenario", "process_coach", "ap_plus_navigator")

    for name in names:
        # The same instance is served every time — one instance per
        # role for the whole process, not one per user.
        assert registry.get(name) is registry.get(name)


@pytest.mark.asyncio
async def test_concurrent_guidance_signal_writes_stay_coherent() -> None:
    model_service, _ = create_shared_model_service()

    runtime = build_orchestration_runtime(model_service=model_service)

    user = await runtime.repositories.identities.create(UserIdentity(student_id="K-GUIDANCE"))

    session = await runtime.conversation_service.create_session(user_id=user.user_id)

    await confirm_scenario(runtime, user=user, session=session)

    initial_user_context = await runtime.user_context_service.get_or_create(user_id=user.user_id)

    barrier = Barrier(2)

    async def one(
        index: int,
    ) -> OrchestrationResponse:
        await barrier.wait()

        return await runtime.orchestration_service.execute(
            OrchestrationRequest(
                session_id=str(session.session_id),
                student_id="K-GUIDANCE",
                user_message=(f"Is my answer {index} correct for the invoice step?"),
            )
        )

    results = await asyncio.gather(one(1), one(2))

    # No signal write may raise out of the graph.
    for result in results:
        assert result.error is None

    final_user_context = await runtime.repositories.user_contexts.get(user.user_id)

    assert final_user_context is not None

    # Counters stay bounded by the number of signals; a conflict may
    # skip one recording (response_signal_recorded=False), but the
    # counters must never exceed the signals sent or go negative.
    assert final_user_context.version >= initial_user_context.version

    assert 0 <= final_user_context.hint_state.consecutive_unproductive_attempts <= 2

    assert 0 <= final_user_context.hint_state.successful_independent_steps <= 2

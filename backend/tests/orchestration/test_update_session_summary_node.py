from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from app.orchestration.factory import (
    build_orchestration_runtime,
)
from app.orchestration.nodes import (
    update_session_summary_node,
)
from app.orchestration.state import (
    OrchestrationState,
)
from app.repositories import (
    InMemoryRepositories,
    create_in_memory_repositories,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.model_service import ModelService
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)
from app.services.session_context_service import (
    SessionContextService,
)
from app.services.session_summary_service import (
    SessionSummaryService,
)


class FakeModelService:
    def __init__(self) -> None:
        self.requests: list[ModelChatRequest] = []

    async def generate(
        self,
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        self.requests.append(request)

        return ModelChatResponse(
            content=(
                '{"overview": "ok", "established_facts": [], '
                '"decisions": [], "blockers": [], '
                '"unresolved_questions": [], '
                '"misconception_candidates": [], '
                '"completed_step_ids": []}'
            ),
            model="fake",
            provider="ollama",
            latency_ms=1.0,
        )


async def create_setup() -> tuple[
    InMemoryRepositories,
    SessionSummaryService,
    FakeModelService,
]:
    repositories = create_in_memory_repositories()

    conversation_service = ConversationService(
        session_repository=(repositories.sessions),
        session_context_repository=(repositories.session_contexts),
        conversation_repository=(repositories.conversations),
    )

    session_context_service = SessionContextService(
        repository=(repositories.session_contexts),
    )

    model_service = FakeModelService()

    summary_service = SessionSummaryService(
        conversation_service=conversation_service,
        session_context_service=session_context_service,
        model_service=(model_service),  # type: ignore[arg-type]
        recent_window=2,
        min_new_messages=2,
    )

    return (
        repositories,
        summary_service,
        model_service,
    )


async def create_user_and_session(
    *,
    repositories: InMemoryRepositories,
    conversation_service: ConversationService,
    student_id: str,
) -> tuple[Any, Any]:
    from app.identity import UserIdentity

    identity = UserIdentity(
        student_id=student_id,
    )

    user = await repositories.identities.create(identity)

    session = await conversation_service.create_session(
        user_id=user.user_id,
    )

    return user, session


@pytest.mark.asyncio
async def test_node_applies_summary_update_and_records_metadata() -> None:
    repositories, summary_service, model_service = await create_setup()

    conversation_service = ConversationService(
        session_repository=repositories.sessions,
        session_context_repository=repositories.session_contexts,
        conversation_repository=repositories.conversations,
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=conversation_service,
        student_id="K10001",
    )

    for index in range(1, 5):
        await conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=(f"msg-{index}"),
        )

    state: OrchestrationState = {
        "user_id": user.user_id,
        "session_uuid": session.session_id,
        "error": None,
        "agent_response": "Done.",
        "metadata": {"assistant_message_persisted": True},
    }

    result = await update_session_summary_node(
        state,
        summary_service=summary_service,
    )

    metadata = result["metadata"]

    assert metadata["summary_update_attempted"] is True
    assert metadata["summary_update_applied"] is True
    assert metadata["summary_update_reason"] == "updated"
    assert metadata["summary_message_count"] == 2
    assert metadata["summary_update_error"] is None
    assert len(model_service.requests) == 1

    context = await summary_service._session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert context.summary is not None
    assert result["agent_response"] == "Done."


@pytest.mark.asyncio
async def test_node_below_threshold_records_skipped() -> None:
    repositories, summary_service, model_service = await create_setup()

    conversation_service = ConversationService(
        session_repository=repositories.sessions,
        session_context_repository=repositories.session_contexts,
        conversation_repository=repositories.conversations,
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=conversation_service,
        student_id="K10002",
    )

    for index in range(1, 4):
        await conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=(f"msg-{index}"),
        )

    state: OrchestrationState = {
        "user_id": user.user_id,
        "session_uuid": session.session_id,
        "error": None,
        "metadata": {},
    }

    result = await update_session_summary_node(
        state,
        summary_service=summary_service,
    )

    assert result["metadata"]["summary_update_applied"] is False
    assert result["metadata"]["summary_update_reason"] == "below_threshold"
    assert model_service.requests == []


@pytest.mark.asyncio
async def test_node_skips_when_request_has_error() -> None:
    _, summary_service, model_service = await create_setup()

    state: OrchestrationState = {
        "error": "Something failed earlier.",
        "metadata": {},
    }

    result = await update_session_summary_node(
        state,
        summary_service=summary_service,
    )

    assert result["error"] == "Something failed earlier."
    assert "summary_update_attempted" not in result["metadata"]
    assert model_service.requests == []


@pytest.mark.asyncio
async def test_node_contains_service_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    _, summary_service, model_service = await create_setup()

    async def failing_update(
        *,
        user_id: Any,
        session_id: Any,
    ) -> None:
        raise RuntimeError("summary service exploded")

    monkeypatch.setattr(
        summary_service,
        "maybe_update",
        failing_update,
    )

    state: OrchestrationState = {
        "user_id": UUID("11111111-1111-1111-1111-111111111111"),
        "session_uuid": UUID("22222222-2222-2222-2222-222222222222"),
        "error": None,
        "agent_response": "Still a valid response.",
        "metadata": {},
    }

    result = await update_session_summary_node(
        state,
        summary_service=summary_service,
    )

    assert result["agent_response"] == "Still a valid response."
    assert result["metadata"]["summary_update_attempted"] is True
    assert result["metadata"]["summary_update_applied"] is False
    assert "summary service exploded" in (result["metadata"]["summary_update_error"])
    assert model_service.requests == []


def test_real_graph_routes_through_summary_node() -> None:
    model_service = cast(
        ModelService,
        MagicMock(spec=ModelService),
    )

    runtime = build_orchestration_runtime(model_service=model_service)

    mermaid = runtime.graph.get_graph().draw_mermaid()

    assert "update_session_summary" in mermaid
    assert "record_response_signal" in mermaid
    assert "apply_state_updates --> record_response_signal;" in mermaid
    assert "record_response_signal --> update_session_summary;" in mermaid
    assert "fallback_response --> update_session_summary;" in mermaid
    assert "update_session_summary --> finalize_response;" in mermaid

    # Scenario onboarding bypasses summary maintenance.
    assert "scenario_onboarding --> finalize_response;" in mermaid


def test_real_graph_summary_node_is_styled() -> None:
    from app.orchestration.visualization import (
        MERMAID_STYLES,
    )

    assert "class update_session_summary contextNode;" in MERMAID_STYLES

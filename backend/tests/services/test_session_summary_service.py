from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import (
    ValidationError,
)

from app.context.schemas import (
    SessionContext,
    SessionSummary,
)
from app.context.summary_prompt import (
    SummaryDraft,
    sanitize_summary_draft,
)
from app.identity import UserIdentity
from app.repositories import (
    InMemoryRepositories,
    RepositoryError,
    VersionConflictError,
    create_in_memory_repositories,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)
from app.services.session_context_service import (
    SessionContextService,
)
from app.services.session_summary_service import (
    SessionSummaryService,
    should_update_summary,
)


class FakeModelService:
    def __init__(
        self,
        *,
        content: str = "{}",
        error: Exception | None = None,
    ) -> None:
        self._content = content
        self._error = error
        self.requests: list[ModelChatRequest] = []

    async def generate(
        self,
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        self.requests.append(request)

        if self._error is not None:
            raise self._error

        return ModelChatResponse(
            content=self._content,
            model="fake",
            provider="ollama",
            latency_ms=1.0,
        )


def draft_content(
    *,
    overview: str = "Student is planning production.",
    decisions: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "overview": (overview),
            "established_facts": (["Fact about the scenario."]),
            "decisions": (decisions or []),
            "blockers": [],
            "unresolved_questions": [],
            "misconception_candidates": [],
            "completed_step_ids": [],
        }
    )


def create_services() -> tuple[
    InMemoryRepositories,
    ConversationService,
    SessionContextService,
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

    return (
        repositories,
        conversation_service,
        session_context_service,
    )


def create_summary_service(
    model_service: FakeModelService,
    *,
    conversation_service: ConversationService,
    session_context_service: SessionContextService,
) -> SessionSummaryService:
    # recent_window=2, min_new_messages=2:
    # an update triggers once total - covered - 2 >= 2.
    return SessionSummaryService(
        conversation_service=conversation_service,
        session_context_service=session_context_service,
        model_service=(model_service),  # type: ignore[arg-type]
        recent_window=2,
        min_new_messages=2,
        max_items_per_list=2,
        max_overview_chars=50,
        model=("fake-summary-model"),
    )


async def create_user(
    *,
    repositories: InMemoryRepositories,
    student_id: str,
) -> UserIdentity:
    identity = UserIdentity(
        student_id=student_id,
    )

    return await repositories.identities.create(identity)


async def create_user_and_session(
    *,
    repositories: InMemoryRepositories,
    conversation_service: ConversationService,
    student_id: str = "K12345",
) -> tuple[UserIdentity, Any]:
    user = await create_user(
        repositories=repositories,
        student_id=student_id,
    )

    session = await conversation_service.create_session(
        user_id=user.user_id,
    )

    return user, session


async def append_messages(
    *,
    conversation_service: ConversationService,
    user_id: Any,
    session_id: Any,
    count: int,
    start: int = 1,
) -> None:
    for index in range(start, start + count):
        await conversation_service.append_user_message(
            user_id=user_id,
            session_id=session_id,
            content=(f"msg-{index}"),
        )


@pytest.mark.parametrize(
    ("total", "covered", "recent_window", "min_new", "expected"),
    [
        (2, 0, 2, 2, False),
        (3, 0, 2, 2, False),
        (4, 0, 2, 2, True),
        (6, 2, 2, 2, True),
        (5, 2, 2, 2, False),
        (5, 0, 0, 0, True),
        (0, 0, 2, 2, False),
    ],
)
def test_should_update_summary_policy(
    total: int,
    covered: int,
    recent_window: int,
    min_new: int,
    expected: bool,
) -> None:
    assert (
        should_update_summary(
            total_message_count=total,
            covered_message_count=covered,
            recent_window=recent_window,
            min_new_messages=min_new,
        )
        is expected
    )


def test_sanitize_summary_draft_bounded_and_deduplicated() -> None:
    draft = SummaryDraft(
        overview=("x" * 80),
        established_facts=(["one", "one", "two", "three"]),
        decisions=(["a", "b", "c"]),
        blockers=[],
        unresolved_questions=[],
        misconception_candidates=[],
        completed_step_ids=[],
    )

    cleaned = sanitize_summary_draft(
        draft,
        max_items_per_list=2,
        max_overview_chars=50,
    )

    assert len(cleaned.overview) == 50  # type: ignore[arg-type]
    assert cleaned.established_facts == ["one", "two"]
    assert cleaned.decisions == ["a", "b"]


@pytest.mark.asyncio
async def test_below_threshold_makes_no_model_call() -> None:
    repositories, conversation_service, session_context_service = create_services()

    model_service = FakeModelService()

    summary_service = create_summary_service(
        model_service,
        conversation_service=conversation_service,
        session_context_service=session_context_service,
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=conversation_service,
    )

    await append_messages(
        conversation_service=conversation_service,
        user_id=user.user_id,
        session_id=session.session_id,
        count=3,
    )

    before = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    result = await summary_service.maybe_update(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert result.updated is False
    assert result.reason == "below_threshold"
    assert model_service.requests == []

    after = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert after.version == before.version
    assert after.summary is None


@pytest.mark.asyncio
async def test_first_summary_covers_only_oldest_batch() -> None:
    repositories, conversation_service, session_context_service = create_services()

    model_service = FakeModelService(
        content=draft_content(decisions=["Student chose AP+ scenario."]),
    )

    summary_service = create_summary_service(
        model_service,
        conversation_service=conversation_service,
        session_context_service=session_context_service,
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=conversation_service,
    )

    # total=4, covered=0, window=2 -> batch = messages 1..2
    await append_messages(
        conversation_service=conversation_service,
        user_id=user.user_id,
        session_id=session.session_id,
        count=4,
    )

    result = await summary_service.maybe_update(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert result.updated is True
    assert result.error is None

    request = model_service.requests[0]
    user_content = request.messages[-1].content

    assert "msg-1" in user_content
    assert "msg-2" in user_content
    assert "msg-3" not in user_content
    assert "EXISTING SUMMARY" not in user_content
    assert request.model == "fake-summary-model"
    assert request.metadata["task"] == "session_summary_update"

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    summary = context.summary

    assert summary is not None
    assert summary.user_id == user.user_id
    assert summary.session_id == session.session_id
    assert summary.source_message_count == 2
    assert summary.decisions == ["Student chose AP+ scenario."]


@pytest.mark.asyncio
async def test_incremental_update_fetches_only_new_messages() -> None:
    repositories, conversation_service, session_context_service = create_services()

    model_service = FakeModelService(
        content=draft_content(decisions=["First decision."]),
    )

    summary_service = create_summary_service(
        model_service,
        conversation_service=conversation_service,
        session_context_service=session_context_service,
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=conversation_service,
    )

    await append_messages(
        conversation_service=conversation_service,
        user_id=user.user_id,
        session_id=session.session_id,
        count=4,
    )

    first = await summary_service.maybe_update(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert first.updated is True

    # Two more messages: total=6, covered=2, window=2 -> batch = 3..4
    await append_messages(
        conversation_service=conversation_service,
        user_id=user.user_id,
        session_id=session.session_id,
        count=2,
        start=5,
    )

    second = await summary_service.maybe_update(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert second.updated is True
    assert len(model_service.requests) == 2

    second_content = model_service.requests[1].messages[-1].content

    assert "msg-3" in second_content
    assert "msg-4" in second_content
    assert "msg-1" not in second_content
    assert "msg-5" not in second_content
    assert "EXISTING SUMMARY" in second_content
    assert "First decision." in second_content

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert context.summary is not None
    assert context.summary.source_message_count == 4
    # The summary keeps its stable identity across incremental updates.
    assert context.summary.summary_id == first.summary.summary_id  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_malformed_model_output_keeps_old_summary() -> None:
    repositories, conversation_service, session_context_service = create_services()

    model_service = FakeModelService(content=("not json at all"))

    summary_service = create_summary_service(
        model_service,
        conversation_service=conversation_service,
        session_context_service=session_context_service,
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=conversation_service,
    )

    await append_messages(
        conversation_service=conversation_service,
        user_id=user.user_id,
        session_id=session.session_id,
        count=4,
    )

    before = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    result = await summary_service.maybe_update(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert result.updated is False
    assert result.reason == "invalid_model_output"
    assert result.error is not None

    after = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert after.summary is None
    assert after.version == before.version


@pytest.mark.asyncio
async def test_model_provider_failure_keeps_old_summary() -> None:
    repositories, conversation_service, session_context_service = create_services()

    model_service = FakeModelService(error=RuntimeError("provider down"))

    summary_service = create_summary_service(
        model_service,
        conversation_service=conversation_service,
        session_context_service=session_context_service,
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=conversation_service,
    )

    await append_messages(
        conversation_service=conversation_service,
        user_id=user.user_id,
        session_id=session.session_id,
        count=4,
    )

    result = await summary_service.maybe_update(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert result.updated is False
    assert result.reason == "model_failed"
    assert "provider down" in (result.error or "")

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert context.summary is None


@pytest.mark.asyncio
async def test_version_conflict_keeps_old_summary() -> None:
    repositories, conversation_service, session_context_service = create_services()

    model_service = FakeModelService(content=draft_content())

    summary_service = create_summary_service(
        model_service,
        conversation_service=conversation_service,
        session_context_service=session_context_service,
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=conversation_service,
    )

    await append_messages(
        conversation_service=conversation_service,
        user_id=user.user_id,
        session_id=session.session_id,
        count=4,
    )

    async def failing_save(**kwargs: Any) -> SessionContext:
        raise VersionConflictError(
            entity_name="SessionContext",
            entity_id=str(session.session_id),
            expected_version=1,
            actual_version=2,
        )

    session_context_service.save = failing_save  # type: ignore[method-assign]

    result = await summary_service.maybe_update(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert result.updated is False
    assert result.reason == "persistence_failed"
    assert result.error is not None

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert context.summary is None


@pytest.mark.asyncio
async def test_cross_user_summary_update_rejected() -> None:
    repositories, conversation_service, session_context_service = create_services()

    model_service = FakeModelService(content=draft_content())

    summary_service = create_summary_service(
        model_service,
        conversation_service=conversation_service,
        session_context_service=session_context_service,
    )

    owner, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=conversation_service,
        student_id="K11111",
    )

    await append_messages(
        conversation_service=conversation_service,
        user_id=owner.user_id,
        session_id=session.session_id,
        count=4,
    )

    intruder = await create_user(
        repositories=repositories,
        student_id="K22222",
    )

    with pytest.raises(RepositoryError):
        await summary_service.maybe_update(
            user_id=intruder.user_id,
            session_id=session.session_id,
        )

    assert model_service.requests == []

    context = await session_context_service.get_required(
        user_id=owner.user_id,
        session_id=session.session_id,
    )

    assert context.summary is None


def test_session_context_rejects_mismatched_summary_ownership() -> None:
    intruder_summary = SessionSummary(
        user_id="33333333-3333-3333-3333-333333333333",
        session_id="22222222-2222-2222-2222-222222222222",
    )

    with pytest.raises(ValidationError):
        SessionContext(
            user_id="11111111-1111-1111-1111-111111111111",
            session_id="22222222-2222-2222-2222-222222222222",
            summary=intruder_summary,
        )

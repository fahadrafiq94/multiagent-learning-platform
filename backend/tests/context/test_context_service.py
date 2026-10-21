from unittest.mock import MagicMock

import pytest

from app.context.budget import (
    ContextBudgetPolicy,
    ContextBudgetService,
)
from app.context.token_estimator import (
    ApproximateTokenEstimator,
)
from app.identity import UserIdentity
from app.orchestration.factory import (
    build_orchestration_runtime,
)
from app.services.context_service import (
    ContextService,
)
from app.services.model_service import ModelService


def create_small_budget_service(
    *,
    user_context_service,
    session_context_service,
    conversation_service,
) -> ContextService:
    """
    Create a deterministic ContextService for budget tests.

    We deliberately use one character per token so tests do not depend on
    provider-specific tokenization behavior.

    The history messages used below are large enough to force truncation while
    leaving enough room for essential FREDi context.
    """

    budget_service = ContextBudgetService(
        estimator=ApproximateTokenEstimator(
            characters_per_token=1.0,
            message_overhead_tokens=0,
        ),
        policy=ContextBudgetPolicy(
            max_context_tokens=4000,
            reserved_output_tokens=500,
            max_recent_messages=20,
        ),
    )

    return ContextService(
        user_context_service=(user_context_service),
        session_context_service=(session_context_service),
        conversation_service=(conversation_service),
        budget_service=budget_service,
    )


@pytest.mark.asyncio
async def test_context_service_reports_real_budget_metadata() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Previous message",
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current question",
    )

    metadata = context.metadata

    assert metadata.estimated_input_tokens is not None
    assert metadata.input_token_budget is not None
    assert metadata.estimated_essential_tokens is not None
    assert metadata.estimated_history_tokens is not None

    assert metadata.estimated_input_tokens > 0
    assert metadata.input_token_budget > 0
    assert metadata.estimated_essential_tokens > 0
    assert metadata.estimated_history_tokens > 0

    assert metadata.candidate_message_count == 1
    assert metadata.recent_message_count == 1
    assert metadata.dropped_message_count == 0
    assert metadata.truncation_applied is False

    assert metadata.estimated_input_tokens <= metadata.input_token_budget


@pytest.mark.asyncio
async def test_token_budget_truncates_history() -> None:
    (
        repositories,
        conversation_service,
        user_context_service,
        session_context_service,
        _,
    ) = create_services()

    context_service = create_small_budget_service(
        user_context_service=(user_context_service),
        session_context_service=(session_context_service),
        conversation_service=(conversation_service),
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    for index in range(6):
        await conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=(f"Message {index}: " + ("x" * 1000)),
        )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
    )

    metadata = context.metadata

    assert len(context.recent_messages) < 6

    assert metadata.candidate_message_count == 6

    assert metadata.recent_message_count == len(context.recent_messages)

    assert metadata.dropped_message_count > 0

    assert metadata.dropped_message_count == (
        metadata.candidate_message_count - metadata.recent_message_count
    )

    assert metadata.truncation_applied is True

    assert metadata.estimated_input_tokens is not None
    assert metadata.input_token_budget is not None

    assert metadata.estimated_input_tokens <= metadata.input_token_budget


@pytest.mark.asyncio
async def test_token_budget_prefers_newest_history() -> None:
    (
        repositories,
        conversation_service,
        user_context_service,
        session_context_service,
        _,
    ) = create_services()

    context_service = create_small_budget_service(
        user_context_service=(user_context_service),
        session_context_service=(session_context_service),
        conversation_service=(conversation_service),
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    for index in range(6):
        await conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=(f"Message {index}: " + ("x" * 1000)),
        )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
    )

    selected_contents = [message.content for message in context.recent_messages]

    assert selected_contents

    # The newest stored message must survive whenever any history fits.
    assert selected_contents[-1].startswith("Message 5:")

    # Because selection is a contiguous recent suffix, if N messages survive,
    # they must be exactly the newest N messages.
    selected_count = len(selected_contents)

    expected_start = 6 - selected_count

    expected_prefixes = [
        f"Message {index}:"
        for index in range(
            expected_start,
            6,
        )
    ]

    actual_prefixes = [" ".join(content.split(" ", 2)[:2]) for content in selected_contents]

    assert actual_prefixes == expected_prefixes


@pytest.mark.asyncio
async def test_token_budget_retains_chronological_order() -> None:
    (
        repositories,
        conversation_service,
        user_context_service,
        session_context_service,
        _,
    ) = create_services()

    context_service = create_small_budget_service(
        user_context_service=(user_context_service),
        session_context_service=(session_context_service),
        conversation_service=(conversation_service),
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    for index in range(6):
        await conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=(f"Message {index}: " + ("y" * 1000)),
        )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
    )

    selected_indexes = [
        int(message.content.split(":", 1)[0].replace("Message ", ""))
        for message in context.recent_messages
    ]

    assert selected_indexes == sorted(selected_indexes)


@pytest.mark.asyncio
async def test_recent_message_limit_is_hard_cap_on_top_of_token_budget() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    for index in range(10):
        await conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=f"Message {index}",
        )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
        recent_message_limit=3,
    )

    assert len(context.recent_messages) <= 3

    assert [item.content for item in context.recent_messages] == [
        "Message 7",
        "Message 8",
        "Message 9",
    ]

    assert context.metadata.candidate_message_count == 3

    assert context.metadata.recent_message_count == 3

    assert context.metadata.dropped_message_count == 0


@pytest.mark.asyncio
async def test_recent_message_limit_zero_returns_no_history() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    for index in range(3):
        await conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=f"Message {index}",
        )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
        recent_message_limit=0,
    )

    assert context.recent_messages == []

    assert context.metadata.candidate_message_count == 0

    assert context.metadata.recent_message_count == 0

    assert context.metadata.estimated_history_tokens == 0


@pytest.mark.asyncio
async def test_negative_recent_message_limit_is_rejected() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    with pytest.raises(
        ValueError,
        match="recent_message_limit",
    ):
        await context_service.build(
            user_id=user.user_id,
            session_id=session.session_id,
            current_message="Current",
            recent_message_limit=-1,
        )


@pytest.mark.asyncio
async def test_current_persisted_message_is_excluded_from_recent_history() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Older message",
    )

    current_message = await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Current persisted message",
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message=("Current persisted message"),
        current_message_id=(current_message.message_id),
    )

    contents = [message.content for message in context.recent_messages]

    assert contents == [
        "Older message",
    ]

    assert "Current persisted message" not in contents


@pytest.mark.asyncio
async def test_budget_metadata_counts_selected_not_candidate_history() -> None:
    (
        repositories,
        conversation_service,
        user_context_service,
        session_context_service,
        _,
    ) = create_services()

    context_service = create_small_budget_service(
        user_context_service=(user_context_service),
        session_context_service=(session_context_service),
        conversation_service=(conversation_service),
    )

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    for index in range(6):
        await conversation_service.append_user_message(
            user_id=user.user_id,
            session_id=session.session_id,
            content=(f"Message {index}: " + ("z" * 1000)),
        )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
    )

    metadata = context.metadata

    assert metadata.candidate_message_count == 6

    assert metadata.recent_message_count == len(context.recent_messages)

    assert metadata.dropped_message_count == (
        metadata.candidate_message_count - metadata.recent_message_count
    )

    assert metadata.truncation_applied == (metadata.dropped_message_count > 0)


@pytest.mark.asyncio
async def test_summary_is_not_marked_as_included_during_4b7() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
    )

    # Session summaries become prompt context in 4B.8.
    assert context.metadata.summary_included is False


@pytest.mark.asyncio
async def test_rag_is_not_part_of_context_during_4b7() -> None:
    (
        repositories,
        conversation_service,
        _,
        _,
        context_service,
    ) = create_services()

    user, session = await create_user_and_session(
        repositories=repositories,
        conversation_service=(conversation_service),
    )

    context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="Current",
    )

    # RAG starts in Sprint 5.
    assert context.knowledge_chunks == []

    assert context.metadata.knowledge_chunk_count == 0


def create_services():
    """
    Build the real FREDi service stack used by ContextService tests.

    Returning services from the orchestration runtime avoids duplicating
    repository/service construction in this test module.
    """

    runtime = build_orchestration_runtime(
        model_service=MagicMock(spec=ModelService),
    )

    return (
        runtime.repositories,
        runtime.conversation_service,
        runtime.user_context_service,
        runtime.session_context_service,
        runtime.context_service,
    )


async def create_user_and_session(
    *,
    repositories,
    conversation_service,
):
    """Persist an isolated identity and create its session through the service."""

    user = await repositories.identities.create(
        UserIdentity(student_id="K12345"),
    )
    session = await conversation_service.create_session(
        user_id=user.user_id,
    )
    return user, session

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.context import ConversationMessage
from app.persistence.turns import TurnStatus
from app.repositories.exceptions import (
    RepositoryConflictError,
    RepositoryNotFoundError,
    RepositoryUnavailableError,
)
from tests.persistence.test_contracts import seed

pytestmark = pytest.mark.asyncio


async def test_concurrent_retries_admit_one_message_and_keep_one_response(pg):
    repositories = pg.repositories
    user, _, session = await seed(repositories)
    request_id = uuid4()
    scope = dict(user_id=user.user_id, session_id=session.session_id, request_id=request_id)
    admissions = await asyncio.gather(
        *(repositories.turns.begin(**scope, content="Why this step?") for _ in range(4))
    )
    assert sum(a.created for a in admissions) == 1
    assert len({a.turn.turn_id for a in admissions}) == 1
    assert (
        await repositories.conversations.count(user_id=user.user_id, session_id=session.session_id)
        == 1
    )
    response = ConversationMessage(
        user_id=user.user_id,
        session_id=session.session_id,
        role="assistant",
        content="Consider the prerequisite.",
        agent_name="process_coach",
    )
    completions = await asyncio.gather(
        *(repositories.turns.complete(**scope, response=response) for _ in range(3))
    )
    completed = completions[0]
    assert all(turn == completed for turn in completions)
    again = await repositories.turns.complete(
        **scope, response=response.model_copy(update={"message_id": uuid4()})
    )
    assert completed == again and completed.status == TurnStatus.COMPLETED
    assert (await repositories.turns.begin(**scope, content="Why this step?")).created is False
    assert (
        await repositories.conversations.count(user_id=user.user_id, session_id=session.session_id)
        == 2
    )
    with pytest.raises(RepositoryConflictError):
        await repositories.turns.begin(**scope, content="Different question")
    with pytest.raises(RepositoryConflictError):
        await repositories.turns.complete(
            **scope, response=response.model_copy(update={"content": "Different answer"})
        )


@pytest.mark.parametrize("status", [TurnStatus.FAILED, TurnStatus.INTERRUPTED])
async def test_partial_turn_is_explicit_and_terminal(pg, status):
    repositories = pg.repositories
    user, other, session = await seed(repositories)
    scope = dict(user_id=user.user_id, session_id=session.session_id, request_id=uuid4())
    await repositories.turns.begin(**scope, content="Help")
    with pytest.raises(RepositoryConflictError):
        await repositories.turns.begin(
            user_id=user.user_id,
            session_id=session.session_id,
            request_id=uuid4(),
            content="Another simultaneous question",
        )
    stopped = await repositories.turns.stop(**scope, status=status, error_code="model_unavailable")
    assert stopped.status == status and stopped.assistant_message_id is None
    assert (
        await repositories.turns.stop(**scope, status=status, error_code="model_unavailable")
        == stopped
    )
    with pytest.raises(RepositoryNotFoundError):
        await repositories.turns.get(**{**scope, "user_id": other.user_id})
    with pytest.raises(RepositoryConflictError):
        await repositories.turns.stop(
            **scope,
            status=TurnStatus.FAILED
            if status == TurnStatus.INTERRUPTED
            else TurnStatus.INTERRUPTED,
            error_code="changed",
        )
    with pytest.raises(ValueError):
        await repositories.turns.stop(
            **scope, status=status, error_code="Raw exception with secrets"
        )
    assert (
        await repositories.turns.begin(
            user_id=user.user_id,
            session_id=session.session_id,
            request_id=uuid4(),
            content="New attempt",
        )
    ).created


async def test_response_failure_rolls_back_message_and_status(pg):
    repositories = pg.repositories
    user, _, session = await seed(repositories)
    scope = dict(user_id=user.user_id, session_id=session.session_id, request_id=uuid4())
    await repositories.turns.begin(**scope, content="Question")
    async with pg.migrator.connection() as connection:
        await connection.execute(
            "CREATE FUNCTION application.reject_completion() RETURNS trigger LANGUAGE "
            "plpgsql AS $$ "
            "BEGIN IF NEW.status='completed' THEN RAISE EXCEPTION 'private-marker'; END "
            "IF; RETURN NEW; END $$; "
            "CREATE TRIGGER reject_completion BEFORE UPDATE ON application.turns "
            "FOR EACH ROW EXECUTE FUNCTION application.reject_completion()"
        )
    response = ConversationMessage(
        user_id=user.user_id, session_id=session.session_id, role="assistant", content="Answer"
    )
    with pytest.raises(RepositoryUnavailableError) as error:
        await repositories.turns.complete(**scope, response=response)
    assert "private-marker" not in str(error.value)
    assert (await repositories.turns.get(**scope)).status == TurnStatus.PROCESSING
    assert (
        await repositories.conversations.count(user_id=user.user_id, session_id=session.session_id)
        == 1
    )
    async with pg.migrator.connection() as connection:
        await connection.execute("DROP TRIGGER reject_completion ON application.turns")
    assert (
        await repositories.turns.complete(**scope, response=response)
    ).status == TurnStatus.COMPLETED


async def test_admission_failure_leaves_no_orphaned_turn(pg):
    repositories = pg.repositories
    user, _, session = await seed(repositories)
    scope = dict(user_id=user.user_id, session_id=session.session_id, request_id=uuid4())
    async with pg.migrator.connection() as connection:
        await connection.execute(
            "CREATE FUNCTION application.reject_message() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'private-marker'; END $$; "
            "CREATE TRIGGER reject_message BEFORE INSERT ON application.conversation_messages "
            "FOR EACH ROW EXECUTE FUNCTION application.reject_message()"
        )
    with pytest.raises(RepositoryUnavailableError):
        await repositories.turns.begin(**scope, content="Question")
    assert await repositories.turns.get(**scope) is None
    assert (
        await repositories.conversations.count(user_id=user.user_id, session_id=session.session_id)
        == 0
    )
    async with pg.migrator.connection() as connection:
        await connection.execute("DROP TRIGGER reject_message ON application.conversation_messages")
    assert (await repositories.turns.begin(**scope, content="Question")).created

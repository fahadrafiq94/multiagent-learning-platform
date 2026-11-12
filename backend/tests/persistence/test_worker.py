import argparse
import asyncio
from uuid import uuid4

import pytest

from app.context import ConversationMessage
from app.context.schemas import MessageSourceReference
from app.persistence.database import ApplicationDatabase
from app.persistence.turn_models import TurnStatus
from app.persistence.worker import ApplicationWorker
from app.repositories.exceptions import RepositoryConflictError, RepositoryUnavailableError
from scripts import application_admin
from tests.persistence.test_contracts import seed

pytestmark = pytest.mark.asyncio


async def test_worker_ownership_recovery_and_old_worker_fencing(pg):
    old = ApplicationWorker(pg.database)
    second_db = ApplicationDatabase(pg.database.config)
    second = ApplicationWorker(second_db)
    try:
        await old.start()
        user, _, session = await seed(pg.repositories)
        scope = dict(user_id=user.user_id, session_id=session.session_id, request_id=uuid4())
        await pg.repositories.turns.begin(**scope, content="Interrupted question")
        # Age is not proof that work has stopped. An active worker prevents recovery.
        async with pg.migrator.connection() as connection:
            await connection.execute("UPDATE application.turns SET updated_at='2000-01-01'")
        with pytest.raises(RepositoryConflictError, match="already running"):
            await second.start()
        assert (await pg.repositories.turns.get(**scope)).status == TurnStatus.PROCESSING
        await old.close()  # Simulate losing the process/ownership connection.
        with pytest.raises(RepositoryUnavailableError, match="no longer owns"):
            await pg.repositories.identities.get_by_user_id(user.user_id)
        await second.start()
        assert second.recovered_turns == 1
        with pytest.raises(RepositoryUnavailableError, match="no longer owns"):
            await pg.repositories.conversations.append(
                ConversationMessage(
                    user_id=user.user_id,
                    session_id=session.session_id,
                    role="assistant",
                    content="Old worker must not save this.",
                )
            )
        from app.repositories.postgres import create_postgres_repositories

        repositories = create_postgres_repositories(second_db)
        recovered = await repositories.turns.get(**scope)
        assert recovered.status == TurnStatus.INTERRUPTED
        assert recovered.error_code == "worker_restarted"
        assert (
            await repositories.turns.begin(
                **{**scope, "request_id": uuid4()},
                content="Deliberate new attempt",
            )
        ).created
    finally:
        await old.close()
        await second_db.close()
        await second.close()


async def test_different_request_ids_racing_admission_always_produce_conflicts(pg):
    user, _, session = await seed(pg.repositories)
    results = await asyncio.gather(
        *(
            pg.repositories.turns.begin(
                user_id=user.user_id,
                session_id=session.session_id,
                request_id=uuid4(),
                content="Question",
            )
            for _ in range(5)
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, RepositoryConflictError) for result in results) == 4
    assert (
        await pg.repositories.conversations.count(
            user_id=user.user_id,
            session_id=session.session_id,
        )
        == 1
    )


async def test_operator_provisioning_is_idempotent_and_recovery_refuses_active_backend(
    pg, monkeypatch
):
    monkeypatch.setattr(application_admin, "configured_dsn", lambda: pg.database.config.dsn)
    args = argparse.Namespace(command="create-user", student_id="operator-test")
    created = await application_admin.execute(args)
    repeated = await application_admin.execute(args)
    assert created.removeprefix("Created user: ") == repeated.removeprefix("Existing user: ")
    user = await pg.repositories.identities.get_by_student_id(student_id="operator-test")
    assert await pg.repositories.user_contexts.get(user.user_id)
    worker = ApplicationWorker(pg.database)
    try:
        await worker.start()
        with pytest.raises(RepositoryConflictError):
            await application_admin.execute(argparse.Namespace(command="recover"))
    finally:
        await worker.close()
    assert (
        await application_admin.execute(argparse.Namespace(command="recover"))
        == "Interrupted abandoned turns: 0."
    )


async def test_citation_provenance_round_trips_with_message_archive(pg):
    user, _, session = await seed(pg.repositories)
    citation = MessageSourceReference(
        document_id="document",
        revision_id="revision",
        chunk_id="chunk",
        page_start=2,
        page_end=3,
        section="Purchasing",
    )
    message = ConversationMessage(
        user_id=user.user_id,
        session_id=session.session_id,
        role="assistant",
        content="Course evidence.",
        source_references=[citation],
    )
    await pg.repositories.conversations.append(message)
    recovered = await pg.repositories.conversations.list_for_session(
        user_id=user.user_id,
        session_id=session.session_id,
    )
    assert recovered[0].source_references == [citation]

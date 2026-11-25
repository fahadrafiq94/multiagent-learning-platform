from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest
from psycopg.types.json import Jsonb

from app.context import ScenarioCard, SessionSummary, StepProgress, UserContext
from app.context.schemas import ScenarioDraft
from app.domain.guidance import GuidanceLevel
from app.orchestration.factory import build_orchestration_runtime
from app.persistence import migrations
from app.persistence.database import ApplicationDatabase, ApplicationDatabaseConfig
from app.repositories.exceptions import (
    RepositoryIntegrityError,
    RepositoryPermissionError,
    RepositoryUnavailableError,
)
from app.repositories.postgres import create_postgres_repositories
from app.services.model_service import ModelService
from scripts.application_credentials import local_application_dsn
from tests.persistence.test_contracts import seed

pytestmark = pytest.mark.asyncio


async def test_state_and_history_reopen_through_existing_services(pg):
    model = Mock(spec=ModelService)
    runtime = build_orchestration_runtime(model_service=model, repositories=pg.repositories)
    from app.identity import UserIdentity

    user = await runtime.repositories.identities.create(UserIdentity(student_id="restart-test"))
    session = await runtime.conversation_service.create_session(
        user_id=user.user_id, title="Procurement"
    )
    user_context = await runtime.user_context_service.get_or_create(user_id=user.user_id)
    user_context.scenario_card = ScenarioCard(user_id=user.user_id, company_context="Demo company")
    user_context.hint_state.guidance_level = GuidanceLevel.DIRECTED
    user_context.hint_state.manually_overridden = True
    user_context.learning_profile.support_preferences.append("Short hints")
    user_context.progress.steps.append(StepProgress(step_id="AP40-04", status="completed"))
    await runtime.repositories.user_contexts.save(user_context, expected_version=1)
    session_context = await runtime.repositories.session_contexts.get(
        user_id=user.user_id, session_id=session.session_id
    )
    session_context.current_step_id = "AP40-05"
    session_context.active_blocker = "Cannot locate the field"
    session_context.summary = SessionSummary(
        user_id=user.user_id, session_id=session.session_id, overview="Earlier explanation"
    )
    session_context.scenario_draft = ScenarioDraft(
        user_id=user.user_id, session_id=session.session_id, learner_role="Planner"
    )
    await runtime.repositories.session_contexts.save(session_context, expected_version=1)
    first = await runtime.conversation_service.append_user_message(
        user_id=user.user_id, session_id=session.session_id, content="Why this prerequisite?"
    )
    reply = await runtime.conversation_service.append_assistant_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Consider the business dependency.",
        agent_name="process_coach",
        route="process_coach",
    )
    await pg.database.close()
    # A new pool/runtime has no access to the original in-process repository objects.
    async with ApplicationDatabase(
        ApplicationDatabaseConfig(dsn=local_application_dsn(database=pg.name))
    ) as reopened:
        runtime = build_orchestration_runtime(
            model_service=model, repositories=create_postgres_repositories(reopened)
        )
        assert (await runtime.conversation_service.list_sessions(user_id=user.user_id))[
            0
        ].title == "Procurement"
        assert await runtime.conversation_service.get_history(
            user_id=user.user_id, session_id=session.session_id
        ) == [first, reply]
        context = await runtime.context_service.build(
            user_id=user.user_id, session_id=session.session_id, current_message="Explain further"
        )
        assert context.guidance_level == GuidanceLevel.DIRECTED
        assert context.session_context.current_step_id == "AP40-05"
        assert context.session_summary.overview == "Earlier explanation"
        assert context.scenario_card.company_context == "Demo company"
        assert context.session_context.scenario_draft.learner_role == "Planner"
        assert context.learning_profile.support_preferences == ["Short hints"]
        assert context.progress.steps[0].status == "completed"
    model.generate.assert_not_called()
    # Also verify storage from an independent interpreter, not just another pool.
    import json
    import subprocess
    import sys

    from scripts.knowledge_infra import ROOT

    script = """
import asyncio,json,sys
from uuid import UUID
from app.persistence.database import ApplicationDatabase,ApplicationDatabaseConfig
from app.repositories.postgres import create_postgres_repositories
from scripts.application_credentials import local_application_dsn
async def main():
    request=json.load(sys.stdin)
    async with ApplicationDatabase(ApplicationDatabaseConfig(
        dsn=local_application_dsn(database=request['database'])
    )) as database:
        repositories=create_postgres_repositories(database)
        context=await repositories.user_contexts.get(UUID(request['user']))
        assert context.hint_state.guidance_level=='directed'
        assert context.progress.steps[0].status=='completed'
        assert await repositories.conversations.count(
            user_id=UUID(request['user']),session_id=UUID(request['session'])
        )==2
asyncio.run(main(),loop_factory=asyncio.SelectorEventLoop)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        input=json.dumps(
            {"database": pg.name, "user": str(user.user_id), "session": str(session.session_id)}
        ),
        cwd=ROOT / "backend",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        "Independent interpreter could not restore persisted context/history."
    )


async def test_mutated_nested_owner_is_revalidated_before_sql(pg):
    user, other, _ = await seed(pg.repositories)
    context = UserContext(user_id=user.user_id)
    await pg.repositories.user_contexts.create(context)
    invalid = context.model_copy(update={"scenario_card": ScenarioCard(user_id=other.user_id)})
    with pytest.raises(RepositoryIntegrityError):
        await pg.repositories.user_contexts.save(invalid, expected_version=1)
    assert (await pg.repositories.user_contexts.get(user.user_id)).version == 1


async def test_invalid_stored_payload_is_sanitized(pg):
    user, _, _ = await seed(pg.repositories)
    await pg.repositories.user_contexts.create(UserContext(user_id=user.user_id))
    async with pg.migrator.connection() as connection:
        await connection.execute(
            "UPDATE application.user_contexts SET payload=%s",
            (Jsonb({"private-marker": "never-log-me"}),),
        )
    with pytest.raises(RepositoryIntegrityError) as error:
        await pg.repositories.user_contexts.get(user.user_id)
    assert "private-marker" not in str(error.value)


async def test_migrations_are_independent_idempotent_and_checksum_checked(
    pg, tmp_path, monkeypatch
):
    assert await migrations.migrate(pg.migrator) == []
    original = migrations.SQL_ROOT / "001_application.sql"
    for source in migrations.SQL_ROOT.glob("*.sql"):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    (tmp_path / original.name).write_text(original.read_text() + "\n-- drift\n")
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "SQL_ROOT", tmp_path)
        with pytest.raises(RepositoryIntegrityError, match="checksum"):
            await migrations.migrate(pg.migrator)
    with pytest.raises(RepositoryIntegrityError, match="migrator"):
        await migrations.migrate(pg.database)
    async with pg.migrator.connection() as connection:
        cursor = await connection.execute(
            "SELECT count(*) AS total FROM pg_extension WHERE extname='vector'"
        )
        assert (await cursor.fetchone())["total"] == 0


async def test_failed_migration_rolls_back_ddl_and_existing_data(pg, tmp_path, monkeypatch):
    user, _, _ = await seed(pg.repositories)
    for source in migrations.SQL_ROOT.glob("*.sql"):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    (tmp_path / "002_probe.sql").write_text(
        "CREATE TABLE application.rollback_probe(id int); SELECT * FROM application.missing_table;"
    )
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "SQL_ROOT", tmp_path)
        with pytest.raises(RepositoryUnavailableError):
            await migrations.migrate(pg.migrator)
    async with pg.migrator.connection() as connection:
        cursor = await connection.execute(
            "SELECT to_regclass('application.rollback_probe') AS probe"
        )
        assert (await cursor.fetchone())["probe"] is None
    assert await pg.repositories.identities.get_by_user_id(user.user_id)


async def test_runtime_role_cannot_change_schema_or_reassign_messages(pg):
    from app.context import ConversationMessage

    user, _, session = await seed(pg.repositories)
    message = ConversationMessage(
        user_id=user.user_id, session_id=session.session_id, role="user", content="Private message"
    )
    await pg.repositories.conversations.append(message)
    for statement in [
        "CREATE TABLE application.forbidden(id int)",
        "UPDATE application.conversation_messages SET role='assistant'",
        "UPDATE application.conversation_messages SET user_id=gen_random_uuid()",
        "SELECT * FROM application.schema_migrations",
    ]:
        with pytest.raises(RepositoryPermissionError):
            async with pg.database.connection() as connection:
                await connection.execute(statement)
    assert (
        await pg.repositories.conversations.count(
            user_id=user.user_id, session_id=session.session_id
        )
        == 1
    )


async def test_cancellation_rolls_back_before_returning_pool_connection(pg):
    user, _, _ = await seed(pg.repositories)
    entered = asyncio.Event()

    async def cancelled_write():
        async with pg.database.connection() as connection:
            await connection.execute(
                "UPDATE application.identities SET student_id='cancelled-marker' WHERE user_id=%s",
                (user.user_id,),
            )
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(cancelled_write())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await pg.repositories.identities.get_by_student_id("cancelled-marker") is None
    assert await pg.repositories.identities.get_by_user_id(user.user_id)


async def test_application_and_knowledge_roles_have_separate_access(pg):
    import psycopg

    from scripts.migrate_knowledge import local_dsn

    async with await psycopg.AsyncConnection.connect(
        local_dsn(role="admin", database=pg.name).get_secret_value()
    ) as admin:
        await admin.execute("CREATE SCHEMA knowledge AUTHORIZATION fredi_knowledge_writer")
        await admin.execute("CREATE TABLE knowledge.probe(id int)")
    with pytest.raises(RepositoryPermissionError):
        async with pg.database.connection() as connection:
            await connection.execute("SELECT * FROM knowledge.probe")
    async with await psycopg.AsyncConnection.connect(
        local_dsn(role="reader", database=pg.name).get_secret_value()
    ) as reader:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await reader.execute("SELECT * FROM application.identities")

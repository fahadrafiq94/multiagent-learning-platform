"""
Sprint 4B.11 layer A: in-memory repository concurrency.

Deterministic interleaving via exact barriers (every worker signals
it is ready, the test releases them together) — no sleeps, no
wall-clock assertions.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.context import (
    ConversationMessage,
    ConversationRole,
    ConversationSession,
    SessionContext,
    UserContext,
)
from app.identity import UserIdentity
from app.repositories import (
    DuplicateRecordError,
    InMemoryRepositories,
    RepositoryNotFoundError,
    VersionConflictError,
    create_in_memory_repositories,
)


async def create_user(
    repositories: InMemoryRepositories,
    *,
    student_id: str,
) -> UserIdentity:
    return await repositories.identities.create(UserIdentity(student_id=student_id))


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


async def seed_session(
    repositories: InMemoryRepositories,
    user: UserIdentity,
) -> ConversationSession:
    session = await repositories.sessions.create(
        ConversationSession(
            user_id=user.user_id,
            session_id=uuid4(),
        )
    )

    return session


@pytest.mark.asyncio
async def test_concurrent_duplicate_identity_creation_single_winner() -> None:
    repositories = create_in_memory_repositories()

    workers = 5

    barrier = Barrier(workers)

    async def worker() -> str:
        await barrier.wait()

        try:
            await repositories.identities.create(UserIdentity(student_id="K-CONCURRENT"))

            return "created"
        except DuplicateRecordError:
            return "duplicate"

    results = await asyncio.gather(*(worker() for _ in range(workers)))

    assert results.count("created") == 1

    assert results.count("duplicate") == workers - 1

    identity = await repositories.identities.get_by_student_id("K-CONCURRENT")

    assert identity is not None


@pytest.mark.asyncio
async def test_concurrent_user_context_saves_single_winner_and_version_advances_once() -> None:
    repositories = create_in_memory_repositories()

    user = await create_user(
        repositories,
        student_id="K-USER-CTX",
    )

    context = await repositories.user_contexts.create(UserContext(user_id=user.user_id))

    workers = 4

    barrier = Barrier(workers)

    async def worker(index: int) -> str:
        await barrier.wait()

        candidate = context.model_copy(deep=True)

        candidate.hint_state.successful_independent_steps = index

        try:
            await repositories.user_contexts.save(
                candidate,
                expected_version=(context.version),
            )

            return "saved"
        except VersionConflictError:
            return "conflict"

    results = await asyncio.gather(*(worker(i) for i in range(workers)))

    assert results.count("saved") == 1

    assert results.count("conflict") == workers - 1

    current = await repositories.user_contexts.get(user.user_id)

    assert current is not None

    assert current.version == context.version + 1

    assert current.hint_state.successful_independent_steps == results.index("saved")


@pytest.mark.asyncio
async def test_concurrent_session_context_saves_single_winner_and_version_advances_once() -> None:
    repositories = create_in_memory_repositories()

    user = await repositories.identities.create(UserIdentity(student_id="K-SESSION-CTX"))

    session = await seed_session(repositories, user)

    context = await repositories.session_contexts.create(
        SessionContext(
            user_id=user.user_id,
            session_id=session.session_id,
        )
    )

    workers = 4

    barrier = Barrier(workers)

    async def worker(index: int) -> str:
        await barrier.wait()

        candidate = context.model_copy(deep=True)

        candidate.current_objective = f"objective-{index}"

        try:
            await repositories.session_contexts.save(
                candidate,
                expected_version=(context.version),
            )

            return "saved"
        except VersionConflictError:
            return "conflict"

    results = await asyncio.gather(*(worker(i) for i in range(workers)))

    assert results.count("saved") == 1

    assert results.count("conflict") == workers - 1

    current = await repositories.session_contexts.get(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert current is not None

    assert current.version == context.version + 1

    assert current.current_objective == f"objective-{results.index('saved')}"


@pytest.mark.asyncio
async def test_concurrent_message_appends_all_succeed_without_loss() -> None:
    repositories = create_in_memory_repositories()

    user = await repositories.identities.create(UserIdentity(student_id="K-MESSAGES"))

    session = await seed_session(repositories, user)

    workers = 6

    barrier = Barrier(workers)

    async def worker(index: int) -> str:
        await barrier.wait()

        canary = f"canary-message-{index}-7f3a"

        try:
            await repositories.conversations.append(
                ConversationMessage(
                    user_id=user.user_id,
                    session_id=session.session_id,
                    role=ConversationRole.USER,
                    content=canary,
                )
            )

            return canary
        except Exception as exc:
            return f"error:{exc}"

    results = await asyncio.gather(*(worker(i) for i in range(workers)))

    assert all(not result.startswith("error:") for result in results)

    count = await repositories.conversations.count(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert count == workers

    history = await repositories.conversations.list_for_session(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    stored_contents = [message.content for message in history]

    assert len(stored_contents) == workers

    assert sorted(stored_contents) == sorted(results)

    assert len(set(stored_contents)) == workers


@pytest.mark.asyncio
async def test_cross_user_reads_during_writes_never_leak() -> None:
    repositories = create_in_memory_repositories()

    user_a = await repositories.identities.create(UserIdentity(student_id="K-CROSS-A"))

    user_b = await repositories.identities.create(UserIdentity(student_id="K-CROSS-B"))

    session = await seed_session(repositories, user_a)

    readers: list[object] = []

    writers: list[str] = []

    stop = asyncio.Event()

    async def reader() -> None:
        while not stop.is_set():
            session_view = await repositories.sessions.get(
                user_id=user_b.user_id,
                session_id=session.session_id,
            )

            readers.append(session_view)

            try:
                messages = await repositories.conversations.list_for_session(
                    user_id=user_b.user_id,
                    session_id=session.session_id,
                )

                readers.append(messages)
            except RepositoryNotFoundError:
                readers.append(None)

            await asyncio.sleep(0)

    async def writer() -> None:
        for index in range(5):
            writers.append(f"secret-canary-{index}")

            await repositories.conversations.append(
                ConversationMessage(
                    user_id=user_a.user_id,
                    session_id=session.session_id,
                    role=ConversationRole.USER,
                    content=f"secret-canary-{index}",
                )
            )

    reader_task = asyncio.create_task(reader())

    writer_task = asyncio.create_task(writer())

    await writer_task

    stop.set()

    await reader_task

    # Every read B attempted against A's session came back empty
    # (session lookup) or was rejected (message listing) — none
    # returned A's data.
    assert readers

    assert all(entry is None for entry in readers)

    history = await repositories.conversations.list_for_session(
        user_id=user_a.user_id,
        session_id=session.session_id,
    )

    assert [message.content for message in history] == sorted(writers)


@pytest.mark.asyncio
async def test_reads_during_appends_return_complete_deep_copies() -> None:
    repositories = create_in_memory_repositories()

    user = await repositories.identities.create(UserIdentity(student_id="K-SNAPSHOT"))

    session = await seed_session(repositories, user)

    snapshots: list[list[ConversationMessage]] = []

    stop = asyncio.Event()

    async def reader() -> None:
        while not stop.is_set():
            snapshot = await repositories.conversations.list_recent(
                user_id=user.user_id,
                session_id=session.session_id,
                limit=10,
            )

            snapshots.append(snapshot)

            await asyncio.sleep(0)

    async def writer() -> None:
        for index in range(5):
            await repositories.conversations.append(
                ConversationMessage(
                    user_id=user.user_id,
                    session_id=session.session_id,
                    role=ConversationRole.USER,
                    content=f"snapshot-canary-{index}",
                )
            )

    reader_task = asyncio.create_task(reader())

    writer_task = asyncio.create_task(writer())

    await writer_task

    stop.set()

    await reader_task

    assert snapshots

    expected = {f"snapshot-canary-{i}" for i in range(5)}

    for snapshot in snapshots:
        for message in snapshot:
            # Every returned message is one of the fully-written
            # canaries — never a partial/half-appended value.
            assert message.content in expected

    # Mutating a returned snapshot must not touch stored state.
    snapshots[0].append(
        ConversationMessage(
            user_id=user.user_id,
            session_id=session.session_id,
            role=ConversationRole.USER,
            content="injected-after-snapshot",
        )
    )

    final = await repositories.conversations.list_for_session(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    assert all(message.content != "injected-after-snapshot" for message in final)

    assert len(final) == 5

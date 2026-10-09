from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from app.context import (
    ConversationMessage,
    ConversationSession,
    SessionContext,
    UserContext,
)
from app.identity import (
    IdentityProvider,
    UserIdentity,
)
from app.repositories.context import (
    ConversationSessionRepository,
)
from app.repositories.exceptions import (
    DuplicateRecordError,
    RepositoryNotFoundError,
    VersionConflictError,
)
from app.repositories.identity import (
    UserIdentityRepository,
)

ExternalIdentityKey = tuple[
    IdentityProvider,
    str,
    str,
]


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(UTC)


def _normalize_key(
    value: str,
) -> str:
    """Normalize repository lookup keys."""

    return value.strip()


class InMemoryUserIdentityRepository:
    """
    In-memory implementation of UserIdentityRepository.

    This repository exists for:

    - Sprint 4 development
    - unit/integration tests
    - repository-contract validation

    It intentionally applies the same uniqueness rules that a
    future relational database implementation should enforce.
    """

    def __init__(self) -> None:
        self._by_user_id: dict[
            UUID,
            UserIdentity,
        ] = {}

        self._student_id_index: dict[
            str,
            UUID,
        ] = {}

        self._external_identity_index: dict[
            ExternalIdentityKey,
            UUID,
        ] = {}

        self._lock = asyncio.Lock()

    async def get_by_user_id(
        self,
        user_id: UUID,
    ) -> UserIdentity | None:
        async with self._lock:
            identity = self._by_user_id.get(user_id)

            if identity is None:
                return None

            return identity.model_copy(deep=True)

    async def get_by_student_id(
        self,
        student_id: str,
    ) -> UserIdentity | None:
        normalized_student_id = _normalize_key(student_id)

        async with self._lock:
            user_id = self._student_id_index.get(normalized_student_id)

            if user_id is None:
                return None

            identity = self._by_user_id[user_id]

            return identity.model_copy(deep=True)

    async def get_by_external_identity(
        self,
        *,
        provider: IdentityProvider,
        provider_instance: str,
        external_user_id: str,
    ) -> UserIdentity | None:
        key = self._external_key(
            provider=provider,
            provider_instance=(provider_instance),
            external_user_id=(external_user_id),
        )

        async with self._lock:
            user_id = self._external_identity_index.get(key)

            if user_id is None:
                return None

            identity = self._by_user_id[user_id]

            return identity.model_copy(deep=True)

    async def create(
        self,
        identity: UserIdentity,
    ) -> UserIdentity:
        async with self._lock:
            self._validate_new_identity(identity)

            stored = identity.model_copy(deep=True)

            self._store_identity(stored)

            return stored.model_copy(deep=True)

    async def update(
        self,
        identity: UserIdentity,
    ) -> UserIdentity:
        async with self._lock:
            current = self._by_user_id.get(identity.user_id)

            if current is None:
                raise RepositoryNotFoundError(
                    entity_name="UserIdentity",
                    entity_id=str(identity.user_id),
                )

            self._validate_updated_identity(
                identity=identity,
                current=current,
            )

            self._remove_indexes(current)

            stored = identity.model_copy(
                update={
                    "updated_at": _utc_now(),
                },
                deep=True,
            )

            self._store_identity(stored)

            return stored.model_copy(deep=True)

    def _validate_new_identity(
        self,
        identity: UserIdentity,
    ) -> None:
        if identity.user_id in self._by_user_id:
            raise DuplicateRecordError(
                entity_name="UserIdentity",
                key=(f"user_id={identity.user_id}"),
            )

        student_id = _normalize_key(identity.student_id)

        if student_id in self._student_id_index:
            raise DuplicateRecordError(
                entity_name="UserIdentity",
                key=(f"student_id={student_id}"),
            )

        self._validate_external_identities(
            identity=identity,
            current_user_id=None,
        )

    def _validate_updated_identity(
        self,
        *,
        identity: UserIdentity,
        current: UserIdentity,
    ) -> None:
        student_id = _normalize_key(identity.student_id)

        owner = self._student_id_index.get(student_id)

        if owner is not None and owner != identity.user_id:
            raise DuplicateRecordError(
                entity_name="UserIdentity",
                key=(f"student_id={student_id}"),
            )

        self._validate_external_identities(
            identity=identity,
            current_user_id=current.user_id,
        )

    def _validate_external_identities(
        self,
        *,
        identity: UserIdentity,
        current_user_id: UUID | None,
    ) -> None:
        seen: set[ExternalIdentityKey] = set()

        for external_identity in identity.external_identities:
            key = self._external_key(
                provider=(external_identity.provider),
                provider_instance=(external_identity.provider_instance),
                external_user_id=(external_identity.external_user_id),
            )

            if key in seen:
                raise DuplicateRecordError(
                    entity_name=("ExternalIdentity"),
                    key=self._format_external_key(key),
                )

            seen.add(key)

            owner = self._external_identity_index.get(key)

            if owner is not None and owner != current_user_id:
                raise DuplicateRecordError(
                    entity_name=("ExternalIdentity"),
                    key=self._format_external_key(key),
                )

    def _store_identity(
        self,
        identity: UserIdentity,
    ) -> None:
        self._by_user_id[identity.user_id] = identity

        student_id = _normalize_key(identity.student_id)

        self._student_id_index[student_id] = identity.user_id

        for external_identity in identity.external_identities:
            key = self._external_key(
                provider=(external_identity.provider),
                provider_instance=(external_identity.provider_instance),
                external_user_id=(external_identity.external_user_id),
            )

            self._external_identity_index[key] = identity.user_id

    def _remove_indexes(
        self,
        identity: UserIdentity,
    ) -> None:
        student_id = _normalize_key(identity.student_id)

        self._student_id_index.pop(
            student_id,
            None,
        )

        for external_identity in identity.external_identities:
            key = self._external_key(
                provider=(external_identity.provider),
                provider_instance=(external_identity.provider_instance),
                external_user_id=(external_identity.external_user_id),
            )

            self._external_identity_index.pop(
                key,
                None,
            )

    @staticmethod
    def _external_key(
        *,
        provider: IdentityProvider,
        provider_instance: str,
        external_user_id: str,
    ) -> ExternalIdentityKey:
        return (
            provider,
            _normalize_key(provider_instance),
            _normalize_key(external_user_id),
        )

    @staticmethod
    def _format_external_key(
        key: ExternalIdentityKey,
    ) -> str:
        (
            provider,
            provider_instance,
            external_user_id,
        ) = key

        return f"{provider.value}:{provider_instance}:{external_user_id}"


class InMemoryUserContextRepository:
    """
    In-memory persistent user-context repository.

    Every context belongs to an existing FREDi UserIdentity.

    Returned models are deep copies so callers cannot accidentally
    mutate repository state without calling save().
    """

    def __init__(
        self,
        *,
        identity_repository: (UserIdentityRepository),
    ) -> None:
        self._identity_repository = identity_repository

        self._contexts: dict[
            UUID,
            UserContext,
        ] = {}

        self._lock = asyncio.Lock()

    async def get(
        self,
        user_id: UUID,
    ) -> UserContext | None:
        await self._require_user(user_id)

        async with self._lock:
            context = self._contexts.get(user_id)

            if context is None:
                return None

            return context.model_copy(deep=True)

    async def create(
        self,
        context: UserContext,
    ) -> UserContext:
        await self._require_user(context.user_id)

        async with self._lock:
            if context.user_id in self._contexts:
                raise DuplicateRecordError(
                    entity_name="UserContext",
                    key=(f"user_id={context.user_id}"),
                )

            stored = context.model_copy(deep=True)

            self._contexts[context.user_id] = stored

            return stored.model_copy(deep=True)

    async def save(
        self,
        context: UserContext,
        *,
        expected_version: int,
    ) -> UserContext:
        await self._require_user(context.user_id)

        async with self._lock:
            current = self._contexts.get(context.user_id)

            if current is None:
                raise RepositoryNotFoundError(
                    entity_name="UserContext",
                    entity_id=str(context.user_id),
                )

            if current.version != expected_version:
                raise VersionConflictError(
                    entity_name="UserContext",
                    entity_id=str(context.user_id),
                    expected_version=(expected_version),
                    actual_version=(current.version),
                )

            stored = context.model_copy(
                update={
                    "version": (current.version + 1),
                    "updated_at": _utc_now(),
                },
                deep=True,
            )

            self._contexts[context.user_id] = stored

            return stored.model_copy(deep=True)

    async def _require_user(
        self,
        user_id: UUID,
    ) -> None:
        identity = await self._identity_repository.get_by_user_id(user_id)

        if identity is None:
            raise RepositoryNotFoundError(
                entity_name="UserIdentity",
                entity_id=str(user_id),
            )


class InMemoryConversationSessionRepository:
    """
    In-memory repository for user-visible conversation sessions.

    session_id is globally unique.

    Every session must belong to an existing FREDi user.
    """

    def __init__(
        self,
        *,
        identity_repository: (UserIdentityRepository),
    ) -> None:
        self._identity_repository = identity_repository

        self._sessions: dict[
            tuple[UUID, UUID],
            ConversationSession,
        ] = {}

        self._session_owners: dict[
            UUID,
            UUID,
        ] = {}

        self._lock = asyncio.Lock()

    async def create(
        self,
        session: ConversationSession,
    ) -> ConversationSession:
        await self._require_user(session.user_id)

        async with self._lock:
            if session.session_id in self._session_owners:
                raise DuplicateRecordError(
                    entity_name=("ConversationSession"),
                    key=(f"session_id={session.session_id}"),
                )

            stored = session.model_copy(deep=True)

            key = (
                session.user_id,
                session.session_id,
            )

            self._sessions[key] = stored

            self._session_owners[session.session_id] = session.user_id

            return stored.model_copy(deep=True)

    async def get(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> ConversationSession | None:
        async with self._lock:
            session = self._sessions.get(
                (
                    user_id,
                    session_id,
                )
            )

            if session is None:
                return None

            return session.model_copy(deep=True)

    async def list_for_user(
        self,
        user_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationSession]:
        await self._require_user(user_id)

        self._validate_pagination(
            limit=limit,
            offset=offset,
        )

        async with self._lock:
            sessions = [
                session.model_copy(deep=True)
                for (
                    stored_user_id,
                    _,
                ), session in (self._sessions.items())
                if (stored_user_id == user_id)
            ]

        sessions.sort(
            key=lambda session: session.updated_at,
            reverse=True,
        )

        return sessions[offset : offset + limit]

    async def update(
        self,
        session: ConversationSession,
    ) -> ConversationSession:
        await self._require_user(session.user_id)

        async with self._lock:
            owner = self._session_owners.get(session.session_id)

            if owner is None or owner != session.user_id:
                raise RepositoryNotFoundError(
                    entity_name=("ConversationSession"),
                    entity_id=str(session.session_id),
                )

            key = (
                session.user_id,
                session.session_id,
            )

            stored = session.model_copy(
                update={
                    "updated_at": _utc_now(),
                },
                deep=True,
            )

            self._sessions[key] = stored

            return stored.model_copy(deep=True)

    async def _require_user(
        self,
        user_id: UUID,
    ) -> None:
        identity = await self._identity_repository.get_by_user_id(user_id)

        if identity is None:
            raise RepositoryNotFoundError(
                entity_name="UserIdentity",
                entity_id=str(user_id),
            )

    @staticmethod
    def _validate_pagination(
        *,
        limit: int,
        offset: int,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        if offset < 0:
            raise ValueError("offset must not be negative")


class InMemorySessionContextRepository:
    """
    In-memory repository for per-session execution state.

    A SessionContext can exist only for a valid user/session pair.

    Writes use optimistic version checking.
    """

    def __init__(
        self,
        *,
        session_repository: (ConversationSessionRepository),
    ) -> None:
        self._session_repository = session_repository

        self._contexts: dict[
            tuple[UUID, UUID],
            SessionContext,
        ] = {}

        self._lock = asyncio.Lock()

    async def get(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> SessionContext | None:
        await self._require_session(
            user_id=user_id,
            session_id=session_id,
        )

        async with self._lock:
            context = self._contexts.get(
                (
                    user_id,
                    session_id,
                )
            )

            if context is None:
                return None

            return context.model_copy(deep=True)

    async def create(
        self,
        context: SessionContext,
    ) -> SessionContext:
        await self._require_session(
            user_id=context.user_id,
            session_id=context.session_id,
        )

        key = (
            context.user_id,
            context.session_id,
        )

        async with self._lock:
            if key in self._contexts:
                raise DuplicateRecordError(
                    entity_name=("SessionContext"),
                    key=(f"session_id={context.session_id}"),
                )

            stored = context.model_copy(deep=True)

            self._contexts[key] = stored

            return stored.model_copy(deep=True)

    async def save(
        self,
        context: SessionContext,
        *,
        expected_version: int,
    ) -> SessionContext:
        await self._require_session(
            user_id=context.user_id,
            session_id=context.session_id,
        )

        key = (
            context.user_id,
            context.session_id,
        )

        async with self._lock:
            current = self._contexts.get(key)

            if current is None:
                raise RepositoryNotFoundError(
                    entity_name="SessionContext",
                    entity_id=str(context.session_id),
                )

            if current.version != expected_version:
                raise VersionConflictError(
                    entity_name="SessionContext",
                    entity_id=str(context.session_id),
                    expected_version=(expected_version),
                    actual_version=(current.version),
                )

            stored = context.model_copy(
                update={
                    "version": (current.version + 1),
                    "updated_at": _utc_now(),
                },
                deep=True,
            )

            self._contexts[key] = stored

            return stored.model_copy(deep=True)

    async def _require_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> None:
        session = await self._session_repository.get(
            user_id=user_id,
            session_id=session_id,
        )

        if session is None:
            raise RepositoryNotFoundError(
                entity_name=("ConversationSession"),
                entity_id=str(session_id),
            )


class InMemoryConversationRepository:
    """
    In-memory complete-message archive.

    Messages are isolated by user_id + session_id.

    Every message operation validates that the session belongs to
    the supplied user.
    """

    def __init__(
        self,
        *,
        session_repository: (ConversationSessionRepository),
    ) -> None:
        self._session_repository = session_repository

        self._messages: dict[
            tuple[UUID, UUID],
            list[ConversationMessage],
        ] = {}

        self._message_ids: dict[
            UUID,
            tuple[UUID, UUID],
        ] = {}

        self._lock = asyncio.Lock()

    async def append(
        self,
        message: ConversationMessage,
    ) -> ConversationMessage:
        await self._require_session(
            user_id=message.user_id,
            session_id=message.session_id,
        )

        key = (
            message.user_id,
            message.session_id,
        )

        async with self._lock:
            if message.message_id in self._message_ids:
                raise DuplicateRecordError(
                    entity_name=("ConversationMessage"),
                    key=(f"message_id={message.message_id}"),
                )

            stored = message.model_copy(deep=True)

            self._messages.setdefault(
                key,
                [],
            ).append(stored)

            self._message_ids[message.message_id] = key

            return stored.model_copy(deep=True)

    async def list_for_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ConversationMessage]:
        await self._require_session(
            user_id=user_id,
            session_id=session_id,
        )

        self._validate_pagination(
            limit=limit,
            offset=offset,
        )

        key = (
            user_id,
            session_id,
        )

        async with self._lock:
            messages = [
                message.model_copy(deep=True)
                for message in (
                    self._messages.get(
                        key,
                        [],
                    )
                )
            ]

        messages.sort(key=lambda message: message.created_at)

        return messages[offset : offset + limit]

    async def list_recent(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int = 10,
    ) -> list[ConversationMessage]:
        await self._require_session(
            user_id=user_id,
            session_id=session_id,
        )

        if limit < 1:
            raise ValueError("limit must be greater than zero")

        key = (
            user_id,
            session_id,
        )

        async with self._lock:
            messages = [
                message.model_copy(deep=True)
                for message in (
                    self._messages.get(
                        key,
                        [],
                    )
                )
            ]

        messages.sort(key=lambda message: message.created_at)

        return messages[-limit:]

    async def count(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> int:
        await self._require_session(
            user_id=user_id,
            session_id=session_id,
        )

        key = (
            user_id,
            session_id,
        )

        async with self._lock:
            return len(
                self._messages.get(
                    key,
                    [],
                )
            )

    async def _require_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> None:
        session = await self._session_repository.get(
            user_id=user_id,
            session_id=session_id,
        )

        if session is None:
            raise RepositoryNotFoundError(
                entity_name=("ConversationSession"),
                entity_id=str(session_id),
            )

    @staticmethod
    def _validate_pagination(
        *,
        limit: int,
        offset: int,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        if offset < 0:
            raise ValueError("offset must not be negative")


@dataclass(slots=True)
class InMemoryRepositories:
    """
    Fully wired Sprint 4 in-memory repository collection.

    Consumers should normally create repositories through
    create_in_memory_repositories() instead of wiring the
    dependencies manually.
    """

    identities: InMemoryUserIdentityRepository

    user_contexts: InMemoryUserContextRepository

    sessions: InMemoryConversationSessionRepository

    session_contexts: InMemorySessionContextRepository

    conversations: InMemoryConversationRepository


def create_in_memory_repositories() -> InMemoryRepositories:
    """
    Create a correctly wired in-memory persistence layer.

    Dependency flow:

        identities
            |
            +--> user contexts
            |
            +--> conversation sessions
                        |
                        +--> session contexts
                        |
                        +--> conversation messages
    """

    identities = InMemoryUserIdentityRepository()

    user_contexts = InMemoryUserContextRepository(
        identity_repository=identities,
    )

    sessions = InMemoryConversationSessionRepository(
        identity_repository=identities,
    )

    session_contexts = InMemorySessionContextRepository(
        session_repository=sessions,
    )

    conversations = InMemoryConversationRepository(
        session_repository=sessions,
    )

    return InMemoryRepositories(
        identities=identities,
        user_contexts=user_contexts,
        sessions=sessions,
        session_contexts=session_contexts,
        conversations=conversations,
    )

"""Storage-independent repository wiring; constructing a bundle performs no I/O."""

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from app.context import ConversationSession
from app.repositories.context import (
    ConversationRepository,
    ConversationSessionRepository,
    SessionContextRepository,
    UserContextRepository,
)
from app.repositories.identity import UserIdentityRepository
from app.repositories.turns import TurnStore


@runtime_checkable
class RepositoryBundle(Protocol):
    @property
    def turns(self) -> TurnStore | None: ...

    @property
    def session_initializer(
        self,
    ) -> Callable[[ConversationSession], Awaitable[ConversationSession]] | None: ...

    @property
    def identities(self) -> UserIdentityRepository: ...

    @property
    def user_contexts(self) -> UserContextRepository: ...

    @property
    def sessions(self) -> ConversationSessionRepository: ...

    @property
    def session_contexts(self) -> SessionContextRepository: ...

    @property
    def conversations(self) -> ConversationRepository: ...

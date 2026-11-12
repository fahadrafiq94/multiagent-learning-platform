"""Connect atomic storage to one graph invocation without retrying student effects."""

import asyncio
from collections.abc import Awaitable, Callable
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.context import ConversationMessage
from app.orchestration.schemas import OrchestrationRequest, OrchestrationResponse
from app.persistence.turn_models import TurnStatus
from app.repositories.exceptions import (
    RepositoryConflictError,
    RepositoryError,
    RepositoryIntegrityError,
    RepositoryUnavailableError,
)
from app.repositories.turns import TurnStore


class DurableTurnService:
    def __init__(self, store: TurnStore):
        self.store = store

    async def execute(
        self,
        request: OrchestrationRequest,
        *,
        user_id: UUID,
        session_id: UUID,
        run: Callable[[UUID, UUID], Awaitable[OrchestrationResponse]],
    ) -> OrchestrationResponse:
        assert request.request_id is not None  # Validated by orchestration service.
        scope = dict(user_id=user_id, session_id=session_id, request_id=request.request_id)
        trace_id = uuid4()
        admitted = await self.store.begin(**scope, content=request.user_message, trace_id=trace_id)
        if not admitted.created:
            if admitted.turn.status == TurnStatus.COMPLETED:
                try:
                    return OrchestrationResponse.model_validate(admitted.turn.response_payload)
                except ValidationError:
                    raise RepositoryIntegrityError("Stored turn response is unavailable.") from None
            if admitted.turn.status == TurnStatus.PROCESSING:
                raise RepositoryConflictError(
                    "Request is still processing; retry the same request ID later."
                )
            raise RepositoryConflictError(
                f"Request is {admitted.turn.status}; "
                "inspect its status before a deliberate new attempt."
            )
        code = "workflow_failed"
        try:
            response = await run(admitted.turn.user_message_id, trace_id)
            if response.error:
                raise RepositoryUnavailableError("The workflow could not complete.")
            code = "invalid_response"
            message = ConversationMessage(
                user_id=user_id,
                session_id=session_id,
                role="assistant",
                content=response.final_response,
                agent_name=response.metadata.get("selected_agent"),
                route=response.route,
                trace_id=response.metadata.get("trace_id"),
                source_references=response.source_references,
            )
            response = response.model_copy(
                update={
                    "final_response": message.content,
                    "metadata": {
                        **response.metadata,
                        "assistant_message_persisted": True,
                        "request_id": str(request.request_id),
                    },
                }
            )
            code = "persistence_failed"
            await self.store.complete(
                **scope,
                response=message,
                response_payload=response.model_dump(mode="json"),
            )
            return response
        except asyncio.CancelledError:
            await self._stop(scope, TurnStatus.INTERRUPTED, "request_cancelled")
            raise
        except Exception as exc:
            await self._stop(scope, TurnStatus.FAILED, code)
            if isinstance(exc, RepositoryError):
                raise
            raise RepositoryUnavailableError(
                "The request did not complete; inspect its turn status."
            ) from None

    async def _stop(self, scope: dict[str, UUID], status: TurnStatus, code: str) -> None:
        try:
            await self.store.stop(**scope, status=status, error_code=code)
        except RepositoryError:
            # A disconnected/committed-unknown DB leaves the turn for replay/recovery.
            # Never mask cancellation or overwrite a completion with a guessed failure.
            pass

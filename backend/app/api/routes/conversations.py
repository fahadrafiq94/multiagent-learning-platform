"""Owner-scoped conversation/history APIs. No student ID in the access scope."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.api.dependencies import CurrentUserDep, RuntimeDep
from app.context import ConversationMessage, ConversationSession, SessionContext
from app.context.schemas import ShortText
from app.persistence.turn_models import TurnStatus

router = APIRouter(prefix="/conversations", tags=["conversations"])
Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


class CreateConversation(BaseModel):
    title: ShortText | None = None


class ConversationPage(BaseModel):
    items: list[ConversationSession]
    limit: int
    offset: int


class ConversationDetail(BaseModel):
    session: ConversationSession
    context: SessionContext


class HistoryPage(BaseModel):
    items: list[ConversationMessage]
    limit: int
    offset: int


class TurnView(BaseModel):
    request_id: UUID
    status: TurnStatus
    error_code: str | None
    user_message_id: UUID
    assistant_message_id: UUID | None


@router.post("", response_model=ConversationSession, status_code=201)
async def create_conversation(body: CreateConversation, user: CurrentUserDep, runtime: RuntimeDep):
    return await runtime.conversation_service.create_session(user_id=user.user_id, title=body.title)


@router.get("", response_model=ConversationPage)
async def list_conversations(
    user: CurrentUserDep, runtime: RuntimeDep, limit: Limit = 50, offset: Offset = 0
):
    items = await runtime.conversation_service.list_sessions(
        user_id=user.user_id, limit=limit, offset=offset
    )
    return ConversationPage(items=items, limit=limit, offset=offset)


@router.get("/{session_id}", response_model=ConversationDetail)
async def reopen_conversation(session_id: UUID, user: CurrentUserDep, runtime: RuntimeDep):
    session = await runtime.conversation_service.get_session(
        user_id=user.user_id, session_id=session_id
    )
    context = await runtime.conversation_service.get_session_context(
        user_id=user.user_id, session_id=session_id
    )
    return ConversationDetail(session=session, context=context)


@router.get("/{session_id}/messages", response_model=HistoryPage)
async def get_history(
    session_id: UUID,
    user: CurrentUserDep,
    runtime: RuntimeDep,
    limit: Limit = 50,
    offset: Offset = 0,
):
    items = await runtime.conversation_service.get_history(
        user_id=user.user_id,
        session_id=session_id,
        limit=limit,
        offset=offset,
    )
    return HistoryPage(items=items, limit=limit, offset=offset)


@router.get("/{session_id}/turns/{request_id}", response_model=TurnView)
async def get_turn(session_id: UUID, request_id: UUID, user: CurrentUserDep, runtime: RuntimeDep):
    if runtime.repositories.turns is None:
        raise HTTPException(503, detail="Durable turn storage is not configured.")
    turn = await runtime.repositories.turns.get(
        user_id=user.user_id,
        session_id=session_id,
        request_id=request_id,
    )
    if turn is None:
        raise HTTPException(404, detail="Turn not found.")
    return TurnView(**turn.model_dump())

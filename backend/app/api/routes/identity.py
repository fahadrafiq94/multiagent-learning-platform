"""Minimal authenticated identity projection for the student browser."""

from uuid import UUID

from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.api.dependencies import CurrentUserDep

router = APIRouter(tags=["identity"])


class CurrentUserView(BaseModel):
    user_id: UUID
    student_id: str


@router.get("/me", response_model=CurrentUserView)
async def current_user(user: CurrentUserDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return CurrentUserView(user_id=user.user_id, student_id=user.student_id)

"""Read-only student projections; internal receipts and summaries stay server-side."""

from uuid import UUID

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from app.api.dependencies import CurrentUserDep, RuntimeDep
from app.context.schemas import ScenarioSetupStatus

router = APIRouter(prefix="/student-api", tags=["student workspace"])


class ScenarioView(BaseModel):
    status: ScenarioSetupStatus
    company_context: str | None = None
    learner_role: str | None = None
    business_problem: str | None = None
    business_goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    version: int | None = None


class WorkspaceView(BaseModel):
    session_id: UUID
    can_regenerate: bool = False
    scenario: ScenarioView
    current_objective: str | None = None
    current_step_id: str | None = None
    current_process_phase: str | None = None
    active_blocker: str | None = None


@router.get("/conversations/{session_id}", response_model=WorkspaceView)
async def workspace(
    session_id: UUID, user: CurrentUserDep, runtime: RuntimeDep, response: Response
):
    await runtime.conversation_service.get_session(user_id=user.user_id, session_id=session_id)
    context = await runtime.conversation_service.get_session_context(
        user_id=user.user_id, session_id=session_id
    )
    profile = await runtime.repositories.user_contexts.get(user.user_id)
    card = profile.scenario_card if profile else None
    source = card or context.scenario_draft
    fields = (
        "company_context",
        "learner_role",
        "business_problem",
        "business_goals",
        "constraints",
    )
    scenario = ScenarioView(
        status=ScenarioSetupStatus.CONFIRMED if card else context.scenario_setup_status,
        **({field: getattr(source, field) for field in fields} if source else {}),
        version=card.version if card else None,
    )
    response.headers["Cache-Control"] = "no-store"
    return WorkspaceView(
        session_id=session_id,
        can_regenerate=runtime.repositories.turns is not None,
        scenario=scenario,
        current_objective=context.current_objective,
        current_step_id=context.current_step_id,
        current_process_phase=context.current_process_phase,
        active_blocker=context.active_blocker,
    )


@router.get("/scenario", response_model=ScenarioView)
async def user_scenario(user: CurrentUserDep, runtime: RuntimeDep, response: Response):
    """The user-level saved scenario card, readable without an open session."""
    profile = await runtime.repositories.user_contexts.get(user.user_id)
    card = profile.scenario_card if profile else None
    scenario = ScenarioView(
        status=ScenarioSetupStatus.CONFIRMED if card else ScenarioSetupStatus.NOT_STARTED,
        **(
            {
                field: getattr(card, field)
                for field in (
                    "company_context",
                    "learner_role",
                    "business_problem",
                    "business_goals",
                    "constraints",
                )
            }
            if card
            else {}
        ),
        version=card.version if card else None,
    )
    response.headers["Cache-Control"] = "no-store"
    return scenario

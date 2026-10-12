from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from app.context.schemas import (
    ScenarioCard,
    ScenarioDraft,
    ScenarioDraftAssessment,
    ScenarioSetupStatus,
)


class ScenarioInteractionOutcome(StrEnum):
    COLLECTING = "collecting"

    REVIEW_REQUIRED = "review_required"

    CORRECTED = "corrected"

    CORRECTION_REQUIRED = "correction_required"

    CLARIFICATION_REQUIRED = "clarification_required"

    CONFIRMED = "confirmed"

    ALREADY_CONFIRMED = "already_confirmed"


class ScenarioOnboardingTurnResult(BaseModel):
    outcome: ScenarioInteractionOutcome

    status: ScenarioSetupStatus

    draft: ScenarioDraft | None = None

    assessment: ScenarioDraftAssessment | None = None

    scenario_card: ScenarioCard | None = None

    changed: bool = False

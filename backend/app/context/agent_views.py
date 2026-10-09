from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.context.schemas import (
    ContextChunk,
    ContextModel,
    ConversationMessage,
    ScenarioCard,
    ScenarioDraft,
    ScenarioSetupStatus,
    SessionSummary,
    UserLearningProfile,
    UserProgress,
    UserSpecificAdaptation,
)
from app.domain import GuidanceLevel


class ContextTarget(StrEnum):
    """
    Runtime consumer of a filtered AgentContext.

    The orchestrator is included even though it is not a specialist
    agent because it also needs a deliberately restricted context.
    """

    ORCHESTRATOR = "orchestrator"
    SCENARIO = "scenario"
    PROCESS_COACH = "process_coach"
    AP_PLUS_NAVIGATOR = "ap_plus_navigator"


class RuntimeSessionContext(ContextModel):
    """
    Runtime session information that may be exposed to specialist
    agents.

    Scenario onboarding state is intentionally excluded because that
    has its own dedicated view and lifecycle owner.
    """

    current_objective: str | None = None
    current_step_id: str | None = None
    current_process_phase: str | None = None
    active_blocker: str | None = None
    last_selected_agent: str | None = None

    unresolved_questions: list[str] = Field(default_factory=list)


class ScenarioSetupContext(ContextModel):
    """
    Temporary scenario-onboarding state.

    This view deliberately keeps ScenarioDraft separate from the
    confirmed ScenarioCard.
    """

    status: ScenarioSetupStatus
    draft: ScenarioDraft | None = None


class AgentContextView(ContextModel):
    """
    Filtered context made available to one runtime consumer.

    All optional sections default to absent. AgentContextPolicyService
    explicitly selects which sections each consumer may receive.
    """

    target: ContextTarget

    user_id: UUID
    session_id: UUID

    current_message: str

    scenario_card: ScenarioCard | None = None

    scenario_setup: ScenarioSetupContext | None = None

    runtime_session: RuntimeSessionContext | None = None

    learning_profile: UserLearningProfile | None = None
    progress: UserProgress | None = None

    recent_messages: list[ConversationMessage] = Field(default_factory=list)

    session_summary: SessionSummary | None = None

    guidance_level: GuidanceLevel | None = None

    user_adaptations: list[UserSpecificAdaptation] = Field(default_factory=list)

    knowledge_chunks: list[ContextChunk] = Field(default_factory=list)

    policy_version: str

    included_sections: list[str] = Field(default_factory=list)

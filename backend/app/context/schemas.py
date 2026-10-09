from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from app.domain.guidance import GuidanceLevel

ShortText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=255,
    ),
]

MediumText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1_000,
    ),
]

LongText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=4_000,
    ),
]


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(UTC)


class ContextModel(BaseModel):
    """
    Base model for Sprint 4 context schemas.

    Context schemas reject unknown fields so accidental state
    does not silently enter persistent or model-facing context.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class ScenarioFactSource(StrEnum):
    """Allowed provenance categories for confirmed scenario facts."""

    USER = "user"
    ASSIGNMENT_DOCUMENT = "assignment_document"
    INSTRUCTOR = "instructor"
    SYSTEM = "system"
    EXTERNAL_PLATFORM = "external_platform"
    USER_CONFIRMED_AGENT_PROPOSAL = "user_confirmed_agent_proposal"


class SessionStatus(StrEnum):
    """Lifecycle state for one conversation session."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class ConversationRole(StrEnum):
    """Persisted conversational roles."""

    USER = "user"
    ASSISTANT = "assistant"


class ProgressStatus(StrEnum):
    """Progress state for one learning/process step."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    NEEDS_REVIEW = "needs_review"


class AdaptationSource(StrEnum):
    """Source of one user-specific behavioral adaptation."""

    EXPLICIT_USER_PREFERENCE = "explicit_user_preference"

    MANUALLY_VALIDATED_PATTERN = "manually_validated_pattern"


class ConfirmedScenarioFact(ContextModel):
    """
    One durable scenario fact with provenance.

    Only confirmed facts belong in ScenarioCard.confirmed_facts.

    Unconfirmed LLM assumptions or proposals must remain outside
    the persistent Scenario Card until confirmed.
    """

    fact_id: UUID = Field(
        default_factory=uuid4,
    )

    fact: LongText

    source_type: ScenarioFactSource

    source_reference: ShortText | None = None

    created_at: datetime = Field(
        default_factory=utc_now,
    )


class ScenarioUpdateCandidate(ContextModel):
    """
    Temporary proposed Scenario Card update.

    A candidate is deliberately outside the persistent
    ScenarioCard.

    Examples:

    - an agent inference
    - ambiguous information extracted from a user message
    - information extracted from a document that still requires
      confirmation

    Only explicit validation moves a candidate into ScenarioCard.
    """

    candidate_id: UUID = Field(
        default_factory=uuid4,
    )

    user_id: UUID

    target: ScenarioCardTarget

    value: LongText

    source: ScenarioCandidateSource

    evidence: LongText | None = None

    source_reference: ShortText | None = None

    created_at: datetime = Field(
        default_factory=utc_now,
    )


class ScenarioCard(ContextModel):
    """
    Compact persistent business-scenario context.

    This intentionally contains only information that is useful
    across sessions and can materially affect an agent response.

    Detailed ERP master data, current workflow state, temporary
    questions, and session-specific information do not belong here.
    """

    user_id: UUID

    company_context: LongText | None = None

    learner_role: ShortText | None = None

    business_problem: LongText | None = None

    business_goals: list[MediumText] = Field(
        default_factory=list,
    )

    constraints: list[MediumText] = Field(
        default_factory=list,
    )

    confirmed_facts: list[ConfirmedScenarioFact] = Field(
        default_factory=list,
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )

    version: int = Field(
        default=1,
        ge=1,
    )


class ScenarioSetupStatus(StrEnum):
    """
    Lifecycle state for scenario onboarding in one session.
    """

    NOT_STARTED = "not_started"
    COLLECTING = "collecting"
    REVIEWING = "reviewing"
    CONFIRMED = "confirmed"


class ScenarioRequiredField(StrEnum):
    """
    Minimum information required before scenario review.
    """

    COMPANY_CONTEXT = "company_context"
    LEARNER_ROLE = "learner_role"
    BUSINESS_PROBLEM = "business_problem"
    BUSINESS_GOALS = "business_goals"


class UserLearningProfile(ContextModel):
    """
    Durable learning-related information about one user.

    The default profile is intentionally empty so every new user
    starts from the same baseline.
    """

    demonstrated_competencies: list[MediumText] = Field(
        default_factory=list,
    )

    recurring_misconceptions: list[MediumText] = Field(
        default_factory=list,
    )

    support_preferences: list[MediumText] = Field(
        default_factory=list,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )


class StepProgress(ContextModel):
    """Persistent progress state for one learning/process step."""

    step_id: ShortText

    status: ProgressStatus = ProgressStatus.NOT_STARTED

    updated_at: datetime = Field(
        default_factory=utc_now,
    )


class UserProgress(ContextModel):
    """
    Cross-session user progress.

    This is different from the active step of one conversation
    session. A session may discuss a specific step while UserProgress
    represents durable progress through the learning environment.
    """

    steps: list[StepProgress] = Field(
        default_factory=list,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )


class HintState(ContextModel):
    """
    Current personalized guidance state.

    Guidance level is controlled by policy rather than being
    permanently chosen by an LLM.
    """

    guidance_level: GuidanceLevel = GuidanceLevel.MINIMAL

    consecutive_blocked_attempts: int = Field(
        default=0,
        ge=0,
    )

    successful_independent_steps: int = Field(
        default=0,
        ge=0,
    )

    manually_overridden: bool = False

    last_change_reason: MediumText | None = None

    updated_at: datetime = Field(
        default_factory=utc_now,
    )


class UserSpecificAdaptation(ContextModel):
    """
    A behavior or response preference that applies only to one user.

    Sprint 4 does not automatically promote interactions into
    global skills.
    """

    adaptation_id: UUID = Field(
        default_factory=uuid4,
    )

    category: ShortText

    instruction: MediumText

    source: AdaptationSource

    active: bool = True

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )


class UserContext(ContextModel):
    """
    Persistent cross-session context for one FREDi user.

    This object is keyed by FREDi's immutable internal user_id,
    not by the institutional student_id.

    Every nested user-owned object must belong to the same user.
    """

    user_id: UUID

    scenario_card: ScenarioCard | None = None

    learning_profile: UserLearningProfile = Field(
        default_factory=UserLearningProfile,
    )

    progress: UserProgress = Field(
        default_factory=UserProgress,
    )

    hint_state: HintState = Field(
        default_factory=HintState,
    )

    adaptations: list[UserSpecificAdaptation] = Field(
        default_factory=list,
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )

    version: int = Field(
        default=1,
        ge=1,
    )

    @model_validator(mode="after")
    def validate_user_owned_context(
        self,
    ) -> UserContext:
        """
        Ensure nested persistent context belongs to this user.

        This prevents a ScenarioCard belonging to one user from
        accidentally being attached to another UserContext.
        """

        if self.scenario_card is not None and self.scenario_card.user_id != self.user_id:
            raise ValueError("scenario_card.user_id must match UserContext.user_id")

        return self


class ConversationSession(ContextModel):
    """
    Persistent metadata for one chatbot conversation.

    This allows the frontend to list and reopen previous chats.
    """

    session_id: UUID = Field(
        default_factory=uuid4,
    )

    user_id: UUID

    title: ShortText | None = None

    status: SessionStatus = SessionStatus.ACTIVE

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )


class ConversationMessage(ContextModel):
    """
    One persisted message in a conversation session.

    Full conversation history can be stored while ContextService
    later selects only a bounded subset for model context.
    """

    message_id: UUID = Field(
        default_factory=uuid4,
    )

    user_id: UUID

    session_id: UUID

    role: ConversationRole

    content: LongText

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    agent_name: ShortText | None = None

    route: ShortText | None = None

    trace_id: UUID | None = None


class SessionSummary(ContextModel):
    """
    Structured compact representation of older conversation history.

    The complete message archive remains the source of truth.
    """

    summary_id: UUID = Field(
        default_factory=uuid4,
    )

    user_id: UUID

    session_id: UUID

    overview: LongText | None = None

    established_facts: list[MediumText] = Field(
        default_factory=list,
    )

    decisions: list[MediumText] = Field(
        default_factory=list,
    )

    blockers: list[MediumText] = Field(
        default_factory=list,
    )

    unresolved_questions: list[MediumText] = Field(
        default_factory=list,
    )

    misconception_candidates: list[MediumText] = Field(
        default_factory=list,
    )

    completed_step_ids: list[ShortText] = Field(
        default_factory=list,
    )

    source_message_count: int = Field(
        default=0,
        ge=0,
    )

    generated_at: datetime = Field(
        default_factory=utc_now,
    )


class SessionContext(ContextModel):
    user_id: UUID

    session_id: UUID

    current_objective: LongText | None = None

    current_step_id: ShortText | None = None

    current_process_phase: ShortText | None = None

    active_blocker: LongText | None = None

    last_selected_agent: ShortText | None = None

    unresolved_questions: list[MediumText] = Field(
        default_factory=list,
    )

    summary: SessionSummary | None = None

    scenario_setup_status: ScenarioSetupStatus = ScenarioSetupStatus.NOT_STARTED

    scenario_draft: ScenarioDraft | None = None

    version: int = Field(
        default=1,
        ge=1,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )

    @model_validator(mode="after")
    def validate_session_owned_context(
        self,
    ) -> SessionContext:
        if self.summary is not None:
            if self.summary.user_id != self.user_id:
                raise ValueError("summary.user_id must match SessionContext.user_id")

            if self.summary.session_id != self.session_id:
                raise ValueError("summary.session_id must match SessionContext.session_id")

        if self.scenario_draft is not None:
            if self.scenario_draft.user_id != self.user_id:
                raise ValueError("scenario_draft.user_id must match SessionContext.user_id")

            if self.scenario_draft.session_id != self.session_id:
                raise ValueError("scenario_draft.session_id must match SessionContext.session_id")

        return self


class SessionContextUpdate(ContextModel):
    """
    Partial update for mutable per-session runtime state.

    Fields that are omitted remain unchanged.

    Explicitly passing None clears nullable scalar fields.

    Example:

        SessionContextUpdate(
            current_step_id="8",
            active_blocker=None,
        )

    updates the step and clears the blocker while leaving all
    other SessionContext fields unchanged.
    """

    current_objective: LongText | None = None

    current_step_id: ShortText | None = None

    current_process_phase: ShortText | None = None

    active_blocker: LongText | None = None

    last_selected_agent: ShortText | None = None

    unresolved_questions: list[MediumText] | None = None


class ContextChunk(ContextModel):
    """
    One knowledge item selected for the current request.

    Sprint 5 will populate these through structured knowledge/RAG.
    Defining the contract now prevents ContextService from later
    depending on retrieval implementation details.
    """

    chunk_id: ShortText

    content: LongText

    source: ShortText

    document_id: ShortText | None = None

    section: ShortText | None = None

    step_id: ShortText | None = None

    version: ShortText | None = None

    retrieval_score: float | None = None


class ContextMetadata(ContextModel):
    """
    Observability metadata describing how AgentContext was built.

    Full prompt contents do not need to be duplicated here.
    """

    trace_id: UUID | None = None

    included_sections: list[ShortText] = Field(
        default_factory=list,
    )

    recent_message_count: int = Field(
        default=0,
        ge=0,
    )

    summary_included: bool = False

    knowledge_chunk_count: int = Field(
        default=0,
        ge=0,
    )

    adaptation_count: int = Field(
        default=0,
        ge=0,
    )

    estimated_input_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    truncation_applied: bool = False


class AgentContext(ContextModel):
    """
    Bounded request-scoped context supplied to one selected agent.

    AgentContext is not the same as UserContext. ContextService
    selects only the fields relevant to the chosen agent.

    Every nested user/session-owned object must belong to the
    same execution identity.
    """

    user_id: UUID

    session_id: UUID

    current_message: LongText

    scenario_card: ScenarioCard | None = None

    learning_profile: UserLearningProfile | None = None

    progress: UserProgress | None = None

    session_context: SessionContext | None = None

    recent_messages: list[ConversationMessage] = Field(
        default_factory=list,
    )

    session_summary: SessionSummary | None = None

    guidance_level: GuidanceLevel = GuidanceLevel.MINIMAL

    user_adaptations: list[UserSpecificAdaptation] = Field(
        default_factory=list,
    )

    knowledge_chunks: list[ContextChunk] = Field(
        default_factory=list,
    )

    metadata: ContextMetadata = Field(
        default_factory=ContextMetadata,
    )

    @model_validator(mode="after")
    def validate_execution_identity(
        self,
    ) -> AgentContext:
        """
        Prevent context belonging to another user or session from
        entering this agent invocation.
        """

        self._validate_scenario_card()
        self._validate_session_context()
        self._validate_session_summary()
        self._validate_recent_messages()

        return self

    def _validate_scenario_card(
        self,
    ) -> None:
        if self.scenario_card is None:
            return

        if self.scenario_card.user_id != self.user_id:
            raise ValueError("scenario_card.user_id must match AgentContext.user_id")

    def _validate_session_context(
        self,
    ) -> None:
        if self.session_context is None:
            return

        if self.session_context.user_id != self.user_id:
            raise ValueError("session_context.user_id must match AgentContext.user_id")

        if self.session_context.session_id != self.session_id:
            raise ValueError("session_context.session_id must match AgentContext.session_id")

    def _validate_session_summary(
        self,
    ) -> None:
        if self.session_summary is None:
            return

        if self.session_summary.user_id != self.user_id:
            raise ValueError("session_summary.user_id must match AgentContext.user_id")

        if self.session_summary.session_id != self.session_id:
            raise ValueError("session_summary.session_id must match AgentContext.session_id")

    def _validate_recent_messages(
        self,
    ) -> None:
        for message in self.recent_messages:
            if message.user_id != self.user_id:
                raise ValueError("recent message user_id must match AgentContext.user_id")

            if message.session_id != self.session_id:
                raise ValueError("recent message session_id must match AgentContext.session_id")


class ScenarioCardTarget(StrEnum):
    """
    A field that ScenarioCardService may update.

    company_size intentionally does not exist as a dedicated
    Scenario Card field.
    """

    COMPANY_CONTEXT = "company_context"
    LEARNER_ROLE = "learner_role"
    BUSINESS_PROBLEM = "business_problem"
    BUSINESS_GOAL = "business_goal"
    CONSTRAINT = "constraint"
    CONFIRMED_FACT = "confirmed_fact"


class ScenarioCandidateSource(StrEnum):
    """
    Origin of an unconfirmed scenario proposal.

    These values describe candidates only. They must not be
    confused with confirmed ScenarioFactSource values.
    """

    AGENT_INFERENCE = "agent_inference"

    USER_MESSAGE_EXTRACTION = "user_message_extraction"

    DOCUMENT_EXTRACTION = "document_extraction"


class ScenarioDraftFact(ContextModel):
    """
    A proposed scenario fact inside an onboarding draft.

    It is not yet a persistent ConfirmedScenarioFact.
    """

    fact: LongText

    source_type: ScenarioFactSource

    source_reference: ShortText | None = None


class ScenarioDraft(ContextModel):
    """
    Temporary complete scenario proposal used during onboarding.

    ScenarioDraft lives in SessionContext.

    It must never be treated as persistent UserContext until the
    user reviews and confirms it.
    """

    draft_id: UUID = Field(
        default_factory=uuid4,
    )

    user_id: UUID

    session_id: UUID

    company_context: LongText | None = None

    learner_role: ShortText | None = None

    business_problem: LongText | None = None

    business_goals: list[MediumText] = Field(
        default_factory=list,
    )

    constraints: list[MediumText] = Field(
        default_factory=list,
    )

    proposed_facts: list[ScenarioDraftFact] = Field(
        default_factory=list,
    )

    source_message_ids: list[UUID] = Field(
        default_factory=list,
    )

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )


class ScenarioDraftAssessment(ContextModel):
    """
    Deterministic assessment of scenario completeness.
    """

    ready_for_review: bool

    missing_required_fields: list[ScenarioRequiredField] = Field(
        default_factory=list,
    )

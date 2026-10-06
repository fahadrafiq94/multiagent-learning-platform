from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.context import (
    ScenarioDraft,
    ScenarioDraftAssessment,
    ScenarioRequiredField,
    ScenarioSetupStatus,
)
from app.services.scenario_card_service import (
    ScenarioCardService,
)
from app.services.session_context_service import (
    SessionContextService,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ScenarioOnboardingService:
    """
    Coordinates first-time scenario setup.

    The service manages lifecycle and persistence rules.

    The LLM may extract ScenarioDraft fields, but deterministic code
    decides whether the draft contains the required minimum.
    """

    def __init__(
        self,
        *,
        session_context_service: SessionContextService,
        scenario_card_service: ScenarioCardService,
    ) -> None:
        self._session_context_service = session_context_service

        self._scenario_card_service = scenario_card_service

    async def needs_onboarding(
        self,
        *,
        user_id: UUID,
    ) -> bool:
        card = await self._scenario_card_service.get(
            user_id=user_id,
        )

        return card is None

    async def start(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> ScenarioSetupStatus:
        """
        Start scenario onboarding.

        If a confirmed Scenario Card already exists, onboarding is
        skipped and this session is marked CONFIRMED.
        """

        context = await self._session_context_service.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        card = await self._scenario_card_service.get(
            user_id=user_id,
        )

        if card is not None:
            context.scenario_setup_status = ScenarioSetupStatus.CONFIRMED

            context.scenario_draft = None

        elif context.scenario_setup_status == ScenarioSetupStatus.NOT_STARTED:
            context.scenario_setup_status = ScenarioSetupStatus.COLLECTING

        saved = await self._session_context_service.save(
            user_id=user_id,
            session_id=session_id,
            context=context,
            expected_version=(context.version),
        )

        return saved.scenario_setup_status

    def assess(
        self,
        draft: ScenarioDraft,
    ) -> ScenarioDraftAssessment:
        """
        Deterministically determine which required fields are absent.
        """

        missing: list[ScenarioRequiredField] = []

        if draft.company_context is None:
            missing.append(ScenarioRequiredField.COMPANY_CONTEXT)

        if draft.learner_role is None:
            missing.append(ScenarioRequiredField.LEARNER_ROLE)

        if draft.business_problem is None:
            missing.append(ScenarioRequiredField.BUSINESS_PROBLEM)

        if not draft.business_goals:
            missing.append(ScenarioRequiredField.BUSINESS_GOALS)

        return ScenarioDraftAssessment(
            ready_for_review=not missing,
            missing_required_fields=missing,
        )

    async def save_draft(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        draft: ScenarioDraft,
    ) -> ScenarioDraftAssessment:
        """
        Save or replace the temporary onboarding draft.

        A complete draft enters REVIEWING.

        An incomplete draft remains COLLECTING.
        """

        if draft.user_id != user_id:
            raise ValueError("draft.user_id must match ScenarioOnboardingService user_id")

        if draft.session_id != session_id:
            raise ValueError("draft.session_id must match ScenarioOnboardingService session_id")

        context = await self._session_context_service.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        if context.scenario_setup_status == ScenarioSetupStatus.CONFIRMED:
            raise ValueError("Scenario onboarding is already confirmed")

        assessment = self.assess(draft)

        draft.updated_at = _utc_now()

        context.scenario_draft = draft

        if assessment.ready_for_review:
            context.scenario_setup_status = ScenarioSetupStatus.REVIEWING

        else:
            context.scenario_setup_status = ScenarioSetupStatus.COLLECTING

        await self._session_context_service.save(
            user_id=user_id,
            session_id=session_id,
            context=context,
            expected_version=(context.version),
        )

        return assessment

    async def get_draft(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> ScenarioDraft | None:
        context = await self._session_context_service.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        if context.scenario_draft is None:
            return None

        return context.scenario_draft.model_copy(deep=True)

    async def confirm(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ):
        """
        Confirm the current complete draft.

        This is the point where scenario information crosses from
        session state into persistent UserContext.
        """

        context = await self._session_context_service.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        draft = context.scenario_draft

        if draft is None:
            raise ValueError("No ScenarioDraft exists for confirmation")

        assessment = self.assess(draft)

        if not assessment.ready_for_review:
            missing = ", ".join(field.value for field in (assessment.missing_required_fields))

            raise ValueError(f"ScenarioDraft is incomplete. Missing: {missing}")

        card = await self._scenario_card_service.confirm_draft(
            user_id=user_id,
            draft=draft,
        )

        # Refetch in case other session-state changes occurred while
        # the persistent Scenario Card was being written.
        current_context = await self._session_context_service.get_required(
            user_id=user_id,
            session_id=session_id,
        )

        current_context.scenario_setup_status = ScenarioSetupStatus.CONFIRMED

        current_context.scenario_draft = None

        await self._session_context_service.save(
            user_id=user_id,
            session_id=session_id,
            context=current_context,
            expected_version=(current_context.version),
        )

        return card

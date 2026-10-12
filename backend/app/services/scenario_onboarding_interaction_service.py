from __future__ import annotations

from uuid import UUID

from app.agents.scenario import (
    ScenarioAgent,
)
from app.agents.scenario_schemas import (
    ScenarioReviewIntent,
)
from app.context.agent_views import (
    AgentContextView,
    ContextTarget,
)
from app.context.schemas import (
    ScenarioDraft,
    ScenarioSetupStatus,
)
from app.services.scenario_onboarding_interaction_schemas import (
    ScenarioInteractionOutcome,
    ScenarioOnboardingTurnResult,
)
from app.services.scenario_onboarding_service import (
    ScenarioOnboardingService,
)


class ScenarioOnboardingInteractionService:
    """
    Coordinate one scenario-onboarding interaction.

    LLM:
        understands and extracts language.

    Application:
        owns lifecycle and persistence.
    """

    def __init__(
        self,
        *,
        scenario_agent: ScenarioAgent,
        onboarding_service: (ScenarioOnboardingService),
    ) -> None:
        self._scenario_agent = scenario_agent

        self._onboarding_service = onboarding_service

    async def handle_turn(
        self,
        *,
        context: AgentContextView,
        source_message_id: (UUID | None) = None,
    ) -> ScenarioOnboardingTurnResult:
        self._validate_context(context)

        assert context.scenario_setup is not None

        status = context.scenario_setup.status

        if status == ScenarioSetupStatus.CONFIRMED:
            return ScenarioOnboardingTurnResult(
                outcome=(ScenarioInteractionOutcome.ALREADY_CONFIRMED),
                status=(ScenarioSetupStatus.CONFIRMED),
                scenario_card=(context.scenario_card),
                changed=False,
            )

        if status == ScenarioSetupStatus.REVIEWING:
            return await self._handle_review_turn(
                context=context,
                source_message_id=(source_message_id),
            )

        return await self._handle_collection_turn(
            context=context,
            source_message_id=(source_message_id),
        )

    async def _handle_collection_turn(
        self,
        *,
        context: AgentContextView,
        source_message_id: (UUID | None),
    ) -> ScenarioOnboardingTurnResult:
        assert context.scenario_setup is not None

        if context.scenario_setup.status == ScenarioSetupStatus.NOT_STARTED:
            await self._onboarding_service.start(
                user_id=context.user_id,
                session_id=(context.session_id),
            )

        draft = await self._scenario_agent.extract_draft(
            context=context,
            source_message_id=(source_message_id),
        )

        await self._onboarding_service.save_draft(
            user_id=context.user_id,
            session_id=context.session_id,
            draft=draft,
        )

        persisted = await self._onboarding_service.get_draft(
            user_id=context.user_id,
            session_id=(context.session_id),
        )

        if persisted is None:
            raise RuntimeError("ScenarioDraft was not persisted.")

        assessment = self._onboarding_service.assess(persisted)

        if assessment.ready_for_review:
            return ScenarioOnboardingTurnResult(
                outcome=(ScenarioInteractionOutcome.REVIEW_REQUIRED),
                status=(ScenarioSetupStatus.REVIEWING),
                draft=persisted,
                assessment=assessment,
                changed=True,
            )

        return ScenarioOnboardingTurnResult(
            outcome=(ScenarioInteractionOutcome.COLLECTING),
            status=(ScenarioSetupStatus.COLLECTING),
            draft=persisted,
            assessment=assessment,
            changed=True,
        )

    async def _handle_review_turn(
        self,
        *,
        context: AgentContextView,
        source_message_id: (UUID | None),
    ) -> ScenarioOnboardingTurnResult:
        assert context.scenario_setup is not None

        existing = context.scenario_setup.draft

        if existing is None:
            raise ValueError("REVIEWING context must contain ScenarioDraft.")

        decision = await self._scenario_agent.assess_review(context=context)

        if decision.intent == ScenarioReviewIntent.CONFIRM:
            card = await self._onboarding_service.confirm(
                user_id=(context.user_id),
                session_id=(context.session_id),
            )

            return ScenarioOnboardingTurnResult(
                outcome=(ScenarioInteractionOutcome.CONFIRMED),
                status=(ScenarioSetupStatus.CONFIRMED),
                scenario_card=card,
                changed=True,
            )

        if decision.intent == ScenarioReviewIntent.REJECT:
            return ScenarioOnboardingTurnResult(
                outcome=(ScenarioInteractionOutcome.CORRECTION_REQUIRED),
                status=(ScenarioSetupStatus.REVIEWING),
                draft=existing,
                assessment=(self._onboarding_service.assess(existing)),
                changed=False,
            )

        if decision.intent == ScenarioReviewIntent.UNCLEAR:
            return ScenarioOnboardingTurnResult(
                outcome=(ScenarioInteractionOutcome.CLARIFICATION_REQUIRED),
                status=(ScenarioSetupStatus.REVIEWING),
                draft=existing,
                assessment=(self._onboarding_service.assess(existing)),
                changed=False,
            )

        correction = decision.correction

        if correction is None:
            raise RuntimeError("CORRECT intent requires correction data.")

        corrected = self._scenario_agent.merge_extraction(
            context=context,
            extraction=correction,
            source_message_id=(source_message_id),
        )

        if not self._business_content_changed(
            before=existing,
            after=corrected,
        ):
            return ScenarioOnboardingTurnResult(
                outcome=(ScenarioInteractionOutcome.CLARIFICATION_REQUIRED),
                status=(ScenarioSetupStatus.REVIEWING),
                draft=existing,
                assessment=(self._onboarding_service.assess(existing)),
                changed=False,
            )

        await self._onboarding_service.save_draft(
            user_id=context.user_id,
            session_id=context.session_id,
            draft=corrected,
        )

        persisted = await self._onboarding_service.get_draft(
            user_id=context.user_id,
            session_id=(context.session_id),
        )

        if persisted is None:
            raise RuntimeError("Corrected ScenarioDraft was not persisted.")

        assessment = self._onboarding_service.assess(persisted)

        status = (
            ScenarioSetupStatus.REVIEWING
            if assessment.ready_for_review
            else ScenarioSetupStatus.COLLECTING
        )

        return ScenarioOnboardingTurnResult(
            outcome=(ScenarioInteractionOutcome.CORRECTED),
            status=status,
            draft=persisted,
            assessment=assessment,
            changed=True,
        )

    @staticmethod
    def _business_content_changed(
        *,
        before: ScenarioDraft,
        after: ScenarioDraft,
    ) -> bool:
        return (
            before.company_context != after.company_context
            or before.learner_role != after.learner_role
            or before.business_problem != after.business_problem
            or before.business_goals != after.business_goals
            or before.constraints != after.constraints
            or [fact.fact for fact in before.proposed_facts]
            != [fact.fact for fact in after.proposed_facts]
        )

    @staticmethod
    def _validate_context(
        context: AgentContextView,
    ) -> None:
        if context.target is not ContextTarget.SCENARIO:
            raise ValueError("Scenario onboarding requires SCENARIO context.")

        if context.scenario_setup is None:
            raise ValueError("Scenario onboarding requires scenario_setup context.")

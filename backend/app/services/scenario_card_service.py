from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.context import (
    ConfirmedScenarioFact,
    ScenarioCandidateSource,
    ScenarioCard,
    ScenarioCardTarget,
    ScenarioDraft,
    ScenarioFactSource,
    ScenarioUpdateCandidate,
    UserContext,
)
from app.repositories import (
    RepositoryNotFoundError,
    VersionConflictError,
)
from app.services.user_context_service import (
    UserContextService,
)
from app.utils.text import (
    text_comparison_key as _normalized_text,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ScenarioCardService:
    """
    Persistent confirmed Scenario Card management.

    Initial Scenario Card creation happens only after a complete
    ScenarioDraft has been reviewed and confirmed.

    Unconfirmed onboarding information remains in SessionContext.
    """

    def __init__(
        self,
        *,
        user_context_service: UserContextService,
    ) -> None:
        self._user_context_service = user_context_service

    async def get(
        self,
        *,
        user_id: UUID,
    ) -> ScenarioCard | None:
        context = await self._user_context_service.get(
            user_id=user_id,
        )

        if context is None or context.scenario_card is None:
            return None

        return context.scenario_card.model_copy(deep=True)

    async def confirm_draft(
        self,
        *,
        user_id: UUID,
        draft: ScenarioDraft,
    ) -> ScenarioCard:
        """
        Convert a reviewed ScenarioDraft into the user's initial
        persistent Scenario Card.

        This is the only initial-card creation path.
        """

        if draft.user_id != user_id:
            raise ValueError("draft.user_id must match ScenarioCardService user_id")

        if draft.company_context is None:
            raise ValueError("company_context is required")

        if draft.learner_role is None:
            raise ValueError("learner_role is required")

        if draft.business_problem is None:
            raise ValueError("business_problem is required")

        if not draft.business_goals:
            raise ValueError("at least one business_goal is required")

        last_conflict: VersionConflictError | None = None

        for _ in range(3):
            context = await self._user_context_service.get_or_create(
                user_id=user_id,
            )

            if context.scenario_card is not None:
                return context.scenario_card.model_copy(deep=True)

            card = ScenarioCard(
                user_id=user_id,
                company_context=(draft.company_context),
                learner_role=(draft.learner_role),
                business_problem=(draft.business_problem),
                business_goals=list(draft.business_goals),
                constraints=list(draft.constraints),
                confirmed_facts=[
                    ConfirmedScenarioFact(
                        fact=fact.fact,
                        source_type=(fact.source_type),
                        source_reference=(fact.source_reference),
                    )
                    for fact in (draft.proposed_facts)
                ],
            )

            context.scenario_card = card

            try:
                saved = await self._user_context_service.save(
                    user_id=user_id,
                    context=context,
                    expected_version=(context.version),
                )

            except VersionConflictError as exc:
                last_conflict = exc
                continue

            if saved.scenario_card is None:
                raise RuntimeError("Scenario Card was not persisted")

            return saved.scenario_card.model_copy(deep=True)

        if last_conflict is not None:
            raise last_conflict

        raise RuntimeError("Unable to persist Scenario Card")

    async def set_confirmed_core_field(
        self,
        *,
        user_id: UUID,
        target: ScenarioCardTarget,
        value: str,
    ) -> ScenarioCard:
        """
        Modify an already confirmed Scenario Card.

        Used after onboarding.
        """

        allowed_targets = {
            ScenarioCardTarget.COMPANY_CONTEXT,
            ScenarioCardTarget.LEARNER_ROLE,
            ScenarioCardTarget.BUSINESS_PROBLEM,
        }

        if target not in allowed_targets:
            raise ValueError(f"{target.value} is not a Scenario Card core field")

        context = await self._get_context_with_existing_card(
            user_id=user_id,
        )

        card = self._copy_card(context)

        if target == ScenarioCardTarget.COMPANY_CONTEXT:
            card.company_context = value

        elif target == ScenarioCardTarget.LEARNER_ROLE:
            card.learner_role = value

        else:
            card.business_problem = value

        self._mark_card_changed(card)

        return await self._save_card(
            context=context,
            card=card,
        )

    async def add_business_goal(
        self,
        *,
        user_id: UUID,
        goal: str,
    ) -> ScenarioCard:
        context = await self._get_context_with_existing_card(
            user_id=user_id,
        )

        card = self._copy_card(context)

        if self._contains_text(
            values=card.business_goals,
            value=goal,
        ):
            return card

        card.business_goals.append(goal)

        self._mark_card_changed(card)

        return await self._save_card(
            context=context,
            card=card,
        )

    async def add_constraint(
        self,
        *,
        user_id: UUID,
        constraint: str,
    ) -> ScenarioCard:
        context = await self._get_context_with_existing_card(
            user_id=user_id,
        )

        card = self._copy_card(context)

        if self._contains_text(
            values=card.constraints,
            value=constraint,
        ):
            return card

        card.constraints.append(constraint)

        self._mark_card_changed(card)

        return await self._save_card(
            context=context,
            card=card,
        )

    async def add_confirmed_fact(
        self,
        *,
        user_id: UUID,
        fact: str,
        source_type: ScenarioFactSource,
        source_reference: str | None = None,
    ) -> ScenarioCard:
        context = await self._get_context_with_existing_card(
            user_id=user_id,
        )

        card = self._copy_card(context)

        if self._contains_fact(
            card=card,
            fact=fact,
        ):
            return card

        card.confirmed_facts.append(
            ConfirmedScenarioFact(
                fact=fact,
                source_type=source_type,
                source_reference=(source_reference),
            )
        )

        self._mark_card_changed(card)

        return await self._save_card(
            context=context,
            card=card,
        )

    def create_candidate(
        self,
        *,
        user_id: UUID,
        target: ScenarioCardTarget,
        value: str,
        source: ScenarioCandidateSource,
        evidence: str | None = None,
        source_reference: str | None = None,
    ) -> ScenarioUpdateCandidate:
        """
        Create one temporary proposed change to an existing card.

        No persistence occurs here.
        """

        return ScenarioUpdateCandidate(
            user_id=user_id,
            target=target,
            value=value,
            source=source,
            evidence=evidence,
            source_reference=(source_reference),
        )

    async def confirm_candidate(
        self,
        *,
        user_id: UUID,
        candidate: ScenarioUpdateCandidate,
    ) -> ScenarioCard:
        """
        Confirm one later change to an existing Scenario Card.
        """

        if candidate.user_id != user_id:
            raise ValueError("candidate.user_id must match ScenarioCardService user_id")

        if candidate.target in {
            ScenarioCardTarget.COMPANY_CONTEXT,
            ScenarioCardTarget.LEARNER_ROLE,
            ScenarioCardTarget.BUSINESS_PROBLEM,
        }:
            return await self.set_confirmed_core_field(
                user_id=user_id,
                target=candidate.target,
                value=candidate.value,
            )

        if candidate.target == ScenarioCardTarget.BUSINESS_GOAL:
            return await self.add_business_goal(
                user_id=user_id,
                goal=candidate.value,
            )

        if candidate.target == ScenarioCardTarget.CONSTRAINT:
            return await self.add_constraint(
                user_id=user_id,
                constraint=candidate.value,
            )

        if candidate.target == ScenarioCardTarget.CONFIRMED_FACT:
            return await self.add_confirmed_fact(
                user_id=user_id,
                fact=candidate.value,
                source_type=(ScenarioFactSource.USER_CONFIRMED_AGENT_PROPOSAL),
                source_reference=(candidate.source_reference),
            )

        raise ValueError(f"Unsupported Scenario Card target: {candidate.target}")

    async def _get_context_with_existing_card(
        self,
        *,
        user_id: UUID,
    ) -> UserContext:
        """
        Return UserContext containing an existing confirmed Scenario Card.

        From the ScenarioCardService boundary, both of these conditions
        mean the same thing:

        - UserContext has not been initialized yet.
        - UserContext exists but has no confirmed Scenario Card.

        In both cases, expose ScenarioCard as the missing resource rather
        than leaking the internal UserContext lifecycle to the caller.
        """

        context = await self._user_context_service.get(
            user_id=user_id,
        )

        if context is None or context.scenario_card is None:
            raise RepositoryNotFoundError(
                entity_name="ScenarioCard",
                entity_id=str(user_id),
            )

        return context

    @staticmethod
    def _copy_card(
        context: UserContext,
    ) -> ScenarioCard:
        if context.scenario_card is None:
            raise RuntimeError("Scenario Card is unexpectedly missing")

        return context.scenario_card.model_copy(deep=True)

    async def _save_card(
        self,
        *,
        context: UserContext,
        card: ScenarioCard,
    ) -> ScenarioCard:
        context.scenario_card = card

        saved_context = await self._user_context_service.save(
            user_id=context.user_id,
            context=context,
            expected_version=(context.version),
        )

        if saved_context.scenario_card is None:
            raise RuntimeError("Scenario Card was not persisted")

        return saved_context.scenario_card.model_copy(deep=True)

    @staticmethod
    def _mark_card_changed(
        card: ScenarioCard,
    ) -> None:
        card.version += 1
        card.updated_at = _utc_now()

    @staticmethod
    def _contains_text(
        *,
        values: list[str],
        value: str,
    ) -> bool:
        candidate = _normalized_text(value)

        return any(_normalized_text(existing) == candidate for existing in values)

    @staticmethod
    def _contains_fact(
        *,
        card: ScenarioCard,
        fact: str,
    ) -> bool:
        candidate = _normalized_text(fact)

        return any(
            _normalized_text(existing.fact) == candidate for existing in (card.confirmed_facts)
        )

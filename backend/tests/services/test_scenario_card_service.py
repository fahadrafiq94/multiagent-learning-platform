from __future__ import annotations

from uuid import uuid4

import pytest

from app.context import (
    ScenarioCandidateSource,
    ScenarioCard,
    ScenarioCardTarget,
    ScenarioDraft,
    ScenarioDraftFact,
    ScenarioFactSource,
)
from app.identity import UserIdentity
from app.repositories import (
    InMemoryRepositories,
    RepositoryNotFoundError,
    create_in_memory_repositories,
)
from app.services.scenario_card_service import (
    ScenarioCardService,
)
from app.services.user_context_service import (
    UserContextService,
)


def create_services() -> tuple[
    InMemoryRepositories,
    UserContextService,
    ScenarioCardService,
]:
    repositories = create_in_memory_repositories()

    user_context_service = UserContextService(
        repository=repositories.user_contexts,
    )

    scenario_service = ScenarioCardService(
        user_context_service=user_context_service,
    )

    return (
        repositories,
        user_context_service,
        scenario_service,
    )


async def create_user(
    *,
    repositories: InMemoryRepositories,
    student_id: str,
) -> UserIdentity:
    identity = UserIdentity(
        student_id=student_id,
    )

    return await repositories.identities.create(identity)


async def create_confirmed_scenario(
    *,
    user: UserIdentity,
    scenario_service: ScenarioCardService,
) -> ScenarioCard:
    """
    Create the initial persistent Scenario Card through the
    reviewed ScenarioDraft confirmation path.
    """

    draft = ScenarioDraft(
        user_id=user.user_id,
        session_id=uuid4(),
        company_context=(
            "UrbanCycle Manufacturing GmbH is a bicycle manufacturer producing city bicycles."
        ),
        learner_role="Production Planner",
        business_problem=(
            "Customer orders are delayed because "
            "planning activities are not sufficiently coordinated."
        ),
        business_goals=[
            "Provide reliable delivery dates.",
        ],
    )

    return await scenario_service.confirm_draft(
        user_id=user.user_id,
        draft=draft,
    )


@pytest.mark.asyncio
async def test_get_returns_none_before_card_is_confirmed() -> None:
    (
        repositories,
        user_context_service,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await user_context_service.get_or_create(
        user_id=user.user_id,
    )

    card = await scenario_service.get(
        user_id=user.user_id,
    )

    assert card is None


@pytest.mark.asyncio
async def test_confirm_draft_creates_persistent_scenario_card() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    draft = ScenarioDraft(
        user_id=user.user_id,
        session_id=uuid4(),
        company_context=(
            "UrbanCycle Manufacturing GmbH is a "
            "bicycle manufacturer producing city bicycles, "
            "spare parts, and maintenance services."
        ),
        learner_role="Production Planner",
        business_problem=(
            "Customer orders are delayed because sales, "
            "material planning, purchasing, and production "
            "planning are not sufficiently coordinated."
        ),
        business_goals=[
            "Provide reliable delivery dates.",
            "Improve material availability.",
        ],
        constraints=[
            "One production site.",
            "Limited assembly capacity.",
        ],
        proposed_facts=[
            ScenarioDraftFact(
                fact=("City Bike is the self-manufactured finished product."),
                source_type=(ScenarioFactSource.ASSIGNMENT_DOCUMENT),
                source_reference=("AP+ pilot assignment"),
            )
        ],
    )

    card = await scenario_service.confirm_draft(
        user_id=user.user_id,
        draft=draft,
    )

    assert card.user_id == user.user_id

    assert card.company_context == (
        "UrbanCycle Manufacturing GmbH is a "
        "bicycle manufacturer producing city bicycles, "
        "spare parts, and maintenance services."
    )

    assert card.learner_role == "Production Planner"

    assert len(card.business_goals) == 2
    assert len(card.constraints) == 2
    assert len(card.confirmed_facts) == 1

    stored = await scenario_service.get(
        user_id=user.user_id,
    )

    assert stored is not None

    assert stored.user_id == user.user_id


@pytest.mark.asyncio
async def test_incomplete_draft_cannot_be_confirmed() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    draft = ScenarioDraft(
        user_id=user.user_id,
        session_id=uuid4(),
        company_context=("UrbanCycle is a bicycle manufacturer."),
    )

    with pytest.raises(
        ValueError,
        match="learner_role is required",
    ):
        await scenario_service.confirm_draft(
            user_id=user.user_id,
            draft=draft,
        )


@pytest.mark.asyncio
async def test_draft_from_another_user_is_rejected() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user_a = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    user_b = await create_user(
        repositories=repositories,
        student_id="K67890",
    )

    draft = ScenarioDraft(
        user_id=user_b.user_id,
        session_id=uuid4(),
        company_context=("UrbanCycle is a bicycle manufacturer."),
        learner_role="Production Planner",
        business_problem=("Customer orders are delayed."),
        business_goals=[
            "Improve delivery reliability.",
        ],
    )

    with pytest.raises(
        ValueError,
        match=("draft.user_id must match ScenarioCardService user_id"),
    ):
        await scenario_service.confirm_draft(
            user_id=user_a.user_id,
            draft=draft,
        )


@pytest.mark.asyncio
async def test_confirming_second_initial_draft_returns_existing_card() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    first_card = await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    second_draft = ScenarioDraft(
        user_id=user.user_id,
        session_id=uuid4(),
        company_context=("A completely different company."),
        learner_role="Buyer",
        business_problem="Another problem.",
        business_goals=[
            "Another goal.",
        ],
    )

    second_result = await scenario_service.confirm_draft(
        user_id=user.user_id,
        draft=second_draft,
    )

    assert second_result.company_context == first_card.company_context

    assert second_result.learner_role == first_card.learner_role


@pytest.mark.asyncio
async def test_confirmed_core_field_can_be_updated_after_onboarding() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    card = await scenario_service.set_confirmed_core_field(
        user_id=user.user_id,
        target=(ScenarioCardTarget.LEARNER_ROLE),
        value="Senior Production Planner",
    )

    assert card.learner_role == "Senior Production Planner"

    assert card.version == 2


@pytest.mark.asyncio
async def test_core_field_update_requires_existing_card() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    with pytest.raises(
        RepositoryNotFoundError,
        match="ScenarioCard not found",
    ):
        await scenario_service.set_confirmed_core_field(
            user_id=user.user_id,
            target=(ScenarioCardTarget.LEARNER_ROLE),
            value="Production Planner",
        )


@pytest.mark.asyncio
async def test_business_goal_can_be_added_after_onboarding() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    card = await scenario_service.add_business_goal(
        user_id=user.user_id,
        goal="Improve material availability.",
    )

    assert "Improve material availability." in card.business_goals


@pytest.mark.asyncio
async def test_duplicate_business_goal_is_not_added_twice() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    first = await scenario_service.add_business_goal(
        user_id=user.user_id,
        goal="Improve material availability.",
    )

    second = await scenario_service.add_business_goal(
        user_id=user.user_id,
        goal=("  improve material availability.  "),
    )

    assert len(second.business_goals) == len(first.business_goals)


@pytest.mark.asyncio
async def test_meaningful_constraint_can_be_added() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    card = await scenario_service.add_constraint(
        user_id=user.user_id,
        constraint=("Limited assembly capacity."),
    )

    assert "Limited assembly capacity." in card.constraints


@pytest.mark.asyncio
async def test_duplicate_constraint_is_not_added_twice() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    first = await scenario_service.add_constraint(
        user_id=user.user_id,
        constraint="Limited assembly capacity.",
    )

    second = await scenario_service.add_constraint(
        user_id=user.user_id,
        constraint=(" limited assembly capacity. "),
    )

    assert len(second.constraints) == len(first.constraints)


@pytest.mark.asyncio
async def test_confirmed_fact_keeps_source_provenance() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    card = await scenario_service.add_confirmed_fact(
        user_id=user.user_id,
        fact=("Frame Set and Wheel Set are purchased components."),
        source_type=(ScenarioFactSource.ASSIGNMENT_DOCUMENT),
        source_reference=("AP+ pilot assignment"),
    )

    fact = card.confirmed_facts[-1]

    assert fact.source_type == ScenarioFactSource.ASSIGNMENT_DOCUMENT

    assert fact.source_reference == "AP+ pilot assignment"


@pytest.mark.asyncio
async def test_duplicate_confirmed_fact_is_not_added_twice() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    first = await scenario_service.add_confirmed_fact(
        user_id=user.user_id,
        fact=("RollingParts supplies the Wheel Set."),
        source_type=(ScenarioFactSource.USER),
    )

    second = await scenario_service.add_confirmed_fact(
        user_id=user.user_id,
        fact=("  rollingparts supplies the wheel set.  "),
        source_type=(ScenarioFactSource.USER),
    )

    assert len(second.confirmed_facts) == len(first.confirmed_facts)


@pytest.mark.asyncio
async def test_relevant_company_size_can_be_stored_as_fact() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    card = await scenario_service.add_confirmed_fact(
        user_id=user.user_id,
        fact=("The company has approximately 180 employees."),
        source_type=(ScenarioFactSource.USER),
    )

    assert any(
        fact.fact == ("The company has approximately 180 employees.")
        for fact in card.confirmed_facts
    )


@pytest.mark.asyncio
async def test_agent_candidate_is_not_persisted_automatically() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    before = await scenario_service.get(
        user_id=user.user_id,
    )

    assert before is not None

    candidate = scenario_service.create_candidate(
        user_id=user.user_id,
        target=(ScenarioCardTarget.CONFIRMED_FACT),
        value=("The company is probably medium-sized."),
        source=(ScenarioCandidateSource.AGENT_INFERENCE),
        evidence=("Several departments are mentioned."),
    )

    assert candidate.value == ("The company is probably medium-sized.")

    after = await scenario_service.get(
        user_id=user.user_id,
    )

    assert after is not None

    assert len(after.confirmed_facts) == len(before.confirmed_facts)


@pytest.mark.asyncio
async def test_confirmed_agent_candidate_becomes_persistent_fact() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    await create_confirmed_scenario(
        user=user,
        scenario_service=scenario_service,
    )

    candidate = scenario_service.create_candidate(
        user_id=user.user_id,
        target=(ScenarioCardTarget.CONFIRMED_FACT),
        value=("RollingParts supplies the Wheel Set."),
        source=(ScenarioCandidateSource.AGENT_INFERENCE),
    )

    card = await scenario_service.confirm_candidate(
        user_id=user.user_id,
        candidate=candidate,
    )

    matching = [
        fact
        for fact in card.confirmed_facts
        if fact.fact == ("RollingParts supplies the Wheel Set.")
    ]

    assert len(matching) == 1

    assert matching[0].source_type == (ScenarioFactSource.USER_CONFIRMED_AGENT_PROPOSAL)


@pytest.mark.asyncio
async def test_candidate_from_another_user_is_rejected() -> None:
    (
        repositories,
        _,
        scenario_service,
    ) = create_services()

    user_a = await create_user(
        repositories=repositories,
        student_id="K12345",
    )

    user_b = await create_user(
        repositories=repositories,
        student_id="K67890",
    )

    await create_confirmed_scenario(
        user=user_a,
        scenario_service=scenario_service,
    )

    candidate = scenario_service.create_candidate(
        user_id=user_b.user_id,
        target=(ScenarioCardTarget.BUSINESS_GOAL),
        value=("Improve material availability."),
        source=(ScenarioCandidateSource.AGENT_INFERENCE),
    )

    with pytest.raises(
        ValueError,
        match=("candidate.user_id must match ScenarioCardService user_id"),
    ):
        await scenario_service.confirm_candidate(
            user_id=user_a.user_id,
            candidate=candidate,
        )

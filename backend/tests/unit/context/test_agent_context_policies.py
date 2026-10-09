from __future__ import annotations

import pytest

from app.context import (
    AgentContextPolicyService,
    ContextTarget,
    ScenarioCard,
    ScenarioDraft,
    ScenarioSetupStatus,
    SessionContextUpdate,
)
from app.identity import UserIdentity
from app.repositories import (
    InMemoryRepositories,
    create_in_memory_repositories,
)
from app.services.context_service import (
    ContextService,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.session_context_service import (
    SessionContextService,
)
from app.services.user_context_service import (
    UserContextService,
)

pytestmark = pytest.mark.asyncio


def create_services() -> tuple[
    InMemoryRepositories,
    ConversationService,
    UserContextService,
    SessionContextService,
    ContextService,
]:
    repositories = create_in_memory_repositories()

    conversation_service = ConversationService(
        session_repository=(repositories.sessions),
        session_context_repository=(repositories.session_contexts),
        conversation_repository=(repositories.conversations),
    )

    user_context_service = UserContextService(
        repository=(repositories.user_contexts),
    )

    session_context_service = SessionContextService(
        repository=(repositories.session_contexts),
    )

    context_service = ContextService(
        user_context_service=(user_context_service),
        session_context_service=(session_context_service),
        conversation_service=(conversation_service),
    )

    return (
        repositories,
        conversation_service,
        user_context_service,
        session_context_service,
        context_service,
    )


async def create_user(
    *,
    repositories: InMemoryRepositories,
) -> UserIdentity:
    identity = UserIdentity(
        student_id="K12345",
    )

    return await repositories.identities.create(identity)


async def build_populated_context():
    (
        repositories,
        conversation_service,
        user_context_service,
        session_context_service,
        context_service,
    ) = create_services()

    user = await create_user(
        repositories=repositories,
    )

    session = await conversation_service.create_session(
        user_id=user.user_id,
    )

    user_context = await user_context_service.get_or_create(
        user_id=user.user_id,
    )

    user_context.scenario_card = ScenarioCard(
        user_id=user.user_id,
        company_context=("UrbanCycle manufactures bicycles."),
        learner_role=("Production Planner"),
        business_problem=("Customer orders are delayed."),
        business_goals=[
            "Improve delivery reliability.",
        ],
        constraints=[
            "Limited assembly capacity.",
        ],
    )

    await user_context_service.save(
        user_id=user.user_id,
        context=user_context,
        expected_version=(user_context.version),
    )

    await session_context_service.update(
        user_id=user.user_id,
        session_id=session.session_id,
        changes=SessionContextUpdate(
            current_objective=("Understand supplier selection."),
            current_step_id="8",
            current_process_phase=("supplier_selection"),
            active_blocker=("Supplier criteria are unclear."),
            last_selected_agent=("process_coach"),
            unresolved_questions=[
                ("Which supplier should be selected?"),
            ],
        ),
    )

    session_context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    session_context.scenario_setup_status = ScenarioSetupStatus.REVIEWING

    session_context.scenario_draft = ScenarioDraft(
        user_id=user.user_id,
        session_id=session.session_id,
        company_context=("Temporary draft company."),
        learner_role=("Production Planner"),
        business_problem=("Temporary draft problem."),
        business_goals=[
            "Temporary draft goal.",
        ],
    )

    await session_context_service.save(
        user_id=user.user_id,
        session_id=session.session_id,
        context=session_context,
        expected_version=(session_context.version),
    )

    await conversation_service.append_user_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Previous user message",
    )

    await conversation_service.append_assistant_message(
        user_id=user.user_id,
        session_id=session.session_id,
        content="Previous assistant message",
    )

    base_context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message=("What should I do next?"),
    )

    return base_context


async def test_orchestrator_receives_minimal_context() -> None:
    base_context = await build_populated_context()

    policy = AgentContextPolicyService()

    view = policy.select(
        target=(ContextTarget.ORCHESTRATOR),
        context=base_context,
    )

    assert view.target == ContextTarget.ORCHESTRATOR

    assert view.current_message == "What should I do next?"

    assert view.runtime_session is not None

    assert view.runtime_session.current_step_id == "8"

    assert view.runtime_session.current_process_phase == "supplier_selection"

    assert view.runtime_session.active_blocker == "Supplier criteria are unclear."

    assert view.runtime_session.unresolved_questions == []

    assert view.scenario_card is None
    assert view.scenario_setup is None

    assert view.learning_profile is None
    assert view.progress is None

    assert view.recent_messages == []

    assert view.session_summary is None

    assert view.guidance_level is None

    assert view.user_adaptations == []

    assert view.knowledge_chunks == []


async def test_scenario_agent_receives_onboarding_context() -> None:
    base_context = await build_populated_context()

    policy = AgentContextPolicyService()

    view = policy.select(
        target=ContextTarget.SCENARIO,
        context=base_context,
    )

    assert view.scenario_card is not None

    assert view.scenario_card.learner_role == "Production Planner"

    assert view.scenario_setup is not None

    assert view.scenario_setup.status == ScenarioSetupStatus.REVIEWING

    assert view.scenario_setup.draft is not None

    assert view.scenario_setup.draft.company_context == "Temporary draft company."

    assert len(view.recent_messages) == 2

    assert view.guidance_level is not None

    assert view.runtime_session is None

    assert view.learning_profile is None
    assert view.progress is None

    assert view.user_adaptations == []

    assert view.knowledge_chunks == []


async def test_process_coach_receives_pedagogical_context() -> None:
    base_context = await build_populated_context()

    policy = AgentContextPolicyService()

    view = policy.select(
        target=(ContextTarget.PROCESS_COACH),
        context=base_context,
    )

    assert view.scenario_card is not None

    assert view.scenario_setup is None

    assert view.runtime_session is not None

    assert view.runtime_session.current_step_id == "8"

    assert view.runtime_session.unresolved_questions == [("Which supplier should be selected?")]

    assert view.learning_profile is not None

    assert view.progress is not None

    assert len(view.recent_messages) == 2

    assert view.guidance_level is not None


async def test_ap_plus_navigator_receives_operational_context() -> None:
    base_context = await build_populated_context()

    policy = AgentContextPolicyService()

    view = policy.select(
        target=(ContextTarget.AP_PLUS_NAVIGATOR),
        context=base_context,
    )

    assert view.scenario_card is not None

    assert view.runtime_session is not None

    assert view.runtime_session.current_step_id == "8"

    assert view.runtime_session.current_process_phase == "supplier_selection"

    assert view.runtime_session.active_blocker == "Supplier criteria are unclear."

    assert view.scenario_setup is None

    assert view.learning_profile is None
    assert view.progress is None

    assert view.user_adaptations == []

    assert len(view.recent_messages) == 2

    assert view.guidance_level is not None


async def test_unconfirmed_scenario_draft_does_not_leak_to_process_coach() -> None:
    base_context = await build_populated_context()

    policy = AgentContextPolicyService()

    view = policy.select(
        target=(ContextTarget.PROCESS_COACH),
        context=base_context,
    )

    assert view.scenario_setup is None

    serialized = view.model_dump()

    assert "Temporary draft company." not in str(serialized)

    assert "Temporary draft problem." not in str(serialized)


async def test_unconfirmed_scenario_draft_does_not_leak_to_navigator() -> None:
    base_context = await build_populated_context()

    policy = AgentContextPolicyService()

    view = policy.select(
        target=(ContextTarget.AP_PLUS_NAVIGATOR),
        context=base_context,
    )

    serialized = view.model_dump()

    assert view.scenario_setup is None

    assert "Temporary draft company." not in str(serialized)


async def test_unconfirmed_scenario_draft_does_not_leak_to_orchestrator() -> None:
    base_context = await build_populated_context()

    policy = AgentContextPolicyService()

    view = policy.select(
        target=(ContextTarget.ORCHESTRATOR),
        context=base_context,
    )

    serialized = view.model_dump()

    assert view.scenario_setup is None

    assert "Temporary draft company." not in str(serialized)


async def test_policy_does_not_mutate_base_context() -> None:
    base_context = await build_populated_context()

    before = base_context.model_dump()

    policy = AgentContextPolicyService()

    policy.select(
        target=(ContextTarget.PROCESS_COACH),
        context=base_context,
    )

    policy.select(
        target=ContextTarget.SCENARIO,
        context=base_context,
    )

    after = base_context.model_dump()

    assert after == before


async def test_each_view_records_policy_version() -> None:
    base_context = await build_populated_context()

    policy = AgentContextPolicyService()

    for target in ContextTarget:
        view = policy.select(
            target=target,
            context=base_context,
        )

        assert view.policy_version == (AgentContextPolicyService.POLICY_VERSION)


async def test_policy_preserves_request_identity() -> None:
    base_context = await build_populated_context()

    policy = AgentContextPolicyService()

    for target in ContextTarget:
        view = policy.select(
            target=target,
            context=base_context,
        )

        assert view.user_id == base_context.user_id

        assert view.session_id == base_context.session_id

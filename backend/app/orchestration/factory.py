from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.agents.factory import (
    create_agent_registry,
)
from app.agents.orchestrator import (
    OrchestratorAgent,
)
from app.agents.scenario import (
    ScenarioAgent,
)
from app.config.settings import settings
from app.context.budget import (
    ContextBudgetPolicy,
    ContextBudgetService,
)
from app.context.policies import (
    AgentContextPolicyService,
)
from app.context.token_estimator import (
    ApproximateTokenEstimator,
)
from app.knowledge.integration import SpecialistGroundingService
from app.orchestration.graph import (
    build_orchestration_graph,
)
from app.orchestration.service import (
    OrchestrationService,
)
from app.repositories.bundle import RepositoryBundle
from app.repositories.in_memory import create_in_memory_repositories
from app.services.context_service import (
    ContextService,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.model_service import (
    ModelService,
    create_model_service,
)
from app.services.scenario_card_service import (
    ScenarioCardService,
)
from app.services.scenario_onboarding_interaction_service import (
    ScenarioOnboardingInteractionService,
)
from app.services.scenario_onboarding_service import (
    ScenarioOnboardingService,
)
from app.services.session_context_service import (
    SessionContextService,
)
from app.services.session_state_update_service import (
    SessionStateUpdateService,
)
from app.services.session_summary_service import (
    SessionSummaryService,
)
from app.services.user_context_service import (
    UserContextService,
)


@dataclass(slots=True)
class OrchestrationRuntime:
    """
    Long-lived services with injectable storage implementations.
    """

    repositories: RepositoryBundle

    model_service: ModelService

    conversation_service: ConversationService

    user_context_service: UserContextService

    session_context_service: SessionContextService

    session_state_update_service: SessionStateUpdateService

    session_summary_service: SessionSummaryService

    scenario_card_service: ScenarioCardService

    scenario_onboarding_service: ScenarioOnboardingService

    context_service: ContextService

    context_policy_service: AgentContextPolicyService

    context_budget_service: ContextBudgetService

    orchestration_service: OrchestrationService

    graph: Any
    grounding_service: SpecialistGroundingService | None = None


def build_orchestration_runtime(
    *,
    model_service: ModelService,
    repositories: (RepositoryBundle | None) = None,
    grounding_service: SpecialistGroundingService | None = None,
) -> OrchestrationRuntime:
    runtime_repositories = repositories or create_in_memory_repositories()

    conversation_service = ConversationService(
        session_repository=(runtime_repositories.sessions),
        session_context_repository=(runtime_repositories.session_contexts),
        conversation_repository=(runtime_repositories.conversations),
        session_initializer=runtime_repositories.session_initializer,
    )

    user_context_service = UserContextService(
        repository=(runtime_repositories.user_contexts),
    )

    session_context_service = SessionContextService(
        repository=(runtime_repositories.session_contexts),
    )
    if grounding_service is not None:
        grounding_service.session_context_service = session_context_service
    session_state_update_service = SessionStateUpdateService(
        session_context_service=(session_context_service),
    )

    scenario_card_service = ScenarioCardService(
        user_context_service=(user_context_service),
    )

    scenario_onboarding_service = ScenarioOnboardingService(
        session_context_service=(session_context_service),
        scenario_card_service=(scenario_card_service),
    )

    token_estimator = ApproximateTokenEstimator(
        characters_per_token=(settings.context_characters_per_token),
        message_overhead_tokens=(settings.context_message_overhead_tokens),
    )

    context_budget_service = ContextBudgetService(
        estimator=token_estimator,
        policy=ContextBudgetPolicy(
            max_context_tokens=(settings.context_max_tokens),
            reserved_output_tokens=(settings.context_reserved_output_tokens),
            max_recent_messages=(settings.context_max_recent_messages),
        ),
    )

    context_service = ContextService(
        user_context_service=user_context_service,
        session_context_service=session_context_service,
        conversation_service=conversation_service,
        budget_service=context_budget_service,
        agent_prompt_reserve_tokens=(settings.context_reserved_agent_prompt_tokens),
    )

    context_policy_service = AgentContextPolicyService()

    session_summary_service = SessionSummaryService(
        conversation_service=(conversation_service),
        session_context_service=(session_context_service),
        model_service=(model_service),
        recent_window=(settings.context_max_recent_messages),
        min_new_messages=(settings.context_summary_min_new_messages),
        max_items_per_list=(settings.context_summary_max_items_per_list),
        max_overview_chars=(settings.context_summary_max_overview_chars),
        # Empty env value means "use the default chat model".
        model=(settings.summary_model or None),
    )

    scenario_agent = ScenarioAgent(
        model_service=model_service,
    )

    registry = create_agent_registry(
        model_service=model_service,
        scenario_agent=scenario_agent,
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
    )

    interaction_service = ScenarioOnboardingInteractionService(
        scenario_agent=(scenario_agent),
        onboarding_service=(scenario_onboarding_service),
    )

    graph = build_orchestration_graph(
        agent_registry=registry,
        orchestrator=orchestrator,
        conversation_service=(conversation_service),
        context_service=(context_service),
        session_state_update_service=(session_state_update_service),
        session_summary_service=(session_summary_service),
        context_policy_service=(context_policy_service),
        onboarding_service=(scenario_onboarding_service),
        onboarding_interaction_service=(interaction_service),
        user_context_service=(user_context_service),
        grounding_service=grounding_service,
    )

    orchestration_service = OrchestrationService(
        graph=graph,
        identity_repository=(runtime_repositories.identities),
        turn_store=runtime_repositories.turns,
    )

    return OrchestrationRuntime(
        repositories=(runtime_repositories),
        model_service=model_service,
        conversation_service=(conversation_service),
        user_context_service=(user_context_service),
        session_context_service=(session_context_service),
        session_state_update_service=(session_state_update_service),
        session_summary_service=(session_summary_service),
        scenario_card_service=(scenario_card_service),
        scenario_onboarding_service=(scenario_onboarding_service),
        context_service=(context_service),
        context_budget_service=(context_budget_service),
        context_policy_service=(context_policy_service),
        orchestration_service=(orchestration_service),
        graph=graph,
        grounding_service=grounding_service,
    )


@lru_cache(maxsize=1)
def get_orchestration_runtime() -> OrchestrationRuntime:
    """
    Return the legacy in-memory runtime for direct callers.

    HTTP uses its lifespan-owned runtime instead of this compatibility cache.
    """

    return build_orchestration_runtime(
        model_service=(create_model_service()),
    )


def create_orchestration_service() -> OrchestrationService:
    return get_orchestration_runtime().orchestration_service

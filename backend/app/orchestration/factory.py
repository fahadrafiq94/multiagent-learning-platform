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
from app.context.policies import (
    AgentContextPolicyService,
)
from app.orchestration.graph import (
    build_orchestration_graph,
)
from app.orchestration.service import (
    OrchestrationService,
)
from app.repositories.in_memory import (
    InMemoryRepositories,
    create_in_memory_repositories,
)
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
from app.services.user_context_service import (
    UserContextService,
)


@dataclass(slots=True)
class OrchestrationRuntime:
    """
    Long-lived runtime services for the current in-memory application.

    PostgreSQL will later replace the repository implementation without
    changing the orchestration contracts.
    """

    repositories: InMemoryRepositories

    model_service: ModelService

    conversation_service: ConversationService

    user_context_service: UserContextService

    session_context_service: SessionContextService

    scenario_card_service: ScenarioCardService

    scenario_onboarding_service: ScenarioOnboardingService

    context_service: ContextService

    context_policy_service: AgentContextPolicyService

    orchestration_service: OrchestrationService

    graph: Any


def build_orchestration_runtime(
    *,
    model_service: ModelService,
    repositories: (InMemoryRepositories | None) = None,
) -> OrchestrationRuntime:
    runtime_repositories = repositories or create_in_memory_repositories()

    conversation_service = ConversationService(
        session_repository=(runtime_repositories.sessions),
        session_context_repository=(runtime_repositories.session_contexts),
        conversation_repository=(runtime_repositories.conversations),
    )

    user_context_service = UserContextService(
        repository=(runtime_repositories.user_contexts),
    )

    session_context_service = SessionContextService(
        repository=(runtime_repositories.session_contexts),
    )

    scenario_card_service = ScenarioCardService(
        user_context_service=(user_context_service),
    )

    scenario_onboarding_service = ScenarioOnboardingService(
        session_context_service=(session_context_service),
        scenario_card_service=(scenario_card_service),
    )

    context_service = ContextService(
        user_context_service=(user_context_service),
        session_context_service=(session_context_service),
        conversation_service=(conversation_service),
    )

    context_policy_service = AgentContextPolicyService()

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
        context_policy_service=(context_policy_service),
        onboarding_service=(scenario_onboarding_service),
        onboarding_interaction_service=(interaction_service),
    )

    orchestration_service = OrchestrationService(
        graph=graph,
        identity_repository=(runtime_repositories.identities),
    )

    return OrchestrationRuntime(
        repositories=(runtime_repositories),
        model_service=model_service,
        conversation_service=(conversation_service),
        user_context_service=(user_context_service),
        session_context_service=(session_context_service),
        scenario_card_service=(scenario_card_service),
        scenario_onboarding_service=(scenario_onboarding_service),
        context_service=(context_service),
        context_policy_service=(context_policy_service),
        orchestration_service=(orchestration_service),
        graph=graph,
    )


@lru_cache(maxsize=1)
def get_orchestration_runtime() -> OrchestrationRuntime:
    """
    Return the process-local runtime.

    This is intentionally cached while repositories are in memory.
    """

    return build_orchestration_runtime(
        model_service=(create_model_service()),
    )


def create_orchestration_service() -> OrchestrationService:
    return get_orchestration_runtime().orchestration_service

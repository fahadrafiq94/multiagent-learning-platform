from __future__ import annotations

from app.agents.factory import (
    create_agent_registry,
)
from app.agents.orchestrator import (
    OrchestratorAgent,
)
from app.config.settings import settings
from app.orchestration.graph import (
    build_orchestration_graph,
)
from app.orchestration.service import (
    OrchestrationService,
)
from app.services.model_service import (
    create_model_service,
)


def create_orchestration_service() -> OrchestrationService:
    """Create the complete configured orchestration service.

    One shared ModelService/provider is used for all agents.

    Individual agents may override the provider's default chat model
    through their ModelChatRequest.
    """

    model_service = create_model_service()

    agent_registry = create_agent_registry(
        model_service=model_service,
        scenario_model=(settings.scenario_agent_model),
        process_coach_model=(settings.process_coach_agent_model),
        ap_plus_navigator_model=(settings.ap_plus_navigator_model),
    )

    orchestrator = OrchestratorAgent(
        model_service=model_service,
        model=settings.orchestrator_model,
    )

    graph = build_orchestration_graph(
        agent_registry=agent_registry,
        orchestrator=orchestrator,
    )

    return OrchestrationService(
        graph=graph,
    )

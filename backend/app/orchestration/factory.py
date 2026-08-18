from __future__ import annotations

from app.agents.factory import create_agent_registry
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
    """Create the complete configured orchestration service."""

    model_service = create_model_service()

    agent_registry = create_agent_registry(
        model_service=model_service,
    )

    graph = build_orchestration_graph(
        agent_registry=agent_registry,
    )

    return OrchestrationService(
        graph=graph,
    )

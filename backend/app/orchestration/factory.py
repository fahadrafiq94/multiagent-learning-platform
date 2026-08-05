from __future__ import annotations

from app.orchestration.graph import build_orchestration_graph
from app.orchestration.service import OrchestrationService
from app.services.model_service import create_model_service


def create_orchestration_service() -> OrchestrationService:
    """Create the application orchestration service."""

    model_service = create_model_service()
    graph = build_orchestration_graph(model_service)

    return OrchestrationService(graph=graph)

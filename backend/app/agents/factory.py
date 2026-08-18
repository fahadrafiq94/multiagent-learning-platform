from __future__ import annotations

from app.agents.ap_plus_navigator import APPlusNavigatorAgent
from app.agents.process_coach import ProcessCoachAgent
from app.agents.registry import AgentRegistry
from app.agents.scenario import ScenarioAgent
from app.services.model_service import ModelService


def create_agent_registry(
    model_service: ModelService,
) -> AgentRegistry:
    """Create the default registry containing all Sprint 3 agents."""

    return AgentRegistry(
        agents=[
            ScenarioAgent(
                model_service=model_service,
            ),
            ProcessCoachAgent(
                model_service=model_service,
            ),
            APPlusNavigatorAgent(
                model_service=model_service,
            ),
        ]
    )

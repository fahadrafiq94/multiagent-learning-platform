from __future__ import annotations

from app.agents.ap_plus_navigator import (
    APPlusNavigatorAgent,
)
from app.agents.process_coach import (
    ProcessCoachAgent,
)
from app.agents.registry import AgentRegistry
from app.agents.scenario import (
    ScenarioAgent,
)
from app.services.model_service import (
    ModelService,
)


def create_agent_registry(
    model_service: ModelService,
    *,
    scenario_model: str | None = None,
    process_coach_model: str | None = None,
    ap_plus_navigator_model: str | None = None,
) -> AgentRegistry:
    """Create the configured specialist-agent registry."""

    return AgentRegistry(
        agents=[
            ScenarioAgent(
                model_service=model_service,
                model=scenario_model,
            ),
            ProcessCoachAgent(
                model_service=model_service,
                model=process_coach_model,
            ),
            APPlusNavigatorAgent(
                model_service=model_service,
                model=ap_plus_navigator_model,
            ),
        ]
    )

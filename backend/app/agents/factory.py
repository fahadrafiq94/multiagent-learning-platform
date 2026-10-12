from __future__ import annotations

from app.agents.ap_plus_navigator import (
    APPlusNavigatorAgent,
)
from app.agents.process_coach import (
    ProcessCoachAgent,
)
from app.agents.registry import (
    AgentRegistry,
)
from app.agents.scenario import (
    ScenarioAgent,
)
from app.services.model_service import (
    ModelService,
)


def create_agent_registry(
    model_service: ModelService,
    *,
    scenario_agent: (ScenarioAgent | None) = None,
    scenario_model: str | None = None,
    process_coach_model: str | None = None,
    ap_plus_navigator_model: (str | None) = None,
) -> AgentRegistry:
    scenario = scenario_agent or ScenarioAgent(
        model_service=model_service,
        model=scenario_model,
    )

    return AgentRegistry(
        agents=[
            scenario,
            ProcessCoachAgent(
                model_service=(model_service),
                model=(process_coach_model),
            ),
            APPlusNavigatorAgent(
                model_service=(model_service),
                model=(ap_plus_navigator_model),
            ),
        ]
    )

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
from app.services.model_service.service import service_for_role


def create_agent_registry(
    model_service: ModelService,
    *,
    scenario_agent: (ScenarioAgent | None) = None,
    scenario_model: str | None = None,
    process_coach_model: str | None = None,
    ap_plus_navigator_model: (str | None) = None,
) -> AgentRegistry:
    scenario = scenario_agent or ScenarioAgent(
        model_service=service_for_role(model_service, "scenario_agent"),
        model=scenario_model,
    )

    return AgentRegistry(
        agents=[
            scenario,
            ProcessCoachAgent(
                model_service=service_for_role(model_service, "process_coach"),
                model=(process_coach_model),
            ),
            APPlusNavigatorAgent(
                model_service=service_for_role(model_service, "ap_plus_navigator"),
                model=(ap_plus_navigator_model),
            ),
        ]
    )

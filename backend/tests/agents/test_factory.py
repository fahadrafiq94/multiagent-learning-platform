from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from app.agents.ap_plus_navigator import APPlusNavigatorAgent
from app.agents.factory import create_agent_registry
from app.agents.process_coach import ProcessCoachAgent
from app.agents.scenario import ScenarioAgent
from app.services.model_service import ModelService


def create_model_service_mock() -> ModelService:
    return cast(
        ModelService,
        MagicMock(spec=ModelService),
    )


def test_create_agent_registry_registers_all_agents() -> None:
    model_service = create_model_service_mock()

    registry = create_agent_registry(
        model_service=model_service,
    )

    assert registry.names == (
        "scenario",
        "process_coach",
        "ap_plus_navigator",
    )


def test_create_agent_registry_creates_scenario_agent() -> None:
    model_service = create_model_service_mock()

    registry = create_agent_registry(model_service)

    agent = registry.get("scenario")

    assert isinstance(
        agent,
        ScenarioAgent,
    )


def test_create_agent_registry_creates_process_coach_agent() -> None:
    model_service = create_model_service_mock()

    registry = create_agent_registry(model_service)

    agent = registry.get("process_coach")

    assert isinstance(
        agent,
        ProcessCoachAgent,
    )


def test_create_agent_registry_creates_ap_plus_navigator_agent() -> None:
    model_service = create_model_service_mock()

    registry = create_agent_registry(model_service)

    agent = registry.get("ap_plus_navigator")

    assert isinstance(
        agent,
        APPlusNavigatorAgent,
    )


def test_agent_registry_returns_distinct_agents() -> None:
    model_service = create_model_service_mock()

    registry = create_agent_registry(model_service)

    scenario = registry.get("scenario")
    process_coach = registry.get("process_coach")
    navigator = registry.get("ap_plus_navigator")

    assert scenario is not process_coach
    assert scenario is not navigator
    assert process_coach is not navigator

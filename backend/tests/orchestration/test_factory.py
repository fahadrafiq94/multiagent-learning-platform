from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.orchestration.factory import (
    create_orchestration_service,
)
from app.orchestration.service import (
    OrchestrationService,
)


@patch("app.orchestration.factory.build_orchestration_graph")
@patch("app.orchestration.factory.create_agent_registry")
@patch("app.orchestration.factory.create_model_service")
def test_create_orchestration_service(
    mock_create_model_service: MagicMock,
    mock_create_agent_registry: MagicMock,
    mock_build_graph: MagicMock,
) -> None:
    fake_model_service = MagicMock()
    fake_agent_registry = MagicMock()
    fake_graph = MagicMock()

    mock_create_model_service.return_value = fake_model_service

    mock_create_agent_registry.return_value = fake_agent_registry

    mock_build_graph.return_value = fake_graph

    service = create_orchestration_service()

    assert isinstance(
        service,
        OrchestrationService,
    )

    mock_create_model_service.assert_called_once_with()

    mock_create_agent_registry.assert_called_once_with(
        model_service=fake_model_service,
    )

    mock_build_graph.assert_called_once_with(
        agent_registry=fake_agent_registry,
    )

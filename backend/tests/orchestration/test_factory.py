from __future__ import annotations

from unittest.mock import (
    MagicMock,
    patch,
)

from app.orchestration.factory import (
    create_orchestration_service,
)
from app.orchestration.service import (
    OrchestrationService,
)


@patch("app.orchestration.factory.get_orchestration_runtime")
def test_create_orchestration_service(
    mock_get_runtime: MagicMock,
) -> None:
    fake_service = MagicMock(spec=OrchestrationService)

    fake_runtime = MagicMock()

    fake_runtime.orchestration_service = fake_service

    mock_get_runtime.return_value = fake_runtime

    result = create_orchestration_service()

    assert result is fake_service

    mock_get_runtime.assert_called_once_with()

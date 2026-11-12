from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from app.config.settings import settings
from app.runtime import runtime_lifespan
from tests.orchestration.test_graph import create_model_service


@pytest.mark.asyncio
async def test_disabled_rag_does_not_read_configuration_or_open_a_knowledge_pool(monkeypatch):
    model, _, _ = create_model_service()
    create = AsyncMock(side_effect=AssertionError("Disabled RAG must not construct resources."))
    monkeypatch.setattr(settings, "application_storage", "memory")
    monkeypatch.setattr(settings, "rag_enabled", False)
    monkeypatch.setattr(settings, "rag_config_file", Path("nonexistent-trusted-config.json"))
    monkeypatch.setattr("app.runtime.create_model_service", lambda: model)
    monkeypatch.setattr("app.runtime.create_grounding_runtime", create)
    app = FastAPI()
    async with runtime_lifespan(app):
        assert app.state.runtime.grounding_service is None
        assert app.state.knowledge_database is None
    create.assert_not_awaited()

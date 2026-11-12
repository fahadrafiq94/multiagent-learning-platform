"""Intake grants and HTTP activation remain explicit deployment decisions."""

from uuid import uuid4

import pytest

from app.config.settings import Settings
from app.knowledge.errors import KnowledgeUnavailable
from app.knowledge.indexer import IndexConfig
from app.knowledge.intake_runtime import create_intake_runtime
from app.knowledge.intake_schemas import IntakeBinding, IntakeConfig
from tests.knowledge.test_indexer import PROFILE, SCOPE


@pytest.mark.parametrize("storage", ["memory", "postgres"])
def test_intake_cannot_run_http_without_persistent_authentication(storage):
    with pytest.raises(KnowledgeUnavailable, match="authenticated"):
        create_intake_runtime(
            Settings(
                _env_file=None,
                application_storage=storage,
                application_pilot_auth_secret=None,
            )
        )


def test_bindings_cannot_grant_management_by_read_scope_or_duplicate_collection():
    value = dict(
        collection_id=uuid4(),
        scope=SCOPE,
        manager_user_ids=[uuid4()],
        index=IndexConfig(embedding=PROFILE),
    )
    with pytest.raises(ValueError, match="management grant"):
        IntakeBinding(
            **{**value, "scope": SCOPE.model_copy(update={"can_manage_knowledge": False})}
        )
    binding = IntakeBinding(**value)
    with pytest.raises(ValueError, match="unambiguous"):
        IntakeConfig(bindings=[binding, binding])


def test_http_intake_is_disabled_by_default():
    assert not Settings(_env_file=None).knowledge_intake_enabled

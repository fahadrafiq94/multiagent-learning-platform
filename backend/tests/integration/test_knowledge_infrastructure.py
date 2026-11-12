"""Opt-in real services; uses isolated scratch objects, never clears application data.

Start services first, then set FREDI_KNOWLEDGE_INTEGRATION=1 and run this module.
The persistence test restarts only the new knowledge PostgreSQL service.
"""

from __future__ import annotations

import json
import os

import pytest

from scripts.knowledge_infra import KnowledgeInfrastructure

pytestmark = pytest.mark.skipif(
    os.environ.get("FREDI_KNOWLEDGE_INTEGRATION") != "1",
    reason="Opt-in: requires the real knowledge Compose services and existing backend.",
)


def test_real_database_cache_and_persistent_storage() -> None:
    infra = KnowledgeInfrastructure()
    infra.check_config()
    results = infra.smoke(restart=True)
    print(json.dumps(results, sort_keys=True))
    assert results["pgvector_version"]
    for check in (
        "cosine_ranking",
        "database_roles",
        "postgres_restart_persistence",
        "redis_auth_and_expiry",
        "backend_network_connectivity",
        "source_volume_persistence",
    ):
        assert results[check] == "passed", check


def test_internal_overlay_has_no_host_ports_or_backend_dependency() -> None:
    infra = KnowledgeInfrastructure()
    # Compile the shared overlay with the base, excluding development port bindings.
    path = str(infra.root / "docker/compose/docker-compose.knowledge.dev.yml")
    position = infra.command.index(path)
    del infra.command[position - 1 : position + 1]
    config = json.loads(infra.run("config", "--format", "json").stdout)
    for service in ("postgres", "redis"):
        assert not config["services"][service].get("ports")
    dependencies = config["services"]["backend"]["depends_on"]
    assert "postgres" not in dependencies
    assert "redis" not in dependencies

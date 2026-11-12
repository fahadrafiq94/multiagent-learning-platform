from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError

from app.knowledge.cache import NoKnowledgeCache
from app.knowledge.database import KnowledgeDatabase, KnowledgeDatabaseConfig
from app.knowledge.errors import KnowledgeIntegrityError, KnowledgeNotFound, KnowledgeUnavailable
from app.knowledge.schemas import KnowledgeChunk, KnowledgeCollection
from app.knowledge.source_store import FileSourceStore


def test_source_roundtrip_idempotence_and_restart(tmp_path: Path) -> None:
    store = FileSourceStore(tmp_path / "sources")
    reference = store.put(b"synthetic source bytes")
    assert store.put(b"synthetic source bytes") == reference
    assert reference.uri == "sha256:" + reference.source_hash
    assert (
        FileSourceStore(tmp_path / "sources").get(reference.source_hash)
        == b"synthetic source bytes"
    )


def test_source_corruption_and_missing_are_visible(tmp_path: Path) -> None:
    store = FileSourceStore(tmp_path)
    reference = store.put(b"original")
    (tmp_path / reference.source_hash[:2] / reference.source_hash).write_bytes(b"corrupt")
    with pytest.raises(KnowledgeIntegrityError):
        store.get(reference.source_hash)
    with pytest.raises(KnowledgeNotFound):
        store.get("0" * 64)


@pytest.mark.parametrize("value", ["../outside", "A" * 64, "", "sha256:abc"])
def test_source_identifier_cannot_escape_root(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError):
        FileSourceStore(tmp_path).get(value)


def test_source_rejects_empty_and_oversize(tmp_path: Path) -> None:
    store = FileSourceStore(tmp_path, max_bytes=3)
    for content in (b"", b"long"):
        with pytest.raises(ValueError):
            store.put(content)


@pytest.mark.parametrize("vector", [[0, 0, 0], [float("nan"), 0, 0], [float("inf"), 0, 0]])
def test_invalid_vectors_cannot_be_evidence(sample: tuple, vector: list[float]) -> None:
    values = next(c for c in sample[2].chunks if c.embedding).model_dump()
    values["embedding"] = vector
    with pytest.raises(ValidationError):
        KnowledgeChunk.model_validate(values)


def test_collection_private_ownership_is_required() -> None:
    with pytest.raises(ValidationError):
        KnowledgeCollection(name="Private", access_domain="course", sharing_class="owner_private")
    with pytest.raises(ValidationError):
        KnowledgeCollection(name="Shared", access_domain="course", owner_user_id=uuid4())


def test_database_construction_does_not_connect_and_hides_dsn() -> None:
    config = KnowledgeDatabaseConfig(
        dsn=SecretStr("postgresql://user:secret-value@invalid/database")
    )
    database = KnowledgeDatabase(config)
    assert database._pool is None
    assert "secret-value" not in repr(config)


@pytest.mark.asyncio
async def test_no_cache_and_invalid_ttl() -> None:
    cache = NoKnowledgeCache()
    await cache.set("key", b"value", ttl_seconds=1)
    assert await cache.get("key") is None
    with pytest.raises(ValueError):
        await cache.set("key", b"value", ttl_seconds=0)


@pytest.mark.asyncio
async def test_malformed_dsn_is_sanitized_without_connection() -> None:
    database = KnowledgeDatabase(KnowledgeDatabaseConfig(dsn=SecretStr("invalid-secret-dsn")))
    with pytest.raises(KnowledgeUnavailable) as error:
        await database.start()
    assert "invalid-secret-dsn" not in str(error.value)
    assert database._pool is None
    await database.close()

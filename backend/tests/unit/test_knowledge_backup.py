"""Bundle validation precedes any privileged restore operation."""

import json
from unittest.mock import AsyncMock

import pytest

from app.knowledge.errors import KnowledgeIntegrityError
from app.knowledge.source_store import FileSourceStore
from scripts import knowledge_backup as tool


def bundle(tmp_path):
    root = tmp_path / "backup"
    root.mkdir()
    reference = FileSourceStore(root / "sources").put(b"%PDF synthetic backup contract")
    (root / "database.dump").write_bytes(b"synthetic pg_dump")
    manifest = dict(
        version=1,
        complete=True,
        dump_sha256=tool.sha_file(root / "database.dump"),
        sources=[reference.model_dump()],
        tables={},
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption", ["dump", "blob", "size", "partial", "duplicate", "path", "shape", "source_shape"]
)
async def test_corrupt_or_partial_backup_never_opens_database(tmp_path, monkeypatch, corruption):
    root, data = bundle(tmp_path)
    if corruption == "dump":
        (root / "database.dump").write_bytes(b"altered")
    elif corruption == "blob":
        FileSourceStore(root / "sources")._path(data["sources"][0]["source_hash"]).write_bytes(
            b"bad"
        )
    elif corruption == "size":
        data["sources"][0]["size_bytes"] += 1
    elif corruption == "partial":
        data["complete"] = False
    elif corruption == "duplicate":
        data["sources"] *= 2
    elif corruption == "path":
        data["sources"][0]["source_hash"] = "../../outside"
    elif corruption == "shape":
        data = []
    else:
        data["sources"][0] = "invalid"
    (root / "manifest.json").write_text(json.dumps(data))
    connect = AsyncMock(side_effect=AssertionError("Must not connect"))
    monkeypatch.setattr(tool.psycopg.AsyncConnection, "connect", connect)
    with pytest.raises((ValueError, KnowledgeIntegrityError)):
        await tool.verify_restore(root)
    connect.assert_not_awaited()


def test_new_private_directory_never_overwrites_existing_content(tmp_path):
    root, manifest = bundle(tmp_path)
    assert tool.verified_manifest(root) == manifest
    with pytest.raises(ValueError, match="never overwritten"):
        tool.private_directory(root)
    assert tool.verified_manifest(root) == manifest


@pytest.mark.parametrize(
    "name",
    [
        "postgres",
        "production",
        "fredi_knowledge; DROP DATABASE x",
        "fredi_restore_0123456789abcdef",
        "fredi_app_test_unknown",
    ],
)
def test_backup_refuses_unconfigured_database_names(name):
    with pytest.raises(ValueError):
        tool.validate_database(name)

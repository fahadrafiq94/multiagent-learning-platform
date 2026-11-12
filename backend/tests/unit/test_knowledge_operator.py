"""Operator decisions, config preservation, source paths and cancellation boundaries."""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.knowledge.intake_schemas import IntakeBinding, IntakeConfig
from app.services.model_service.schemas import EmbeddingModelInfo
from scripts import knowledge_operator_setup as setup
from scripts.knowledge_operator_ui import Console, pdf_paths, upload_flow
from scripts.manage_knowledge import verified_model
from tests.knowledge.test_indexer import PROFILE, SCOPE, IndexConfig


class ScriptedConsole(Console):
    def __init__(self, answers=()):
        self.answers, self.output = iter(answers), []

    def say(self, text):
        self.output.append(text)

    def ask(self, text):
        self.output.append(text)
        return next(self.answers)


def test_setup_preserves_valid_manager_grants_and_only_replaces_known_placeholder():
    student, operator = uuid4(), uuid4()
    assert setup.manager_ids([str(student), "REPLACE_WITH_YOUR_USER_UUID"], operator) == [
        str(student),
        str(operator),
    ]
    assert setup.manager_ids([str(operator)], operator) == [str(operator)]
    with pytest.raises(ValueError):
        setup.manager_ids(["invalid-uuid"], operator)


def test_private_atomic_config_creation_preserves_existing_choices_and_concurrent_edits(
    tmp_path, monkeypatch
):
    protected = []
    monkeypatch.setattr(setup, "secure_credentials_file", lambda path: protected.append(path))
    path = tmp_path / "config.json"
    assert setup.read_local_config(path) == ({}, None)
    first = {"bindings": [], "custom": "preserve me"}
    setup.write_local_config(path, first, None)
    raw, original = setup.read_local_config(path)
    assert raw == first and protected
    changed = {**raw, "custom": "edited during setup"}
    path.write_text(json.dumps(changed))
    with pytest.raises(setup.InfrastructureError, match="changed during"):
        setup.write_local_config(path, first, original)
    assert json.loads(path.read_text()) == changed


def test_invalid_existing_json_is_not_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "secure_credentials_file", lambda path: None)
    path = tmp_path / "config.json"
    path.write_text("{unfinished")
    with pytest.raises(setup.OperatorError, match="not overwritten"):
        setup.read_local_config(path)
    assert path.read_text() == "{unfinished"
    calls = []
    monkeypatch.setattr(setup, "secure_credentials_file", lambda path: calls.append(path))
    with pytest.raises(setup.InfrastructureError, match="regular JSON file"):
        setup.read_local_config(tmp_path)
    assert not calls


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink/permissions contract")
def test_config_rejects_symlink_and_has_owner_only_mode(tmp_path):
    target = tmp_path / "private.json"
    setup.write_local_config(target, {"bindings": []}, None)
    assert target.stat().st_mode & 0o777 == 0o600
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(setup.InfrastructureError):
        setup.read_local_config(alias)


@pytest.mark.parametrize("choice", ["0", "1", "2"])
def test_menu_uses_explicit_numbered_choices(choice):
    console = ScriptedConsole(["invalid", choice])
    assert (
        console.choose("Select document", ["First", "Second"])
        == {"0": None, "1": "First", "2": "Second"}[choice]
    )


def test_folder_collects_only_pdf_files_and_rejects_empty_inputs(tmp_path):
    one, two = tmp_path / "first.pdf", tmp_path / "second.PDF"
    one.write_bytes(b"%PDF-first")
    two.write_bytes(b"%PDF-second")
    (tmp_path / "unrelated.txt").write_text("ignored")
    assert pdf_paths("", tmp_path) == [one, two]
    assert pdf_paths(f'"{one}"', tmp_path) == [one]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(setup.OperatorError, match="between 1 and 100"):
        pdf_paths(str(empty), tmp_path)


@pytest.mark.asyncio
async def test_cancelled_replacement_never_submits_or_processes(tmp_path):
    path = tmp_path / "updated.pdf"
    path.write_bytes(b"%PDF-synthetic")
    document = SimpleNamespace(title="Existing manual", document_id=uuid4())
    service = SimpleNamespace(
        jobs=SimpleNamespace(
            head=AsyncMock(return_value=(uuid4(), uuid4())),
            knowledge=SimpleNamespace(
                get_revision_data=AsyncMock(return_value=SimpleNamespace(documents=[document]))
            ),
        ),
        submit=AsyncMock(),
        run_once=AsyncMock(),
    )
    binding = SimpleNamespace(scope=SCOPE)
    console = ScriptedConsole([str(path), "1", "n"])
    await upload_flow(service, binding, uuid4(), tmp_path, console, replace=True)
    service.submit.assert_not_awaited()
    service.run_once.assert_not_awaited()
    assert any("Existing manual" in line for line in console.output)


@pytest.mark.asyncio
async def test_duplicate_replacement_targets_are_rejected_before_submission(tmp_path):
    (tmp_path / "one.pdf").write_bytes(b"%PDF-one")
    (tmp_path / "two.pdf").write_bytes(b"%PDF-two")
    document = SimpleNamespace(title="Existing manual", document_id=uuid4())
    service = SimpleNamespace(
        jobs=SimpleNamespace(
            head=AsyncMock(return_value=(uuid4(), uuid4())),
            knowledge=SimpleNamespace(
                get_revision_data=AsyncMock(return_value=SimpleNamespace(documents=[document]))
            ),
        ),
        submit=AsyncMock(),
    )
    with pytest.raises(setup.OperatorError, match="same document"):
        await upload_flow(
            service,
            SimpleNamespace(scope=SCOPE),
            uuid4(),
            tmp_path,
            ScriptedConsole([str(tmp_path), "1", "1"]),
            replace=True,
        )
    service.submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_serving_digest_mismatch_does_not_change_pinned_profile(monkeypatch):
    config = IntakeConfig(
        bindings=[
            IntakeBinding(
                collection_id=uuid4(),
                scope=SCOPE,
                manager_user_ids=[uuid4()],
                index=IndexConfig(embedding=PROFILE),
            )
        ]
    )
    model = SimpleNamespace(
        inspect_embedding_model=AsyncMock(
            return_value=EmbeddingModelInfo(
                provider="ollama",
                model=PROFILE.model,
                revision="different",
                dimension=3,
                context_length=512,
            )
        )
    )
    monkeypatch.setattr("scripts.manage_knowledge.create_model_service", lambda profile: model)
    with pytest.raises(setup.OperatorError, match="pinned profile"):
        await verified_model(config, "http://explicit-model:11434")
    assert config.bindings[0].index.embedding == PROFILE

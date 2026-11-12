"""Operator setup and interactive replacement over disposable application/knowledge PG."""

import asyncio
import json
from argparse import Namespace
from uuid import uuid4

import pytest

from app.identity import IdentityProvider, UserIdentity
from app.knowledge.errors import KnowledgePermissionError
from app.repositories.exceptions import RepositoryIntegrityError
from scripts.knowledge_operator_setup import OPERATOR_INSTANCE, provision_operator
from scripts.knowledge_operator_ui import jobs_flow, upload_flow
from scripts.manage_knowledge import complete_config, execute
from tests.knowledge.conftest import postgres_fixture as postgres_fixture
from tests.knowledge.test_extraction import pdf_bytes
from tests.knowledge.test_intake import setup, submit
from tests.unit.test_knowledge_operator import ScriptedConsole

pytestmark = pytest.mark.asyncio


async def test_operator_is_new_os_link_not_a_student_and_concurrent_setup_reuses_it(pg):
    student = await pg.repositories.identities.create(UserIdentity(student_id="knowledge-operator"))
    ids = await asyncio.gather(
        *[provision_operator(pg.database, "test-local-owner") for _ in range(3)]
    )
    assert len(set(ids)) == 1 and ids[0] != student.user_id
    linked = await pg.repositories.identities.get_by_external_identity(
        provider=IdentityProvider.LOCAL,
        provider_instance=OPERATOR_INSTANCE,
        external_user_id="test-local-owner",
    )
    assert linked.user_id == ids[0]
    assert await pg.repositories.user_contexts.get(ids[0]) is not None
    assert (
        await pg.repositories.identities.get_by_user_id(student.user_id)
    ).external_identities == []


async def test_disabled_operator_is_not_reenabled_or_replaced(pg):
    operator = await provision_operator(pg.database, "test-owner")
    async with pg.database.connection() as connection:
        await connection.execute(
            "UPDATE application.identities SET status='disabled' WHERE user_id=%s", (operator,)
        )
    with pytest.raises(RepositoryIntegrityError, match="disabled"):
        await provision_operator(pg.database, "test-owner")


async def initial(kp, tmp_path):
    collection, binding, model, service = await setup(kp, tmp_path)
    first = await submit(
        service, collection, "Original generic manual before operator replacement."
    )
    ready = await service.run_once(collection.collection_id)
    async with kp.writer_database.connection() as connection:
        await connection.execute(
            "UPDATE knowledge.knowledge_collections SET active_revision_id=NULL "
            "WHERE collection_id=%s",
            (collection.collection_id,),
        )
    return collection, binding, model, service, first, ready


async def test_config_discovery_preserves_ready_baseline_and_does_not_broaden_scope(
    postgres_fixture, tmp_path
):
    collection, binding, _, service, _, ready = await initial(postgres_fixture, tmp_path)
    operator = uuid4()
    config = await complete_config({}, service.jobs, operator, ScriptedConsole(), root=tmp_path)
    actual = config.bindings[0]
    assert actual.collection_id == collection.collection_id
    assert actual.initial_revision_id == ready.result_revision_id
    assert actual.index.embedding == binding.index.embedding
    assert actual.manager_user_ids == [operator]
    assert (await service.jobs.head(actual))[0] is None
    existing = {"bindings": [binding.model_dump(mode="json")]}
    existing["bindings"][0]["scope"]["can_manage_knowledge"] = False
    with pytest.raises(ValueError, match="not broadened"):
        await complete_config(existing, service.jobs, operator, ScriptedConsole(), root=tmp_path)


async def test_existing_index_tuning_and_manager_grants_are_preserved(postgres_fixture, tmp_path):
    _, binding, _, service, _, ready = await initial(postgres_fixture, tmp_path)
    original = binding.model_dump(mode="json")
    original["initial_revision_id"] = str(ready.result_revision_id)
    original["index"]["chunking"]["max_chars"] = 1234
    original["manager_user_ids"].append("REPLACE_WITH_YOUR_USER_UUID")
    operator = uuid4()
    config = await complete_config(
        {"bindings": [original]}, service.jobs, operator, ScriptedConsole()
    )
    assert config.bindings[0].index.chunking.max_chars == 1234
    assert config.bindings[0].manager_user_ids == [*binding.manager_user_ids, operator]


async def test_scoped_bounded_job_list(postgres_fixture, tmp_path):
    collection, binding, _, service = await setup(postgres_fixture, tmp_path)
    first = await submit(service, collection, "One queued job.")
    second = await submit(service, collection, "Another queued job.")
    assert [j.job_id for j in await service.jobs.list(binding, limit=1)] == [second.job_id]
    assert {j.job_id for j in await service.jobs.list(binding)} == {first.job_id, second.job_id}
    with pytest.raises(ValueError):
        await service.jobs.list(binding, limit=True)
    denied = binding.model_copy(
        update={"scope": binding.scope.model_copy(update={"access_domain": "other"})}
    )
    with pytest.raises(KnowledgePermissionError):
        await service.jobs.list(denied)


async def test_setup_only_reuses_operator_and_never_submits_pdf_jobs(
    pg, postgres_fixture, tmp_path, monkeypatch
):
    _, binding, model, service, _, _ = await initial(postgres_fixture, tmp_path)
    args = Namespace(
        config=tmp_path / "operator.json",
        source_store=tmp_path / "operator-sources",
        inbox=tmp_path / "inbox",
        model_url=None,
        setup_only=True,
    )
    monkeypatch.setattr(
        "scripts.manage_knowledge.import_dsn", lambda: postgres_fixture.writer_database.config.dsn
    )
    monkeypatch.setattr("scripts.manage_knowledge.configured_dsn", lambda: pg.database.config.dsn)

    async def owner():
        return "operator-setup-only"

    async def verify(config, model_url):
        return model

    monkeypatch.setattr("scripts.manage_knowledge.owner_identity", owner)
    monkeypatch.setattr("scripts.manage_knowledge.verified_model", verify)
    assert await execute(args, ScriptedConsole()) == 0
    before = args.config.read_bytes()
    assert await execute(args, ScriptedConsole()) == 0
    assert args.config.read_bytes() == before
    assert len(await service.jobs.list(binding)) == 1
    async with pg.database.connection() as connection:
        assert (
            await (
                await connection.execute("SELECT count(*) AS n FROM application.identities")
            ).fetchone()
        )["n"] == 1


async def test_interactive_explicit_mapping_replaces_pdf_and_preserves_existing_config(
    pg, postgres_fixture, tmp_path, monkeypatch
):
    collection, binding, model, service, original, ready = await initial(postgres_fixture, tmp_path)
    config_path = tmp_path / "operator.json"
    value = binding.model_dump(mode="json")
    value["initial_revision_id"] = str(ready.result_revision_id)
    value["manager_user_ids"] = ["REPLACE_WITH_YOUR_USER_UUID"]
    config_path.write_text(json.dumps({"bindings": [value]}))
    pdf = tmp_path / "arbitrary-new-file-name.pdf"
    pdf.write_bytes(pdf_bytes("Updated generic knowledge submitted by the interactive operator."))
    args = Namespace(
        config=config_path,
        source_store=tmp_path / "operator-sources",
        inbox=tmp_path / "inbox",
        model_url=None,
        setup_only=False,
    )
    monkeypatch.setattr(
        "scripts.manage_knowledge.import_dsn", lambda: postgres_fixture.writer_database.config.dsn
    )
    monkeypatch.setattr("scripts.manage_knowledge.configured_dsn", lambda: pg.database.config.dsn)

    async def owner():
        return "interactive-operator"

    async def verify(config, model_url):
        return model

    monkeypatch.setattr("scripts.manage_knowledge.owner_identity", owner)
    monkeypatch.setattr("scripts.manage_knowledge.verified_model", verify)
    console = ScriptedConsole(["2", str(pdf), "1", "y", "3", "", "0"])
    assert await execute(args, console) == 0
    active, _ = await service.jobs.head(binding)
    assert active != ready.result_revision_id
    data = await postgres_fixture.reader.get_revision_data(active, binding.scope)
    assert (
        len(data.documents) == 1 and data.documents[0].document_id == original.payload.document_id
    )
    assert "Updated generic knowledge" in data.sections[0].text
    assert "REPLACE_" not in config_path.read_text()
    assert any("succeeded" in line for line in console.output)


async def test_menu_add_failure_retry_and_duplicate_use_existing_intake_contract(
    postgres_fixture, tmp_path
):
    collection, binding, model, service, _, ready = await initial(postgres_fixture, tmp_path)
    binding = binding.model_copy(update={"initial_revision_id": ready.result_revision_id})
    service.bindings[collection.collection_id] = binding
    pdf = tmp_path / "additional-source.pdf"
    pdf.write_bytes(pdf_bytes("An additional generic document from the operator menu."))
    model.failure = "provider"
    await upload_flow(
        service, binding, binding.manager_user_ids[0], tmp_path, ScriptedConsole([str(pdf), "y"])
    )
    failed = (await service.jobs.list(binding))[0]
    assert failed.status == "failed"
    model.failure = None
    await jobs_flow(service, binding, ScriptedConsole(["1", "y"]))
    assert (await service.jobs.get(binding, failed.job_id)).status == "succeeded"
    active, _ = await service.jobs.head(binding)
    data = await postgres_fixture.reader.get_revision_data(active, binding.scope)
    assert len(data.documents) == 2
    calls = len(model.requests)
    output = ScriptedConsole([str(pdf), "y"])
    await upload_flow(service, binding, binding.manager_user_ids[0], tmp_path, output)
    assert len(model.requests) == calls
    assert any("duplicate; reused" in line for line in output.output)

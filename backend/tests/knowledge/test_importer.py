from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.knowledge.errors import (
    KnowledgeIntegrityError,
    KnowledgeNotFound,
    KnowledgePermissionError,
)
from app.knowledge.extraction import ExtractionResult, PdfExtractor, StructureProfile
from app.knowledge.importer import ImportRegistration, RegisteredSource, import_registered
from app.knowledge.repository import validate_data
from app.knowledge.schemas import (
    AccessScope,
    ExtractedPage,
    ExtractionDiagnostic,
    ExtractionStatus,
    KnowledgeWriteScope,
    ProcessingStatus,
)
from app.knowledge.source_store import FileSourceStore
from tests.knowledge.test_extraction import pdf_bytes

WRITE_SCOPE = KnowledgeWriteScope(access_domain="course-test", can_manage_knowledge=True)


@pytest.fixture
def registered(tmp_path, sample):
    collection, revision, _ = sample
    (tmp_path / "source.pdf").write_bytes(pdf_bytes("Step ID\nGEN-01\nUseful explanation"))
    registration = ImportRegistration(
        collection=collection,
        embedding=revision.embedding,
        sources=[
            RegisteredSource(
                document_id=uuid4(),
                path="source.pdf",
                title="General source",
                profile=StructureProfile(expected_references=["GEN-01"], required_hint_levels=[4]),
            )
        ],
    )
    return registration


async def run_import(registration, repository, tmp_path):
    return await import_registered(
        registration,
        repository=repository,
        source_root=tmp_path,
        source_store=FileSourceStore(tmp_path / "stored"),
        extractor=PdfExtractor(),
        scope=KnowledgeWriteScope(
            can_manage_knowledge=True, access_domain=registration.collection.access_domain
        ),
    )


@pytest.mark.asyncio
async def test_import_roundtrip_rename_reuse_and_changed_bytes(repository, registered, tmp_path):
    first = await run_import(registered, repository, tmp_path)
    assert first.documents == 1 and first.sections == 1 and not first.reused
    scope = AccessScope(access_domain=registered.collection.access_domain)
    data = await repository.get_revision_data(first.revision_id, scope)
    assert data and not data.chunks and not data.steps
    assert data.documents[0].extraction_status == ExtractionStatus.EXTRACTED
    assert data.sections[0].processing_status == ProcessingStatus.READY
    digest = data.documents[0].source_hash
    assert (
        FileSourceStore(tmp_path / "stored").get(digest) == (tmp_path / "source.pdf").read_bytes()
    )
    (tmp_path / "source.pdf").rename(tmp_path / "renamed.pdf")
    renamed = registered.model_copy(
        update={
            "sources": [
                registered.sources[0].model_copy(update={"path": "renamed.pdf"}),
            ]
        }
    )
    assert (await run_import(renamed, repository, tmp_path)).reused
    (tmp_path / "renamed.pdf").write_bytes(pdf_bytes("Step ID\nGEN-01\nChanged explanation"))
    changed = await run_import(renamed, repository, tmp_path)
    assert changed.revision_id != first.revision_id and not changed.reused
    assert await repository.get_revision_data(first.revision_id, scope) == data


@pytest.mark.asyncio
async def test_failed_and_scanned_documents_are_durable(repository, registered, tmp_path):
    (tmp_path / "source.pdf").write_bytes(b"bad-pdf")
    failed = await run_import(registered, repository, tmp_path)
    assert failed.failed_documents == 1 and failed.sections == 0
    scope = AccessScope(access_domain=registered.collection.access_domain)
    data = await repository.get_revision_data(failed.revision_id, scope)
    assert data.documents[0].diagnostics[0].code == "malformed_pdf"
    (tmp_path / "source.pdf").write_bytes(pdf_bytes(""))
    scanned = await run_import(registered, repository, tmp_path)
    data = await repository.get_revision_data(scanned.revision_id, scope)
    assert data.documents[0].extraction_status == ExtractionStatus.FAILED
    assert not data.sections


@pytest.mark.asyncio
async def test_parallel_duplicate_import_has_one_snapshot(repository, registered, tmp_path):
    results = await asyncio.gather(
        *[run_import(registered, repository, tmp_path) for _ in range(2)]
    )
    assert results[0].revision_id == results[1].revision_id
    assert sorted(r.reused for r in results) == [False, True]


@pytest.mark.asyncio
async def test_scope_and_path_escape_fail_before_database_write(repository, registered, tmp_path):
    with pytest.raises(KnowledgePermissionError):
        await import_registered(
            registered,
            repository=repository,
            source_root=tmp_path,
            source_store=FileSourceStore(tmp_path / "stored"),
            extractor=PdfExtractor(),
            scope=KnowledgeWriteScope(can_manage_knowledge=True, access_domain="different"),
        )
    escaped = registered.model_copy(
        update={
            "sources": [
                registered.sources[0].model_copy(update={"path": "../outside.pdf"}),
            ]
        }
    )
    with pytest.raises(KnowledgeIntegrityError):
        await run_import(escaped, repository, tmp_path)


@pytest.mark.asyncio
async def test_provenance_tampering_rejected(repository, registered, tmp_path):
    result = await run_import(registered, repository, tmp_path)
    scope = AccessScope(access_domain=registered.collection.access_domain)
    revision = await repository.get_revision(result.revision_id, scope)
    data = await repository.get_revision_data(result.revision_id, scope)
    corrupted = data.model_copy(
        update={
            "sections": [
                data.sections[0].model_copy(update={"text": "invented source text"}),
            ]
        }
    )
    with pytest.raises(KnowledgeIntegrityError):
        validate_data(revision, corrupted)


@pytest.mark.asyncio
async def test_reimport_preserves_existing_snapshot(repository, registered, tmp_path):
    result = await run_import(registered, repository, tmp_path)
    scope = AccessScope(access_domain=registered.collection.access_domain)
    data = await repository.get_revision_data(result.revision_id, scope)
    modified = data.model_copy(
        update={
            "sections": [
                data.sections[0].model_copy(
                    update={
                        "title": "Persisted structural title",
                    }
                )
            ]
        }
    )
    await repository.write_revision_data(
        result.revision_id, modified, expected_version=1, scope=WRITE_SCOPE
    )
    assert (await run_import(registered, repository, tmp_path)).reused
    assert await repository.get_revision_data(result.revision_id, scope) == modified


@pytest.mark.asyncio
async def test_interrupted_extraction_resumes_version_zero(repository, registered, tmp_path):
    class InterruptedExtractor:
        def extract(self, content):
            raise RuntimeError("Simulated process interruption")

    with pytest.raises(RuntimeError, match="Simulated"):
        await import_registered(
            registered,
            repository=repository,
            source_root=tmp_path,
            source_store=FileSourceStore(tmp_path / "stored"),
            extractor=InterruptedExtractor(),
            scope=KnowledgeWriteScope(
                can_manage_knowledge=True, access_domain=registered.collection.access_domain
            ),
        )
    resumed = await run_import(registered, repository, tmp_path)
    assert not resumed.reused and resumed.sections == 1


@pytest.mark.asyncio
async def test_profile_change_creates_new_revision(repository, registered, tmp_path):
    first = await run_import(registered, repository, tmp_path)
    updated = registered.model_copy(
        update={
            "sources": [
                registered.sources[0].model_copy(
                    update={
                        "profile": StructureProfile(expected_references=["GEN-01", "GEN-02"]),
                    }
                )
            ]
        }
    )
    second = await run_import(updated, repository, tmp_path)
    assert not second.reused and second.revision_id != first.revision_id
    assert first.diagnostics == 0 and second.diagnostics == 1
    stored = await repository.get_revision_data(second.revision_id, WRITE_SCOPE)
    assert stored.documents[0].extraction_status == ExtractionStatus.EXTRACTED
    assert stored.sections[0].processing_status == ProcessingStatus.READY


@pytest.mark.asyncio
async def test_combined_diagnostics_are_bounded(repository, registered, tmp_path):
    class BlankExtractor:
        def extract(self, content):
            return ExtractionResult(
                pages=[ExtractedPage(number=i, text="") for i in range(1, 501)],
                diagnostics=[
                    ExtractionDiagnostic(code="page_without_text", page=i) for i in range(1, 501)
                ]
                + [ExtractionDiagnostic(code="ocr_required")],
            )

    registered = registered.model_copy(
        update={
            "sources": [
                registered.sources[0].model_copy(
                    update={
                        "profile": StructureProfile(
                            expected_references=[f"REF-{i}" for i in range(500)],
                        )
                    }
                ),
            ]
        }
    )
    result = await import_registered(
        registered,
        repository=repository,
        source_root=tmp_path,
        source_store=FileSourceStore(tmp_path / "stored"),
        extractor=BlankExtractor(),
        scope=KnowledgeWriteScope(
            can_manage_knowledge=True, access_domain=registered.collection.access_domain
        ),
    )
    assert result.diagnostics == 1000 and result.failed_documents == 1
    data = await repository.get_revision_data(
        result.revision_id, AccessScope(access_domain=registered.collection.access_domain)
    )
    assert data.documents[0].diagnostics[-1].code == "diagnostics_truncated"


@pytest.mark.asyncio
async def test_missing_source_has_sanitized_domain_error(repository, registered, tmp_path):
    (tmp_path / "source.pdf").unlink()
    with pytest.raises(KnowledgeNotFound, match="Registered source file is missing") as error:
        await run_import(registered, repository, tmp_path)
    assert str(tmp_path) not in str(error.value)
    assert await repository.get_collection(registered.collection.collection_id, WRITE_SCOPE) is None


@pytest.mark.asyncio
async def test_read_scope_cannot_import_or_reuse(repository, registered, tmp_path):
    for scope in [
        AccessScope(access_domain=registered.collection.access_domain),
        KnowledgeWriteScope(access_domain=registered.collection.access_domain),
    ]:
        with pytest.raises(KnowledgePermissionError):
            await import_registered(
                registered,
                repository=repository,
                source_root=tmp_path,
                source_store=FileSourceStore(tmp_path / "denied"),
                extractor=PdfExtractor(),
                scope=scope,
            )
    assert not (tmp_path / "denied").exists()
    assert not (await run_import(registered, repository, tmp_path)).reused
    with pytest.raises(KnowledgePermissionError):
        await import_registered(
            registered,
            repository=repository,
            source_root=tmp_path,
            source_store=FileSourceStore(tmp_path / "denied"),
            extractor=PdfExtractor(),
            scope=KnowledgeWriteScope(access_domain=registered.collection.access_domain),
        )

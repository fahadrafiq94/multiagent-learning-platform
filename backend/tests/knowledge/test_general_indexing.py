from __future__ import annotations

import hashlib
import json

import pytest
from psycopg.types.json import Jsonb

from app.knowledge import migrations
from app.knowledge.errors import KnowledgeUnavailable
from app.knowledge.repository import validate_data
from app.knowledge.schemas import (
    ChunkKind,
    ExtractedPage,
    ExtractionStatus,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeRevision,
    KnowledgeSection,
    KnowledgeWriteScope,
    PageSpan,
    ProcessingStatus,
    RevisionData,
)

SCOPE = KnowledgeWriteScope(access_domain="course-test", can_manage_knowledge=True)


def test_mixed_general_text_needs_no_type_or_approval(sample):
    _, revision, data = sample
    section = data.sections[0].model_copy(
        update={"text": "Reservations prevent overselling. Click Save in the order screen."}
    )
    evidence = data.chunks[0].model_copy(update={"content": section.text})
    stored = validate_data(
        revision,
        RevisionData(
            documents=data.documents,
            sections=[section],
            chunks=[evidence.model_copy(update={"namespace": None, "step_id": None})],
        ),
    )
    assert stored.sections[0].processing_status == ProcessingStatus.READY
    assert "semantic_type" not in stored.sections[0].model_dump()
    assert "review_status" not in stored.sections[0].model_dump()


def test_authored_hint_is_optional_and_needs_no_course_step(sample):
    _, revision, data = sample
    hint = KnowledgeChunk(
        revision_id=revision.revision_id,
        document_id=data.documents[0].document_id,
        section_id=data.sections[0].section_id,
        kind=ChunkKind.HINT,
        content="Consider the effect of a reservation.",
        hint_level=4,
    )
    assert (
        validate_data(
            revision, RevisionData(documents=data.documents, sections=data.sections, chunks=[hint])
        )
        .chunks[0]
        .step_id
        is None
    )


def test_historical_annotations_are_preserved_but_not_policy(sample):
    section = sample[2].sections[0].model_dump()
    section.update(
        semantic_type="unknown",
        review_status="needs_review",
        classification={"proposal": {"split_before": []}, "reviewed_at": "pending"},
    )
    restored = KnowledgeSection.model_validate(section)
    assert restored.processing_status == ProcessingStatus.READY
    assert restored.legacy_annotations["classification"] == section["classification"]
    assert not hasattr(restored, "classification")
    document = sample[2].documents[0].model_dump()
    document.update(extraction_status="needs_review", pages=[{"number": 1, "text": "Useful"}])
    assert (
        KnowledgeDocument.model_validate(document).extraction_status == ExtractionStatus.EXTRACTED
    )
    document["pages"] = [{"number": 1, "text": ""}]
    assert KnowledgeDocument.model_validate(document).extraction_status == ExtractionStatus.FAILED


def test_historical_pipeline_keeps_revision_fingerprint(sample):
    raw = sample[1].model_dump(mode="json")
    # Actual historical payloads predate operational indexing state.
    raw.pop("indexing_status")
    raw.pop("indexing_error")
    raw["pipeline"] = {
        "extractor": "old-v1",
        "classifier": "not-run",
        "taxonomy": "knowledge-v1",
        "chunker": "not-run",
        "policy": "knowledge-v1",
    }
    expected = hashlib.sha256(
        json.dumps(
            {k: v for k, v in raw.items() if k not in {"revision_id", "version", "status"}},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    restored = KnowledgeRevision.model_validate(raw)
    assert restored.revision_id == sample[1].revision_id
    assert restored.fingerprint == expected
    assert not hasattr(restored.pipeline, "classifier")


@pytest.mark.asyncio
@pytest.mark.parametrize("postgres_fixture", ["foundation"], indirect=True)
async def test_forward_upgrade_preserves_imports_and_rolls_back_on_failure(
    postgres_fixture, sample, tmp_path, monkeypatch
):
    repository = postgres_fixture.writer
    collection, revision, data = sample
    await repository.create_collection(collection, scope=SCOPE)
    raw = revision.model_dump(mode="json")
    raw["pipeline"].pop("legacy_versions")
    raw["pipeline"].update(classifier="not-run", taxonomy="knowledge-v1")
    revision = KnowledgeRevision.model_validate(raw)
    await repository.create_revision(revision, scope=SCOPE)
    text = "Business purpose and screen instructions coexist."
    document = data.documents[0].model_copy(update={"pages": [ExtractedPage(number=1, text=text)]})
    section = KnowledgeSection(
        revision_id=revision.revision_id,
        document_id=document.document_id,
        title="Original source",
        text=text,
        page_start=1,
        page_end=1,
        page_spans=[PageSpan(page=1, start=0, end=len(text), section_start=0)],
    )
    doc_payload = document.model_dump(
        mode="json", exclude={"revision_id", "document_id", "source_hash", "workflow_authority"}
    )
    doc_payload["extraction_status"] = "needs_review"
    doc_payload["diagnostics"] = [{"code": "missing_hint_level_4"}]
    section_payload = section.model_dump(
        mode="json", exclude={"revision_id", "document_id", "section_id", "processing_status"}
    )
    section_payload["classification"] = {"proposal": {"split_before": []}}
    database = postgres_fixture.writer_database
    async with database.connection() as connection:
        await connection.execute(
            "INSERT INTO knowledge.knowledge_documents VALUES (%s,%s,%s,%s,%s)",
            (
                revision.revision_id,
                document.document_id,
                document.source_hash,
                document.workflow_authority,
                Jsonb(doc_payload),
            ),
        )
        await connection.execute(
            "INSERT INTO knowledge.knowledge_sections "
            "VALUES (%s,%s,%s,'unknown','needs_review',%s)",
            (
                revision.revision_id,
                document.document_id,
                section.section_id,
                Jsonb(section_payload),
            ),
        )
        await connection.execute(
            "UPDATE knowledge.knowledge_revisions SET version=1 WHERE revision_id=%s",
            (revision.revision_id,),
        )
    sql_root = migrations.SQL_ROOT
    for source in sql_root.glob("*.sql"):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    failed = tmp_path / "002_general_indexing.sql"
    failed.write_text(failed.read_text() + "\nSELECT * FROM knowledge.intentionally_missing;\n")
    with monkeypatch.context() as patch:
        patch.setattr(migrations, "SQL_ROOT", tmp_path)
        with pytest.raises(KnowledgeUnavailable):
            await migrations.migrate(database, embedding_dimension=3)
    async with database.connection() as connection:
        cursor = await connection.execute(
            "SELECT semantic_type,review_status,payload FROM knowledge.knowledge_sections"
        )
        row = await cursor.fetchone()
        assert row["review_status"] == "needs_review" and row["payload"] == section_payload
    assert await migrations.migrate(database, embedding_dimension=3) == [
        "002_general_indexing",
        "003_chunk_index",
        "004_intake_jobs",
    ]
    assert await migrations.migrate(database, embedding_dimension=3) == []
    restored = await repository.get_revision_data(revision.revision_id, SCOPE)
    assert restored.sections[0].text == text
    assert restored.sections[0].processing_status == ProcessingStatus.READY
    assert restored.sections[0].legacy_annotations["review_status"] == "needs_review"
    assert (
        restored.sections[0].legacy_annotations["classification"]
        == section_payload["classification"]
    )
    assert restored.documents[0].source_hash == document.source_hash
    assert restored.documents[0].pages == document.pages
    assert (
        await repository.get_revision(revision.revision_id, SCOPE)
    ).fingerprint == revision.fingerprint
    # This previously unclassified source can now support evidence without approval.
    evidence = KnowledgeChunk(
        revision_id=revision.revision_id,
        document_id=document.document_id,
        section_id=section.section_id,
        kind=ChunkKind.EVIDENCE,
        content=text,
        embedding=[1, 0, 0],
    )
    await repository.write_revision_data(
        revision.revision_id,
        restored.model_copy(update={"chunks": [evidence]}),
        scope=SCOPE,
        expected_version=1,
    )
    assert (await repository.get_revision_data(revision.revision_id, SCOPE)).chunks == [evidence]
    assert (
        await repository.get_revision_data(
            revision.revision_id, SCOPE.model_copy(update={"access_domain": "other-course"})
        )
        is None
    )

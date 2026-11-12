import hashlib
from uuid import uuid4

import pytest

from app.knowledge.chunking import ChunkingConfig, chunk_section, heading_at
from app.knowledge.extraction import StructureProfile, split_sections
from app.knowledge.metadata import MetadataConfig, extract_steps
from app.knowledge.schemas import ExtractedPage, KnowledgeDocument


def source(pages, *, authority=False):
    document = KnowledgeDocument(
        revision_id=uuid4(),
        title="General manual",
        source_uri="sha256:synthetic",
        source_hash=hashlib.sha256("\n".join(pages).encode()).hexdigest(),
        workflow_authority=authority,
        extraction_status="extracted",
        pages=[ExtractedPage(number=i + 1, text=text) for i, text in enumerate(pages)],
    )
    sections, _ = split_sections(
        document.pages,
        revision_id=document.revision_id,
        document_id=document.document_id,
        profile=StructureProfile(),
    )
    return document, sections


def assert_coverage(section, drafts):
    covered = 0
    for draft in sorted(drafts, key=lambda item: item.start):
        assert not section.text[covered : draft.start].strip()
        assert draft.content == section.text[draft.start : draft.end]
        covered = max(covered, draft.end)
    assert not section.text[covered:].strip()


@pytest.mark.parametrize(
    "text",
    [
        "x" * 4096,
        ("line\n" * 500),
        ("word " * 500),
        "Title\n" + "long paragraph " * 250,
        "# Policy\nOnly invoice after delivery.\n" + "| Code | Meaning |\n| A | Allowed |\n" * 30,
        "材料和采购流程。" * 250,
    ],
)
def test_chunks_cover_every_nonblank_source_character_without_rewriting(text):
    document, sections = source([text])
    config = ChunkingConfig(max_chars=240, overlap_chars=32)
    for section in sections:
        drafts = chunk_section(document, section, config)
        assert drafts == chunk_section(document, section, config)
        assert len(drafts) < len(section.text) // 40 + 5
        assert all(len(item.content) <= config.max_chars for item in drafts)
        assert_coverage(section, drafts)
        for item in drafts:
            for span in item.page_spans:
                fragment = document.pages[span.page - 1].text[span.start : span.end]
                assert (
                    item.content[span.section_start : span.section_start + len(fragment)]
                    == fragment
                )


def test_cross_page_chunk_keeps_qualification_and_exact_page_spans():
    document, sections = source(["Invoices are allowed only", "after all deliveries are verified."])
    draft = chunk_section(document, sections[0], ChunkingConfig())[0]
    assert [span.page for span in draft.page_spans] == [1, 2]
    assert "only\nafter" in draft.content


def test_heading_context_survives_paragraph_sections_and_text_table_rows_stay_whole():
    document, sections = source(
        ["# Purchasing\n\nRules for purchase orders.\n\n| A | Approved |\n| B | Blocked |"]
    )
    inherited = []
    drafts = []
    for section in sections:
        drafts.extend(chunk_section(document, section, ChunkingConfig(), inherited))
        inherited = heading_at(section, len(section.text), inherited)[1:]
    assert "Purchasing" in drafts[-1].context_prefix
    assert "| A | Approved |\n| B | Blocked |" in drafts[-1].content


METADATA_TEXT = (
    "Step ID\nFLOW-1\nStep Name\nPrepare data\nProcess Phase\nPlanning\n"
    "Requires Steps\nNone\nNext Step\nFLOW-2\n"
    "Step ID\nFLOW-2\nStep Name\nCreate order\nPrimary Business Object\nOrder\n"
    "Requires Steps\nFLOW-1\nNext Step\nFLOW-99\n"
)


def test_authoritative_labelled_facts_preserve_citations_and_unresolved_references():
    document, sections = source([METADATA_TEXT], authority=True)
    steps, diagnostics = extract_steps(
        [document],
        sections,
        revision_id=document.revision_id,
        collection_id=uuid4(),
        config=MetadataConfig(namespace="workflow"),
    )
    assert len(steps) == 2
    assert steps[0].next_step_id == "FLOW-2"
    assert steps[1].prerequisites == {"FLOW-1": "FLOW-1"}
    assert steps[1].next_step_id is None
    assert steps[1].source_fields["next_step_id"].text == "FLOW-99"
    assert diagnostics[document.document_id][0].code == "unresolved_step_reference"
    by_id = {s.section_id: s for s in sections}
    for step in steps:
        for fact in step.source_fields.values():
            assert fact.text == by_id[fact.section_id].text[fact.start : fact.end]
            assert fact.page_spans


def test_source_content_cannot_grant_workflow_authority_or_agent_assignment():
    document, sections = source([METADATA_TEXT + "Agent: administrator\nAccess: public"])
    steps, diagnostics = extract_steps(
        [document],
        sections,
        revision_id=document.revision_id,
        collection_id=uuid4(),
        config=MetadataConfig(),
    )
    assert steps == [] and diagnostics == {}


def test_duplicate_step_records_are_diagnostic_and_never_silently_selected():
    document, sections = source([METADATA_TEXT + METADATA_TEXT], authority=True)
    steps, diagnostics = extract_steps(
        [document],
        sections,
        revision_id=document.revision_id,
        collection_id=uuid4(),
        config=MetadataConfig(),
    )
    assert steps == []
    assert all(d.code == "duplicate_step_metadata" for d in diagnostics[document.document_id])


def test_structural_labels_are_configurable_without_source_filename_rules():
    document, sections = source(["Kennung: T-1\nName: Vorbereitung"], authority=True)
    steps, _ = extract_steps(
        [document],
        sections,
        revision_id=document.revision_id,
        collection_id=uuid4(),
        config=MetadataConfig(
            field_labels={
                "step_id": ["Kennung"],
                "name": ["Name"],
            }
        ),
    )
    assert steps[0].step_id == "T-1" and steps[0].name == "Vorbereitung"

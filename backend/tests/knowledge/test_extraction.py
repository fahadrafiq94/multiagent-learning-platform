from __future__ import annotations

import io
import subprocess
import sys
from uuid import uuid4

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.knowledge import pdf_worker
from app.knowledge.extraction import (
    ExtractionResult,
    PdfExtractor,
    StructureProfile,
    split_sections,
)
from app.knowledge.pdf_worker import parse
from app.knowledge.schemas import ExtractedPage


def pdf_bytes(*texts: str, encrypted: bool = False) -> bytes:
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): writer._add_object(font)}
                ),
            }
        )
        stream = DecodedStreamObject()
        lines = [
            line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            for line in text.splitlines()
        ]
        stream.set_data(
            (
                "BT /F1 12 Tf 20 750 Td 15 TL "
                + " ".join(f"({line}) Tj T*" for line in lines)
                + " ET"
            ).encode()
        )
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("secret-test-password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_real_pdf_worker_preserves_page_order_and_cross_page_record():
    result = PdfExtractor().extract(
        pdf_bytes(
            "Preface\nStep ID\nEX-01\nFirst page", "Continuation\nStep ID\nEX-02\nSecond record"
        )
    )
    assert not result.failed
    assert len(result.pages) == 2
    sections, diagnostics = split_sections(
        result.pages,
        revision_id=uuid4(),
        document_id=uuid4(),
        profile=StructureProfile(
            expected_references=["EX-01", "EX-02"],
        ),
    )
    assert not diagnostics
    record = next(s for s in sections if s.reference_id == "EX-01")
    assert (record.page_start, record.page_end) == (1, 2)
    assert "Continuation" in record.text
    for span in record.page_spans:
        source = result.pages[span.page - 1].text[span.start : span.end]
        assert record.text[span.section_start : span.section_start + len(source)] == source


def test_worker_emits_utf8_bytes_under_non_utf8_stdout(monkeypatch):
    """Regression: on Windows the worker's text stdout defaults to the ANSI
    codepage (e.g. cp1252); non-ASCII extraction then raised
    UnicodeEncodeError, the worker exited nonzero, and the job was
    misreported as intake_invalid_source. The worker must emit UTF-8 bytes
    regardless of the host text encoding."""
    text = "café ✓ 中"  # ✓ and 中 are unencodable in cp1252
    monkeypatch.setattr(
        pdf_worker,
        "parse",
        lambda content: ExtractionResult(pages=[ExtractedPage(number=1, text=text)]),
    )
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"")))

    sink = io.BytesIO()
    hostile_stdout = io.TextIOWrapper(sink, encoding="cp1252", newline="")
    monkeypatch.setattr(sys, "stdout", hostile_stdout)

    pdf_worker.main()
    hostile_stdout.detach()

    decoded = ExtractionResult.model_validate_json(sink.getvalue())
    assert not decoded.failed
    assert decoded.pages[0].text == text


@pytest.mark.parametrize(
    "content,code",
    [
        (b"malformed private text", "malformed_pdf"),
        (pdf_bytes("Confidential", encrypted=True), "encrypted_pdf"),
        (pdf_bytes(), "empty_pdf"),
        (b"", "source_size_limit"),
    ],
)
def test_parser_failures_are_stable_and_sanitized(content, code):
    result = PdfExtractor().extract(content)
    assert result.failed
    assert [d.code for d in result.diagnostics] == [code]
    assert not result.pages


def test_scanned_or_blank_pdf_requires_ocr_without_inventing_text():
    result = parse(pdf_bytes(""))
    assert not result.failed
    assert result.pages[0].text == ""
    assert [d.code for d in result.diagnostics] == ["page_without_text", "ocr_required"]


def test_worker_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert PdfExtractor().extract(b"pdf").diagnostics[0].code == "extraction_timeout"


def test_generic_document_and_large_record_preserve_all_text():
    pages = [ExtractedPage(number=1, text="General explanation. " * 3000)]
    sections, diagnostics = split_sections(
        pages,
        revision_id=uuid4(),
        document_id=uuid4(),
        profile=StructureProfile(),
    )
    assert not diagnostics
    assert len(sections) == 2
    assert "".join(s.text for s in sections) == pages[0].text
    assert all(s.reference_id is None for s in sections)


def test_structure_profile_reports_reference_diagnostics_without_hint_requirements():
    sections, diagnostics = split_sections(
        [
            ExtractedPage(
                number=1,
                text=(
                    "Step ID\nX-01\nHint Ladder\nLevel 1: one\nLevel 2: two\nLevel 3: three\n"
                    "Step ID\nX-01\nRepeated record\nStep ID\nX-99\nOther record"
                ),
            )
        ],
        revision_id=uuid4(),
        document_id=uuid4(),
        profile=StructureProfile(
            expected_references=["X-01", "X-02"],
            required_hint_levels=[1, 2, 3, 4],
        ),
    )
    assert {d.code for d in diagnostics} >= {
        "duplicate_reference",
        "missing_reference",
        "unexpected_reference",
    }
    assert sections[0].reference_id == "X-01"


def test_general_paragraph_boundaries_and_authored_hint_levels():
    body = "Introduction\n\nAnother paragraph\n\nFinal paragraph"
    sections, diagnostics = split_sections(
        [ExtractedPage(number=1, text=body)],
        revision_id=uuid4(),
        document_id=uuid4(),
        profile=StructureProfile(),
    )
    assert len(sections) == 3 and not diagnostics
    assert "".join(s.text for s in sections) == body
    _, diagnostics = split_sections(
        [ExtractedPage(number=1, text="Step ID\nX-1\nLevel 1: a\nLevel 2: b\nLevel 3: c")],
        revision_id=uuid4(),
        document_id=uuid4(),
        profile=StructureProfile(required_hint_levels=[1, 2, 3, 4]),
    )
    assert not diagnostics


def test_pdf_page_limit():
    assert parse(pdf_bytes(*([""] * 501))).diagnostics[0].code == "page_count_limit"

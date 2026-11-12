"""Private subprocess entry point: PDF bytes in, bounded extraction JSON out."""

from __future__ import annotations

import io
import logging
import sys
import warnings

from pypdf import PdfReader

from app.knowledge.extraction import MAX_SOURCE_BYTES, ExtractionResult, failure
from app.knowledge.schemas import ExtractedPage, ExtractionDiagnostic


def parse(content: bytes) -> ExtractionResult:
    if not content or len(content) > MAX_SOURCE_BYTES:
        return failure("source_size_limit")
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            return failure("encrypted_pdf")
        if not reader.pages:
            return failure("empty_pdf")
        if len(reader.pages) > 500:
            return failure("page_count_limit")
        pages = []
        diagnostics = []
        total = 0
        for number, page in enumerate(reader.pages, 1):
            # Cap decoded content before text layout; decompression itself can
            # allocate memory, so container resource limits remain necessary.
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > 10 * 1024 * 1024:
                return failure("page_content_limit")
            text = page.extract_text() or ""
            total += len(text)
            if len(text) > 100000 or total > 2000000:
                return failure("extracted_text_limit")
            pages.append(ExtractedPage(number=number, text=text))
            if not text.strip():
                diagnostics.append(ExtractionDiagnostic(code="page_without_text", page=number))
        if all(not p.text.strip() for p in pages):
            diagnostics.append(ExtractionDiagnostic(code="ocr_required"))
        return ExtractionResult(pages=pages, diagnostics=diagnostics)
    except Exception:
        # Parser exceptions may contain source bytes; only stable codes cross
        # the process boundary. Never return tracebacks or source text as errors.
        return failure("malformed_pdf")


def main() -> None:
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    result = parse(sys.stdin.buffer.read(MAX_SOURCE_BYTES + 1))
    # Emit UTF-8 bytes directly: text stdout would use the host locale's
    # codepage (e.g. cp1252 on Windows), crashing on non-ASCII extraction.
    sys.stdout.buffer.write(result.model_dump_json().encode("utf-8"))


if __name__ == "__main__":
    main()

"""Explicit labelled fields from trusted workflow sources; no semantic classifier."""

import re
from uuid import UUID

from pydantic import Field, model_validator

from app.knowledge.chunking import sliced_spans
from app.knowledge.schemas import (
    ExtractionDiagnostic,
    FactSource,
    KnowledgeDocument,
    KnowledgeModel,
    KnowledgeSection,
    Label,
    ProcessStep,
)

METADATA_VERSION = "labelled-facts-v1"
DEFAULT_LABELS = {
    "step_id": ["Step ID"],
    "name": ["Step Name"],
    "course_label": ["Course Overview Label"],
    "phase": ["Process Phase", "Phase"],
    "business_object": ["Primary Business Object", "Business Object"],
    "prerequisites": ["Requires Steps", "Prerequisites"],
    "next_step_id": ["Next Step", "Next Step ID"],
}


class MetadataConfig(KnowledgeModel):
    namespace: Label | None = None
    field_labels: dict[str, list[Label]] = Field(
        default_factory=lambda: {key: list(value) for key, value in DEFAULT_LABELS.items()}
    )

    @model_validator(mode="after")
    def labels(self):
        aliases = [label.casefold() for labels in self.field_labels.values() for label in labels]
        if set(self.field_labels) - set(DEFAULT_LABELS) or len(aliases) != len(set(aliases)):
            raise ValueError("Metadata labels must be unique and use known structural fields.")
        if not self.field_labels.get("step_id") or not self.field_labels.get("name"):
            raise ValueError("Step identity/name labels are required for canonical metadata.")
        return self


def labelled_fields(section: KnowledgeSection, config: MetadataConfig) -> dict[str, FactSource]:
    aliases = {
        label.casefold(): field for field, labels in config.field_labels.items() for label in labels
    }
    pattern = re.compile(
        r"(?mi)^[ \t]*("
        + "|".join(re.escape(label) for label in sorted(aliases, key=len, reverse=True))
        + r")[ \t]*(?::[ \t]*(.*))?[ \t]*$"
    )
    matches = list(pattern.finditer(section.text))
    found = {}
    for index, match in enumerate(matches):
        field = aliases[match[1].casefold()]
        start = match.start(2) if match[2] else match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section.text)
        while start < end and section.text[start].isspace():
            start += 1
        while end > start and section.text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        if field in found:
            # Duplicate labels are structurally ambiguous; never silently choose one.
            return {}
        spans = sliced_spans(section, start, end)
        if spans:
            found[field] = FactSource(
                section_id=section.section_id,
                start=start,
                end=end,
                text=section.text[start:end],
                page_spans=spans,
            )
    return found


def extract_steps(
    documents: list[KnowledgeDocument],
    sections: list[KnowledgeSection],
    *,
    revision_id: UUID,
    collection_id: UUID,
    config: MetadataConfig,
) -> tuple[list[ProcessStep], dict[UUID, list[ExtractionDiagnostic]]]:
    trusted = {d.document_id for d in documents if d.workflow_authority}
    candidates: dict[str, list[tuple[KnowledgeSection, dict[str, FactSource]]]] = {}
    diagnostics: dict[UUID, list[ExtractionDiagnostic]] = {}

    def diagnostic(document_id, code, reference=None):
        diagnostics.setdefault(document_id, []).append(
            ExtractionDiagnostic(
                code=code,
                reference_id=reference,
            )
        )

    for section in sorted(sections, key=lambda s: (str(s.document_id), s.ordinal)):
        if section.document_id not in trusted or section.processing_status != "ready":
            continue
        fields = labelled_fields(section, config)
        if "step_id" not in fields or "name" not in fields:
            continue
        step_id = fields["step_id"].text
        if (
            len(step_id) > 255
            or "\n" in step_id
            or len(fields["name"].text) > 255
            or section.reference_id is not None
            and section.reference_id != step_id
        ):
            diagnostic(section.document_id, "invalid_step_fields", section.reference_id)
            continue
        candidates.setdefault(step_id, []).append((section, fields))
    unambiguous = {key: rows[0] for key, rows in candidates.items() if len(rows) == 1}
    for key, rows in candidates.items():
        if len(rows) > 1:
            for section, _ in rows:
                diagnostic(section.document_id, "duplicate_step_metadata", key)
    result = []
    for step_id, (section, fields) in sorted(unambiguous.items()):

        def links(field, fields=fields, section=section, step_id=step_id):
            if field not in fields:
                return []
            text = fields[field].text
            known = [
                key
                for key in unambiguous
                if re.search(
                    r"(?<![\w.-])" + re.escape(key) + r"(?![\w.-])",
                    text,
                )
            ]
            explicit = re.findall(r"\b[A-Z0-9]+(?:[-_.][A-Z0-9]+)+\b", text)
            if any(key not in unambiguous for key in explicit):
                diagnostic(section.document_id, "unresolved_step_reference", step_id)
            return known

        prerequisites = {key: fields["prerequisites"].text for key in links("prerequisites")}
        next_steps = links("next_step_id")
        if len(next_steps) > 1:
            diagnostic(section.document_id, "ambiguous_next_step", step_id)
        attributes = {}
        for field in ("course_label", "phase", "business_object"):
            if field in fields:
                if len(fields[field].text) <= 255:
                    attributes[field] = fields[field].text
                else:
                    diagnostic(section.document_id, "metadata_field_too_long", step_id)
        result.append(
            ProcessStep(
                revision_id=revision_id,
                namespace=config.namespace or str(collection_id),
                step_id=step_id,
                source_document_id=section.document_id,
                name=fields["name"].text,
                prerequisites=prerequisites,
                next_step_id=next_steps[0] if len(next_steps) == 1 else None,
                source_fields=fields,
                **attributes,
            )
        )
    return result, diagnostics

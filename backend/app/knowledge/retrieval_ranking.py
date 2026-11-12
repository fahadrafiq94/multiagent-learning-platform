"""Deterministic rank fusion, source-preserving deduplication and evidence budgeting."""

import re

from app.knowledge.retrieval_schemas import (
    RetrievalConfig,
    RetrievalRequest,
    RetrievalStats,
    RetrievedEvidence,
    SearchCandidate,
    SourceBundle,
)


def references(request: RetrievalRequest) -> list[str]:
    values = [
        *request.reference_ids,
        *re.findall(r"\b[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)+\b", request.query),
    ]
    return list(dict.fromkeys(value.lower() for value in values if len(value) <= 255))[:20]


def rank_candidates(
    candidates: list[SearchCandidate], request: RetrievalRequest, config: RetrievalConfig
):
    refs = references(request)
    identifier_only = request.query.strip().lower() in refs
    explicit_matches = any(candidate.identifier_rank is not None for candidate in candidates)
    ranked = []
    for candidate in candidates:
        exact = candidate.identifier_rank is not None
        relevant = exact or (
            not identifier_only
            and (
                candidate.vector_similarity >= config.min_vector_similarity
                or (
                    candidate.lexical_score > 0
                    and candidate.vector_similarity >= config.lexical_vector_floor
                )
            )
        )
        if not relevant:
            continue
        score = sum(
            weight / (config.rrf_constant + rank)
            for rank, weight in [
                (candidate.vector_rank, 1),
                (candidate.lexical_rank, 1),
                (candidate.identifier_rank, 3),
            ]
            if rank is not None
        )
        # Preferences never establish relevance or exclude a different step/document.
        if candidate.current_step_match and not explicit_matches:
            score += 0.002
        if candidate.phase_match:
            score += 0.001
        score += min(candidate.purpose_score, 1) * 0.001
        ranked.append((candidate, score))
    return sorted(
        ranked, key=lambda item: (-item[1], -item[0].vector_similarity, str(item[0].chunk_id))
    )


def item_chars(item: RetrievedEvidence) -> int:
    return len(item.model_dump_json())


def select_evidence(
    ranked: list[tuple[SearchCandidate, float]],
    items: list[RetrievedEvidence],
    request: RetrievalRequest,
    config: RetrievalConfig,
    stats: RetrievalStats,
) -> tuple[list[RetrievedEvidence], RetrievalStats]:
    by_id = {item.evidence_id: item for item in items}
    # Preserve at least one matching passage for each explicitly named known step
    # when k/budget permits, then fill with the remaining fused ranking.
    priority = []
    for ref in references(request):
        for candidate, _score in ranked:
            if ref in {
                str(candidate.reference_id or "").lower(),
                str(candidate.step_id or "").lower(),
            }:
                if candidate.chunk_id not in priority:
                    priority.append(candidate.chunk_id)
                break
    order = {key: i for i, key in enumerate(priority)}
    processing = sorted(
        enumerate(ranked),
        key=lambda entry: (
            0 if entry[1][0].chunk_id in order else 1,
            order.get(entry[1][0].chunk_id, entry[0]),
        ),
    )
    selected: list[RetrievedEvidence] = []
    used = 0
    deduplicated = 0
    omissions = 0
    for _, (candidate, score) in processing:
        item = by_id[candidate.chunk_id].model_copy(
            update={
                "score": score,
                "vector_similarity": candidate.vector_similarity,
                "lexical_score": candidate.lexical_score,
            }
        )
        duplicate = next((i for i, old in enumerate(selected) if old.content == item.content), None)
        if duplicate is not None:
            old = selected[duplicate]
            deduplicated += 1
            citations = [*old.citations, *[c for c in item.citations if c not in old.citations]]
            merged = old.model_copy(
                update={"citations": citations[: config.max_citations_per_item]}
            )
            cost = used - item_chars(old) + item_chars(merged)
            if cost <= request.max_chars:
                selected[duplicate] = merged
                used = cost
            else:
                omissions += 1
            continue
        if any(
            old.citations[0].document_id == item.citations[0].document_id
            and old.citations[0].section_id == item.citations[0].section_id
            and old.source_start is not None
            and old.source_end is not None
            and item.source_start is not None
            and item.source_end is not None
            and old.source_start <= item.source_start
            and old.source_end >= item.source_end
            for old in selected
        ):
            deduplicated += 1
            continue
        if len(selected) >= request.top_k:
            continue
        cost = item_chars(item)
        if used + cost > request.max_chars:
            omissions += 1
            continue
        selected.append(item)
        used += cost
    selected.sort(key=lambda item: (-item.score, str(item.evidence_id)))
    represented = {str(c.reference_id or "").lower() for item in selected for c in item.citations}
    return selected, stats.model_copy(
        update={
            "returned_count": len(selected),
            "deduplicated": deduplicated,
            "budget_omissions": omissions,
            "omitted_references": [ref for ref in references(request) if ref not in represented],
        }
    )


def source_bundles(items: list[RetrievedEvidence]) -> list[SourceBundle]:
    groups = {}
    for item in items:
        for citation in item.citations:
            key = (citation.revision_id, citation.document_id)
            if key not in groups:
                groups[key] = SourceBundle(
                    revision_id=citation.revision_id,
                    document_id=citation.document_id,
                    title=citation.title,
                    evidence_ids=[],
                    pages=[],
                )
            group = groups[key]
            if item.evidence_id not in group.evidence_ids:
                group.evidence_ids.append(item.evidence_id)
            group.pages[:] = sorted(set(group.pages) | {span.page for span in citation.page_spans})
    return list(groups.values())

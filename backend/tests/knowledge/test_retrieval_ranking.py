from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.knowledge.retrieval import KnowledgeRetrievalService
from app.knowledge.retrieval_ranking import (
    item_chars,
    rank_candidates,
    select_evidence,
    source_bundles,
)
from app.knowledge.retrieval_schemas import (
    EvidenceCitation,
    RetrievalConfig,
    RetrievalRequest,
    RetrievalStats,
    RetrievedEvidence,
    RevisionPin,
    SearchCandidate,
)
from app.knowledge.schemas import (
    AccessScope,
    KnowledgeRevision,
    PageSpan,
    PipelineVersions,
    ProcessingStatus,
)
from app.knowledge.search_repository import SearchRepository
from tests.knowledge.test_indexer import PROFILE
from tests.knowledge.test_retrieval import QueryModel


def request(**kwargs):
    return RetrievalRequest(collection_id=uuid4(), query="purchase order", **kwargs)


def candidate(number, **kwargs):
    return SearchCandidate(
        chunk_id=UUID(int=number),
        vector_similarity=0.8,
        lexical_score=0,
        vector_rank=number,
        **kwargs,
    )


def evidence(number, *, content="Source evidence.", document=None, start=0):
    return RetrievedEvidence(
        evidence_id=UUID(int=number),
        kind="chunk",
        content=content,
        source_start=start,
        source_end=start + len(content),
        citations=[
            EvidenceCitation(
                revision_id=UUID(int=50),
                document_id=document or UUID(int=100 + number),
                section_id=UUID(int=60),
                chunk_id=UUID(int=number),
                title="Course manual",
                source_hash="a" * 64,
                page_start=1,
                page_end=1,
                page_spans=[
                    PageSpan(page=1, start=start, end=start + len(content), section_start=0)
                ],
            )
        ],
    )


def test_rank_fusion_respects_identifiers_without_turning_preferences_into_gates():
    config = RetrievalConfig()
    items = [
        candidate(1, current_step_match=True),
        candidate(2, reference_id="FLOW-2", identifier_rank=1),
    ]
    ranked = rank_candidates(items, request(reference_ids=["FLOW-2"]), config)
    assert ranked[0][0].chunk_id == UUID(int=2)
    assert len(ranked) == 2
    weak = SearchCandidate(
        chunk_id=uuid4(),
        vector_similarity=0.1,
        lexical_score=0,
        vector_rank=1,
        phase_match=True,
        current_step_match=True,
        purpose_score=100,
    )
    assert rank_candidates([weak], request(), config) == []
    hyphenated = request().model_copy(update={"query": "business-process guidance"})
    preferred = rank_candidates(
        [candidate(1), candidate(2, current_step_match=True)], hyphenated, config
    )
    assert preferred[0][0].chunk_id == UUID(int=2)


def test_exact_duplicate_quotes_merge_sources_and_keep_unique_relevant_text():
    config = RetrievalConfig()
    req = request(top_k=3)
    candidates = [candidate(1), candidate(2), candidate(3)]
    ranked = rank_candidates(candidates, req, config)
    items = [evidence(1), evidence(2), evidence(3, content="Different useful evidence.")]
    chosen, stats = select_evidence(ranked, items, req, config, RetrievalStats())
    assert len(chosen) == 2 and stats.deduplicated == 1
    assert len(chosen[0].citations) == 2 and len(source_bundles(chosen)) == 3
    assert sum(item_chars(item) for item in chosen) <= req.max_chars


def test_contained_overlap_is_deduplicated_but_distinct_qualification_is_retained():
    req = request(top_k=4)
    doc = uuid4()
    items = [
        evidence(1, document=doc, content="Only after validation."),
        evidence(2, document=doc, content="after validation.", start=5),
        evidence(3, document=doc, content="Except in the explicitly documented case.", start=23),
    ]
    chosen, stats = select_evidence(
        rank_candidates([candidate(1), candidate(2), candidate(3)], req, RetrievalConfig()),
        items,
        req,
        RetrievalConfig(),
        RetrievalStats(),
    )
    assert [item.evidence_id for item in chosen] == [UUID(int=1), UUID(int=3)]
    assert stats.deduplicated == 1


def test_budget_keeps_whole_quotes_and_citations_instead_of_truncating():
    item = evidence(1)
    scored = item.model_copy(update={"score": 0.1, "vector_similarity": 0.8, "lexical_score": 0.0})
    req = request(max_chars=item_chars(scored) - 1)
    chosen, stats = select_evidence(
        [(candidate(1), 0.1)], [item], req, RetrievalConfig(), RetrievalStats()
    )
    assert chosen == [] and stats.budget_omissions == 1


@pytest.mark.parametrize(
    "kwargs", [{"query": "  "}, {"top_k": 21}, {"max_chars": 24001}, {"reference_ids": ["x"] * 21}]
)
def test_invalid_requests_are_rejected_before_retrieval(kwargs):
    with pytest.raises(ValidationError):
        RetrievalRequest(collection_id=uuid4(), **{"query": "valid", **kwargs})


@pytest.mark.asyncio
async def test_unavailable_revision_does_not_call_model_and_has_no_sources():
    repository = AsyncMock(spec=SearchRepository)
    repository.admit.return_value = None
    model = QueryModel()
    result = await KnowledgeRetrievalService(repository, model).search(
        request(), scope=AccessScope(access_domain="test")
    )
    assert result.status == "unavailable" and not result.items and not result.sources
    assert not model.requests and model.inspections == 0


@pytest.mark.asyncio
async def test_empty_candidate_result_is_no_evidence_and_never_an_error():
    req = request()
    revision = KnowledgeRevision(
        collection_id=req.collection_id,
        embedding=PROFILE,
        pipeline=PipelineVersions(extractor="v1", chunker="v1", policy="v1"),
        manifest={},
        version=2,
        indexing_status=ProcessingStatus.READY,
    )
    repository = AsyncMock(spec=SearchRepository)
    repository.admit.return_value = RevisionPin(revision=revision)
    repository.candidates.return_value = []
    repository.evidence.return_value = []
    result = await KnowledgeRetrievalService(repository, QueryModel()).search(
        req, scope=AccessScope(access_domain="test")
    )
    assert result.status == "no_evidence" and result.stats.model_calls == 1
    repository.check_pin.assert_awaited_once()

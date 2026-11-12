"""Release metrics must fail closed on missing observations and unsupported evidence."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.knowledge.evaluation import EvaluationCase, EvaluationSuite, retrieval_score, summarize

SUITE = Path(__file__).resolve().parents[3] / "docs/knowledge/rag-evaluation-v1.json"


def item(reference=None, document=None):
    return SimpleNamespace(
        reference_id=reference,
        citations=[SimpleNamespace(document_id=document)] if document else [],
    )


def test_recall_counts_distinct_gold_only_in_first_four_and_requires_found():
    case = EvaluationCase(
        case_id="cross",
        split="heldout",
        specialist="process_coach",
        question="Why?",
        expected_references=["AP40-11", "AP40-12"],
    )
    results = [item("AP40-11"), item("AP40-11"), item("AP40-99"), item(), item("AP40-12")]
    assert retrieval_score(case, SimpleNamespace(status="found", items=results)) == 0.5
    assert retrieval_score(case, SimpleNamespace(status="error", items=results)) == 0
    with pytest.raises(ValidationError, match="unique"):
        EvaluationCase(**{**case.model_dump(), "expected_references": ["AP40-11"] * 2})


def test_rejection_does_not_treat_provider_failure_or_unrelated_evidence_as_success():
    case = EvaluationCase(
        case_id="no", split="heldout", specialist="scenario", question="Key?", answerable=False
    )
    assert retrieval_score(case, SimpleNamespace(status="no_evidence", items=[])) == 1
    assert retrieval_score(case, SimpleNamespace(status="unavailable", items=[])) == 0
    assert retrieval_score(case, SimpleNamespace(status="found", items=[item()])) == 0


def test_generic_gold_document_identity_is_checked_and_counts_citations():
    suite, digest = EvaluationSuite.read(SUITE)
    assert len(digest) == 64
    case = next(c for c in suite.cases if c.corpus == "generic" and c.answerable)
    assert (
        retrieval_score(
            case, SimpleNamespace(status="found", items=[item(document=suite.generic_document_id)])
        )
        == 1
    )
    assert (
        retrieval_score(case, SimpleNamespace(status="found", items=[item(document=uuid4())])) == 0
    )
    data = suite.model_dump()
    data["generic_document_id"] = uuid4()
    with pytest.raises(ValidationError, match="Generic gold"):
        EvaluationSuite.model_validate(data)


def test_partial_results_never_pass_complete_release_gates_and_duplicates_rejected():
    suite, _ = EvaluationSuite.read(SUITE)
    rows = []
    for case in suite.cases:
        if case.split != "heldout":
            continue
        if case.retrieval:
            rows.append(dict(stage="retrieval", case_id=case.case_id, score=1))
        rows.append(
            dict(
                stage="planner",
                case_id=case.case_id,
                passed=True,
                origin="model",
                has_plan=True,
                latency_ms=100,
                actions=case.allowed_actions[:1],
            )
        )
    result = summarize(suite, rows, split="heldout")
    assert result["retrieval_gate"] and result["rejection_gate"] and result["planner_gate"]
    assert result["release_authorized"] is False
    partial = summarize(suite, rows[:-2], split="heldout")
    assert partial["planner_gate"] is False
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(suite, [*rows, rows[0]], split="heldout")


def test_fallback_plans_cannot_satisfy_valid_model_gate():
    suite, _ = EvaluationSuite.read(SUITE)
    rows = [
        dict(
            stage="planner",
            case_id=c.case_id,
            passed=True,
            origin="fallback",
            has_plan=True,
            latency_ms=100,
            actions=["document_search"],
        )
        for c in suite.cases
        if c.split == "heldout"
    ]
    result = summarize(suite, rows, split="heldout")
    assert result["planner_complete"] is True
    assert result["planner_expected_fraction"] == 1
    assert result["planner_valid_fraction"] == 0
    assert result["planner_gate"] is False
    assert result["unnecessary_searches_on_context_only_cases"] == 3

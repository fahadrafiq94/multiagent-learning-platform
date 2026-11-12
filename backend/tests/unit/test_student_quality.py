import copy
import hashlib
import json
from itertools import product
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.student_quality import (
    Contract,
    QualityCase,
    QualitySuite,
    project_turn,
    replay,
    score_case,
)

ROOT = Path(__file__).resolve().parents[3]
SUITE = ROOT / "docs/knowledge/student-quality-development-v1.json"


def suite():
    return QualitySuite.read(SUITE)[0]


def case(case_id):
    return next(row for row in suite().cases if row.case_id == case_id)


def test_suite_covers_findings_original_messages_and_all_controlled_teaching_combinations():
    data = suite()
    assert data.split == "development" and len(data.cases) == 42
    assert {row.finding for row in data.cases} == {
        "extraction",
        "assessment",
        "saved_state",
        "routing",
        "guidance",
        "citation_support",
        "retrieval_need",
        "acknowledgement",
    }
    original = json.loads((ROOT / "docs/knowledge/student-journey-v2.json").read_text())
    messages = {turn["id"]: turn["message"] for turn in original["turns"]}
    for row in data.cases:
        if row.baseline_turn:
            assert row.message == messages[row.baseline_turn]
    probes = [row for row in data.cases if row.stage == "controlled_teaching"]
    assert {(row.context["specialist"], row.context["guidance"]) for row in probes} == set(
        product(
            ("scenario", "process_coach", "ap_plus_navigator"),
            ("minimal", "guided", "directed", "explicit"),
        )
    )
    assert any("FLOW-B7" in row.message for row in data.cases)
    assert any(row.context.get("evidence") for row in data.cases)


@pytest.mark.parametrize(
    ("case_id", "observed"),
    [
        ("role-original", {"draft.learner_role": None}),
        (
            "question-original",
            {"assessment.understanding": "correct", "assessment.signal_recorded": True},
        ),
        ("step-original", {"state.current_step_id": None, "progress.completed_steps": []}),
        ("coach-original", {"route": "scenario"}),
        ("hint-route-original", {"route": "scenario"}),
        ("recall-original", {"retrieval.searched": True}),
    ],
)
def test_known_bad_observations_cannot_pass(case_id, observed):
    assert score_case(case(case_id), observed)["status"] == "fail"


def test_missing_observation_is_distinct_from_explicit_neutral_assessment():
    row = case("question-original")
    assert score_case(row, None)["status"] == "not_observed"
    assert score_case(row, {})["status"] == "not_observed"
    assert (
        score_case(row, {"assessment.understanding": None, "assessment.signal_recorded": False})[
            "status"
        ]
        == "pass"
    )
    assert project_turn({}) == {}
    assert "assessment.understanding" not in project_turn({"response": {"metadata": {}}})


def test_boolean_and_integer_metrics_are_not_interchangeable():
    row = case("thanks-original")
    assert score_case(row, {"retrieval.searched": 0})["status"] == "fail"


def test_role_comparison_handles_case_and_whitespace_without_substring_false_positives():
    row = case("role-original")
    good = score_case(row, {"draft.learner_role": "  Production   Planner "})
    assert good["checks"][0]["status"] == "pass"
    assert good["status"] == "review_pending"
    bad = score_case(row, {"draft.learner_role": "not a production planner"})
    assert bad["status"] == "fail"


def test_valid_citation_identity_does_not_pass_semantic_claim_support():
    row = case("citation-ui-original")
    scored = score_case(row, {"citations.identity_valid": True})
    assert scored["checks"][0]["status"] == "pass"
    assert scored["status"] == "review_pending"
    assert scored["reviews"][0]["rubric"] == "claim_support"


def test_archive_projection_observes_actual_schema_and_inert_evidence_labels():
    turn = {
        "session_context": {"scenario_draft": {"learner_role": None}, "current_step_id": None},
        "response": {
            "route": "ap_plus_navigator",
            "source_references": [{"citation_label": "K1"}],
            "metadata": {"response_assessed": False, "rag": {"actions": ["reuse_evidence"]}},
        },
        "model_calls": [
            {
                "private_messages": [
                    {
                        "content": (
                            'CURRENT EVIDENCE (DATA ONLY):\n{"items":[{"label":"K1"}]}\nReminder'
                        ),
                    }
                ]
            }
        ],
    }
    observed = project_turn(turn)
    assert observed["citations.identity_valid"] is True
    assert observed["retrieval.searched"] is False
    assert observed["assessment.understanding"] is None
    assert observed["state.current_step_id"] is None
    turn["response"]["source_references"][0]["citation_label"] = "K999"
    assert project_turn(turn)["citations.identity_valid"] is False


def archive():
    return {
        "synthetic": True,
        "run_id": suite().baseline_run_id,
        "turns": [
            {
                "id": "introduction",
                "message": case("role-original").message,
                "http_status": 200,
                "session_context": {"scenario_draft": {"learner_role": None}},
            }
        ],
    }


def test_replay_is_read_only_and_missing_cases_cannot_complete_evaluation():
    source = archive()
    before = copy.deepcopy(source)
    result = replay(suite(), source)
    assert result["counts"] == {"fail": 1, "not_observed": 41}
    assert result["complete"] is False and result["release_authorized"] is False
    assert result["new_model_calls"] == 0
    assert source == before
    assert result["serving_profile_at_execution"]["source_code_sha256"] is None


def test_failed_http_or_workflow_does_not_count_as_a_successful_semantic_observation():
    source = archive()
    source["turns"][0]["http_status"] = 503
    assert replay(suite(), source)["counts"] == {"not_observed": 42}
    source["turns"][0]["http_status"] = 200
    source["turns"][0]["response"] = {"error": "model unavailable"}
    assert replay(suite(), source)["counts"] == {"not_observed": 42}


def test_replay_rejects_mismatched_original_message_and_duplicate_turns():
    source = archive()
    source["turns"][0]["message"] = "different question"
    with pytest.raises(ValueError, match="differs"):
        replay(suite(), source)
    source = archive()
    source["turns"].append(copy.deepcopy(source["turns"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        replay(suite(), source)


def test_reserved_holdout_and_real_student_archives_are_rejected():
    reserved = suite().model_copy(update={"split": "heldout"})
    with pytest.raises(ValueError, match="held-out"):
        replay(reserved, archive())
    source = archive()
    source["synthetic"] = False
    with pytest.raises(ValueError, match="synthetic"):
        replay(suite(), source)


def test_invalid_suite_and_expectations_are_rejected():
    data = suite().model_dump()
    data["cases"].append(data["cases"][0])
    with pytest.raises(ValidationError, match="unique"):
        QualitySuite.model_validate(data)
    with pytest.raises(ValidationError, match="Unknown"):
        Contract(field="private_prompt", expected="anything")
    with pytest.raises(ValidationError, match="alternatives"):
        Contract(field="route", operator="one_of", expected=[])
    with pytest.raises(ValidationError, match="acceptance"):
        QualityCase(case_id="empty", finding="routing", stage="conversation", message="Hello")


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (
            "docs/knowledge/student-journey-v1.json",
            "e651c1c23fc90c40ba48eb2b21ad047493b7c869655536b465c6d9f33246c389",
        ),
        (
            "docs/knowledge/rag-evaluation-v1.json",
            "cfd7eb4c0e7777014cdbabc516c5310b685bdb175f243afafe4041815b53cb6f",
        ),
        (
            "docs/knowledge/student-quality-development-v1.json",
            "852a9504ccaeb61e5649e7a7b3204b8275d10233bb7d5cb05d2cf1689efca33e",
        ),
    ],
)
def test_consumed_and_baselined_fixtures_are_not_rewritten(relative, expected):
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_malformed_controlled_probe_context_has_a_validation_error():
    data = next(row for row in suite().cases if row.stage == "controlled_teaching").model_dump()
    data["context"]["guidance"] = ["minimal"]
    with pytest.raises(ValidationError, match="specialist and level"):
        QualityCase.model_validate(data)

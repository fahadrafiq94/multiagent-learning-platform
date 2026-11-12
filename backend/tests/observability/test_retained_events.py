import json
from uuid import uuid4

from app.observability.events import build_request_event


def test_retained_event_correlates_rag_without_retaining_model_prose():
    request_id = uuid4()
    event = build_request_event(
        {
            "route_reason": "PRIVATE STUDENT QUOTE",
            "error": "PRIVATE PROVIDER ERROR",
            "metadata": {
                "request_id": request_id,
                "summary_update_reason": "PRIVATE SUMMARY REASON",
                "rag": {
                    "status": "found",
                    "actions": ["metadata_lookup", "document_search"],
                    "planner_calls": 1,
                    "planner_latency_ms": 12.3,
                    "embedding_calls": 1,
                    "provided_count": 2,
                    "reason": "PRIVATE PLANNER QUOTE",
                    "query": "PRIVATE QUERY",
                },
            },
        },
        duration_ms=45,
    )
    payload = event.log_payload()
    assert payload["schema_version"] == 2
    assert payload["recorded_at"].endswith("Z")
    assert payload["request_id"] == str(request_id)
    assert payload["rag_status"] == "found"
    assert payload["rag_actions"] == ["metadata_lookup", "document_search"]
    assert payload["rag_provided_count"] == 2
    assert payload["error_present"] is True
    assert "PRIVATE" not in json.dumps(payload)


def test_unknown_retrieval_fields_cannot_export_free_text():
    event = build_request_event(
        {
            "metadata": {
                "rag": {
                    "status": "PRIVATE STATUS",
                    "actions": ["PRIVATE ACTION"],
                    "provided_count": True,
                }
            }
        },
        duration_ms=1,
    )
    assert event.rag_status is None
    assert event.rag_actions == []
    assert event.rag_provided_count is None


def test_citation_support_exports_only_safe_verdict_counts_and_duration():
    event = build_request_event(
        {
            "metadata": {
                "rag": {
                    "status": "found",
                    "support_status": "unsupported",
                    "support_calls": 1,
                    "support_latency_ms": 19.5,
                    "support_claim_count": 2,
                    "support_abstained": True,
                    "support_link_mode": "whole_answer",
                    "claims": [{"text": "PRIVATE ANSWER"}],
                    "quotes": ["PRIVATE QUOTATION"],
                    "support_reason": "PRIVATE DIAGNOSIS",
                }
            }
        },
        duration_ms=30,
    )
    payload = event.log_payload()
    assert payload["citation_support_status"] == "unsupported"
    assert payload["citation_support_calls"] == 1
    assert payload["citation_support_latency_ms"] == 19.5
    assert payload["citation_support_claim_count"] == 2
    assert payload["citation_support_abstained"] is True
    assert payload["citation_support_link_mode"] == "whole_answer"
    assert "PRIVATE" not in json.dumps(payload)
    invalid = build_request_event(
        {
            "metadata": {
                "rag": {
                    "support_status": "PRIVATE STATUS",
                    "support_calls": True,
                    "support_abstained": "PRIVATE FLAG",
                    "support_claim_count": "PRIVATE COUNT",
                    "support_link_mode": "PRIVATE MODE",
                }
            }
        },
        duration_ms=1,
    )
    assert invalid.citation_support_status is None
    assert invalid.citation_support_calls is None
    assert invalid.citation_support_abstained is None
    assert invalid.citation_support_claim_count is None
    assert invalid.citation_support_link_mode is None


def test_event_uses_final_grounded_budget_and_recognizes_onboarding_route():
    event = build_request_event(
        {
            "onboarding_required": True,
            "route": "scenario",
            "metadata": {
                "estimated_input_tokens": 4567,
                "input_token_budget": 7168,
                "recent_message_count": 2,
                "dropped_message_count": 8,
                "truncation_applied": True,
            },
        },
        duration_ms=1,
    )
    assert event.onboarding_handled is True
    assert event.estimated_input_tokens == 4567
    assert event.input_token_budget == 7168
    assert event.dropped_message_count == 8
    assert event.truncation_applied is True


def test_malformed_rag_choices_degrade_to_missing_metrics():
    event = build_request_event(
        {
            "metadata": {
                "rag": {
                    "status": ["invalid"],
                    "actions": [{"unexpected": "data"}],
                    "origin": {"unexpected": "data"},
                }
            }
        },
        duration_ms=1,
    )
    assert event.rag_status is None and event.rag_actions == [] and event.rag_origin is None


def test_student_step_and_role_status_export_has_no_source_quote():
    event = build_request_event(
        {
            "metadata": {
                "student_step_status": "persisted",
                "scenario_role_status": "persisted",
                "student_step_source_quote": "PRIVATE STUDENT TEXT",
            }
        },
        duration_ms=1,
    )
    payload = event.log_payload()
    assert payload["student_step_status"] == "persisted"
    assert payload["scenario_role_status"] == "persisted"
    assert "PRIVATE" not in json.dumps(payload)

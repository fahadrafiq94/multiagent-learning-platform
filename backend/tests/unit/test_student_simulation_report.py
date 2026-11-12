import pytest

from scripts.run_student_simulation import technical_checks
from scripts.student_simulation_report import render_report


def test_presentation_escapes_model_html_and_omits_private_diagnostics(tmp_path):
    run = {
        "synthetic": True,
        "run_id": "example",
        "started_at": "2026-11-06",
        "model": "real",
        "planner_model": "real",
        "user_id": "PRIVATE UUID",
        "turns": [
            {
                "id": "example",
                "message": "<script>alert(1)</script>",
                "expected": "Review",
                "http_status": 200,
                "duration_ms": 1000,
                "response": {"final_response": '<img src=x onerror="alert(2)">', "metadata": {}},
                "model_calls": [{"private_model_output": "PRIVATE OUTPUT"}],
            }
        ],
    }
    render_report(run, tmp_path)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html
    assert "PRIVATE" not in html
    assert "Human review" in html
    assert (tmp_path / "metrics.csv").exists()


def test_real_student_records_cannot_be_exported_with_synthetic_report(tmp_path):
    with pytest.raises(ValueError, match="synthetic"):
        render_report({"synthetic": False}, tmp_path)


def test_http_success_alone_cannot_pass_simulation_checks():
    turn = {"id": "example", "http_status": 200, "response": {}, "request_id": "id"}
    checks = technical_checks(turn, 1, [])
    assert checks["http_ok"]
    assert not checks["assistant_persisted"]
    assert not checks["event_request_correlated"]
    assert not checks["history_pair_added"]
    assert not checks["workflow_ok"]
    assert not checks["citations_persisted"]


def test_failed_request_with_only_user_history_cannot_pass_workflow_or_citations():
    turn = {
        "id": "failed",
        "http_status": 503,
        "response": {"detail": "Service unavailable"},
        "request_id": "id",
        "history": [{"role": "user", "content": "Question", "source_references": []}],
    }
    checks = technical_checks(turn, 1, [])
    assert not checks["workflow_ok"]
    assert not checks["citations_persisted"]


def test_valid_empty_citations_are_verified_against_the_persisted_assistant():
    turn = {
        "id": "valid",
        "http_status": 200,
        "request_id": "id",
        "response": {"final_response": "Thanks!", "source_references": [], "metadata": {}},
        "history": [
            {"role": "user", "content": "Thank you"},
            {"role": "assistant", "content": "Thanks!", "source_references": []},
        ],
    }
    checks = technical_checks(turn, 1, [])
    assert checks["workflow_ok"] and checks["citations_persisted"]


def _confirmation_turn(role, company):
    return {
        "id": "confirmation",
        "http_status": 200,
        "response": {},
        "request_id": "id",
        "onboarding_status": "confirmed",
        "user_context": {"scenario_card": {"learner_role": role, "company_context": company}},
    }


def test_confirmation_check_uses_fixture_declared_values():
    turn = _confirmation_turn("Receiving Supervisor", "Bridgemark Foods")
    checks = technical_checks(
        turn, 1, [], {"learner_role": "receiving supervisor", "company_context": "bridgemark"}
    )
    assert checks["onboarding_confirmed"]
    assert checks["corrected_role_saved"]
    assert checks["company_saved"]


def test_confirmation_check_defaults_preserve_original_v1_journey():
    turn = _confirmation_turn("Purchasing Coordinator", "UrbanCycle GmbH")
    checks = technical_checks(turn, 1, [])
    assert checks["corrected_role_saved"]
    assert checks["company_saved"]

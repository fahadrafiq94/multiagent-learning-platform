import argparse
import json

import pytest

from app.services.model_service.schemas import ModelChatResponse
from scripts import evaluate_student_teaching as adapter
from scripts.student_quality import QualitySuite
from tests.agents.support_fixtures import support_reply


@pytest.mark.asyncio
async def test_probe_runs_actual_nodes_and_counter_guard_without_grading_from_expected_labels(
    tmp_path, monkeypatch
):
    requests = []

    class Model:
        async def generate(self, req):
            requests.append(req)
            if req.metadata["task"] == "citation_support":
                return support_reply(req)
            return ModelChatResponse(
                content=json.dumps(
                    {
                        "content": "Which aspect of your goal needs clarification?",
                        "citation_ids": [],
                        "claims": [],
                        "response_assessment": {
                            "assessability": "not_assessable",
                            "student_claim": None,
                            "understanding": "not_applicable",
                        },
                    }
                ),
                model="controlled",
                provider="ollama",
                latency_ms=0,
            )

    monkeypatch.setattr(adapter, "create_model_service", lambda settings: Model())
    monkeypatch.setattr(adapter, "private_directory", lambda path: path.mkdir())
    args = argparse.Namespace(
        output=tmp_path / "new-probe",
        case_id=["question-original"],
        model="controlled",
        model_url="http://unused",
    )
    result = await adapter.execute(args)
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["before"] == row["after"]
    assert row["observed"]["assessment.signal_recorded"] is False
    assert row["calls"] == 2 and row["score"]["status"] == "pass"
    assert [req.metadata["task"] for req in requests] == [
        "scenario_agent_response",
        "citation_support",
    ]
    assert "checks" not in requests[0].messages[0].content
    assert "reviews" not in requests[0].messages[0].content


@pytest.mark.asyncio
async def test_probe_rejects_heldout_before_creating_output_or_models(tmp_path, monkeypatch):
    suite, digest = QualitySuite.read(
        adapter.ROOT / "docs/knowledge/student-quality-development-v1.json"
    )
    monkeypatch.setattr(
        adapter.QualitySuite,
        "read",
        lambda path: (suite.model_copy(update={"split": "heldout"}), digest),
    )
    args = argparse.Namespace(
        output=tmp_path / "never", case_id=None, model="unused", model_url="unused"
    )
    with pytest.raises(ValueError, match="Development"):
        await adapter.execute(args)
    assert not args.output.exists()

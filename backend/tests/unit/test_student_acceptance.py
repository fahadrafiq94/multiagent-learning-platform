import hashlib
import json
from typing import cast

import pytest

from app.services.model_service import ModelService
from app.services.model_service.schemas import ModelChatRequest, ModelChatResponse, ModelMessage
from scripts import evaluate_student_acceptance as adapter
from scripts.student_quality import QualityCase
from tests.agents.support_fixtures import support_reply


def test_frozen_candidate_check_rejects_changed_code(tmp_path, monkeypatch):
    path = tmp_path / "candidate.py"
    path.write_text("original")
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    snapshot = {"files": {"candidate.py": hashlib.sha256(path.read_bytes()).hexdigest()}}
    adapter.verify_snapshot(snapshot)
    path.write_text("changed")
    with pytest.raises(ValueError, match="candidate.py"):
        adapter.verify_snapshot(snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 512])
async def test_acceptance_profile_applies_to_uncontrolled_calls_without_expanding_verifier(
    limit, tmp_path
):
    seen = []

    class Model:
        async def generate(self, request):
            seen.append(request)
            return ModelChatResponse(
                content="{}", provider="ollama", model="controlled", latency_ms=0
            )

    model = adapter.AcceptanceModel(cast(ModelService, Model()), tmp_path / "calls.jsonl")
    original = ModelChatRequest(
        messages=[ModelMessage(role="user", content="Extract this role.")], max_tokens=limit
    )
    await model.generate(original)
    assert seen[0].context_window_tokens == 8192
    assert seen[0].reasoning_enabled is False
    assert seen[0].max_tokens == (512 if limit == 512 else 1024)
    assert original.reasoning_enabled is None and original.context_window_tokens is None
    archived = json.loads((tmp_path / "calls.jsonl").read_text())
    assert archived["request"]["reasoning_enabled"] is False


@pytest.mark.parametrize(
    "status,regression,expected",
    [
        ("reserved_unexecuted", False, "heldout"),
        ("consumed", True, "consumed_regression"),
        ("consumed", False, None),
        ("reserved_unexecuted", True, None),
    ],
)
def test_consumed_regression_cannot_be_presented_as_fresh_heldout(status, regression, expected):
    if expected is None:
        with pytest.raises(ValueError):
            adapter.execution_split(status, consumed_regression=regression)
    else:
        assert adapter.execution_split(status, consumed_regression=regression) == expected


@pytest.mark.asyncio
async def test_acceptance_observes_real_signal_node_without_passing_gold_to_model():
    requests = []

    class Model:
        async def generate(self, request):
            requests.append(request)
            if request.metadata.get("task") == "citation_support":
                return support_reply(request)
            return ModelChatResponse(
                provider="ollama",
                model="controlled",
                latency_ms=0,
                content=json.dumps(
                    {
                        "content": {"question": "Which goal would you like to revisit?"},
                        "response_assessment": {
                            "student_claim": None,
                            "assessability": "not_assessable",
                            "understanding": "not_applicable",
                        },
                        "citation_ids": [],
                        "claims": [],
                    }
                ),
            )

    case = QualityCase.model_validate(
        {
            "case_id": "synthetic-neutral",
            "finding": "assessment",
            "stage": "conversation",
            "message": "What is our goal?",
            "checks": [{"field": "assessment.understanding", "expected": "GOLD-LABEL-ONLY"}],
        }
    )
    row = await adapter.observe(case, cast(ModelService, Model()))
    assert row["observed"]["assessment.understanding"] == "not_applicable"
    assert row["observed"]["assessment.signal_recorded"] is False
    assert row["before"] == row["after"]
    assert len(requests) == 2
    assert all("GOLD-LABEL-ONLY" not in req.model_dump_json() for req in requests)

"""Probe instrumentation must observe real in-memory persistence, not hide API errors."""

import argparse
import json
from unittest.mock import AsyncMock

import pytest

from app.services.model_service.schemas import ModelChatResponse
from scripts import evaluate_student_interactions as adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["mixed", "recall"])
async def test_adapter_observes_step_and_reply_through_actual_nodes(tmp_path, monkeypatch, kind):
    message = "I am now doing FLOW-Z. Why is approval required?"
    step = {"action": "set", "step_id": "FLOW-Z", "source_quote": "I am now doing FLOW-Z."}
    interaction = {"kind": "mixed", "target": None, "source_quote": None}
    route = "process_coach"
    checks = [
        {"field": "route", "expected": route},
        {"field": "state.current_step_id", "expected": "FLOW-Z"},
    ]
    if kind == "recall":
        message = "Which step am I doing?"
        step = {"action": "unchanged", "step_id": None, "source_quote": None}
        interaction = {"kind": "recall", "target": "current_step_id", "source_quote": None}
        route = "fallback"
        checks = [
            {"field": "state.current_step_id", "expected": "FLOW-Z"},
            {"field": "retrieval.searched", "expected": False},
        ]
    suite = tmp_path / "development.json"
    suite.write_text(
        json.dumps(
            {
                "version": "controlled-adapter-test",
                "split": "development",
                "prepared_on": "2026-11-09",
                "baseline_run_id": "synthetic-adapter-test",
                "cases": [
                    {
                        "case_id": "probe",
                        "finding": "routing",
                        "stage": "conversation",
                        "message": message,
                        "context": {"current_step_id": "FLOW-Z" if kind == "recall" else "FLOW-X"},
                        "checks": checks,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "archive.json"
    archive.write_text(
        json.dumps({"synthetic": True, "run_id": "synthetic-adapter-test", "turns": []})
    )
    model = AsyncMock()
    model.generate.return_value = ModelChatResponse(
        content=json.dumps(
            {
                "route": route,
                "reason": "Controlled adapter contract.",
                "current_step": step,
                "interaction": interaction,
            }
        ),
        model="controlled",
        provider="ollama",
        latency_ms=0,
    )
    monkeypatch.setattr(adapter, "create_model_service", lambda profile: model)
    # ACL protection has its own operator tests; this unit test uses pytest's scratch directory.
    monkeypatch.setattr(adapter, "private_directory", lambda path: path.mkdir())
    result = await adapter.execute(
        argparse.Namespace(
            suite=suite,
            archive=archive,
            output=tmp_path / "result",
            case_id=None,
            model="controlled",
            model_url="http://127.0.0.1:1",
        )
    )
    row = result["rows"][0]
    assert "error_type" not in row
    assert row["score"]["status"] == "pass"
    assert row["saved_step"] == "FLOW-Z"
    assert model.generate.await_count == 1
    assert row["student_step_status"] == ("persisted" if kind == "mixed" else "unchanged")
    if kind == "recall":
        assert "FLOW-Z" in row["reply"]
    assert (tmp_path / "result/model-calls.jsonl").exists()

"""Diagnostic telemetry must preserve errors/cancellation and keep content private."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.model_service.exceptions import ModelTimeoutError
from app.services.model_service.schemas import ModelChatRequest, ModelChatResponse, ModelMessage
from scripts.evaluate_rag import ProfileModel, select_cases


def request():
    return ModelChatRequest(
        messages=[ModelMessage(role="user", content="private student input")],
        max_tokens=321,
        metadata={"task": "synthetic"},
    )


def response(**kwargs):
    return ModelChatResponse(
        content="private model answer",
        model="installed-model",
        provider="ollama",
        latency_ms=20,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_timings_count_output_and_dont_copy_private_provider_fields():
    underlying = SimpleNamespace(
        generate=AsyncMock(
            return_value=response(
                raw={
                    "total_duration": 5_000_000_000,
                    "load_duration": 1_000_000_000,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_duration": 2_000_000_000,
                    "eval_count": 10,
                    "prompt_eval_count": 300,
                    "done_reason": "stop",
                    "message": {"thinking": "private thinking"},
                    "password": "private password",
                }
            )
        )
    )
    model = ProfileModel(underlying)
    model.scope = {"case_id": "development-example"}
    original = request()
    result = await model.generate(original)
    call = model.calls[0]
    assert call["load_duration_ms"] == 1000
    assert call["prompt_eval_duration_ms"] == 2000
    assert call["output_tokens_per_second"] == 5
    assert call["done_reason"] == "stop"
    assert call["case_id"] == "development-example"
    assert "private" not in json.dumps(model.calls)
    assert model.responses == [result]
    forwarded = underlying.generate.await_args.args[0]
    assert forwarded.max_tokens == original.max_tokens
    assert forwarded.messages == original.messages
    assert forwarded.context_window_tokens == 8192 and forwarded.reasoning_enabled is False
    assert original.context_window_tokens is None  # Caller request remains untouched.


@pytest.mark.asyncio
async def test_diagnostic_output_is_explicit_bounded_and_rejects_bad_metrics():
    underlying = SimpleNamespace(
        generate=AsyncMock(
            return_value=response(
                raw={
                    "eval_duration": float("nan"),
                    "load_duration": -1,
                    "eval_count": True,
                    "prompt_eval_count": "wrong",
                    "done_reason": "private",
                }
            )
        )
    )
    underlying.generate.return_value.content = "x" * 20000
    model = ProfileModel(underlying, capture_output=True)
    await model.generate(request())
    call = model.calls[0]
    assert len(call["private_model_output"]) == 16000 and call["output_truncated"] is True
    assert not {
        "eval_duration_ms",
        "load_duration_ms",
        "eval_count",
        "prompt_eval_count",
        "done_reason",
        "output_tokens_per_second",
    }.intersection(call)


@pytest.mark.asyncio
async def test_timeout_preserves_exception_without_copying_sensitive_details():
    error = ModelTimeoutError("ollama", details={"request": "private credential"})
    model = ProfileModel(SimpleNamespace(generate=AsyncMock(side_effect=error)))
    with pytest.raises(ModelTimeoutError) as caught:
        await model.generate(request())
    assert caught.value is error
    assert model.calls[0]["outcome"] == "failed"
    assert model.calls[0]["failure_code"] == "model_timeout"
    assert "private" not in json.dumps(model.calls) and model.responses == []


@pytest.mark.asyncio
async def test_outer_planner_deadline_cancellation_remains_visible_and_propagates():
    async def pending(_):
        await asyncio.Future()

    model = ProfileModel(SimpleNamespace(generate=pending))
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await model.generate(request())
    assert model.calls[0]["outcome"] == "cancelled"
    assert model.calls[0]["failure_type"] == "CancelledError"
    assert model.responses == []


def test_subset_rejects_wrong_split_and_extra_holdout_probes():
    dev = SimpleNamespace(case_id="development", split="development")
    held = SimpleNamespace(case_id="held", split="heldout")
    suite = SimpleNamespace(cases=[dev, held])
    args = SimpleNamespace(
        split="development", case_id=["development"], probe_teaching=True, guidance=["minimal"]
    )
    assert select_cases(suite, args) == [dev]
    args.case_id = ["held"]
    with pytest.raises(ValueError, match="selected split"):
        select_cases(suite, args)
    args.split = "heldout"
    with pytest.raises(ValueError, match="development-only"):
        select_cases(suite, args)
    args.split, args.case_id, args.guidance = "development", None, ["minimal"] * 2
    with pytest.raises(ValueError, match="unique"):
        select_cases(suite, args)

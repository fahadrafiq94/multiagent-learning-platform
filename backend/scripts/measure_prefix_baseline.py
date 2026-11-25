"""Measure sequential vLLM prefix reuse with synthetic messages from the backend host."""

import argparse
import json
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from app.agents.scenario import SCENARIO_AGENT_SYSTEM_PROMPT


def counters(client, model):
    response = client.get("/metrics")
    response.raise_for_status()
    result = {}
    for name in ("prefix_cache_queries_total", "prefix_cache_hits_total", "request_success_total"):
        values = re.findall(
            rf'^vllm:{name}\{{[^}}]*model_name="{re.escape(model)}"[^}}]*\}} ([\d.eE+-]+)$',
            response.text,
            re.MULTILINE,
        )
        if values:
            result[name] = int(sum(map(float, values)))
    return result


def measure(client, model, system, question):
    start = time.perf_counter()
    first = None
    usage = {}
    finished = False
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                finished = True
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError("Inference server returned a stream error.")
            for choice in event.get("choices", []):
                if first is None and choice["delta"].get("content"):
                    first = time.perf_counter()
            usage = event.get("usage") or usage
    elapsed = time.perf_counter() - start
    if first is None or not finished:
        raise RuntimeError("Incomplete model response.")
    return {
        "first_text_ms": round((first - start) * 1000),
        "total_ms": round(elapsed * 1000),
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "reported_cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Server root, without /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--long-prefix-control",
        action="store_true",
        help="Include a long, shared synthetic lesson for a cache control run",
    )
    args = parser.parse_args()
    now = datetime.now(UTC)
    rows = []
    with httpx.Client(base_url=args.url, timeout=90, trust_env=False) as client:
        before = counters(client, args.model)
        for trial in range(1, 4):
            # A fresh shared prefix per trial avoids flushing the running server's cache.
            system = f"Exercise {uuid4().hex}\n{SCENARIO_AGENT_SYSTEM_PROMPT}"
            if args.long_prefix_control:
                system += "\nSynthetic lesson examples:\n" + "\n".join(
                    f"Example {i}: a fictional shop compares planned demand with available stock."
                    for i in range(180)
                )
            for case, company in (
                ("first", "bakery"),
                ("exact_repeat", "bakery"),
                ("changed_question", "bookshop"),
            ):
                question = f"Suggest one business goal for a {company} in one short sentence."
                row = measure(client, args.model, system, question)
                rows.append({"trial": trial, "case": case, **row})
        # Some installations publish counters periodically.
        for _ in range(12):
            after = counters(client, args.model)
            successes = after.get("request_success_total", 0) - before.get(
                "request_success_total", 0
            )
            if successes >= len(rows):
                break
            time.sleep(1)
    medians = {
        case: {
            metric: statistics.median(row[metric] for row in rows if row["case"] == case)
            for metric in ("first_text_ms", "total_ms")
        }
        for case in ("first", "exact_repeat", "changed_question")
    }
    report = {
        "utc": now.isoformat(),
        "model": args.model,
        "url": args.url,
        "workload": "Single client, three trials, Scenario Agent policy and synthetic questions",
        "medians_ms": medians,
        "long_prefix_control": args.long_prefix_control,
        "requests": rows,
        "server_before": before,
        "server_after": after,
        "server_delta": {k: after[k] - v for k, v in before.items() if k in after},
    }
    directory = Path(__file__).resolve().parents[1] / "reports"
    directory.mkdir(exist_ok=True)
    path = directory / f"prefix-baseline-{now:%Y%m%dT%H%M%S%fZ}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps(medians))
    print(json.dumps(report["server_delta"]))


if __name__ == "__main__":
    main()

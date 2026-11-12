"""Offline presentation export for explicitly synthetic journeys only."""

import csv
import html
import json
from pathlib import Path


def render_report(run: dict, output: Path) -> None:
    """No raw prompts, real identifiers, endpoints, or credentials in shared exports."""
    if run.get("synthetic") is not True:
        raise ValueError("Presentation export requires an explicitly synthetic run.")
    output.mkdir(parents=True, exist_ok=True)
    escape = lambda value: html.escape(str(value), quote=True)  # noqa: E731
    cards, rows = [], []
    lines = [
        "# FREDi synthetic student journey",
        "",
        f"Run: {run['run_id']} | Started: {run['started_at']}",
        "",
        "Real local models, signed ASGI HTTP routes and PostgreSQL. Synthetic student.",
        "Development evidence only: teaching/source fidelity require human review.",
        "",
        f"Status: {run.get('status', 'running')}",
        "",
        f"Model: {run['model']} | Planner: {run['planner_model']}",
        "",
    ]
    for index, turn in enumerate(run.get("turns", []), 1):
        data = turn.get("response", {})
        meta = data.get("metadata", {})
        rag = meta.get("rag") or {}
        metrics = {
            "turn": index,
            "case": turn["id"],
            "http_status": turn["http_status"],
            "seconds": round(turn["duration_ms"] / 1000, 2),
            "route": data.get("route"),
            "guidance_used": meta.get("agent_metadata", {}).get("guidance_level"),
            "guidance_saved": turn.get("guidance"),
            "current_step_saved": turn.get("session_context", {}).get("current_step_id"),
            "rag": rag.get("status"),
            "actions": ", ".join(rag.get("actions", [])),
            "sources": len(data.get("source_references", [])),
            "input_tokens_estimated": meta.get("estimated_input_tokens"),
            "input_budget": meta.get("input_token_budget"),
            "model_calls": len(turn.get("model_calls", [])),
            "onboarding": turn.get("onboarding_status"),
            "assessment": meta.get("response_understanding"),
        }
        rows.append(metrics)
        answer = data.get("final_response") or data.get("detail") or "No response."
        facts = " | ".join(f"{key}: {value}" for key, value in metrics.items())
        source_rows = [
            {
                key: source.get(key)
                for key in (
                    "citation_label",
                    "title",
                    "page_start",
                    "page_end",
                    "reference_id",
                    "field_name",
                    "kind",
                )
            }
            for source in data.get("source_references", [])
        ]
        call_rows = [
            {
                key: call.get(key)
                for key in (
                    "task",
                    "model",
                    "wall_ms",
                    "load_duration_ms",
                    "prompt_eval_count",
                    "eval_count",
                    "output_tokens_per_second",
                    "outcome",
                    "failure_code",
                )
            }
            for call in turn.get("model_calls", [])
        ]
        cards.append(
            f'<article id="turn-{index}"><h2>Turn {index}: {escape(turn["id"])}</h2>'
            f'<p class="metrics">{escape(facts)}</p>'
            f"<h3>Student</h3><pre>{escape(turn['message'])}</pre>"
            f"<h3>FREDi</h3><pre>{escape(answer)}</pre>"
            f"<p><b>Review expectation:</b> {escape(turn['expected'])}</p>"
            f"<details><summary>Sources and technical checks</summary>"
            f"<pre>{escape(json.dumps(source_rows, indent=2))}</pre>"
            f"<pre>{escape(json.dumps(turn.get('checks', {}), indent=2))}</pre>"
            f"<h3>Chat model timings and tokens</h3>"
            f"<pre>{escape(json.dumps(call_rows, indent=2))}</pre>"
            "</details></article>"
        )
        lines += [
            f"## Turn {index}: {turn['id']}",
            "",
            facts,
            "",
            f"Student: {html.escape(turn['message'], quote=False)}",
            "",
            f"FREDi: {html.escape(str(answer), quote=False)}",
            "",
            f"Review expectation: {turn['expected']}",
            "",
            f"Technical checks: {json.dumps(turn.get('checks', {}))}",
            "",
        ]
    checks = run.get("checks", {})
    durations = [turn["duration_ms"] / 1000 for turn in run.get("turns", [])]
    maximum = max(durations, default=1) or 1
    timeline = "".join(
        f'<div class="bar-row"><span>Turn {index}</span>'
        f'<meter min="0" max="{maximum:.3f}" value="{duration:.3f}"></meter>'
        f"<span>{duration:.1f} s</span></div>"
        for index, duration in enumerate(durations, 1)
    )
    lines += [
        "## Run checks",
        "",
        json.dumps(checks, indent=2),
        "",
        "Passing transport/storage checks does not certify teaching accuracy.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>FREDi synthetic student journey</title><style>"
        "body{font:17px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;"
        "background:#f5f7fa;color:#18263c}article{background:white;border-radius:12px;"
        "padding:24px;margin:24px 0;border:1px solid #ccd7e5}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.5}"
        ".metrics{font-size:14px;color:#425a75}h1,h2{color:#133e64}</style>"
        "<h1>FREDi synthetic student journey</h1>"
        f"<p>{escape(run['run_id'])} | {escape(run['started_at'])}</p>"
        "<p>Real local models · signed ASGI HTTP routes · PostgreSQL · synthetic student</p>"
        f"<p>Status: <b>{escape(run.get('status', 'running'))}</b></p>"
        f"<p>Teacher/router: {escape(run['model'])} · Planner: "
        f"{escape(run['planner_model'])}</p>"
        "<p><b>Development simulation. Human review of teaching and source fidelity is "
        "still required. This is not release acceptance or a held-out evaluation.</b></p>"
        f"<p>{len(rows)} recorded turns · {sum(durations):.1f} seconds in student requests. "
        "The timings include multiple sequential model calls and are observations "
        "from this development laptop, not a controlled benchmark.</p>"
        "<h2>Request duration</h2><style>.bar-row{display:flex;gap:12px;"
        "align-items:center;margin:6px 0}.bar-row span{min-width:70px}"
        "meter{width:65%;height:24px}</style>"
        + timeline
        + "".join(cards)
        + "<h2>Run checks</h2><pre>"
        + escape(json.dumps(checks, indent=2))
        + "</pre></html>",
        encoding="utf-8",
    )
    if rows:
        with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

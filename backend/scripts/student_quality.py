"""Development-only quality contracts and read-only replay of a synthetic archive.

No model requests, database writes or application policy changes occur here.
Semantic reviews remain pending until a reviewer evaluates the declared rubric.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from scripts.knowledge_backup import private_directory
from scripts.knowledge_infra import ROOT

Finding = Literal[
    "extraction",
    "assessment",
    "saved_state",
    "routing",
    "guidance",
    "citation_support",
    "retrieval_need",
    "acknowledgement",
]
Status = Literal["pass", "fail", "not_observed", "review_pending"]
FIELDS = {
    "draft.learner_role",
    "state.current_step_id",
    "route",
    "assessment.understanding",
    "assessment.signal_recorded",
    "retrieval.searched",
    "guidance.used",
    "citations.identity_valid",
    "progress.completed_steps",
}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    operator: Literal["equals", "one_of", "text_equals"] = "equals"
    expected: JsonValue

    @model_validator(mode="after")
    def valid(self):
        if self.field not in FIELDS:
            raise ValueError("Unknown observable field.")
        if self.operator == "one_of" and (not isinstance(self.expected, list) or not self.expected):
            raise ValueError("one_of requires nonempty alternatives.")
        if self.operator == "text_equals" and not isinstance(self.expected, str):
            raise ValueError("text_equals requires text.")
        return self


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rubric: Literal[
        "fact_fidelity",
        "state_intent",
        "routing",
        "teaching_style",
        "claim_support",
        "acknowledgement",
        "retrieval_need",
    ]
    expected: str = Field(min_length=1)


class QualityCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1, max_length=100)
    finding: Finding
    stage: Literal["onboarding", "conversation", "controlled_teaching"]
    message: str = Field(min_length=1)
    context: dict[str, JsonValue] = Field(default_factory=dict)
    baseline_turn: str | None = None
    checks: list[Contract] = Field(default_factory=list)
    reviews: list[Review] = Field(default_factory=list)

    @model_validator(mode="after")
    def explicit_expectations(self):
        if not self.checks and not self.reviews:
            raise ValueError("Each case needs an explicit acceptance condition.")
        level, specialist = self.context.get("guidance"), self.context.get("specialist")
        valid_probe = (
            isinstance(level, str)
            and level in {"minimal", "guided", "directed", "explicit"}
            and isinstance(specialist, str)
            and specialist in {"scenario", "process_coach", "ap_plus_navigator"}
        )
        if self.stage == "controlled_teaching" and not valid_probe:
            raise ValueError("Controlled teaching needs an explicit specialist and level.")
        return self


class QualitySuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    split: Literal["development", "heldout"]
    prepared_on: str
    baseline_run_id: str | None = None
    cases: list[QualityCase] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def unique_cases(self):
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Case IDs must be unique.")
        return self

    @classmethod
    def read(cls, path: Path):
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:
            raise ValueError("Quality suite is too large.")
        return cls.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


def _equal(actual: Any, expected: Any) -> bool:
    # Python considers True == 1; metrics and labels must not.
    if type(actual) is not type(expected):
        return False
    return bool(actual == expected)


def score_case(case: QualityCase, observed: dict[str, Any] | None) -> dict[str, Any]:
    checks = []
    for check in case.checks:
        if observed is None or check.field not in observed:
            status = "not_observed"
        else:
            actual = observed[check.field]
            if check.operator == "text_equals":
                passed = isinstance(actual, str) and (
                    " ".join(actual.casefold().split())
                    == " ".join(str(check.expected).casefold().split())
                )
            elif check.operator == "one_of":
                options = check.expected if isinstance(check.expected, list) else []
                passed = any(_equal(actual, option) for option in options)
            else:
                passed = _equal(actual, check.expected)
            status = "pass" if passed else "fail"
        checks.append(
            {
                **check.model_dump(),
                "status": status,
                "observed": observed.get(check.field) if observed else None,
            }
        )
    reviews = [{**review.model_dump(), "status": "review_pending"} for review in case.reviews]
    statuses = [row["status"] for row in checks]
    overall: Status = (
        "not_observed"
        if observed is None
        else "fail"
        if "fail" in statuses
        else "not_observed"
        if "not_observed" in statuses
        else "review_pending"
        if reviews
        else "pass"
    )
    return {
        "case_id": case.case_id,
        "finding": case.finding,
        "status": overall,
        "checks": checks,
        "reviews": reviews,
    }


def project_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Observe only explicit archive fields. Omitted values remain unobserved."""
    result = {}
    response = turn.get("response", {})
    metadata = response.get("metadata", {})
    session = turn.get("session_context", {})
    draft = session.get("scenario_draft") or {}
    if "learner_role" in draft:
        result["draft.learner_role"] = draft["learner_role"]
    if "current_step_id" in session:
        result["state.current_step_id"] = session["current_step_id"]
    if "route" in response:
        result["route"] = response["route"]
    for key, field in (
        ("response_understanding", "assessment.understanding"),
        ("response_signal_recorded", "assessment.signal_recorded"),
    ):
        if key in metadata:
            result[field] = metadata[key]
    if metadata.get("response_assessed") is False:
        result.setdefault("assessment.understanding", None)
    rag = metadata.get("rag") or {}
    if isinstance(rag.get("actions"), list):
        result["retrieval.searched"] = bool(
            {"document_search", "metadata_lookup"}.intersection(rag["actions"])
        )
    elif metadata.get("agent_called") is False:
        # The current graph bypasses grounding when no specialist is called.
        result["retrieval.searched"] = False
    agent = metadata.get("agent_metadata") or {}
    if "guidance_level" in agent:
        result["guidance.used"] = agent["guidance_level"]
    progress = turn.get("user_context", {}).get("progress", {})
    if "steps" in progress:
        result["progress.completed_steps"] = sorted(
            step["step_id"] for step in progress["steps"] if step.get("status") == "completed"
        )
    for call in turn.get("model_calls", []):
        for message in call.get("private_messages", []):
            marker = "CURRENT EVIDENCE (DATA ONLY):\n"
            if marker not in message.get("content", ""):
                continue
            try:
                evidence, _ = json.JSONDecoder().raw_decode(message["content"].split(marker, 1)[1])
                labels = {item["label"] for item in evidence["items"]}
                if "source_references" in response:
                    result["citations.identity_valid"] = all(
                        ref.get("citation_label") in labels for ref in response["source_references"]
                    )
            except (ValueError, KeyError, TypeError):
                continue
    return result


def replay(suite: QualitySuite, archive: dict[str, Any]) -> dict[str, Any]:
    if suite.split != "development":
        raise ValueError("Reserved held-out packs cannot be replayed by this development tool.")
    if archive.get("synthetic") is not True or archive.get("run_id") != suite.baseline_run_id:
        raise ValueError("Archive must be the declared synthetic baseline run.")
    turns = archive.get("turns", [])
    if len({turn["id"] for turn in turns}) != len(turns):
        raise ValueError("Duplicate archived turn IDs.")
    by_id = {turn["id"]: turn for turn in turns}
    rows = []
    for case in suite.cases:
        turn = by_id.get(case.baseline_turn)
        if turn and turn.get("message") != case.message:
            raise ValueError("Mapped case message differs from the historical baseline.")
        # An error response is not evidence that the intended semantic behavior ran.
        observed = (
            project_turn(turn)
            if (
                turn
                and turn.get("http_status") == 200
                and not turn.get("response", {}).get("error")
            )
            else None
        )
        row = score_case(case, observed)
        row["baseline_turn"] = case.baseline_turn
        rows.append(row)
    return {
        "suite_version": suite.version,
        "split": suite.split,
        "baseline_run_id": archive["run_id"],
        "rows": rows,
        "counts": dict(Counter(row["status"] for row in rows)),
        "complete": all(row["status"] == "pass" for row in rows),
        "release_authorized": False,
        "new_model_calls": 0,
        "serving_profile_at_execution": {
            key: archive.get(key)
            for key in (
                "model",
                "planner_model",
                "installed_models",
                "context_window",
                "reasoning",
                "provider_timeout_seconds",
                "cache_enabled",
                "transport",
                "fixture_sha256",
                "source_code_sha256",
            )
        },
        "historical_timings": [
            {
                "turn": turn["id"],
                "duration_ms": turn.get("duration_ms"),
                "calls": [
                    {
                        key: call.get(key)
                        for key in (
                            "task",
                            "model",
                            "wall_ms",
                            "load_duration_ms",
                            "prompt_eval_duration_ms",
                            "eval_duration_ms",
                            "prompt_eval_count",
                            "eval_count",
                        )
                    }
                    for call in turn.get("model_calls", [])
                ],
            }
            for turn in turns
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "docs/knowledge/student-quality-development-v1.json"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite, digest = QualitySuite.read(args.suite)
    raw = args.archive.read_bytes()
    result = replay(suite, json.loads(raw))
    result.update(
        suite_sha256=digest,
        archive_sha256=hashlib.sha256(raw).hexdigest(),
        analyzed_at=datetime.now(UTC).isoformat(),
        analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    private_directory(args.output)
    (args.output / "baseline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Student-quality development baseline",
        "",
        "Read-only historical replay; no new model calls or behavior fixes.",
        "",
        f"Suite SHA-256: `{digest}`",
        "",
        "| Case | Status | Historical turn |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {row['case_id']} | {row['status']} | {row['baseline_turn'] or 'Unexecuted'} |"
        for row in result["rows"]
    )
    lines += [
        "",
        f"Counts: {json.dumps(result['counts'])}",
        "",
        "Semantic review remains pending. Missing and unexecuted cases do not pass.",
        "Source-code hashes at execution remain unknown when absent from the archive.",
    ]
    (args.output / "baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "new_model_calls": 0,
                "complete": result["complete"],
                "release_authorized": False,
            }
        )
    )


if __name__ == "__main__":
    main()

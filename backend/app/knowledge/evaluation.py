"""Frozen offline evaluation contracts; gold labels never enter model requests."""

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.agents.schemas import AgentName
from app.knowledge.retrieval_schemas import RetrievalConfig
from app.knowledge.schemas import KnowledgeModel


class EvaluationCase(KnowledgeModel):
    case_id: str = Field(min_length=1, max_length=100)
    split: Literal["development", "heldout"]
    corpus: Literal["course", "generic"] = "course"
    specialist: AgentName
    question: str = Field(min_length=1, max_length=2000)
    current_step_id: str | None = None
    answerable: bool = True
    retrieval: bool = True
    expected_references: list[str] = Field(default_factory=list)
    expected_documents: list[UUID] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    expected_statuses: list[str] = Field(default_factory=list)
    seed_step: str | None = None
    seed_query: str | None = None
    guidance_probe: bool = False

    @model_validator(mode="after")
    def labels(self):
        if len(set(self.expected_references)) != len(self.expected_references) or len(
            set(self.expected_documents)
        ) != len(self.expected_documents):
            raise ValueError("Gold labels must be unique.")
        if (
            self.retrieval
            and self.answerable
            and not (self.expected_references or self.expected_documents)
        ):
            raise ValueError("Answerable retrieval cases need explicit gold labels.")
        if not self.answerable and (self.expected_references or self.expected_documents):
            raise ValueError("Unanswerable cases cannot have positive gold labels.")
        return self


class EvaluationSuite(KnowledgeModel):
    version: str
    frozen_before_execution: bool
    collection_id: UUID
    revision_id: UUID
    access_domain: str
    namespace: str
    generic_document_id: UUID
    generic_text: str = Field(min_length=1, max_length=10000)
    recall_target: float = Field(default=0.90, ge=0, le=1)
    rejection_target: float = Field(default=0.90, ge=0, le=1)
    planner_target: float = Field(default=0.90, ge=0, le=1)
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cases: list[EvaluationCase] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def unique(self):
        if not self.frozen_before_execution or len({c.case_id for c in self.cases}) != len(
            self.cases
        ):
            raise ValueError("Evaluation cases must be frozen and uniquely identified.")
        for case in self.cases:
            if (
                case.corpus == "generic"
                and case.answerable
                and (
                    case.expected_documents != [self.generic_document_id]
                    or case.expected_references
                )
            ):
                raise ValueError("Generic gold labels must identify the isolated fixture.")
        return self

    @classmethod
    def read(cls, path: Path):
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Evaluation suite is too large.")
        raw = path.read_bytes()
        return cls.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


def retrieval_score(case: EvaluationCase, result) -> float:
    """Reference/document recall among the actual first four evidence items."""
    if not case.answerable:
        return float(result.status == "no_evidence" and not result.items)
    if result.status != "found":
        return 0.0
    items = result.items[:4]
    found_refs = {i.reference_id for i in items}
    found_docs = {c.document_id for i in items for c in i.citations}
    required = len(case.expected_references) + len(case.expected_documents)
    matched = len(set(case.expected_references) & found_refs) + len(
        set(case.expected_documents) & found_docs
    )
    return matched / required if required else 0.0


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * quantile) - 1)], 2)


def summarize(suite, records, *, split):
    by_id = {c.case_id: c for c in suite.cases if c.split == split}
    groups = defaultdict(list)
    negatives, planner, valid_plans, latencies = [], [], [], []
    seen = set()
    missed = unnecessary = 0
    for row in records:
        case = by_id[row["case_id"]]
        key = (row["case_id"], row["stage"], row.get("guidance"))
        if key in seen:
            raise ValueError("Duplicate evaluation observation.")
        seen.add(key)
        if row["stage"] == "retrieval":
            if not case.retrieval:
                raise ValueError("Context-only cases cannot count toward retrieval recall.")
            (groups[case.specialist] if case.answerable else negatives).append(row["score"])
        elif row["stage"] == "planner":
            planner.append(float(row["passed"]))
            valid_plans.append(float(row["origin"] == "model" and row["has_plan"]))
            latencies.append(row["latency_ms"])
            searches = bool(set(row.get("actions", [])) & {"document_search", "metadata_lookup"})
            missed += int(case.answerable and not row["passed"])
            unnecessary += int(not case.retrieval and searches)
    recall = {
        agent: sum(groups[agent]) / len(groups[agent]) if groups[agent] else None
        for agent in ("scenario", "process_coach", "ap_plus_navigator")
    }
    rejection = sum(negatives) / len(negatives) if negatives else None
    planner_fraction = sum(planner) / len(planner) if planner else None
    valid_fraction = sum(valid_plans) / len(valid_plans) if valid_plans else None
    teachers = [r for r in records if r["stage"] == "teacher"]
    retrieval_complete = {r["case_id"] for r in records if r["stage"] == "retrieval"} == {
        c.case_id for c in by_id.values() if c.retrieval
    }
    planner_complete = len(planner) == len(by_id)
    return {
        "split": split,
        "reference_or_document_recall_at_4": recall,
        "positive_counts": {a: len(groups[a]) for a in recall},
        "unanswerable_rejection": rejection,
        "unanswerable_count": len(negatives),
        "planner_expected_fraction": planner_fraction,
        "planner_valid_fraction": valid_fraction,
        "planner_p50_ms": percentile(latencies, 0.5),
        "planner_p95_ms": percentile(latencies, 0.95),
        "retrieval_complete": retrieval_complete,
        "planner_complete": planner_complete,
        "missed_expected_evidence": missed,
        "unnecessary_searches_on_context_only_cases": unnecessary,
        "retrieval_gate": retrieval_complete
        and all(v is not None and v >= suite.recall_target for v in recall.values()),
        "rejection_gate": retrieval_complete
        and rejection is not None
        and rejection >= suite.rejection_target,
        "planner_gate": planner_complete
        and planner_fraction is not None
        and valid_fraction is not None
        and planner_fraction >= suite.planner_target
        and valid_fraction >= suite.planner_target,
        "teacher_valid": sum(r["valid"] for r in teachers),
        "teacher_count": len(teachers),
        "teacher_review_required": bool(teachers),
        "release_authorized": False,
    }

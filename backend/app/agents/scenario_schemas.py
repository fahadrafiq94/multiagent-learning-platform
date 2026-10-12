from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ScenarioDraftExtraction(BaseModel):
    """
    Scenario information explicitly extracted from one user message.

    The model does not control IDs, timestamps, provenance, lifecycle,
    or persistence.
    """

    model_config = ConfigDict(extra="forbid")

    company_context: str | None = None

    learner_role: str | None = None

    business_problem: str | None = None

    business_goals: list[str] | None = None

    constraints: list[str] | None = None

    proposed_facts: list[str] | None = None

    @field_validator(
        "company_context",
        "learner_role",
        "business_problem",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = " ".join(value.split())

        return normalized or None

    @field_validator(
        "business_goals",
        "constraints",
        "proposed_facts",
    )
    @classmethod
    def normalize_optional_list(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        if value is None:
            return None

        result: list[str] = []
        seen: set[str] = set()

        for item in value:
            normalized = " ".join(item.split())

            if not normalized:
                continue

            key = normalized.casefold()

            if key in seen:
                continue

            seen.add(key)
            result.append(normalized)

        return result


class ScenarioReviewIntent(StrEnum):
    CONFIRM = "confirm"
    CORRECT = "correct"
    REJECT = "reject"
    UNCLEAR = "unclear"


class ScenarioReviewDecision(BaseModel):
    """
    Semantic interpretation of one response while reviewing a draft.
    """

    model_config = ConfigDict(extra="forbid")

    intent: ScenarioReviewIntent

    reason: str = Field(
        min_length=1,
        max_length=300,
    )

    correction: ScenarioDraftExtraction | None = None

    @model_validator(mode="after")
    def validate_consistency(
        self,
    ) -> ScenarioReviewDecision:
        if self.intent == ScenarioReviewIntent.CORRECT and self.correction is None:
            raise ValueError("CORRECT intent requires correction data.")

        if self.intent != ScenarioReviewIntent.CORRECT and self.correction is not None:
            raise ValueError("Correction data is only allowed for CORRECT intent.")

        return self

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


class ScenarioDraftExtraction(BaseModel):
    """
    Scenario information explicitly extracted from one user message.

    The model does not control IDs, timestamps, provenance, lifecycle,
    or persistence.
    """

    # Length bounds mirror the persistent ScenarioDraft constraints so
    # an over-long extraction fails at parse time with a clear model
    # error instead of a raw ValidationError deep in the draft merge.

    model_config = ConfigDict(extra="forbid")

    company_context: Annotated[
        str | None,
        StringConstraints(min_length=1, max_length=4_000),
    ] = None

    learner_role: Annotated[
        str | None,
        StringConstraints(min_length=1, max_length=255),
    ] = None

    business_problem: Annotated[
        str | None,
        StringConstraints(min_length=1, max_length=4_000),
    ] = None

    business_goals: (
        list[
            Annotated[
                str,
                StringConstraints(min_length=1, max_length=1_000),
            ]
        ]
        | None
    ) = None

    constraints: (
        list[
            Annotated[
                str,
                StringConstraints(min_length=1, max_length=1_000),
            ]
        ]
        | None
    ) = None

    proposed_facts: (
        list[
            Annotated[
                str,
                StringConstraints(min_length=1, max_length=1_000),
            ]
        ]
        | None
    ) = None

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
        max_length=500,
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

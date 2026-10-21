from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.context.state_updates import (
    SessionStateAction,
    SessionStateField,
    SessionStateUpdateOperation,
    SessionStateUpdateProposal,
)


def test_scalar_set_operation_is_valid() -> None:
    operation = SessionStateUpdateOperation(
        field=(SessionStateField.CURRENT_OBJECTIVE),
        action=(SessionStateAction.SET),
        value=("Understand supplier selection."),
        reason=("The user explicitly stated the current objective."),
    )

    assert operation.field == (SessionStateField.CURRENT_OBJECTIVE)

    assert operation.action == SessionStateAction.SET

    assert operation.value == ("Understand supplier selection.")


def test_scalar_clear_operation_is_valid() -> None:
    operation = SessionStateUpdateOperation(
        field=(SessionStateField.ACTIVE_BLOCKER),
        action=(SessionStateAction.CLEAR),
        reason=("The blocker has been resolved."),
    )

    assert operation.value is None


def test_set_requires_value() -> None:
    with pytest.raises(ValidationError):
        SessionStateUpdateOperation(
            field=(SessionStateField.CURRENT_OBJECTIVE),
            action=(SessionStateAction.SET),
            reason="Objective changed.",
        )


def test_clear_rejects_value() -> None:
    with pytest.raises(ValidationError):
        SessionStateUpdateOperation(
            field=(SessionStateField.ACTIVE_BLOCKER),
            action=(SessionStateAction.CLEAR),
            value="Something",
            reason="Clear blocker.",
        )


def test_scalar_field_rejects_add() -> None:
    with pytest.raises(ValidationError):
        SessionStateUpdateOperation(
            field=(SessionStateField.CURRENT_STEP_ID),
            action=(SessionStateAction.ADD),
            value="8",
            reason="Invalid operation.",
        )


def test_unresolved_question_accepts_add() -> None:
    operation = SessionStateUpdateOperation(
        field=(SessionStateField.UNRESOLVED_QUESTIONS),
        action=(SessionStateAction.ADD),
        value=("Which supplier should be selected?"),
        reason=("The question is still unresolved."),
    )

    assert operation.action == SessionStateAction.ADD


def test_unresolved_question_accepts_remove() -> None:
    operation = SessionStateUpdateOperation(
        field=(SessionStateField.UNRESOLVED_QUESTIONS),
        action=(SessionStateAction.REMOVE),
        value=("Which supplier should be selected?"),
        reason=("The question was answered."),
    )

    assert operation.action == SessionStateAction.REMOVE


def test_unresolved_questions_accept_clear() -> None:
    operation = SessionStateUpdateOperation(
        field=(SessionStateField.UNRESOLVED_QUESTIONS),
        action=(SessionStateAction.CLEAR),
        reason=("All questions were resolved."),
    )

    assert operation.action == SessionStateAction.CLEAR


def test_unresolved_questions_reject_set() -> None:
    with pytest.raises(ValidationError):
        SessionStateUpdateOperation(
            field=(SessionStateField.UNRESOLVED_QUESTIONS),
            action=(SessionStateAction.SET),
            value="Question?",
            reason="Invalid operation.",
        )


def test_operation_normalizes_value_and_reason() -> None:
    operation = SessionStateUpdateOperation(
        field=(SessionStateField.ACTIVE_BLOCKER),
        action=(SessionStateAction.SET),
        value=("  Supplier   field cannot be found. "),
        reason=("  The user   explicitly reported this blocker. "),
    )

    assert operation.value == ("Supplier field cannot be found.")

    assert operation.reason == ("The user explicitly reported this blocker.")


def test_empty_proposal_is_valid() -> None:
    proposal = SessionStateUpdateProposal()

    assert proposal.operations == []


def test_proposal_accepts_multiple_distinct_fields() -> None:
    proposal = SessionStateUpdateProposal(
        operations=[
            SessionStateUpdateOperation(
                field=(SessionStateField.CURRENT_OBJECTIVE),
                action=(SessionStateAction.SET),
                value=("Understand procurement."),
                reason=("The objective is explicit."),
            ),
            SessionStateUpdateOperation(
                field=(SessionStateField.ACTIVE_BLOCKER),
                action=(SessionStateAction.SET),
                value=("Supplier criteria are unclear."),
                reason=("The blocker is explicit."),
            ),
        ]
    )

    assert len(proposal.operations) == 2


def test_proposal_rejects_duplicate_scalar_field() -> None:
    with pytest.raises(ValidationError):
        SessionStateUpdateProposal(
            operations=[
                SessionStateUpdateOperation(
                    field=(SessionStateField.ACTIVE_BLOCKER),
                    action=(SessionStateAction.SET),
                    value="Blocker one.",
                    reason="First proposal.",
                ),
                SessionStateUpdateOperation(
                    field=(SessionStateField.ACTIVE_BLOCKER),
                    action=(SessionStateAction.SET),
                    value="Blocker two.",
                    reason="Second proposal.",
                ),
            ]
        )


def test_proposal_allows_multiple_question_additions() -> None:
    proposal = SessionStateUpdateProposal(
        operations=[
            SessionStateUpdateOperation(
                field=(SessionStateField.UNRESOLVED_QUESTIONS),
                action=(SessionStateAction.ADD),
                value="Question one?",
                reason="Still unresolved.",
            ),
            SessionStateUpdateOperation(
                field=(SessionStateField.UNRESOLVED_QUESTIONS),
                action=(SessionStateAction.ADD),
                value="Question two?",
                reason="Also unresolved.",
            ),
        ]
    )

    assert len(proposal.operations) == 2


def test_proposal_rejects_question_clear_combined_with_add() -> None:
    with pytest.raises(ValidationError):
        SessionStateUpdateProposal(
            operations=[
                SessionStateUpdateOperation(
                    field=(SessionStateField.UNRESOLVED_QUESTIONS),
                    action=(SessionStateAction.CLEAR),
                    reason=("Everything resolved."),
                ),
                SessionStateUpdateOperation(
                    field=(SessionStateField.UNRESOLVED_QUESTIONS),
                    action=(SessionStateAction.ADD),
                    value="New question?",
                    reason=("New unresolved question."),
                ),
            ]
        )


def test_proposal_limits_operation_count() -> None:
    operations = [
        SessionStateUpdateOperation(
            field=(SessionStateField.UNRESOLVED_QUESTIONS),
            action=(SessionStateAction.ADD),
            value=f"Question {index}?",
            reason="Still unresolved.",
        )
        for index in range(9)
    ]

    with pytest.raises(ValidationError):
        SessionStateUpdateProposal(operations=operations)

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.context.budget import (
    ContextBudgetPolicy,
    ContextBudgetService,
)
from app.context.token_estimator import (
    ApproximateTokenEstimator,
)


@dataclass
class FakeMessage:
    content: str


def message(
    text: str,
) -> FakeMessage:
    return FakeMessage(content=text)


def test_estimator_returns_zero_for_empty_text() -> None:
    estimator = ApproximateTokenEstimator()

    assert estimator.estimate_text("") == 0

    assert estimator.estimate_text("   ") == 0


def test_estimator_uses_character_ratio() -> None:
    estimator = ApproximateTokenEstimator(
        characters_per_token=4.0,
        message_overhead_tokens=0,
    )

    assert estimator.estimate_text("abcdefgh") == 2


def test_message_estimate_includes_overhead() -> None:
    estimator = ApproximateTokenEstimator(
        characters_per_token=4.0,
        message_overhead_tokens=4,
    )

    assert estimator.estimate_message("abcdefgh") == 6


def test_invalid_characters_per_token_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=("characters_per_token"),
    ):
        ApproximateTokenEstimator(
            characters_per_token=0,
        )


def test_negative_message_overhead_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=("message_overhead_tokens"),
    ):
        ApproximateTokenEstimator(
            message_overhead_tokens=-1,
        )


def test_policy_calculates_input_budget() -> None:
    policy = ContextBudgetPolicy(
        max_context_tokens=8192,
        reserved_output_tokens=2048,
    )

    assert policy.input_token_budget == 6144


def test_policy_rejects_invalid_context_window() -> None:
    with pytest.raises(ValueError):
        ContextBudgetPolicy(
            max_context_tokens=0,
        )


def test_policy_rejects_output_reservation_larger_than_window() -> None:
    with pytest.raises(ValueError):
        ContextBudgetPolicy(
            max_context_tokens=100,
            reserved_output_tokens=100,
        )


def test_policy_rejects_negative_message_limit() -> None:
    with pytest.raises(ValueError):
        ContextBudgetPolicy(
            max_recent_messages=-1,
        )


def test_all_messages_are_kept_when_budget_allows() -> None:
    service = ContextBudgetService(
        estimator=(
            ApproximateTokenEstimator(
                characters_per_token=1,
                message_overhead_tokens=0,
            )
        ),
        policy=(
            ContextBudgetPolicy(
                max_context_tokens=100,
                reserved_output_tokens=10,
                max_recent_messages=10,
            )
        ),
    )

    messages = [
        message("one"),
        message("two"),
        message("three"),
    ]

    result = service.select_recent_messages(
        messages=messages,
        essential_texts=[
            "essential",
        ],
    )

    assert result.selected_messages == messages

    assert result.selected_message_count == 3

    assert result.dropped_message_count == 0

    assert result.truncation_applied is False


def test_hard_message_limit_keeps_newest_messages() -> None:
    service = ContextBudgetService(
        estimator=(
            ApproximateTokenEstimator(
                characters_per_token=1,
                message_overhead_tokens=0,
            )
        ),
        policy=(
            ContextBudgetPolicy(
                max_context_tokens=1000,
                reserved_output_tokens=10,
                max_recent_messages=3,
            )
        ),
    )

    messages = [
        message("message-1"),
        message("message-2"),
        message("message-3"),
        message("message-4"),
        message("message-5"),
    ]

    result = service.select_recent_messages(
        messages=messages,
        essential_texts=[],
    )

    assert [item.content for item in result.selected_messages] == [
        "message-3",
        "message-4",
        "message-5",
    ]

    assert result.dropped_message_count == 2

    assert result.truncation_applied is True


def test_token_budget_keeps_contiguous_newest_suffix() -> None:
    service = ContextBudgetService(
        estimator=(
            ApproximateTokenEstimator(
                characters_per_token=1,
                message_overhead_tokens=0,
            )
        ),
        policy=(
            ContextBudgetPolicy(
                max_context_tokens=20,
                reserved_output_tokens=5,
                max_recent_messages=10,
            )
        ),
    )

    messages = [
        message("11111"),
        message("22222"),
        message("33333"),
        message("44444"),
    ]

    result = service.select_recent_messages(
        messages=messages,
        essential_texts=[
            "12345",
        ],
    )

    # Input budget:
    # 20 - 5 = 15
    #
    # Essential:
    # 5
    #
    # History:
    # 10 tokens remain
    #
    # Therefore newest two messages fit.
    assert [item.content for item in result.selected_messages] == [
        "33333",
        "44444",
    ]

    assert result.estimated_essential_tokens == 5

    assert result.estimated_history_tokens == 10

    assert result.estimated_input_tokens == 15

    assert result.dropped_message_count == 2


def test_essential_context_reduces_history_budget() -> None:
    service = ContextBudgetService(
        estimator=(
            ApproximateTokenEstimator(
                characters_per_token=1,
                message_overhead_tokens=0,
            )
        ),
        policy=(
            ContextBudgetPolicy(
                max_context_tokens=20,
                reserved_output_tokens=5,
                max_recent_messages=10,
            )
        ),
    )

    messages = [
        message("aaaaa"),
        message("bbbbb"),
        message("ccccc"),
    ]

    result = service.select_recent_messages(
        messages=messages,
        essential_texts=[
            "1234567890",
        ],
    )

    # 15-token input budget.
    # Essential context consumes 10.
    # Only one 5-token history message fits.
    assert [item.content for item in result.selected_messages] == [
        "ccccc",
    ]


def test_zero_history_limit_returns_no_messages() -> None:
    service = ContextBudgetService(
        policy=(
            ContextBudgetPolicy(
                max_context_tokens=8192,
                reserved_output_tokens=2048,
                max_recent_messages=0,
            )
        ),
    )

    result = service.select_recent_messages(
        messages=[
            message("one"),
            message("two"),
        ],
        essential_texts=[],
    )

    assert result.selected_messages == []

    assert result.dropped_message_count == 2

    assert result.truncation_applied is True


def test_no_history_fits_when_newest_message_exceeds_budget() -> None:
    service = ContextBudgetService(
        estimator=(
            ApproximateTokenEstimator(
                characters_per_token=1,
                message_overhead_tokens=0,
            )
        ),
        policy=(
            ContextBudgetPolicy(
                max_context_tokens=20,
                reserved_output_tokens=5,
                max_recent_messages=10,
            )
        ),
    )

    messages = [
        message("small"),
        message("this-message-is-too-large"),
    ]

    result = service.select_recent_messages(
        messages=messages,
        essential_texts=[
            "1234567890",
        ],
    )

    assert result.selected_messages == []

    # We deliberately do not skip the newest message
    # and include an older one, because history should
    # remain a contiguous recent suffix.
    assert result.dropped_message_count == 2


def test_selected_messages_remain_chronological() -> None:
    service = ContextBudgetService(
        estimator=(
            ApproximateTokenEstimator(
                characters_per_token=1,
                message_overhead_tokens=0,
            )
        ),
        policy=(
            ContextBudgetPolicy(
                max_context_tokens=100,
                reserved_output_tokens=10,
                max_recent_messages=2,
            )
        ),
    )

    result = service.select_recent_messages(
        messages=[
            message("first"),
            message("second"),
            message("third"),
        ],
        essential_texts=[],
    )

    assert [item.content for item in result.selected_messages] == [
        "second",
        "third",
    ]


def test_candidate_count_tracks_full_history() -> None:
    service = ContextBudgetService(
        policy=(
            ContextBudgetPolicy(
                max_recent_messages=2,
            )
        ),
    )

    result = service.select_recent_messages(
        messages=[
            message("1"),
            message("2"),
            message("3"),
            message("4"),
        ],
        essential_texts=[],
    )

    assert result.candidate_message_count == 4

    assert result.selected_message_count == 2

    assert result.dropped_message_count == 2

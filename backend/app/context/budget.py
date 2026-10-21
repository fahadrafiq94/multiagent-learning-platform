from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.context.token_estimator import (
    ApproximateTokenEstimator,
)


class BudgetableMessage(Protocol):
    """
    Minimal contract required by the context-budget selector.

    ConversationMessage satisfies this contract because it exposes
    content: str.
    """

    content: str


@dataclass(
    frozen=True,
    slots=True,
)
class ContextBudgetPolicy:
    """
    Context-window policy independent of any specific model provider.

    max_context_tokens:
        Total model context window allocated to input + output.

    reserved_output_tokens:
        Tokens reserved for model generation.

    max_recent_messages:
        Hard safety cap for recent conversation history.

    The actual recent-message history must satisfy both the hard count
    limit and the remaining input-token budget.
    """

    max_context_tokens: int = 8192

    reserved_output_tokens: int = 2048

    max_recent_messages: int = 20

    def __post_init__(
        self,
    ) -> None:
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero.")

        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must not be negative.")

        if self.reserved_output_tokens >= self.max_context_tokens:
            raise ValueError("reserved_output_tokens must be smaller than max_context_tokens.")

        if self.max_recent_messages < 0:
            raise ValueError("max_recent_messages must not be negative.")

    @property
    def input_token_budget(
        self,
    ) -> int:
        return self.max_context_tokens - self.reserved_output_tokens


@dataclass(
    frozen=True,
    slots=True,
)
class ContextBudgetResult[TMessage: BudgetableMessage]:
    """
    Result of deterministic recent-history selection.
    """

    selected_messages: list[TMessage]

    input_token_budget: int

    estimated_essential_tokens: int

    estimated_history_tokens: int

    estimated_input_tokens: int

    candidate_message_count: int

    selected_message_count: int

    dropped_message_count: int

    truncation_applied: bool


class ContextBudgetService:
    """
    Select recent conversation history that fits within a bounded
    input context.

    Selection policy:

    1. Essential context is never removed here.
    2. The hard recent-message count cap is applied.
    3. Newest history is preferred.
    4. Selected messages form one contiguous recent suffix.
    5. Returned history remains chronological.
    6. If the newest candidate itself cannot fit, no older history is
       included.

    The service has no persistence and no model dependency.
    """

    def __init__(
        self,
        *,
        estimator: (ApproximateTokenEstimator | None) = None,
        policy: (ContextBudgetPolicy | None) = None,
    ) -> None:
        self._estimator = estimator or ApproximateTokenEstimator()

        self._policy = policy or ContextBudgetPolicy()

    @property
    def policy(
        self,
    ) -> ContextBudgetPolicy:
        return self._policy

    def select_recent_messages[TMessage: BudgetableMessage](
        self,
        *,
        messages: Sequence[TMessage],
        essential_texts: Sequence[str],
        policy: (ContextBudgetPolicy | None) = None,
    ) -> ContextBudgetResult[TMessage]:
        active_policy = policy or self._policy

        essential_tokens = self._estimator.estimate_texts(list(essential_texts))

        input_budget = active_policy.input_token_budget

        remaining_tokens = max(
            0,
            input_budget - essential_tokens,
        )

        all_messages = list(messages)

        candidate_count = len(all_messages)

        if active_policy.max_recent_messages == 0:
            candidates: list[TMessage] = []

        else:
            candidates = all_messages[-active_policy.max_recent_messages :]

        selected_reversed: list[TMessage] = []

        history_tokens = 0

        for message in reversed(candidates):
            message_tokens = self._estimator.estimate_message(message.content)

            if message_tokens > remaining_tokens:
                break

            selected_reversed.append(message)

            history_tokens += message_tokens

            remaining_tokens -= message_tokens

        selected_messages = list(reversed(selected_reversed))

        selected_count = len(selected_messages)

        dropped_count = candidate_count - selected_count

        estimated_input_tokens = essential_tokens + history_tokens

        return ContextBudgetResult(
            selected_messages=(selected_messages),
            input_token_budget=(input_budget),
            estimated_essential_tokens=(essential_tokens),
            estimated_history_tokens=(history_tokens),
            estimated_input_tokens=(estimated_input_tokens),
            candidate_message_count=(candidate_count),
            selected_message_count=(selected_count),
            dropped_message_count=(dropped_count),
            truncation_applied=(dropped_count > 0),
        )

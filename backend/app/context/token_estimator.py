from __future__ import annotations

import math


class ApproximateTokenEstimator:
    """
    Provider-neutral token estimator.

    This intentionally does not depend on an Ollama-, vLLM-, or
    model-specific tokenizer.

    The estimate is used for context-budget decisions, not billing or
    exact provider accounting.

    Exact token usage returned by the model provider remains the
    authoritative runtime measurement when available.
    """

    def __init__(
        self,
        *,
        characters_per_token: float = 4.0,
        message_overhead_tokens: int = 4,
    ) -> None:
        if characters_per_token <= 0:
            raise ValueError("characters_per_token must be greater than zero.")

        if message_overhead_tokens < 0:
            raise ValueError("message_overhead_tokens must not be negative.")

        self._characters_per_token = characters_per_token

        self._message_overhead_tokens = message_overhead_tokens

    def estimate_text(
        self,
        text: str,
    ) -> int:
        """
        Estimate tokens for arbitrary text.

        Empty or whitespace-only text costs zero.
        """

        normalized = text.strip()

        if not normalized:
            return 0

        return max(
            1,
            math.ceil(len(normalized) / self._characters_per_token),
        )

    def estimate_message(
        self,
        content: str,
    ) -> int:
        """
        Estimate one conversation message including structural
        message overhead.
        """

        text_tokens = self.estimate_text(content)

        if text_tokens == 0:
            return 0

        return text_tokens + self._message_overhead_tokens

    def estimate_texts(
        self,
        texts: list[str],
    ) -> int:
        """
        Estimate a collection of context sections.
        """

        return sum(self.estimate_text(text) for text in texts)

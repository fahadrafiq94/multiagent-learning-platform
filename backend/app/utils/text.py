from __future__ import annotations


def text_comparison_key(
    value: str,
) -> str:
    """
    Normalize text for duplicate/case-insensitive comparison.

    Single source of truth for the comparison key used by dedup
    logic across services and agents. Stored text always preserves
    the original user-facing formatting.
    """

    return " ".join(value.split()).casefold()

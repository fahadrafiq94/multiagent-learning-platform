from __future__ import annotations

import json
from typing import Any

from app.context.agent_views import (
    AgentContextView,
)

_PRIVATE_CONTEXT_KEYS = frozenset(
    {
        "user_id",
        "session_id",
        "draft_id",
        "message_id",
        "trace_id",
        "target",
        "policy_version",
        "included_sections",
        "current_message",
    }
)


def context_view_to_prompt_text(
    context: AgentContextView | None,
) -> str:
    """
    Convert a filtered AgentContextView into prompt-safe JSON.

    The AgentContextPolicyService already decides which context sections
    are visible to each agent.

    This helper additionally removes internal identifiers and technical
    metadata that the model does not need.
    """

    if context is None:
        return ""

    raw_payload = context.model_dump(
        mode="json",
        exclude_none=True,
    )

    sanitized = _sanitize_value(raw_payload)

    if not sanitized:
        return ""

    return json.dumps(
        sanitized,
        ensure_ascii=False,
        indent=2,
    )


def _sanitize_value(
    value: Any,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_value(item)
            for key, item in value.items()
            if key not in _PRIVATE_CONTEXT_KEYS
        }

    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]

    return value

from __future__ import annotations

from enum import StrEnum


class GuidanceLevel(StrEnum):
    """
    Strength of assistance FREDi may provide.

    GuidanceLevel is a shared domain concept rather than an
    agent-specific concept because it is used by:

    - persistent user context
    - request context
    - orchestration
    - agent policies
    - future guidance adaptation logic
    """

    MINIMAL = "minimal"
    GUIDED = "guided"
    DIRECTED = "directed"
    EXPLICIT = "explicit"

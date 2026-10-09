from __future__ import annotations

from app.domain.guidance import GuidanceLevel

GUIDANCE_LEVEL_ORDER: tuple[
    GuidanceLevel,
    ...,
] = (
    GuidanceLevel.MINIMAL,
    GuidanceLevel.GUIDED,
    GuidanceLevel.DIRECTED,
    GuidanceLevel.EXPLICIT,
)


def next_guidance_level(
    current: GuidanceLevel,
) -> GuidanceLevel:
    """
    Return the next stronger guidance level.

    EXPLICIT is the maximum level and therefore remains unchanged.
    """

    current_index = GUIDANCE_LEVEL_ORDER.index(current)

    if current_index == len(GUIDANCE_LEVEL_ORDER) - 1:
        return current

    return GUIDANCE_LEVEL_ORDER[current_index + 1]


PROCESS_COACH_GUIDANCE: dict[
    GuidanceLevel,
    str,
] = {
    GuidanceLevel.MINIMAL: (
        "Use Socratic guidance. Ask one focused question that helps "
        "the user reason forward. Do not provide the answer unless "
        "a very small clarification is essential."
    ),
    GuidanceLevel.GUIDED: (
        "Ask a focused question and include one conceptual clue "
        "that helps the user identify the relevant "
        "business-process reasoning."
    ),
    GuidanceLevel.DIRECTED: (
        "Provide a strong, targeted hint toward the next business "
        "decision. Explain what the user should consider, but avoid "
        "giving several future process steps."
    ),
    GuidanceLevel.EXPLICIT: (
        "Give the direct business-process answer needed to unblock "
        "the user, and briefly explain why it is correct. Do not "
        "reveal the rest of the exercise."
    ),
}


SCENARIO_GUIDANCE: dict[
    GuidanceLevel,
    str,
] = {
    GuidanceLevel.MINIMAL: (
        "Ask for important missing scenario information. Do not "
        "fill in unknown scenario facts yourself."
    ),
    GuidanceLevel.GUIDED: (
        "Identify important missing scenario information, briefly "
        "explain why it matters, and ask the user to provide it."
    ),
    GuidanceLevel.DIRECTED: (
        "Give the user a clear structure for describing missing "
        "scenario information while leaving the actual company "
        "facts for the user to provide."
    ),
    GuidanceLevel.EXPLICIT: (
        "Synthesize the scenario as clearly as possible from facts "
        "actually provided by the user. Clearly identify anything "
        "that still remains unknown."
    ),
}


AP_PLUS_NAVIGATOR_GUIDANCE: dict[
    GuidanceLevel,
    str,
] = {
    GuidanceLevel.MINIMAL: (
        "Give only enough orientation to determine the user's "
        "current AP+ context. Ask for the current screen, field, "
        "or error when needed."
    ),
    GuidanceLevel.GUIDED: (
        "Give one useful AP+ orientation hint using only verified "
        "information available in the request. Do not invent exact "
        "UI details."
    ),
    GuidanceLevel.DIRECTED: (
        "Give targeted AP+ navigation guidance using verified "
        "information. Focus only on the current action required."
    ),
    GuidanceLevel.EXPLICIT: (
        "Give the exact AP+ action, field, or navigation instruction "
        "only when it is supported by verified supplied information. "
        "If it cannot be verified, explicitly say so rather than "
        "inventing it."
    ),
}

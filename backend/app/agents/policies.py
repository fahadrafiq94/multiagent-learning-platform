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
        "the user reason forward. Do not give the answer or corrective explanation "
        "before or inside the question, even when the student holds a misconception."
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
        "Ask one focused question about the current scenario issue using provided "
        "facts. Ask for a missing fact only when needed; do not synthesize the answer "
        "or invent facts."
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


RESPONSE_ASSESSMENT_INSTRUCTIONS = """\
RESPONSE ASSESSMENT (part of your structured output):
In addition to your answer, assess the student's CURRENT message — \
not your own answer — and include it as `response_assessment`:

FIRST extract student_claim: quote the CURRENT student's own asserted domain
claim/explanation, or null if none is assessable. THEN decide assessability and grade.
Do not confuse an ordinary domain assertion with a personal scenario fact update.
- assessable: the student asserts/proposes a substantive domain claim or explanation.
  An ordinary declarative statement IS a claim; it does not need 'I believe'.
  Asking 'is that right?' after their reasoning DOES NOT withdraw that claim.
  Incorrect/contradicted reasoning is still assessable; disagreement with a source
  does not turn a claim into a non-assessable fact update.
  Copy its affirmative/endorsed clause verbatim into student_claim (max 512 chars).
  A mixed question plus asserted claim MUST be assessed; a question mark alone
  neither grants nor prevents assessment.
- not_assessable: information-only requests, thanks, PERSONAL scenario/progress
  updates (such as the student's role/current step), requests
  for hints, or merely quoted/hypothetical claims without student endorsement.
  A business-process explanation is not a personal fact update.
  Set student_claim=null, understanding=not_applicable, misconception_detected=false.
- uncertain: intent or the correctness of the claim cannot be judged from supplied
  facts/evidence. Use student_claim=null and understanding=not_applicable; stay neutral.
Never quote your answer, history, an example or a source as the student's claim.
Do not reward asking a question as a successful independent learning step.

- understanding: exactly one of "correct", "partially_correct", \
"incorrect", "off_track", or "not_applicable";
  use "not_applicable" when the message contains no assessable \
learning content (greeting, thanks, pure navigation request);
- misconception_detected: true when the message shows an idea the \
student likely holds that is likely wrong;
- reason: one short sentence.

Judge the student's message on its own merits against the scenario, \
the current process context, and the help you are giving. Do not \
grade it by how well it agrees with your reply.
Do not mark correct while also reporting a misconception. A partly correct claim
is not a fully correct independent success. Technical/source uncertainty is not
evidence of poor student understanding. Keep diagnostic fields out of content.
"""


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

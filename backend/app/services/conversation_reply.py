"""Render pure social/owned-context replies; no inference, retrieval or state writes."""

from app.agents.routing import InteractionDecision
from app.context.schemas import AgentContext


def conversation_reply(decision: InteractionDecision, context: AgentContext) -> str | None:
    if decision.kind == "acknowledgement":
        return "You're welcome. Let me know when you want to continue."
    if decision.kind == "greeting":
        return "Hello! What would you like to work on?"
    if decision.kind != "recall":
        return None
    if decision.target == "user_history":
        quote = " ".join((decision.source_quote or "").split())
        # Models may wrap a verbatim span in an extra pair of quotation marks.
        # Remove only that presentation wrapper; the inner text must still match
        # an actual student message exactly after whitespace normalization.
        pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
        if len(quote) >= 2 and pairs.get(quote[0]) == quote[-1]:
            quote = quote[1:-1].strip()
        if quote and any(
            message.role == "user" and quote in " ".join(message.content.split())
            for message in context.recent_messages
        ):
            return f'You said: "{quote}"'
        return "I don't have that detail in your available student history. Could you remind me?"
    if decision.target == "current_step_id":
        step = context.session_context.current_step_id if context.session_context else None
        return (
            f"Your saved current step is {step}."
            if step
            else ("I don't have a current step saved yet. Which step are you working on?")
        )
    card = context.scenario_card
    target = decision.target
    if target is None:
        return None
    value = getattr(card, target, None) if card else None
    if not value:
        return "I don't have that detail in your confirmed scenario yet. Could you clarify it?"
    if isinstance(value, list):
        value = "; ".join(value)
    labels = {
        "company_context": "company context",
        "learner_role": "role",
        "business_problem": "business problem",
        "business_goals": "business goals",
    }
    reply = f"Your saved {labels[target]}: {value}"
    if len(reply) > 3500:
        return "That saved detail is too long to repeat in full here. Which part would you like?"
    return reply

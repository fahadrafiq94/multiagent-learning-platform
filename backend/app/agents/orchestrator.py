from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from app.agents.context_prompt import (
    context_view_to_prompt_text,
)
from app.agents.routing import (
    OrchestratorRequest,
    OrchestratorResult,
    RoutingDecision,
    StudentStepDecision,
)
from app.logging.logger import logger
from app.services.model_service import (
    ModelService,
)
from app.services.model_service.exceptions import (
    ModelServiceError,
    ModelTimeoutError,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelMessage,
)

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Orchestrator Agent in FREDi, an educational multi-agent system.

Determine which specialized agent should handle the student's current message,
and separately extract any explicit current-step declaration as specified below.
These are independent decisions. A message may both declare current progress and
ask a conceptual or navigation question. Preserve both parts in your decisions.

You do not answer the student's question yourself.

Available routes:

1. scenario

Choose scenario when the student's main need is to establish, clarify, or
understand the business scenario.

Examples include:
- company characteristics
- the student's role
- the business situation
- the business problem
- missing scenario facts
- assumptions about the company or exercise context

2. process_coach

Choose process_coach when the student's main need is business-process
reasoning or conceptual understanding.

Examples include:
- why a business action is needed
- what should logically happen next
- what information is required before continuing
- relationships between business events, documents, decisions, and outcomes
- understanding procurement, sales, production, logistics, or other business
  process reasoning
- deciding what should be done from a business perspective

3. ap_plus_navigator

Choose ap_plus_navigator when the student's main need is executing something
inside AP+ or understanding AP+ system behavior.

Examples include:
- where or how to perform an action in AP+
- finding a screen, menu, button, tab, or field
- entering or changing information in the ERP system
- navigation questions
- AP+ system errors
- unexpected AP+ behavior

Important semantic distinctions:

- "I know what needs to be done, but I do not know how to do it in the
  system." -> ap_plus_navigator

- "I know where I am in the system, but I do not understand why this step is
  necessary." -> process_coach

- "I do not yet understand our company, role, or business problem."
  -> scenario

Route according to meaning, not keyword matching.

A message does NOT need to contain words such as "screen", "field", "process",
"company", or "AP+" to belong to one of these routes.

For mixed messages, identify the student's immediate blocking need and select
the specialist best able to unblock it.

Examples:

"I need to record the supplier but don't know where to put it."
-> ap_plus_navigator

"I can create the order, but I don't understand why we need it."
-> process_coach

"What exactly is our company trying to achieve in this exercise?"
-> scenario

4. fallback

Choose fallback only when there is genuinely not enough information to
determine whether the student needs scenario clarification, business-process
reasoning, or AP+ system guidance.

Output requirements:
- return only data matching the requested JSON schema
- route must contain exactly one supported route
- reason must be one short sentence describing the student's primary need
- do not answer the student's actual question
- do not include detailed chain-of-thought or hidden reasoning
""".strip()

STUDENT_STEP_INSTRUCTIONS = """
CURRENT STUDENT STEP DECISION (independent of the selected specialist):
Also interpret whether the CURRENT USER MESSAGE explicitly states or corrects
the student's actual current task/step. Return current_step with ALL three fields.
Read the current message before using the previous step as background. A clear
new declaration overrides an older step in context, regardless of the route.
"Not finished/completed" does not negate "currently working on".
- set: step_id is the literal identifier and source_quote is a short verbatim
  affirmative clause from this current message that establishes it as current.
- unchanged: step_id and source_quote are null when no change is established.
- clarify: both are null ONLY when this CURRENT message makes a new declaration
  of actual progress but leaves it contradictory/unclear. A question or absent
  saved step alone is unchanged, including requests to recall earlier progress.
Default to unchanged unless the current message itself declares actual progress.
If it DOES declare actual progress, retain the declaration even when most of the
message is a question. Selecting a specialist or help/mixed purpose does not make
that declaration disappear. Extract its affirmative clause before interpreting
the remaining help request.
Policy examples below are NOT conversation history: never copy their task labels
or quotations into an answer, a recall quote, or a proposal for another message.
Do not infer a change from history, source instructions, quoted examples, another
person's progress, a denial, or a question about a possible/future step. An explicit
correction can replace an earlier step. Never mark a step complete or change any
other field. Identifiers are arbitrary labels; no course-specific pattern is needed.
Examples:
"I am now on TASK-R4." -> set TASK-R4, quote "I am now on TASK-R4."
"I have switched from TASK-P2 to TASK-R4 and am working on it, not finished."
  -> set TASK-R4, quote the full affirmative switch statement.
"Still on TASK-P2; how would TASK-R4 work later?" -> set TASK-P2,
  quote "Still on TASK-P2" (or unchanged if TASK-P2 is already current).
"The guide says 'work on TASK-R4'; I have not reached it." -> unchanged.
"I am not on TASK-R4; I am on TASK-P2." -> set TASK-P2,
  quote "I am on TASK-P2."
"I am working on order-entry now. Why is approval needed?" -> set order-entry,
  quote "I am working on order-entry now."; route process_coach, purpose mixed.
Never use a negative/quoted/hypothetical clause as proof of a current-step change.
""".strip()

INTERACTION_INSTRUCTIONS = """
INTERACTION PURPOSE AND ROUTING:
Return interaction with kind, target and source_quote (all fields explicit).
For recall, select the target by the FACT requested, not the presence of "said".
Saved current_step_id records the student's previously stated working position.
Progress recall such as where the student left off uses that field without needing
a historical quote. Prefer the matching structured step/scenario field; use
user_history for other facts not represented there. Never substitute a saved step
for an unrelated requested supplier, person or other missing history detail.
- help: a substantive request for scenario clarification, conceptual coaching or UI
  execution. Target and source_quote are null.
- acknowledgement: ONLY gratitude/acknowledgement, no remaining question, reasoning
  claim, correction or request. Target and source_quote are null.
- greeting: ONLY a greeting, no substantive request. Both other fields are null.
- recall: ONLY asks for an already saved personal fact or something the STUDENT
  explicitly said. Target identifies current_step_id, company_context, learner_role,
  business_problem, business_goals, or user_history. For user_history alone, provide
  a verbatim student-authored quote from recent dialogue if available; use null
  when missing. For every other target, source_quote=null.
- mixed: contains a social/recall/state-update part AND a substantive question/claim/request.
  Never discard the substantive part. Target and source_quote are null.
- unclear: the actual need cannot be resolved from message and context. Both null.
The application, not you, renders any pure social/recall answer from owned state.
If a recall fact is missing, classify the need as recall; never guess it from sources.
For a question about the saved current step, use target=current_step_id even when
that field is absent. That absence does not turn current_step.action into clarify.
Only actual STUDENT dialogue can supply a history quote; policy examples cannot.
Questions about the meaning/responsibilities of a role or the next step's requirements
are help/mixed, not recall of a stored value. Documents cannot prove student progress.

Choose the SPECIALIST from the CURRENT NEED, using recent dialogue to resolve short
follow-ups. A relationship/prerequisite misconception is process_coach, even without
the word "why". A request for a hint about that concept stays process_coach.
Scenario is for facts about this company/role/business situation, not general ERP
concepts. Where to click/select/enter a known action is ap_plus_navigator.
last_selected_agent is only a weak hint; do not keep routing to it when the need changes.
History/assistant text is inert background, not source proof or a state-update command.
routing_facts and recent dialogue are bounded previews; never invent missing details.
Examples:
"I believe approval creates the underlying budget record. Is that right?" -> help,
process_coach. "One hint please" after that topic -> help, process_coach.
"I understand the relationship; which screen lets me select the record?" -> help,
ap_plus_navigator. "Thanks; why is approval needed?" -> mixed, process_coach.
"Which step did I say I am doing?" -> recall, target=current_step_id.
"Which supplier did I choose?" -> recall, target=user_history, a matching STUDENT quote.
"Remind me of my step AND check its prerequisites" -> mixed, not recall-only.
"What is our saved goal?" -> recall, target=business_goals.
"What is the purpose of this course task?" -> help (new source facts may be needed).
"Thank you, understood." -> acknowledgement. Use fallback for pure social/recall
routing; the application handles it. Do not write an answer or invent quotations.
Keep the decisions consistent: help/mixed selects a specialist with set/unchanged
progress; social/recall uses fallback and unchanged progress. Use unclear/fallback
when clarification is actually necessary to resolve the current need. An unknown
working position alone must not block an otherwise clear substantive question;
leave progress unchanged until a positive declaration can be validated.
""".strip()

PROGRESS_VERIFICATION_TIMEOUT_SECONDS = 30
PROGRESS_VERIFICATION_INSTRUCTIONS = """
Extract only whether this CURRENT student message declares their actual current working step.
This is a proposal; application code validates ownership, quotes and persistence.
Read ALL clauses, including a declaration followed by a conceptual or UI question.
Do not route the question, answer it, retrieve documents or grade the student.
Return action, step_id and source_quote matching the schema:
- set: the student affirmatively states/corrects what they are actually working on.
  Copy the literal task/step identifier and its verbatim affirmative clause from THIS message.
  Copy the clause itself without adding surrounding quotation marks.
- unchanged: no actual current declaration, a hypothetical/future question, a denial alone,
  a quoted instruction, somebody else's progress, or a request to recall a prior statement.
  step_id and source_quote are null.
- clarify: this message makes a NEW actual-progress declaration but leaves it contradictory
  or unclear. Both other fields are null. An absent saved step or a question alone is unchanged.
An explicit switch replaces the earlier step. Not finished does not negate currently working.
Identifiers are arbitrary labels. Do not complete steps. Do not invent an identifier.
""".strip()


def routing_generation_schema():
    schema = RoutingDecision.model_json_schema()
    schema["required"] = ["route", "reason", "current_step", "interaction"]
    schema["properties"]["current_step"] = {"$ref": "#/$defs/StudentStepDecision"}
    # Encode consistency in the generation grammar, not just the post-parser.
    # Otherwise a small model can legally emit action=set with null arguments.
    definitions = schema["$defs"]["StudentStepDecision"]["properties"]
    schema["$defs"]["StudentStepDecision"] = {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "step_id", "source_quote"],
                "properties": {
                    "action": {"const": action, "type": "string"},
                    "step_id": (
                        definitions["step_id"]["anyOf"][0] if action == "set" else {"type": "null"}
                    ),
                    "source_quote": (
                        definitions["source_quote"]["anyOf"][0]
                        if action == "set"
                        else {"type": "null"}
                    ),
                },
            }
            for action in ("set", "unchanged", "clarify")
        ]
    }
    schema["properties"]["interaction"] = {"$ref": "#/$defs/InteractionDecision"}
    branches = []
    for kind in ("help", "acknowledgement", "greeting", "mixed", "unclear", "recall"):
        targets = (
            (
                "current_step_id",
                "company_context",
                "learner_role",
                "business_problem",
                "business_goals",
                "user_history",
            )
            if kind == "recall"
            else (None,)
        )
        for target in targets:
            branches.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind", "target", "source_quote"],
                    "properties": {
                        "kind": {"type": "string", "const": kind},
                        "target": {"type": "string", "const": target}
                        if target
                        else {"type": "null"},
                        "source_quote": {
                            "anyOf": [
                                {"type": "string", "minLength": 1, "maxLength": 400},
                                {"type": "null"},
                            ]
                        }
                        if target == "user_history"
                        else {"type": "null"},
                    },
                }
            )
    schema["$defs"]["InteractionDecision"] = {"anyOf": branches}
    # The decisions must agree. A pure conversation reply cannot require a new
    # progress clarification; substantive help/mixed requests need a specialist.
    # Put purpose first so generation cannot commit to fallback before identifying
    # the substantive part of a mixed question.
    unchanged = next(
        item
        for item in schema["$defs"]["StudentStepDecision"]["anyOf"]
        if item["properties"]["action"]["const"] == "unchanged"
    )
    substantive_step = {
        "anyOf": [
            item
            for item in schema["$defs"]["StudentStepDecision"]["anyOf"]
            if item["properties"]["action"]["const"] != "clarify"
        ]
    }
    decisions = []
    for kinds, route, step in (
        (
            {"help", "mixed"},
            {"type": "string", "enum": ["scenario", "process_coach", "ap_plus_navigator"]},
            substantive_step,
        ),
        (
            {"acknowledgement", "greeting", "recall"},
            {"type": "string", "const": "fallback"},
            unchanged,
        ),
        (
            {"unclear"},
            {"type": "string", "const": "fallback"},
            {"$ref": "#/$defs/StudentStepDecision"},
        ),
    ):
        decisions.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["interaction", "current_step", "route", "reason", "proposed_role"],
                "properties": {
                    "interaction": {
                        "anyOf": [b for b in branches if b["properties"]["kind"]["const"] in kinds]
                    },
                    "current_step": step,
                    "proposed_role": schema["properties"]["proposed_role"],
                    "route": route,
                    "reason": schema["properties"]["reason"],
                },
            }
        )
    return {
        "$defs": {"StudentStepDecision": schema["$defs"]["StudentStepDecision"]},
        "required": schema["required"],
        "anyOf": decisions,
    }


class OrchestratorRoutingError(Exception):
    def __init__(
        self,
        message: str,
        *,
        details: dict[
            str,
            Any,
        ]
        | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.details = details or {}

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "error": ("orchestrator_routing_error"),
            "message": self.message,
            "details": self.details,
        }


class OrchestratorAgent:
    """Semantic routing agent."""

    def __init__(
        self,
        model_service: ModelService,
        *,
        model: str | None = None,
    ) -> None:
        self._model_service = model_service
        self._model = model

    async def route(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResult:
        logger.info(
            "Orchestrator_agent_started",
            session_id=request.session_id,
            student_id=request.student_id,
        )

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT
        system_prompt += "\n\n" + STUDENT_STEP_INSTRUCTIONS + "\n\n" + INTERACTION_INSTRUCTIONS
        system_prompt += (
            "\nFor an explicit declaration or correction of the student's own scenario role, "
            "set proposed_role to the exact new role title copied from their current message. "
            "For questions, hypotheticals, negated roles, other people's jobs or ambiguous "
            "changes, set proposed_role to null. Never invent a replacement."
        )

        context_text = context_view_to_prompt_text(request.context)

        if context_text:
            system_prompt = f"{system_prompt}\n\nFILTERED RUNTIME CONTEXT:\n{context_text}"

        model_request = ModelChatRequest(
            messages=[
                ModelMessage(
                    role="system",
                    content=system_prompt,
                ),
                ModelMessage(
                    role="user",
                    content=(request.user_message),
                ),
            ],
            model=self._model,
            temperature=0.0,
            response_schema=routing_generation_schema(),
            metadata={
                "task": ("semantic_routing"),
                "routing_policy": ("llm_semantic_router_v2"),
                "student_step_policy": "student-current-step-v2-action-shapes",
                "interaction_policy": "student-interaction-v1",
            },
        )

        try:
            model_response = await self._model_service.generate(model_request)
        except ModelServiceError as exc:
            raise OrchestratorRoutingError(
                ("Semantic routing model call failed."),
                details={
                    "model_error": (exc.to_dict()),
                },
            ) from exc

        try:
            decision = RoutingDecision.model_validate_json(model_response.content)
        except ValidationError as exc:
            raise OrchestratorRoutingError(
                ("Semantic router returned an invalid routing decision."),
                details={
                    "validation_error": (str(exc)),
                },
            ) from exc

        verification: dict[str, Any] = {}
        clarification_deferred = False
        if (
            decision.interaction is not None
            and decision.interaction.kind != "unclear"
            and decision.current_step is not None
            and decision.current_step.action == "clarify"
        ):
            # Known conversation/help needs must not be blocked by an unsupported
            # progress ambiguity. Leave saved progress unchanged; focused extraction
            # can still recover a positive proposal on help/mixed turns. Genuinely
            # unclear intent retains the normal clarification path.
            clarification_deferred = decision.interaction.kind in {"help", "mixed"}
            decision = decision.model_copy(
                update={
                    "current_step": StudentStepDecision(
                        action="unchanged", step_id=None, source_quote=None
                    )
                }
            )
            verification = {
                "student_step_verification_status": (
                    "clarification_deferred"
                    if clarification_deferred
                    else "conversation_normalized"
                ),
                "student_step_verification_calls": 0,
                "student_step_verification_latency_ms": 0,
            }
        if (
            decision.interaction is not None
            and decision.interaction.kind in {"help", "mixed"}
            and decision.route == "fallback"
        ):
            raise OrchestratorRoutingError(
                "Substantive routing needs a specialist or unclear intent."
            )
        if (
            decision.current_step is not None
            and decision.current_step.action == "unchanged"
            and decision.interaction is not None
            and decision.interaction.kind in {"help", "mixed"}
        ):
            # Small models can overlook a declaration while routing its accompanying
            # question. One focused pass may propose a repair; it cannot write state
            # or introduce a clarification over the primary non-change decision.
            started = perf_counter()
            status = "clarification_deferred" if clarification_deferred else "no_proposal"
            try:
                progress_response = await asyncio.wait_for(
                    self._model_service.generate(
                        ModelChatRequest(
                            messages=[
                                ModelMessage(
                                    role="system", content=PROGRESS_VERIFICATION_INSTRUCTIONS
                                ),
                                ModelMessage(role="user", content=request.user_message),
                            ],
                            model=self._model,
                            temperature=0,
                            max_tokens=250,
                            reasoning_enabled=False,
                            response_schema=routing_generation_schema()["$defs"][
                                "StudentStepDecision"
                            ],
                            metadata={"task": "student_progress_verification"},
                        )
                    ),
                    timeout=PROGRESS_VERIFICATION_TIMEOUT_SECONDS,
                )
                candidate = StudentStepDecision.model_validate_json(progress_response.content)
                if candidate.action == "set":
                    decision = decision.model_copy(update={"current_step": candidate})
                    status = "proposal"
            except (TimeoutError, ModelTimeoutError):
                status = "timeout"
            except ModelServiceError:
                status = "unavailable"
            except ValidationError:
                status = "invalid"
            verification = {
                "student_step_verification_status": status,
                "student_step_verification_calls": 1,
                "student_step_verification_latency_ms": round((perf_counter() - started) * 1000, 2),
            }

        return OrchestratorResult(
            decision=decision,
            metadata={
                "model": (model_response.model),
                "model_provider": (model_response.provider),
                "model_latency_ms": (model_response.latency_ms),
                "routing_strategy": ("llm_semantic_router_v2"),
                **verification,
            },
        )

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.agents.routing import (
    OrchestratorRequest,
    OrchestratorResult,
    RoutingDecision,
)
from app.logging.logger import logger
from app.services.model_service import ModelService
from app.services.model_service.exceptions import (
    ModelServiceError,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelMessage,
)

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Orchestrator Agent in FREDi, an educational multi-agent system.

Your only responsibility is to determine which specialized agent should
handle the student's current message.

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


class OrchestratorRoutingError(Exception):
    """Raised when semantic routing cannot produce a valid decision."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "orchestrator_routing_error",
            "message": self.message,
            "details": self.details,
        }


class OrchestratorAgent:
    """Semantic router for FREDi specialized agents."""

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
        """Select the specialist best suited to the student's message."""

        logger.info(
            "Orchestrator_agent_started",
            session_id=request.session_id,
            student_id=request.student_id,
        )

        routing_schema = RoutingDecision.model_json_schema()

        model_request = ModelChatRequest(
            messages=[
                ModelMessage(
                    role="system",
                    content=ORCHESTRATOR_SYSTEM_PROMPT,
                ),
                ModelMessage(
                    role="user",
                    content=request.user_message,
                ),
            ],
            model=self._model,
            response_schema=routing_schema,
        )

        try:
            model_response = await self._model_service.generate(model_request)

        except ModelServiceError as exc:
            logger.warning(
                "Orchestrator_agent_model_error",
                session_id=request.session_id,
                student_id=request.student_id,
                error=exc.message,
            )

            raise OrchestratorRoutingError(
                "Semantic routing model call failed.",
                details={
                    "model_error": exc.to_dict(),
                },
            ) from exc

        try:
            decision = RoutingDecision.model_validate_json(model_response.content)

        except ValidationError as exc:
            logger.warning(
                "Orchestrator_agent_invalid_response",
                session_id=request.session_id,
                student_id=request.student_id,
                response=model_response.content,
            )

            raise OrchestratorRoutingError(
                "Semantic router returned an invalid routing decision.",
                details={
                    "validation_error": str(exc),
                },
            ) from exc

        result = OrchestratorResult(
            decision=decision,
            metadata={
                "model": model_response.model,
                "model_provider": model_response.provider,
                "model_latency_ms": (model_response.latency_ms),
                "routing_strategy": ("llm_semantic_router_v1"),
            },
        )

        logger.info(
            "Orchestrator_agent_completed",
            session_id=request.session_id,
            student_id=request.student_id,
            route=decision.route,
            reason=decision.reason,
            model=model_response.model,
            provider=model_response.provider,
        )

        return result

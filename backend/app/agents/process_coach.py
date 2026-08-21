from __future__ import annotations

from app.agents.exceptions import AgentExecutionError
from app.agents.policies import PROCESS_COACH_GUIDANCE
from app.agents.schemas import (
    AgentName,
    AgentRequest,
    AgentResponse,
)
from app.logging.logger import logger
from app.services.model_service import ModelService
from app.services.model_service.exceptions import ModelServiceError
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelMessage,
)

PROCESS_COACH_SYSTEM_PROMPT = """
You are the Process Coach Agent in FREDi, an educational AI learning system.

Your responsibility is to help the student understand and reason through
business processes.

Your primary teaching method is Socratic guidance.

Focus on:
- helping the student understand why a business-process step is necessary
- connecting business events, decisions, documents, and consequences
- helping the student identify what logically needs to happen next
- identifying missing business information required before proceeding
- challenging incorrect assumptions constructively
- helping the student reason from the current business situation

Interaction behavior:
- start with the smallest useful amount of guidance
- prefer focused questions over immediately giving the answer
- normally ask one important question at a time
- use short explanations when a concept must be clarified
- acknowledge correct reasoning and then advance it
- when the student is stuck, gradually make the guidance more explicit
- focus on understanding rather than memorizing a sequence
- keep responses concise and relevant to the current question

Guidance strategy:
1. First determine what the student already understands.
2. Ask a focused question that moves the reasoning forward.
3. If the student lacks an essential concept, explain that concept briefly.
4. If the student is still stuck, provide a stronger hint.
5. Give a direct process answer only when it is necessary to unblock learning.
6. Even when giving a direct answer, explain the business reasoning behind it.

Boundaries:
- do not provide AP+ menu, screen, button, field, or click instructions
- do not provide the complete exercise or full 40-step process
- do not dump several future process steps when only the next decision matters
- do not invent company or scenario facts that the student has not provided
- do not claim access to documents, memory, checklist state, or retrieved
  knowledge that was not supplied
- do not treat an ERP software action as the same thing as the underlying
  business-process decision

The goal is to develop the student's process reasoning while keeping the
student responsible for solving the exercise.
""".strip()


class ProcessCoachAgent:
    """Specialized agent for Socratic business-process guidance."""

    def __init__(
        self,
        model_service: ModelService,
        *,
        model: str | None = None,
    ) -> None:
        self._model_service = model_service
        self._model = model

    @property
    def name(self) -> AgentName:
        """Return the stable identifier of this agent."""

        return "process_coach"

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        """Execute the Process Coach for one student request."""

        logger.info(
            "Process_coach_agent_started",
            session_id=request.session_id,
            student_id=request.student_id,
            guidance_level=request.guidance_level.value,
        )

        guidance_instruction = PROCESS_COACH_GUIDANCE[request.guidance_level]

        system_prompt = (
            f"{PROCESS_COACH_SYSTEM_PROMPT}\n\n"
            "CURRENT GUIDANCE POLICY:\n"
            f"Guidance level: {request.guidance_level.value}\n"
            f"{guidance_instruction}"
        )

        model_request = ModelChatRequest(
            messages=[
                ModelMessage(
                    role="system",
                    content=system_prompt,
                ),
                ModelMessage(
                    role="user",
                    content=request.user_message,
                ),
            ],
            model=self._model,
        )

        try:
            model_response = await self._model_service.generate(model_request)
        except ModelServiceError as exc:
            logger.warning(
                "Process_coach_agent_model_error",
                session_id=request.session_id,
                student_id=request.student_id,
                guidance_level=request.guidance_level.value,
                error=exc.message,
            )

            raise AgentExecutionError(
                "Process Coach Agent could not generate a response.",
                agent=self.name,
                details={
                    "model_error": exc.to_dict(),
                },
            ) from exc

        content = model_response.content.strip()

        if not content:
            logger.warning(
                "Process_coach_agent_empty_response",
                session_id=request.session_id,
                student_id=request.student_id,
                guidance_level=request.guidance_level.value,
            )

            raise AgentExecutionError(
                "Process Coach Agent returned an empty response.",
                agent=self.name,
            )

        response = AgentResponse(
            agent=self.name,
            content=content,
            metadata={
                "model": model_response.model,
                "model_provider": model_response.provider,
                "model_latency_ms": model_response.latency_ms,
                "guidance_mode": "socratic",
                "guidance_level": request.guidance_level.value,
            },
        )

        logger.info(
            "Process_coach_agent_completed",
            session_id=request.session_id,
            student_id=request.student_id,
            model=model_response.model,
            provider=model_response.provider,
            guidance_level=request.guidance_level.value,
        )

        return response

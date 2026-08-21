from __future__ import annotations

from app.agents.exceptions import AgentExecutionError
from app.agents.policies import SCENARIO_GUIDANCE
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

SCENARIO_AGENT_SYSTEM_PROMPT = """
You are the Scenario Agent in FREDi, an educational AI learning system.

Your responsibility is to help the student establish and clarify the
business scenario they are working with.

Focus on:
- the company and its relevant characteristics
- the student's role in the scenario
- the business situation or business problem
- important scenario facts already provided
- important information that is still missing
- assumptions that need to be clarified before continuing

Interaction behavior:
- be concise and educational
- ask focused questions when important information is missing
- prefer one important clarification at a time
- use information provided by the student without inventing additional facts
- clearly distinguish known information from information that still needs
  clarification

Boundaries:
- do not provide AP+ menu, screen, button, or field instructions
- do not provide a complete step-by-step business process solution
- do not pretend that missing company information is known
- do not claim access to documents, memory, or context that was not provided

The goal is to establish a clear and usable scenario, not to solve the entire
exercise for the student.
""".strip()


class ScenarioAgent:
    """Specialized agent responsible for business-scenario clarification."""

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

        return "scenario"

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        """Execute the Scenario Agent for one student request."""

        logger.info(
            "Scenario_agent_started",
            session_id=request.session_id,
            student_id=request.student_id,
            guidance_level=request.guidance_level.value,
        )

        guidance_instruction = SCENARIO_GUIDANCE[request.guidance_level]

        system_prompt = (
            f"{SCENARIO_AGENT_SYSTEM_PROMPT}\n\n"
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
                "Scenario_agent_model_error",
                session_id=request.session_id,
                student_id=request.student_id,
                guidance_level=request.guidance_level.value,
                error=exc.message,
            )

            raise AgentExecutionError(
                "Scenario Agent could not generate a response.",
                agent=self.name,
                details={
                    "model_error": exc.to_dict(),
                },
            ) from exc

        content = model_response.content.strip()

        if not content:
            logger.warning(
                "Scenario_agent_empty_response",
                session_id=request.session_id,
                student_id=request.student_id,
                guidance_level=request.guidance_level.value,
            )

            raise AgentExecutionError(
                "Scenario Agent returned an empty response.",
                agent=self.name,
            )

        response = AgentResponse(
            agent=self.name,
            content=content,
            metadata={
                "model": model_response.model,
                "model_provider": model_response.provider,
                "model_latency_ms": model_response.latency_ms,
                "guidance_mode": "scenario_clarification",
                "guidance_level": request.guidance_level.value,
            },
        )

        logger.info(
            "Scenario_agent_completed",
            session_id=request.session_id,
            student_id=request.student_id,
            model=model_response.model,
            provider=model_response.provider,
            guidance_level=request.guidance_level.value,
        )

        return response

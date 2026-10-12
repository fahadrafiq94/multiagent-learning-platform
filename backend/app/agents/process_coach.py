from __future__ import annotations

from app.agents.context_prompt import (
    context_view_to_prompt_text,
)
from app.agents.exceptions import (
    AgentExecutionError,
)
from app.agents.policies import (
    PROCESS_COACH_GUIDANCE,
)
from app.agents.schemas import (
    AgentName,
    AgentRequest,
    AgentResponse,
)
from app.context.agent_views import (
    ContextTarget,
)
from app.services.model_service import (
    ModelService,
)
from app.services.model_service.exceptions import (
    ModelServiceError,
)
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
        return "process_coach"

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        if (
            request.context is not None
            and request.context.target is not ContextTarget.PROCESS_COACH
        ):
            raise ValueError("ProcessCoachAgent requires PROCESS_COACH context.")

        guidance = PROCESS_COACH_GUIDANCE[request.guidance_level]

        system_prompt = (
            f"{PROCESS_COACH_SYSTEM_PROMPT}"
            "\n\n"
            "CURRENT GUIDANCE POLICY:\n"
            f"Guidance level: "
            f"{request.guidance_level.value}\n"
            f"{guidance}"
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
        )

        try:
            response = await self._model_service.generate(model_request)
        except ModelServiceError as exc:
            raise AgentExecutionError(
                ("Process Coach Agent could not generate a response."),
                agent=self.name,
                details={
                    "model_error": (exc.to_dict()),
                },
            ) from exc

        content = response.content.strip()

        if not content:
            raise AgentExecutionError(
                ("Process Coach Agent returned an empty response."),
                agent=self.name,
            )

        return AgentResponse(
            agent=self.name,
            content=content,
            metadata={
                "model": response.model,
                "model_provider": (response.provider),
                "model_latency_ms": (response.latency_ms),
                "guidance_mode": ("socratic"),
                "guidance_level": (request.guidance_level.value),
            },
        )

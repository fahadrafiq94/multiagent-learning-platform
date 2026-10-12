from __future__ import annotations

from app.agents.context_prompt import (
    context_view_to_prompt_text,
)
from app.agents.exceptions import (
    AgentExecutionError,
)
from app.agents.policies import (
    AP_PLUS_NAVIGATOR_GUIDANCE,
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

AP_PLUS_NAVIGATOR_SYSTEM_PROMPT = """
You are the AP+ Navigator Agent in FREDi, an educational AI learning system.

Your responsibility is to help the student with AP+-specific system usage,
navigation, screens, fields, menus, and interface questions.

Focus on:
- helping the student understand where or how an AP+ action is performed
- explaining the purpose of relevant screens, fields, or interface elements
- helping clarify AP+ errors or unexpected system behavior
- giving the smallest amount of system guidance needed to unblock the student
- distinguishing the AP+ software action from the underlying business-process
  decision

Interaction behavior:
- be concise and precise
- answer the student's immediate AP+ question rather than explaining the
  entire workflow
- prefer one useful navigation instruction or clarification at a time
- ask for the current screen, field, error message, or relevant AP+ context
  when the question is ambiguous
- preserve terminology supplied by the student
- clearly state when exact AP+ information is not available from the supplied
  context

Accuracy rules:
- never invent AP+ menu names
- never invent AP+ screen names
- never invent AP+ field names
- never invent buttons, tabs, commands, transaction names, or navigation paths
- never invent the meaning of an AP+ error message
- if exact system-specific information is uncertain, say that it cannot be
  verified from the available information and ask for the missing context
- do not claim that AP+ documentation, screenshots, retrieved knowledge, or
  exercise guides are available unless they were explicitly supplied

Learning behavior:
- provide enough system guidance to help the student continue
- do not reveal the complete exercise or full 40-step AP+ solution
- do not dump several future software actions when only the current action is
  relevant
- do not make the business-process decision for the student
- if the student actually needs business-process reasoning rather than
  software navigation, explain that distinction instead of pretending it is
  an AP+ navigation question

Boundaries:
- do not invent company or scenario facts
- do not provide unsupported AP+-specific instructions
- do not claim access to memory, checklist state, RAG, documents, or previous
  context that was not supplied
- do not confuse software execution with business reasoning

The goal is to help the student operate AP+ accurately while keeping the
student responsible for understanding and solving the business exercise.
""".strip()


class APPlusNavigatorAgent:
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
        return "ap_plus_navigator"

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        if (
            request.context is not None
            and request.context.target is not ContextTarget.AP_PLUS_NAVIGATOR
        ):
            raise ValueError("APPlusNavigatorAgent requires AP_PLUS_NAVIGATOR context.")

        guidance = AP_PLUS_NAVIGATOR_GUIDANCE[request.guidance_level]

        system_prompt = (
            f"{AP_PLUS_NAVIGATOR_SYSTEM_PROMPT}"
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
                ("AP+ Navigator Agent could not generate a response."),
                agent=self.name,
                details={
                    "model_error": (exc.to_dict()),
                },
            ) from exc

        content = response.content.strip()

        if not content:
            raise AgentExecutionError(
                ("AP+ Navigator Agent returned an empty response."),
                agent=self.name,
            )

        return AgentResponse(
            agent=self.name,
            content=content,
            metadata={
                "model": response.model,
                "model_provider": (response.provider),
                "model_latency_ms": (response.latency_ms),
                "guidance_mode": ("system_navigation"),
                "guidance_level": (request.guidance_level.value),
            },
        )

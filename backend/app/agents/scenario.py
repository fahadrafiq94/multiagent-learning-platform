from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from pydantic import ValidationError

from app.agents.context_prompt import (
    context_view_to_prompt_text,
)
from app.agents.exceptions import (
    AgentExecutionError,
)
from app.agents.policies import (
    SCENARIO_GUIDANCE,
)
from app.agents.scenario_schemas import (
    ScenarioDraftExtraction,
    ScenarioReviewDecision,
)
from app.agents.schemas import (
    AgentName,
    AgentRequest,
    AgentResponse,
)
from app.context.agent_views import (
    AgentContextView,
    ContextTarget,
)
from app.context.schemas import (
    ScenarioDraft,
    ScenarioDraftFact,
    ScenarioFactSource,
    ScenarioSetupStatus,
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

SCENARIO_AGENT_SYSTEM_PROMPT = """
You are the Scenario Agent in FREDi, an educational AI learning system.

Your responsibility is to help the user establish and clarify the
business scenario they are working with.

Focus on:
- the company and its relevant characteristics
- the user's role in the scenario
- the business situation or business problem
- important scenario facts already provided
- important information that is still missing
- assumptions that need to be clarified before continuing

Interaction behavior:
- be concise and educational
- use information provided by the user without inventing facts
- clearly distinguish known information from information that still
  needs clarification
- do not treat model assumptions as confirmed scenario information

Boundaries:
- do not provide AP+ menu, screen, button, or field instructions
- do not provide a complete step-by-step business process solution
- do not pretend that missing company information is known
- do not claim access to information that was not provided

The goal is to establish a clear and usable scenario, not to solve the
entire exercise for the user.
""".strip()


SCENARIO_EXTRACTION_SYSTEM_PROMPT = """
You extract structured business-scenario information for FREDi.

Your task is extraction only.

Use ONLY information explicitly stated in the CURRENT USER MESSAGE.

Rules:

1. Never invent, infer, assume, or complete missing scenario facts.

2. Do not decide whether the scenario is complete.

3. Do not decide whether the scenario should be confirmed.

4. Do not create identifiers, timestamps, lifecycle states, or
   provenance metadata.

5. For company_context, learner_role, and business_problem:
   - return the value only when the current message explicitly
     provides or corrects it;
   - otherwise return null.

6. For business_goals, constraints, and proposed_facts:
   - return only NEW items explicitly stated in the current message;
   - do not copy items merely because they already exist in the
     existing draft;
   - use an empty list or null when there are no new items.

7. proposed_facts should contain concrete scenario facts that are
   useful later and are not simply repetitions of the main role,
   business problem, or goals.

8. Existing draft information is supplied only so you can avoid
   unnecessarily repeating already-collected information.

9. If the current user message explicitly corrects an existing scalar
   value, extract the corrected value.

10. Return only data that conforms to the supplied JSON schema.

Do not write conversational text.
""".strip()

SCENARIO_REVIEW_SYSTEM_PROMPT = """
You assess a user's intention while they are reviewing a proposed
ScenarioDraft in FREDi.

Your task is semantic interpretation only.

The user has already been shown a structured scenario draft and is now
responding to it.

Classify the CURRENT USER MESSAGE into exactly one intent:

CONFIRM
- The user clearly accepts the current draft as correct as-is.
- There must be no correction, disagreement, exception, or requested
  change in the message.

CORRECT
- The user provides new, corrected, replaced, or additional scenario
  information.
- Extract only the information explicitly provided in the current
  message into the correction object.

REJECT
- The user says or implies that the current draft is incorrect or
  should not be accepted, but does not provide enough concrete
  scenario information to update it.

UNCLEAR
- The user's intention cannot safely be understood as confirmation,
  correction, or rejection.
- Use this for questions, ambiguous responses, unrelated messages,
  or uncertainty.

Important semantic rules:

- Do not classify based only on individual words such as "yes",
  "confirm", "correct", or "no".
- Interpret the meaning of the complete message.
- If a message contains both apparent agreement and a correction,
  classify it as CORRECT.
- If the user accepts only part of the scenario but rejects or changes
  another part, classify it as CORRECT or REJECT, never CONFIRM.
- CONFIRM means the user accepts the reviewed draft without changes.
- Never infer missing scenario information.
- Never decide whether the scenario should be persisted.
- Return only structured data matching the supplied JSON schema.

Examples:

User:
"Yes, that's correct."
Intent:
CONFIRM

User:
"Yes, but my role is actually Buyer."
Intent:
CORRECT

User:
"I confirm that this is not correct."
Intent:
REJECT

User:
"No, this is wrong."
Intent:
REJECT

User:
"No, my role is Senior Production Planner."
Intent:
CORRECT

User:
"The company is correct, but the business problem should be delivery
delays caused by material shortages."
Intent:
CORRECT

User:
"I'm not sure what you mean by business problem."
Intent:
UNCLEAR

User:
"Can you explain this before I confirm?"
Intent:
UNCLEAR
""".strip()


class ScenarioAgent:
    EXTRACTION_POLICY_VERSION = "scenario-extraction-v1"

    REVIEW_POLICY_VERSION = "scenario-review-v1"

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
        return "scenario"

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        if request.context is not None and request.context.target is not ContextTarget.SCENARIO:
            raise ValueError("ScenarioAgent requires SCENARIO context.")

        guidance_instruction = SCENARIO_GUIDANCE[request.guidance_level]

        system_prompt = (
            f"{SCENARIO_AGENT_SYSTEM_PROMPT}"
            "\n\n"
            "CURRENT GUIDANCE POLICY:\n"
            f"Guidance level: "
            f"{request.guidance_level.value}\n"
            f"{guidance_instruction}"
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
            model_response = await self._model_service.generate(model_request)
        except ModelServiceError as exc:
            raise AgentExecutionError(
                ("Scenario Agent could not generate a response."),
                agent=self.name,
                details={
                    "model_error": (exc.to_dict()),
                },
            ) from exc

        content = model_response.content.strip()

        if not content:
            raise AgentExecutionError(
                ("Scenario Agent returned an empty response."),
                agent=self.name,
            )

        return AgentResponse(
            agent=self.name,
            content=content,
            metadata={
                "model": (model_response.model),
                "model_provider": (model_response.provider),
                "model_latency_ms": (model_response.latency_ms),
                "guidance_mode": ("scenario_clarification"),
                "guidance_level": (request.guidance_level.value),
            },
        )

    async def extract_draft(
        self,
        *,
        context: AgentContextView,
        source_message_id: (UUID | None) = None,
    ) -> ScenarioDraft:
        if context.target is not ContextTarget.SCENARIO:
            raise ValueError("ScenarioAgent.extract_draft requires scenario context.")

        request = self._build_extraction_request(context=context)

        try:
            response = await self._model_service.generate(request)
        except ModelServiceError as exc:
            raise AgentExecutionError(
                ("Scenario Agent could not extract scenario information."),
                agent=self.name,
                details={
                    "model_error": (exc.to_dict()),
                },
            ) from exc

        content = response.content.strip()

        if not content:
            raise AgentExecutionError(
                ("Scenario Agent returned an empty structured extraction."),
                agent=self.name,
            )

        try:
            extraction = ScenarioDraftExtraction.model_validate_json(content)
        except ValidationError as exc:
            raise AgentExecutionError(
                ("Scenario Agent returned invalid structured scenario information."),
                agent=self.name,
                details={
                    "validation_errors": (exc.errors()),
                },
            ) from exc

        return self.merge_extraction(
            context=context,
            extraction=extraction,
            source_message_id=(source_message_id),
        )

    async def assess_review(
        self,
        *,
        context: AgentContextView,
    ) -> ScenarioReviewDecision:
        if context.target is not ContextTarget.SCENARIO:
            raise ValueError("ScenarioAgent.assess_review requires scenario context.")

        if context.scenario_setup is None:
            raise ValueError("Scenario review requires scenario_setup.")

        if context.scenario_setup.status != ScenarioSetupStatus.REVIEWING:
            raise ValueError("ScenarioAgent.assess_review requires REVIEWING status.")

        draft = context.scenario_setup.draft

        if draft is None:
            raise ValueError("Scenario review requires an existing ScenarioDraft.")

        user_content = (
            "CURRENT SCENARIO DRAFT:\n"
            f"{json.dumps(self._draft_business_payload(draft), ensure_ascii=False, indent=2)}"
            "\n\nCURRENT USER MESSAGE:\n"
            f"{context.current_message}"
        )

        request = ModelChatRequest(
            messages=[
                ModelMessage(
                    role="system",
                    content=(SCENARIO_REVIEW_SYSTEM_PROMPT),
                ),
                ModelMessage(
                    role="user",
                    content=user_content,
                ),
            ],
            model=self._model,
            temperature=0.0,
            response_schema=(ScenarioReviewDecision.model_json_schema()),
            metadata={
                "task": ("scenario_review_intent"),
                "policy_version": (self.REVIEW_POLICY_VERSION),
            },
        )

        try:
            response = await self._model_service.generate(request)
        except ModelServiceError as exc:
            raise AgentExecutionError(
                ("Scenario Agent could not assess the scenario review."),
                agent=self.name,
                details={
                    "model_error": (exc.to_dict()),
                },
            ) from exc

        content = response.content.strip()

        if not content:
            raise AgentExecutionError(
                ("Scenario Agent returned an empty scenario review assessment."),
                agent=self.name,
            )

        try:
            return ScenarioReviewDecision.model_validate_json(content)
        except ValidationError as exc:
            raise AgentExecutionError(
                ("Scenario Agent returned an invalid scenario review assessment."),
                agent=self.name,
                details={
                    "validation_errors": (exc.errors()),
                },
            ) from exc

    def _build_extraction_request(
        self,
        *,
        context: AgentContextView,
    ) -> ModelChatRequest:
        existing_draft = (
            context.scenario_setup.draft if context.scenario_setup is not None else None
        )

        draft_json = json.dumps(
            self._draft_business_payload(existing_draft),
            ensure_ascii=False,
            indent=2,
        )
        user_content = (
            "EXISTING SCENARIO DRAFT:\n"
            f"{draft_json}\n\n"
            "CURRENT USER MESSAGE:\n"
            f"{context.current_message}"
        )

        return ModelChatRequest(
            messages=[
                ModelMessage(
                    role="system",
                    content=(SCENARIO_EXTRACTION_SYSTEM_PROMPT),
                ),
                ModelMessage(
                    role="user",
                    content=user_content,
                ),
            ],
            model=self._model,
            temperature=0.0,
            response_schema=(ScenarioDraftExtraction.model_json_schema()),
            metadata={
                "task": ("scenario_draft_extraction"),
                "policy_version": (self.EXTRACTION_POLICY_VERSION),
            },
        )

    @staticmethod
    def _draft_business_payload(
        draft: ScenarioDraft | None,
    ) -> dict[str, object]:
        if draft is None:
            return {
                "company_context": None,
                "learner_role": None,
                "business_problem": None,
                "business_goals": [],
                "constraints": [],
                "proposed_facts": [],
            }

        return {
            "company_context": (draft.company_context),
            "learner_role": (draft.learner_role),
            "business_problem": (draft.business_problem),
            "business_goals": list(draft.business_goals),
            "constraints": list(draft.constraints),
            "proposed_facts": [fact.fact for fact in draft.proposed_facts],
        }

    def merge_extraction(
        self,
        *,
        context: AgentContextView,
        extraction: ScenarioDraftExtraction,
        source_message_id: (UUID | None) = None,
    ) -> ScenarioDraft:
        existing = context.scenario_setup.draft if context.scenario_setup is not None else None

        if existing is None:
            draft = ScenarioDraft(
                user_id=context.user_id,
                session_id=(context.session_id),
            )
        else:
            self._validate_existing_draft(
                context=context,
                draft=existing,
            )

            draft = existing.model_copy(deep=True)

        if extraction.company_context is not None:
            draft.company_context = extraction.company_context

        if extraction.learner_role is not None:
            draft.learner_role = extraction.learner_role

        if extraction.business_problem is not None:
            draft.business_problem = extraction.business_problem

        if extraction.business_goals is not None:
            draft.business_goals = self._merge_unique_strings(
                current=(draft.business_goals),
                incoming=(extraction.business_goals),
            )

        if extraction.constraints is not None:
            draft.constraints = self._merge_unique_strings(
                current=(draft.constraints),
                incoming=(extraction.constraints),
            )

        if extraction.proposed_facts is not None:
            draft.proposed_facts = self._merge_proposed_facts(
                current=(draft.proposed_facts),
                incoming=(extraction.proposed_facts),
                source_message_id=(source_message_id),
            )

        if source_message_id is not None and source_message_id not in draft.source_message_ids:
            draft.source_message_ids.append(source_message_id)

        draft.updated_at = datetime.now(UTC)

        return draft

    @staticmethod
    def _validate_existing_draft(
        *,
        context: AgentContextView,
        draft: ScenarioDraft,
    ) -> None:
        if draft.user_id != context.user_id:
            raise ValueError("ScenarioDraft user_id does not match context.")

        if draft.session_id != context.session_id:
            raise ValueError("ScenarioDraft session_id does not match context.")

    @staticmethod
    def _merge_unique_strings(
        *,
        current: list[str],
        incoming: list[str],
    ) -> list[str]:
        merged = list(current)

        seen = {ScenarioAgent._comparison_key(item) for item in current}

        for item in incoming:
            key = ScenarioAgent._comparison_key(item)

            if key in seen:
                continue

            seen.add(key)
            merged.append(item)

        return merged

    @staticmethod
    def _merge_proposed_facts(
        *,
        current: list[ScenarioDraftFact],
        incoming: list[str],
        source_message_id: (UUID | None),
    ) -> list[ScenarioDraftFact]:
        result = [fact.model_copy(deep=True) for fact in current]

        seen = {ScenarioAgent._comparison_key(fact.fact) for fact in current}

        source_reference = str(source_message_id) if source_message_id is not None else None

        for fact_text in incoming:
            key = ScenarioAgent._comparison_key(fact_text)

            if key in seen:
                continue

            seen.add(key)

            result.append(
                ScenarioDraftFact(
                    fact=fact_text,
                    source_type=(ScenarioFactSource.USER),
                    source_reference=(source_reference),
                )
            )

        return result

    @staticmethod
    def _comparison_key(
        value: str,
    ) -> str:
        return " ".join(value.casefold().split())

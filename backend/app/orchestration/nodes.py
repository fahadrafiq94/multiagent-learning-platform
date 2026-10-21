from __future__ import annotations

from app.agents.exceptions import (
    AgentError,
)
from app.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorRoutingError,
)
from app.agents.registry import (
    AgentRegistry,
)
from app.agents.routing import (
    OrchestratorRequest,
)
from app.agents.schemas import (
    AgentName,
    AgentRequest,
)
from app.context.agent_views import (
    ContextTarget,
)
from app.context.policies import (
    AgentContextPolicyService,
)
from app.context.schemas import (
    ScenarioDraft,
    ScenarioRequiredField,
)
from app.domain.guidance import (
    GuidanceLevel,
)
from app.orchestration.router import (
    classify_route,
)
from app.orchestration.state import (
    OrchestrationState,
)
from app.repositories.exceptions import (
    RepositoryError,
)
from app.services.context_service import (
    ContextService,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.scenario_onboarding_interaction_schemas import (
    ScenarioInteractionOutcome,
    ScenarioOnboardingTurnResult,
)
from app.services.scenario_onboarding_interaction_service import (
    ScenarioOnboardingInteractionService,
)
from app.services.scenario_onboarding_service import (
    ScenarioOnboardingService,
)
from app.services.session_state_update_service import (
    SessionStateUpdateService,
)


def prepare_input_node(
    state: OrchestrationState,
) -> OrchestrationState:
    user_message = state.get(
        "user_message",
        "",
    ).strip()

    metadata = dict(
        state.get(
            "metadata",
            {},
        )
    )

    if not user_message:
        return {
            **state,
            "user_message": "",
            "error": ("User message cannot be empty."),
            "metadata": {
                **metadata,
                "prepared": False,
            },
        }

    return {
        **state,
        "user_message": user_message,
        "error": None,
        "metadata": {
            **metadata,
            "prepared": True,
        },
    }


async def load_context_node(
    state: OrchestrationState,
    *,
    conversation_service: (ConversationService),
    context_service: ContextService,
    onboarding_service: (ScenarioOnboardingService),
) -> OrchestrationState:
    if state.get("error"):
        return state

    metadata = dict(
        state.get(
            "metadata",
            {},
        )
    )

    user_id = state.get("user_id")

    session_id = state.get("session_uuid")

    if user_id is None or session_id is None:
        return {
            **state,
            "error": ("Internal request identity is missing."),
            "metadata": {
                **metadata,
                "context_loaded": False,
            },
        }

    try:
        current_message = await conversation_service.append_user_message(
            user_id=user_id,
            session_id=session_id,
            content=(state["user_message"]),
        )

        context = await context_service.build(
            user_id=user_id,
            session_id=session_id,
            current_message=(state["user_message"]),
            trace_id=state.get("trace_id"),
            current_message_id=(current_message.message_id),
        )

        onboarding_required = await onboarding_service.needs_onboarding(user_id=user_id)

    except (
        RepositoryError,
        ValueError,
    ) as exc:
        return {
            **state,
            "error": str(exc),
            "metadata": {
                **metadata,
                "context_loaded": False,
            },
        }

    return {
        **state,
        "current_message_id": (current_message.message_id),
        "base_context": context,
        "onboarding_required": (onboarding_required),
        "guidance_level": (context.guidance_level),
        "metadata": {
            **metadata,
            "context_loaded": True,
            "onboarding_required": (onboarding_required),
            "recent_message_count": (len(context.recent_messages)),
        },
    }


async def scenario_onboarding_node(
    state: OrchestrationState,
    *,
    context_policy_service: (AgentContextPolicyService),
    interaction_service: (ScenarioOnboardingInteractionService),
) -> OrchestrationState:
    if state.get("error"):
        return state

    metadata = dict(
        state.get(
            "metadata",
            {},
        )
    )

    base_context = state.get("base_context")

    if base_context is None:
        return {
            **state,
            "error": ("Base context is missing."),
        }

    scenario_context = context_policy_service.select(
        target=(ContextTarget.SCENARIO),
        context=base_context,
    )

    try:
        result = await interaction_service.handle_turn(
            context=scenario_context,
            source_message_id=(state.get("current_message_id")),
        )
    except (
        AgentError,
        ValueError,
        RuntimeError,
    ) as exc:
        message = (
            exc.message
            if isinstance(
                exc,
                AgentError,
            )
            else str(exc)
        )

        return {
            **state,
            "route": "scenario",
            "error": message,
            "metadata": {
                **metadata,
                "selected_agent": ("scenario"),
                "scenario_onboarding": (True),
            },
        }

    response_text = _render_onboarding_result(result)

    return {
        **state,
        "route": "scenario",
        "route_reason": ("Scenario onboarding is required before normal routing."),
        "onboarding_result": result,
        "agent_response": response_text,
        "metadata": {
            **metadata,
            "selected_agent": "scenario",
            "scenario_onboarding": True,
            "onboarding_outcome": (result.outcome.value),
            "scenario_status": (result.status.value),
            "context_target": (scenario_context.target.value),
            "context_policy_version": (scenario_context.policy_version),
        },
    }


async def route_request_node(
    state: OrchestrationState,
    *,
    orchestrator: OrchestratorAgent,
    context_policy_service: (AgentContextPolicyService),
) -> OrchestrationState:
    if state.get("error"):
        return state

    metadata = dict(
        state.get(
            "metadata",
            {},
        )
    )

    metadata["scenario_onboarding"] = False

    base_context = state.get("base_context")

    if base_context is None:
        return {
            **state,
            "error": ("Base context is missing."),
        }

    orchestrator_context = context_policy_service.select(
        target=(ContextTarget.ORCHESTRATOR),
        context=base_context,
    )

    request = OrchestratorRequest(
        session_id=state["session_id"],
        student_id=(state.get("student_id")),
        user_message=(state["user_message"]),
        context=orchestrator_context,
        metadata=metadata,
    )

    try:
        result = await orchestrator.route(request)

        decision = result.decision

        metadata.update(
            {
                "orchestrator_metadata": (dict(result.metadata)),
                "semantic_routing_failed": (False),
                "context_target": (orchestrator_context.target.value),
            }
        )

        if decision.route != "fallback":
            return {
                **state,
                "route": decision.route,
                "route_reason": (decision.reason),
                "metadata": {
                    **metadata,
                    "routing_completed": (True),
                    "routing_strategy": ("llm_semantic_router_v2"),
                    "keyword_fallback_used": (False),
                },
            }

        keyword = classify_route(state["user_message"])

        return {
            **state,
            "route": keyword.route,
            "route_reason": (keyword.reason if keyword.route != "fallback" else decision.reason),
            "metadata": {
                **metadata,
                "routing_completed": True,
                "routing_strategy": ("llm_semantic_router_v2"),
                "keyword_fallback_used": (True),
                "keyword_fallback_trigger": ("semantic_fallback"),
                "keyword_route": (keyword.route),
            },
        }

    except OrchestratorRoutingError as exc:
        keyword = classify_route(state["user_message"])

        return {
            **state,
            "route": keyword.route,
            "route_reason": (keyword.reason),
            "metadata": {
                **metadata,
                "routing_completed": True,
                "routing_strategy": ("deterministic_fallback_v1"),
                "semantic_routing_failed": (True),
                "orchestrator_error": (exc.to_dict()),
                "keyword_fallback_used": (True),
                "keyword_fallback_trigger": ("orchestrator_error"),
                "keyword_route": (keyword.route),
            },
        }


async def execute_agent_node(
    state: OrchestrationState,
    *,
    agent_registry: AgentRegistry,
    agent_name: AgentName,
    context_policy_service: (AgentContextPolicyService),
) -> OrchestrationState:
    if state.get("error"):
        return state

    base_context = state.get("base_context")

    if base_context is None:
        return {
            **state,
            "error": ("Base context is missing."),
        }

    targets: dict[
        AgentName,
        ContextTarget,
    ] = {
        "scenario": (ContextTarget.SCENARIO),
        "process_coach": (ContextTarget.PROCESS_COACH),
        "ap_plus_navigator": (ContextTarget.AP_PLUS_NAVIGATOR),
    }

    view = context_policy_service.select(
        target=targets[agent_name],
        context=base_context,
    )

    metadata = dict(
        state.get(
            "metadata",
            {},
        )
    )

    request = AgentRequest(
        session_id=state["session_id"],
        student_id=(state.get("student_id")),
        user_message=(state["user_message"]),
        guidance_level=(
            view.guidance_level or base_context.guidance_level or GuidanceLevel.MINIMAL
        ),
        context=view,
        metadata=metadata,
    )

    try:
        agent = agent_registry.get(agent_name)

        response = await agent.execute(request)

    except AgentError as exc:
        return {
            **state,
            "error": exc.message,
            "metadata": {
                **metadata,
                "selected_agent": (agent_name),
                "agent_error": (exc.to_dict()),
            },
        }

    return {
        **state,
        "selected_agent": (response.agent),
        "agent_response": (response.content),
        "state_update_proposal": (response.state_update_proposal),
        "metadata": {
            **metadata,
            "selected_agent": (response.agent),
            "agent_called": True,
            "agent_metadata": (dict(response.metadata)),
            "context_target": (view.target.value),
            "context_policy_version": (view.policy_version),
            "state_update_proposed": (
                bool(response.state_update_proposal and response.state_update_proposal.operations)
            ),
        },
    }


def fallback_response_node(
    state: OrchestrationState,
) -> OrchestrationState:
    if state.get("error"):
        return state

    metadata = dict(
        state.get(
            "metadata",
            {},
        )
    )

    return {
        **state,
        "agent_response": (
            "I'm not sure which type of "
            "help you need yet. Are you "
            "asking about your business "
            "scenario, business-process "
            "reasoning, or AP+?"
        ),
        "metadata": {
            **metadata,
            "selected_agent": None,
            "fallback_used": True,
        },
    }


async def finalize_response_node(
    state: OrchestrationState,
    *,
    conversation_service: (ConversationService),
) -> OrchestrationState:
    metadata = dict(
        state.get(
            "metadata",
            {},
        )
    )

    if state.get("error"):
        final_response = f"I could not process the message because: {state['error']}"
    else:
        final_response = state.get("agent_response") or state.get("model_response") or ""

    if not final_response:
        final_response = "I could not produce a response for this request."

    if (
        state.get("current_message_id") is not None
        and state.get("user_id") is not None
        and state.get("session_uuid") is not None
    ):
        try:
            await conversation_service.append_assistant_message(
                user_id=state["user_id"],
                session_id=(state["session_uuid"]),
                content=final_response,
            )

            metadata["assistant_message_persisted"] = True

        except RepositoryError as exc:
            metadata["assistant_message_persisted"] = False

            metadata["assistant_persistence_error"] = str(exc)

    return {
        **state,
        "final_response": final_response,
        "metadata": {
            **metadata,
            "finalized": True,
        },
    }


async def apply_state_updates_node(
    state: OrchestrationState,
    *,
    session_state_update_service: (SessionStateUpdateService),
) -> OrchestrationState:
    """
    Validate and persist specialist-proposed runtime state.

    State update failure must not discard an otherwise valid
    specialist response.
    """

    if state.get("error"):
        return state

    user_id = state.get("user_id")

    session_id = state.get("session_uuid")

    agent_name = state.get("selected_agent")

    if user_id is None or session_id is None or agent_name is None:
        return state

    metadata = dict(
        state.get(
            "metadata",
            {},
        )
    )

    try:
        result = await session_state_update_service.apply(
            user_id=user_id,
            session_id=session_id,
            agent_name=agent_name,
            proposal=(state.get("state_update_proposal")),
        )

    except Exception as exc:
        return {
            **state,
            "metadata": {
                **metadata,
                "state_update_applied": (False),
                "state_update_error": (str(exc)),
            },
        }

    return {
        **state,
        "metadata": {
            **metadata,
            "state_update_applied": (result.applied),
            "state_update_accepted_count": (len(result.accepted_operations)),
            "state_update_rejected_count": (len(result.rejected_operations)),
            "session_context_version": (result.session_context_version),
            "state_update_rejections": [
                {
                    "field": (item.operation.field.value),
                    "action": (item.operation.action.value),
                    "reason": item.reason,
                }
                for item in result.rejected_operations
            ],
        },
    }


def _render_onboarding_result(
    result: ScenarioOnboardingTurnResult,
) -> str:
    if result.outcome == ScenarioInteractionOutcome.COLLECTING:
        if result.assessment is None:
            return "I need some more information about your scenario."

        labels = [
            _required_field_label(field) for field in (result.assessment.missing_required_fields)
        ]

        return (
            "I still need the following "
            "scenario information: "
            f"{', '.join(labels)}. "
            "Please provide the missing "
            "information together if you can."
        )

    if result.outcome in {
        ScenarioInteractionOutcome.REVIEW_REQUIRED,
        ScenarioInteractionOutcome.CORRECTED,
    }:
        if result.draft is None:
            return "Please review the scenario information before confirming it."

        prefix = (
            "I updated the scenario."
            if result.outcome == ScenarioInteractionOutcome.CORRECTED
            else "I understood your scenario."
        )

        return (
            f"{prefix}\n\n"
            f"{_render_draft(result.draft)}"
            "\n\nIs this correct? You can "
            "confirm it or tell me what "
            "should be changed."
        )

    if result.outcome == ScenarioInteractionOutcome.CORRECTION_REQUIRED:
        return (
            "I won't confirm the scenario "
            "yet. Please tell me which "
            "part is incorrect and what it "
            "should be instead."
        )

    if result.outcome == ScenarioInteractionOutcome.CLARIFICATION_REQUIRED:
        return (
            "I couldn't safely determine "
            "whether you were confirming "
            "the scenario or changing it. "
            "Please tell me whether it is "
            "correct or what should change."
        )

    if result.outcome == ScenarioInteractionOutcome.CONFIRMED:
        return "Your scenario is confirmed. We can now continue with the normal learning workflow."

    return "Your scenario is already confirmed."


def _render_draft(
    draft: ScenarioDraft,
) -> str:
    lines = [
        (f"Company context: {draft.company_context or 'Not provided'}"),
        (f"Role: {draft.learner_role or 'Not provided'}"),
        (f"Business problem: {draft.business_problem or 'Not provided'}"),
    ]

    if draft.business_goals:
        lines.append("Business goals: " + "; ".join(draft.business_goals))

    if draft.constraints:
        lines.append("Constraints: " + "; ".join(draft.constraints))

    if draft.proposed_facts:
        lines.append("Additional facts: " + "; ".join(fact.fact for fact in draft.proposed_facts))

    return "\n".join(lines)


def _required_field_label(
    field: ScenarioRequiredField,
) -> str:
    labels = {
        ScenarioRequiredField.COMPANY_CONTEXT: ("company context"),
        ScenarioRequiredField.LEARNER_ROLE: ("your scenario role"),
        ScenarioRequiredField.BUSINESS_PROBLEM: ("business problem"),
        ScenarioRequiredField.BUSINESS_GOALS: ("at least one business goal"),
    }

    return labels[field]

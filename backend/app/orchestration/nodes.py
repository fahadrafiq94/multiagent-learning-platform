from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.agents.assessment_guard import guarded_assessment
from app.agents.exceptions import (
    AgentError,
)
from app.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorRoutingError,
)
from app.agents.policies import (
    next_guidance_level,
)
from app.agents.registry import (
    AgentRegistry,
)
from app.agents.routing import (
    OrchestratorRequest,
    RoutingDecision,
)
from app.agents.schemas import (
    AgentName,
    AgentRequest,
)
from app.context.agent_views import (
    ContextTarget,
)
from app.context.grounding import GroundingContext
from app.context.policies import (
    AgentContextPolicyService,
)
from app.context.schemas import (
    ScenarioDraft,
    ScenarioRequiredField,
)
from app.context.state_updates import (
    SessionStateAction,
    SessionStateField,
)
from app.domain.guidance import (
    GuidanceLevel,
)
from app.domain.guidance_policy import (
    assessment_signal,
    grounded_abstention_signal,
    select_guidance,
)
from app.knowledge.integration import GroundingAdmission, SpecialistGroundingService
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
from app.services.conversation_reply import conversation_reply
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
from app.services.session_summary_service import (
    SessionSummaryService,
)
from app.services.user_context_service import (
    UserContextService,
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
    metadata = {
        key: value
        for key, value in metadata.items()
        if not key.startswith(
            (
                "student_step_",
                "scenario_role_",
                "interaction_",
                "context_reply_",
                "assessment_",
                "guidance_",
                "teaching_",
            )
        )
        and key != "scenario_source_recorded"
        and key != "active_blocker_after"
    }

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
        current_message_id = state.get("current_message_id")
        if not state.get("turn_managed"):
            current_message = await conversation_service.append_user_message(
                user_id=user_id,
                session_id=session_id,
                content=(state["user_message"]),
                trace_id=(state.get("trace_id")),
            )
            current_message_id = current_message.message_id
        if current_message_id is None:
            raise ValueError("Admitted turn message is missing.")

        context = await context_service.build(
            user_id=user_id,
            session_id=session_id,
            current_message=(state["user_message"]),
            trace_id=state.get("trace_id"),
            current_message_id=current_message_id,
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
        "current_message_id": current_message_id,
        "base_context": context,
        "onboarding_required": (onboarding_required),
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
    saved_draft = result.draft
    saved_role = (
        saved_draft.learner_role
        if saved_draft
        else result.scenario_card.learner_role
        if result.scenario_card
        else None
    )

    return {
        **state,
        "route": "scenario",
        "route_reason": ("Scenario onboarding is required before normal routing."),
        "agent_response": response_text,
        "metadata": {
            **metadata,
            "selected_agent": "scenario",
            "scenario_onboarding": True,
            "onboarding_outcome": (result.outcome.value),
            "scenario_status": (result.status.value),
            "scenario_role_status": "persisted" if saved_role else "missing",
            "scenario_source_recorded": bool(
                saved_draft and state.get("current_message_id") in saved_draft.source_message_ids
            ),
            "context_target": (scenario_context.target.value),
            "context_policy_version": (scenario_context.policy_version),
        },
    }


async def _apply_student_step_before_specialist(
    state: OrchestrationState,
    decision: RoutingDecision,
    *,
    service: SessionStateUpdateService | None,
    context_service: ContextService | None,
) -> OrchestrationState:
    metadata = dict(state.get("metadata", {}))
    metadata["student_step_guard"] = True
    step = decision.current_step
    if step is None:
        metadata["student_step_status"] = (
            "rejected" if "current_step" in decision.model_fields_set else "omitted"
        )
        return {**state, "metadata": metadata}
    metadata["student_step_proposed"] = step.action == "set"
    base = state.get("base_context")
    if service is None or context_service is None or base is None or base.session_context is None:
        metadata["student_step_status"] = "unavailable"
        return {**state, "metadata": metadata}
    try:
        status, _ = await service.apply_student_step(
            user_id=base.user_id,
            session_id=base.session_id,
            message_id=state.get("current_message_id"),
            current_message=state["user_message"],
            decision=step,
            expected_version=base.session_context.version,
        )
        metadata["student_step_status"] = status
        metadata["student_step_applied"] = status == "persisted"
        if status == "persisted":
            metadata["student_step_source_message_id"] = str(state["current_message_id"])
            # Rebuild through the ordinary budget/context service. Grounding and
            # specialist views must see the committed step and a fresh version.
            base = await context_service.build(
                user_id=base.user_id,
                session_id=base.session_id,
                current_message=state["user_message"],
                trace_id=state.get("trace_id"),
                current_message_id=state.get("current_message_id"),
            )
            metadata["student_step_context_refreshed"] = True
        if status in {"rejected", "clarification_required"}:
            metadata["student_step_clarification"] = True
    except (RepositoryError, ValueError):
        metadata["student_step_status"] = (
            "persisted" if metadata.get("student_step_applied") else "failed"
        )
        return {
            **state,
            "error": "Student state could not be saved; please retry.",
            "route": "fallback",
            "metadata": metadata,
        }
    return {**state, "base_context": base, "metadata": metadata}


async def route_request_node(
    state: OrchestrationState,
    *,
    orchestrator: OrchestratorAgent,
    context_policy_service: (AgentContextPolicyService),
    session_state_update_service: SessionStateUpdateService | None = None,
    context_service: ContextService | None = None,
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
    metadata["student_step_guard"] = session_state_update_service is not None
    metadata["student_step_status"] = "unavailable"

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
                key: result.metadata[key]
                for key in (
                    "student_step_verification_status",
                    "student_step_verification_calls",
                    "student_step_verification_latency_ms",
                )
                if key in result.metadata
            }
        )

        if session_state_update_service is not None:
            state = await _apply_student_step_before_specialist(
                {**state, "metadata": metadata},
                decision,
                service=session_state_update_service,
                context_service=context_service,
            )
            metadata = dict(state.get("metadata", {}))
            if state.get("error"):
                return state
            if metadata.get("student_step_clarification"):
                return {
                    **state,
                    "route": "fallback",
                    "route_reason": "The current-step statement needs clarification.",
                }

        metadata.update(
            {
                "orchestrator_metadata": (dict(result.metadata)),
                "semantic_routing_failed": (False),
                "context_target": (orchestrator_context.target.value),
            }
        )

        purpose = decision.interaction
        metadata["interaction_kind"] = purpose.kind if purpose else None
        metadata["interaction_status"] = (
            "normal_workflow"
            if purpose
            else "rejected"
            if "interaction" in decision.model_fields_set
            else "omitted"
        )
        # Explicit step changes/ambiguity take precedence over a contradictory
        # social/recall classification. Mixed substantive requests never shortcut.
        if purpose and not (decision.current_step and decision.current_step.action != "unchanged"):
            reply = conversation_reply(purpose, state.get("base_context") or base_context)
            if reply is not None:
                return {
                    **state,
                    "route": "fallback",
                    "route_reason": "Pure conversation/context reply.",
                    "agent_response": reply,
                    "source_references": [],
                    "metadata": {
                        **metadata,
                        "context_reply_handled": True,
                        "interaction_status": "direct_reply",
                        "routing_completed": True,
                        "routing_strategy": "llm_semantic_router_v2",
                        "keyword_fallback_used": False,
                        "agent_called": False,
                        "response_assessed": False,
                        "response_signal_recorded": False,
                        "retrieval_searched": False,
                        "rag": {
                            "status": "skipped",
                            "reason": "owned_context_reply",
                            "planner_calls": 0,
                            "embedding_calls": 0,
                            "actions": [],
                        },
                    },
                }

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
    grounding_service: SpecialistGroundingService | None = None,
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

    admission: GroundingAdmission | None = None
    if grounding_service is not None:
        try:
            admission = await grounding_service.prepare(
                base_context,
                agent_name,
                current_message_id=state.get("current_message_id"),
            )
        except Exception:
            admission = GroundingAdmission(
                GroundingContext(status="error"),
                {"status": "error", "reason": "grounding_preparation_failed"},
            )
        metadata["rag"] = admission.trace
        if admission.clarification:
            return {
                **state,
                "selected_agent": agent_name,
                "agent_response": admission.clarification,
                "state_update_proposal": None,
                "response_assessment": None,
                "source_references": [],
                "metadata": {
                    **metadata,
                    "selected_agent": agent_name,
                    "agent_called": False,
                    "response_assessed": False,
                },
            }
        view = view.model_copy(deep=True)
        view.grounding = admission.context

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

    called = False
    try:
        if admission is not None and grounding_service is not None:
            await grounding_service.authorize(admission)
        agent = agent_registry.get(agent_name)

        called = True
        response = await agent.execute(request)
        if admission is not None and grounding_service is not None:
            await grounding_service.authorize(admission)

    except AgentError as exc:
        return {
            **state,
            "error": (
                "Grounded specialist response could not be generated."
                if admission is not None
                else exc.message
            ),
            "metadata": {
                **metadata,
                "selected_agent": (agent_name),
                "agent_error": (
                    {"error": "grounded_specialist_failed"}
                    if admission is not None
                    else exc.to_dict()
                ),
            },
        }

    except (ValueError, RuntimeError) as exc:
        # Context-target guards and repository errors must degrade
        # into a clean error state, not an unhandled graph exception
        # after the user message was already persisted.
        # RepositoryError is a RuntimeError subclass.
        if admission is not None:
            return {
                **state,
                "selected_agent": agent_name,
                "agent_response": (
                    "I couldn't verify the course context for this request. Please try again."
                ),
                "source_references": [],
                "state_update_proposal": None,
                "response_assessment": None,
                "metadata": {
                    **metadata,
                    "selected_agent": agent_name,
                    "agent_called": called,
                    "response_assessed": False,
                    "rag": {"status": "unavailable", "reason": "grounding_unavailable"},
                },
            }
        return {
            **state,
            "error": str(exc),
            "metadata": {
                **metadata,
                "selected_agent": (agent_name),
            },
        }
    except Exception:
        if admission is None:
            raise
        return {
            **state,
            "error": "Grounded specialist response could not be generated.",
            "source_references": [],
            "metadata": {
                **metadata,
                "selected_agent": agent_name,
                "agent_error": {"error": "grounded_specialist_failed"},
            },
        }

    neutral = admission is not None and (
        admission.context.status in {"error", "unavailable", "no_evidence"}
        or response.metadata.get("grounding", {}).get("status")
        in {"error", "unavailable", "no_evidence"}
        or response.metadata.get("grounding", {}).get("support_abstained") is True
        or response.metadata.get("grounding", {}).get("teaching_abstained") is True
    )
    proposal = None if neutral else response.state_update_proposal
    assessment, assessment_status = guarded_assessment(
        None if neutral else response.response_assessment, state["user_message"]
    )
    if admission is not None:
        metadata["rag"] = {
            **admission.trace,
            **response.metadata.get("grounding", {}),
            "learning_signal_suppressed": neutral,
        }
        if (
            grounding_service is not None
            and admission.context.status == "found"
            and (
                metadata["rag"].get("support_abstained") is True
                or metadata["rag"].get("teaching_abstained") is True
            )
        ):
            # Evidence was retrieved but the answer was not verified, so no
            # citations were archived. Bridge the same references to the
            # student's next turn (app-owned, single-turn, best-effort).
            await grounding_service.record_unverified_receipt(
                user_id=base_context.user_id,
                session_id=base_context.session_id,
                admission=admission,
            )
        final_budget = response.metadata.get("grounding", {})
        for key in (
            "estimated_input_tokens",
            "input_token_budget",
            "estimated_essential_tokens",
            "estimated_history_tokens",
        ):
            if key in final_budget:
                metadata[key] = final_budget[key]
        if "dropped_message_count" in final_budget:
            metadata["dropped_message_count"] = (
                base_context.metadata.dropped_message_count + final_budget["dropped_message_count"]
            )
            metadata["recent_message_count"] = final_budget["retained_message_count"]
            metadata["truncation_applied"] = (
                base_context.metadata.truncation_applied
                or final_budget["dropped_message_count"] > 0
                or final_budget["budget_omissions"] > 0
            )
    return {
        **state,
        "selected_agent": (response.agent),
        "agent_response": (response.content),
        "state_update_proposal": proposal,
        "response_assessment": assessment,
        "source_references": response.source_references if admission is not None else [],
        "metadata": {
            **metadata,
            "selected_agent": (response.agent),
            "agent_called": True,
            "response_assessed": assessment is not None,
            "assessment_status": "suppressed"
            if neutral
            else response.metadata.get("assessment_status", assessment_status),
            "guidance_level_used": request.guidance_level.value,
            "guidance_level_next": request.guidance_level.value,
            "agent_metadata": (dict(response.metadata)),
            "context_target": (view.target.value),
            "context_policy_version": (view.policy_version),
            "state_update_proposed": (bool(proposal and proposal.operations)),
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
            state.get("agent_response", "")
            if metadata.get("context_reply_handled")
            else "Which step are you currently working on? Please state its identifier."
            if metadata.get("student_step_clarification")
            else "I'm not sure which type of "
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
        final_response = state.get("agent_response") or ""

    if not final_response:
        final_response = "I could not produce a response for this request."

    if (
        not state.get("turn_managed")
        and state.get("current_message_id") is not None
        and state.get("user_id") is not None
        and state.get("session_uuid") is not None
    ):
        try:
            await conversation_service.append_assistant_message(
                user_id=state["user_id"],
                session_id=(state["session_uuid"]),
                content=final_response,
                agent_name=(state.get("selected_agent")),
                route=(state.get("route")),
                trace_id=(state.get("trace_id")),
                source_references=state.get("source_references", [])
                if not state.get("error")
                else [],
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
    user_context_service: (UserContextService),
) -> OrchestrationState:
    """
    Validate and persist specialist-proposed runtime state.

    State update failure must not discard an otherwise valid
    specialist response.

    A persisted active blocker also escalates the user's guidance
    level (see _update_hint_state_after_blocker), because graduated
    support must follow observed blocking, not LLM judgment.
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
            allow_specialist_step_updates=not metadata.get("student_step_guard", False),
        )

    except Exception as exc:
        return {
            **state,
            "metadata": {
                **metadata,
                "state_update_applied": bool(metadata.get("student_step_applied", False)),
                "state_update_error": (str(exc)),
            },
        }

    blocker_operations = [
        operation
        for operation in (result.accepted_operations)
        if operation.field == SessionStateField.ACTIVE_BLOCKER
    ]

    if blocker_operations:
        blocked = any(
            operation.action == SessionStateAction.SET for operation in (blocker_operations)
        )
        metadata["active_blocker_after"] = blocked

        try:
            metadata["guidance_level_updated"] = await _update_hint_state_after_blocker(
                user_context_service=(user_context_service),
                user_id=(user_id),
                blocked=(blocked),
            )

        except Exception as exc:
            # Guidance escalation is best-effort: a hint-state write
            # failure must not replace the otherwise successful turn.
            metadata["guidance_level_error"] = str(exc)

    return {
        **state,
        "metadata": {
            **metadata,
            "state_update_applied": result.applied or metadata.get("student_step_applied", False),
            "state_update_accepted_count": (
                len(result.accepted_operations) + int(metadata.get("student_step_applied", False))
            ),
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


async def record_response_signal_node(
    state: OrchestrationState,
    *,
    user_context_service: (UserContextService),
) -> OrchestrationState:
    """
    Persist deterministic response-signal counters from the
    specialist's assessment of the student's message (4B.9).

    The raw assessment is LLM output and is never persisted. Only the
    bounded counter movements produced by assessment_signal() touch
    HintState. Signal persistence is best-effort: a failure is recorded
    in metadata and must not affect the specialist response.
    """

    if state.get("error"):
        return state

    user_id = state.get("user_id")

    session_id = state.get("session_uuid")

    if user_id is None or session_id is None:
        return state

    metadata = dict(
        state.get(
            "metadata",
            {},
        )
    )

    # Defense in depth for custom agents as well as normal specialist entry points.
    assessment, guard_status = guarded_assessment(
        state.get("response_assessment"), state.get("user_message", "")
    )
    metadata["assessment_status"] = metadata.get("assessment_status", guard_status)
    metadata["guidance_level_next"] = metadata.get(
        "guidance_level_updated", metadata.get("guidance_level_used")
    )

    if assessment is not None:
        update = assessment_signal(assessment)
        assessment_metadata = {
            **metadata,
            "response_understanding": (assessment.understanding.value),
            "response_misconception_detected": (assessment.misconception_detected),
        }
    else:
        # Suppressed LLM assessment: only the application-owned grounding outcome
        # may still move the counters (found evidence, unverified answer).
        rag = metadata.get("rag") or {}
        update = grounded_abstention_signal(
            rag_status=rag.get("status"),
            support_abstained=rag.get("support_abstained"),
            teaching_abstained=rag.get("teaching_abstained"),
        )
        assessment_metadata = {
            **metadata,
            "response_reason": (
                "unverified_grounded_abstention"
                if update.struggle_delta or update.reset_productive
                else "no_assessment"
            ),
        }

    if not (
        update.success_delta
        or update.struggle_delta
        or update.reset_struggle
        or update.reset_productive
    ):
        return {
            **state,
            "metadata": {
                **assessment_metadata,
                "response_signal_recorded": (False),
                "response_reason": ("no_assessment" if assessment is None else "no_signal"),
            },
        }

    try:
        user_context = await user_context_service.get_or_create(
            user_id=user_id,
        )

        hint_state = user_context.hint_state

        hint_state.successful_independent_steps += update.success_delta
        if update.reset_productive:
            hint_state.consecutive_productive_attempts = 0
        elif update.success_delta:
            hint_state.consecutive_productive_attempts += 1

        if update.reset_struggle:
            hint_state.consecutive_unproductive_attempts = 0

        else:
            hint_state.consecutive_unproductive_attempts += update.struggle_delta

        base_context = state.get("base_context")
        if base_context is not None and base_context.session_context is not None:
            session_context = base_context.session_context.model_copy(deep=True)
            # Only accepted blocker operations set this server-owned projection.
            if "active_blocker_after" in metadata:
                session_context.active_blocker = (
                    "Active blocker" if metadata["active_blocker_after"] else None
                )
            if session_context.active_blocker is not None or hint_state.manually_overridden:
                hint_state.consecutive_productive_attempts = 0
            if not hint_state.manually_overridden:
                used = base_context.guidance_level
                if used is not None and list(GuidanceLevel).index(used) > list(GuidanceLevel).index(
                    hint_state.guidance_level
                ):
                    hint_state.guidance_level = used
            decision = select_guidance(hint_state=hint_state, session_context=session_context)
            hint_state.guidance_level = decision.level
            hint_state.last_change_reason = decision.reason
            if decision.reason == "productive_reasoning":
                hint_state.consecutive_productive_attempts = 0

        hint_state.updated_at = datetime.now(UTC)

        await user_context_service.save(
            user_id=user_id,
            context=user_context,
            expected_version=(user_context.version),
        )
        assessment_metadata["guidance_level_next"] = hint_state.guidance_level.value

    except Exception as exc:
        return {
            **state,
            "metadata": {
                **assessment_metadata,
                "response_signal_recorded": (False),
                "response_signal_error": (str(exc)),
            },
        }

    return {
        **state,
        "metadata": {
            **assessment_metadata,
            "response_signal_recorded": (True),
        },
    }


async def update_session_summary_node(
    state: OrchestrationState,
    *,
    summary_service: (SessionSummaryService),
) -> OrchestrationState:
    """
    Best-effort incremental session-summary update at the end of a turn.

    Runs on any completed turn (specialist or fallback response); it
    does not require an agent to have run. Scenario onboarding is not
    wired through this node.

    Summary maintenance must never fail the turn: all errors are
    recorded in metadata and the previous summary is kept.
    """

    if state.get("error"):
        return state

    if state.get("metadata", {}).get("context_reply_handled"):
        return {
            **state,
            "metadata": {
                **state.get("metadata", {}),
                "summary_update_attempted": False,
                "summary_update_applied": False,
                "summary_update_reason": "deferred_context_reply",
            },
        }

    user_id = state.get("user_id")

    session_id = state.get("session_uuid")

    if user_id is None or session_id is None:
        return state

    metadata = dict(
        state.get(
            "metadata",
            {},
        )
    )

    try:
        result = await summary_service.maybe_update(
            user_id=user_id,
            session_id=session_id,
        )

    except Exception as exc:
        return {
            **state,
            "metadata": {
                **metadata,
                "summary_update_attempted": True,
                "summary_update_applied": False,
                "summary_update_error": (str(exc)),
            },
        }

    return {
        **state,
        "metadata": {
            **metadata,
            "summary_update_attempted": True,
            "summary_update_applied": (result.updated),
            "summary_update_reason": (result.reason),
            "summary_message_count": (result.covered_message_count),
            "summary_update_error": (result.error),
        },
    }


async def _update_hint_state_after_blocker(
    *,
    user_context_service: (UserContextService),
    user_id: UUID,
    blocked: bool,
) -> str:
    """
    Deterministically update the persisted guidance level after a
    blocker change was accepted into session runtime state.

    SET blocker -> one step toward EXPLICIT (unless manually
    overridden) and a longer blocked streak.
    CLEAR blocker -> reset the blocked streak; the level itself is
    kept so a student who needed support does not bounce back to
    minimal guidance after the very next message.
    """

    user_context = await user_context_service.get_or_create(
        user_id=user_id,
    )

    hint_state = user_context.hint_state

    if not blocked and hint_state.consecutive_blocked_attempts == 0:
        return hint_state.guidance_level.value

    if not hint_state.manually_overridden and blocked:
        hint_state.guidance_level = next_guidance_level(
            hint_state.guidance_level,
        )

    if blocked:
        hint_state.consecutive_blocked_attempts += 1

        hint_state.last_change_reason = "Active blocker persisted in session runtime state."

    else:
        hint_state.consecutive_blocked_attempts = 0

        hint_state.last_change_reason = "Active blocker cleared in session runtime state."

    hint_state.consecutive_productive_attempts = 0

    hint_state.updated_at = datetime.now(UTC)

    await user_context_service.save(
        user_id=user_id,
        context=user_context,
        expected_version=(user_context.version),
    )

    return hint_state.guidance_level.value


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

from __future__ import annotations

from app.agents.exceptions import AgentError
from app.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorRoutingError,
)
from app.agents.policies import GuidanceLevel
from app.agents.registry import AgentRegistry
from app.agents.routing import OrchestratorRequest
from app.agents.schemas import AgentName, AgentRequest
from app.logging.logger import logger
from app.orchestration.router import classify_route
from app.orchestration.state import OrchestrationState


def prepare_input_node(
    state: OrchestrationState,
) -> OrchestrationState:
    """Prepare and validate initial user input."""

    user_message = state.get("user_message", "").strip()
    metadata = dict(state.get("metadata", {}))

    if not user_message:
        return {
            **state,
            "user_message": user_message,
            "error": "User message cannot be empty.",
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


async def route_request_node(
    state: OrchestrationState,
    orchestrator: OrchestratorAgent,
) -> OrchestrationState:
    """Select the orchestration route.

    Routing uses the semantic LLM orchestrator first.

    If the semantic orchestrator:
    - explicitly selects fallback, or
    - fails to produce a valid routing decision,

    the deterministic keyword router is used as a secondary fallback.

    If both routing strategies return fallback, the graph continues to the
    fallback response node so that the student can clarify the type of help
    they need.
    """

    if state.get("error"):
        return state

    metadata = dict(state.get("metadata", {}))

    orchestrator_request = OrchestratorRequest(
        session_id=state["session_id"],
        student_id=state.get("student_id"),
        user_message=state["user_message"],
        metadata=dict(metadata),
    )

    try:
        orchestrator_result = await orchestrator.route(orchestrator_request)

    except OrchestratorRoutingError as exc:
        logger.warning(
            "Semantic_routing_failed_using_keyword_fallback",
            session_id=state["session_id"],
            student_id=state.get("student_id"),
            error=exc.message,
        )

        return _route_with_keyword_fallback(
            state=state,
            metadata={
                **metadata,
                "semantic_routing_attempted": True,
                "semantic_routing_failed": True,
                "orchestrator_error": exc.to_dict(),
            },
            fallback_trigger="orchestrator_error",
        )

    decision = orchestrator_result.decision

    semantic_metadata = {
        **metadata,
        "semantic_routing_attempted": True,
        "semantic_routing_failed": False,
        "semantic_route": decision.route,
        "semantic_route_reason": decision.reason,
        "orchestrator_metadata": dict(orchestrator_result.metadata),
    }

    if decision.route != "fallback":
        logger.info(
            "Semantic_routing_completed",
            session_id=state["session_id"],
            student_id=state.get("student_id"),
            route=decision.route,
            reason=decision.reason,
        )

        return {
            **state,
            "route": decision.route,
            "route_reason": decision.reason,
            "metadata": {
                **semantic_metadata,
                "routing_completed": True,
                "routing_strategy": ("llm_semantic_router_v1"),
                "keyword_fallback_used": False,
            },
        }

    logger.info(
        "Semantic_routing_returned_fallback",
        session_id=state["session_id"],
        student_id=state.get("student_id"),
        reason=decision.reason,
    )

    return _route_with_keyword_fallback(
        state=state,
        metadata=semantic_metadata,
        fallback_trigger="semantic_fallback",
    )


def _route_with_keyword_fallback(
    *,
    state: OrchestrationState,
    metadata: dict,
    fallback_trigger: str,
) -> OrchestrationState:
    """Run deterministic routing after semantic routing cannot decide."""

    keyword_decision = classify_route(state["user_message"])

    logger.info(
        "Keyword_fallback_routing_completed",
        session_id=state["session_id"],
        student_id=state.get("student_id"),
        route=keyword_decision.route,
        reason=keyword_decision.reason,
        fallback_trigger=fallback_trigger,
    )

    return {
        **state,
        "route": keyword_decision.route,
        "route_reason": keyword_decision.reason,
        "metadata": {
            **metadata,
            "routing_completed": True,
            "routing_strategy": ("llm_semantic_router_v1->deterministic_keywords_v1"),
            "keyword_fallback_used": True,
            "keyword_fallback_trigger": fallback_trigger,
            "keyword_route": keyword_decision.route,
            "keyword_route_reason": keyword_decision.reason,
        },
    }


async def execute_agent_node(
    state: OrchestrationState,
    agent_registry: AgentRegistry,
    agent_name: AgentName,
) -> OrchestrationState:
    """Execute one specialized agent selected by orchestration."""

    if state.get("error"):
        return state

    metadata = dict(state.get("metadata", {}))

    agent_request = AgentRequest(
        session_id=state["session_id"],
        student_id=state.get("student_id"),
        user_message=state["user_message"],
        guidance_level=state.get(
            "guidance_level",
            GuidanceLevel.MINIMAL,
        ),
        metadata=dict(metadata),
    )

    try:
        agent = agent_registry.get(agent_name)

        response = await agent.execute(agent_request)

    except AgentError as exc:
        return {
            **state,
            "error": exc.message,
            "metadata": {
                **metadata,
                "agent_error": exc.to_dict(),
                "selected_agent": agent_name,
            },
        }

    return {
        **state,
        "agent_response": response.content,
        "metadata": {
            **metadata,
            "selected_agent": response.agent,
            "agent_called": True,
            "agent_metadata": dict(response.metadata),
        },
    }


def fallback_response_node(
    state: OrchestrationState,
) -> OrchestrationState:
    """Return clarification when no routing strategy can decide."""

    if state.get("error"):
        return state

    metadata = dict(state.get("metadata", {}))

    return {
        **state,
        "agent_response": (
            "I'm not sure which type of help you need yet. "
            "Are you asking about your business scenario, "
            "the business-process reasoning, or how to work in AP+?"
        ),
        "metadata": {
            **metadata,
            "selected_agent": None,
            "fallback_used": True,
        },
    }


def finalize_response_node(
    state: OrchestrationState,
) -> OrchestrationState:
    """Create the final response returned by orchestration."""

    metadata = dict(state.get("metadata", {}))

    if state.get("error"):
        return {
            **state,
            "final_response": (f"I could not process the message because: {state['error']}"),
            "metadata": {
                **metadata,
                "finalized": True,
            },
        }

    final_response = state.get("agent_response") or state.get("model_response") or ""

    return {
        **state,
        "final_response": final_response,
        "metadata": {
            **metadata,
            "finalized": True,
        },
    }

from __future__ import annotations

from app.agents.exceptions import AgentError
from app.agents.policies import GuidanceLevel
from app.agents.registry import AgentRegistry
from app.agents.schemas import AgentName, AgentRequest
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


def route_request_node(
    state: OrchestrationState,
) -> OrchestrationState:
    """Select the orchestration route."""

    if state.get("error"):
        return state

    decision = classify_route(state["user_message"])

    metadata = dict(state.get("metadata", {}))

    return {
        **state,
        "route": decision.route,
        "route_reason": decision.reason,
        "metadata": {
            **metadata,
            "routing_completed": True,
            "routing_strategy": "deterministic_keywords_v1",
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
    """Return a deterministic clarification for unmatched requests."""

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

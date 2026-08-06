from __future__ import annotations

from app.orchestration.router import classify_route
from app.orchestration.state import OrchestrationState, RouteName
from app.services.model_service.exceptions import ModelServiceError
from app.services.model_service.schemas import ModelChatRequest, ModelMessage
from app.services.model_service.service import ModelService

ROUTE_SYSTEM_INSTRUCTIONS: dict[RouteName, str] = {
    "scenario": (
        "You are currently operating through the Scenario placeholder path. "
        "Help clarify the company context and business problem. "
        "Do not provide AP+ procedural instructions."
    ),
    "process_coach": (
        "You are currently operating through the Process Coach placeholder path. "
        "Support business-process reasoning using brief guiding questions. "
        "Do not provide a step-by-step tutorial."
    ),
    "ap_plus_navigator": (
        "You are currently operating through the AP+ Navigator placeholder path. "
        "Provide concise AP+-specific clarification without revealing an entire "
        "step-by-step process."
    ),
    "fallback": (
        "You are currently operating through the fallback placeholder path. "
        "Respond briefly, acknowledge the request, and ask for clarification "
        "when the learning intent is unclear."
    ),
}


def route_request_node(
    state: OrchestrationState,
) -> OrchestrationState:
    """Select a placeholder orchestration path deterministically."""

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


def scenario_path_node(
    state: OrchestrationState,
) -> OrchestrationState:
    return _mark_selected_path(state, "scenario")


def process_coach_path_node(
    state: OrchestrationState,
) -> OrchestrationState:
    return _mark_selected_path(state, "process_coach")


def ap_plus_navigator_path_node(
    state: OrchestrationState,
) -> OrchestrationState:
    return _mark_selected_path(state, "ap_plus_navigator")


def fallback_path_node(
    state: OrchestrationState,
) -> OrchestrationState:
    return _mark_selected_path(state, "fallback")


def _mark_selected_path(
    state: OrchestrationState,
    route: RouteName,
) -> OrchestrationState:
    """Record which placeholder route node was executed."""

    metadata = dict(state.get("metadata", {}))

    return {
        **state,
        "metadata": {
            **metadata,
            "selected_path": route,
        },
    }


def prepare_input_node(state: OrchestrationState) -> OrchestrationState:
    """Prepare and validate initial user input.

    Sprint 2 version:
    - trims the user message
    - initializes metadata if missing
    - records that input preparation happened

    Later:
    - load active context
    - load session state
    - attach checklist state
    - attach retrieved context
    """

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


async def model_response_node(
    state: OrchestrationState,
    model_service: ModelService,
) -> OrchestrationState:
    """Generate a response through the selected Sprint 2 route."""

    if state.get("error"):
        return state

    metadata = dict(state.get("metadata", {}))
    route = state.get("route", "fallback")

    request = ModelChatRequest(
        messages=[
            ModelMessage(
                role="system",
                content=(
                    "You are FREDi, an educational AI learning environment. "
                    f"{ROUTE_SYSTEM_INSTRUCTIONS[route]}"
                ),
            ),
            ModelMessage(
                role="user",
                content=state["user_message"],
            ),
        ],
    )

    try:
        response = await model_service.generate(request)
    except ModelServiceError as exc:
        return {
            **state,
            "error": exc.message,
            "metadata": {
                **metadata,
                "model_error": exc.to_dict(),
            },
        }

    return {
        **state,
        "model_response": response.content,
        "metadata": {
            **metadata,
            "model_called": True,
            "model": response.model,
            "model_provider": response.provider,
            "model_latency_ms": response.latency_ms,
        },
    }


def finalize_response_node(state: OrchestrationState) -> OrchestrationState:
    """Create the final response returned by the graph."""

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

    return {
        **state,
        "final_response": state.get("model_response", ""),
        "metadata": {
            **metadata,
            "finalized": True,
        },
    }

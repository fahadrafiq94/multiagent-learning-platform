from __future__ import annotations

from app.orchestration.state import OrchestrationState
from app.services.model_service.exceptions import ModelServiceError
from app.services.model_service.schemas import ModelChatRequest, ModelMessage
from app.services.model_service.service import ModelService


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
    """Generate a response using the Sprint 1 Model Service.

    This is not an agent yet. It is only a minimal graph-to-model connection.
    """

    if state.get("error"):
        return state

    metadata = dict(state.get("metadata", {}))

    request = ModelChatRequest(
        messages=[
            ModelMessage(
                role="system",
                content=(
                    "You are FREDi, a concise educational AI assistant. "
                    "For now, respond briefly and clearly. "
                    "Do not claim to be a full AP+ tutor yet."
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

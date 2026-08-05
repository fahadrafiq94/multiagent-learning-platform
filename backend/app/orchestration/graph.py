from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.orchestration.nodes import (
    finalize_response_node,
    model_response_node,
    prepare_input_node,
)
from app.orchestration.state import OrchestrationState
from app.services.model_service.service import ModelService


def build_orchestration_graph(model_service: ModelService):
    """Build the Sprint 2 orchestration graph.

    Current graph:
    START -> prepare_input -> model_response -> finalize_response -> END

    This connects LangGraph to the Sprint 1 Model Service without implementing
    real agents yet.
    """

    async def call_model_response_node(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await model_response_node(state, model_service)

    graph = StateGraph(OrchestrationState)

    graph.add_node("prepare_input", prepare_input_node)
    graph.add_node("model_response", call_model_response_node)
    graph.add_node("finalize_response", finalize_response_node)

    graph.add_edge(START, "prepare_input")
    graph.add_edge("prepare_input", "model_response")
    graph.add_edge("model_response", "finalize_response")
    graph.add_edge("finalize_response", END)

    return graph.compile()

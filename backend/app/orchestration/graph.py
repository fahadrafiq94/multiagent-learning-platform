from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.orchestration.nodes import (
    ap_plus_navigator_path_node,
    fallback_path_node,
    finalize_response_node,
    model_response_node,
    prepare_input_node,
    process_coach_path_node,
    route_request_node,
    scenario_path_node,
)
from app.orchestration.router import (
    route_after_input_preparation,
    route_to_graph_target,
)
from app.orchestration.state import OrchestrationState
from app.services.model_service import ModelService


def build_orchestration_graph(model_service: ModelService):
    """Build the Sprint 2 orchestration graph with conditional routing."""

    async def call_model_response_node(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await model_response_node(state, model_service)

    graph = StateGraph(OrchestrationState)

    graph.add_node("prepare_input", prepare_input_node)
    graph.add_node("route_request", route_request_node)

    graph.add_node("scenario_path", scenario_path_node)
    graph.add_node("process_coach_path", process_coach_path_node)
    graph.add_node(
        "ap_plus_navigator_path",
        ap_plus_navigator_path_node,
    )
    graph.add_node("fallback_path", fallback_path_node)

    graph.add_node("model_response", call_model_response_node)
    graph.add_node("finalize_response", finalize_response_node)

    graph.add_edge(START, "prepare_input")

    graph.add_conditional_edges(
        "prepare_input",
        route_after_input_preparation,
        {
            "route_request": "route_request",
            "finalize_response": "finalize_response",
        },
    )

    graph.add_conditional_edges(
        "route_request",
        route_to_graph_target,
        {
            "scenario_path": "scenario_path",
            "process_coach_path": "process_coach_path",
            "ap_plus_navigator_path": "ap_plus_navigator_path",
            "fallback_path": "fallback_path",
        },
    )

    graph.add_edge("scenario_path", "model_response")
    graph.add_edge("process_coach_path", "model_response")
    graph.add_edge("ap_plus_navigator_path", "model_response")
    graph.add_edge("fallback_path", "model_response")

    graph.add_edge("model_response", "finalize_response")
    graph.add_edge("finalize_response", END)

    return graph.compile()

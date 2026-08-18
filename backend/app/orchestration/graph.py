from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents.registry import AgentRegistry
from app.orchestration.nodes import (
    execute_agent_node,
    fallback_response_node,
    finalize_response_node,
    prepare_input_node,
    route_request_node,
)
from app.orchestration.router import (
    route_after_input_preparation,
    route_to_graph_target,
)
from app.orchestration.state import OrchestrationState


def build_orchestration_graph(
    agent_registry: AgentRegistry,
):
    """Build the Sprint 3 orchestration graph."""

    async def scenario_agent_node(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await execute_agent_node(
            state=state,
            agent_registry=agent_registry,
            agent_name="scenario",
        )

    async def process_coach_agent_node(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await execute_agent_node(
            state=state,
            agent_registry=agent_registry,
            agent_name="process_coach",
        )

    async def ap_plus_navigator_agent_node(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await execute_agent_node(
            state=state,
            agent_registry=agent_registry,
            agent_name="ap_plus_navigator",
        )

    graph = StateGraph(OrchestrationState)

    graph.add_node(
        "prepare_input",
        prepare_input_node,
    )

    graph.add_node(
        "route_request",
        route_request_node,
    )

    graph.add_node(
        "scenario_agent",
        scenario_agent_node,
    )

    graph.add_node(
        "process_coach_agent",
        process_coach_agent_node,
    )

    graph.add_node(
        "ap_plus_navigator_agent",
        ap_plus_navigator_agent_node,
    )

    graph.add_node(
        "fallback_response",
        fallback_response_node,
    )

    graph.add_node(
        "finalize_response",
        finalize_response_node,
    )

    graph.add_edge(
        START,
        "prepare_input",
    )

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
            "scenario_agent": "scenario_agent",
            "process_coach_agent": "process_coach_agent",
            "ap_plus_navigator_agent": ("ap_plus_navigator_agent"),
            "fallback_response": "fallback_response",
        },
    )

    graph.add_edge(
        "scenario_agent",
        "finalize_response",
    )

    graph.add_edge(
        "process_coach_agent",
        "finalize_response",
    )

    graph.add_edge(
        "ap_plus_navigator_agent",
        "finalize_response",
    )

    graph.add_edge(
        "fallback_response",
        "finalize_response",
    )

    graph.add_edge(
        "finalize_response",
        END,
    )

    return graph.compile()

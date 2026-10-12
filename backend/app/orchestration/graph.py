from __future__ import annotations

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.agents.orchestrator import (
    OrchestratorAgent,
)
from app.agents.registry import (
    AgentRegistry,
)
from app.context.policies import (
    AgentContextPolicyService,
)
from app.orchestration.nodes import (
    execute_agent_node,
    fallback_response_node,
    finalize_response_node,
    load_context_node,
    prepare_input_node,
    route_request_node,
    scenario_onboarding_node,
)
from app.orchestration.router import (
    route_after_context_load,
    route_after_input_preparation,
    route_to_graph_target,
)
from app.orchestration.state import (
    OrchestrationState,
)
from app.services.context_service import (
    ContextService,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.scenario_onboarding_interaction_service import (
    ScenarioOnboardingInteractionService,
)
from app.services.scenario_onboarding_service import (
    ScenarioOnboardingService,
)


def build_orchestration_graph(
    *,
    agent_registry: AgentRegistry,
    orchestrator: OrchestratorAgent,
    conversation_service: (ConversationService),
    context_service: ContextService,
    context_policy_service: (AgentContextPolicyService),
    onboarding_service: (ScenarioOnboardingService),
    onboarding_interaction_service: (ScenarioOnboardingInteractionService),
):
    """
    Build the Sprint 4B.5 runtime graph.
    """

    async def load_context(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await load_context_node(
            state,
            conversation_service=(conversation_service),
            context_service=(context_service),
            onboarding_service=(onboarding_service),
        )

    async def scenario_onboarding(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await scenario_onboarding_node(
            state,
            context_policy_service=(context_policy_service),
            interaction_service=(onboarding_interaction_service),
        )

    async def route_request(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await route_request_node(
            state,
            orchestrator=orchestrator,
            context_policy_service=(context_policy_service),
        )

    async def scenario_agent(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await execute_agent_node(
            state,
            agent_registry=agent_registry,
            agent_name="scenario",
            context_policy_service=(context_policy_service),
        )

    async def process_coach_agent(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await execute_agent_node(
            state,
            agent_registry=agent_registry,
            agent_name=("process_coach"),
            context_policy_service=(context_policy_service),
        )

    async def ap_plus_navigator_agent(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await execute_agent_node(
            state,
            agent_registry=agent_registry,
            agent_name=("ap_plus_navigator"),
            context_policy_service=(context_policy_service),
        )

    async def finalize_response(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await finalize_response_node(
            state,
            conversation_service=(conversation_service),
        )

    graph = StateGraph(OrchestrationState)

    graph.add_node(
        "prepare_input",
        prepare_input_node,
    )

    graph.add_node(
        "load_context",
        load_context,
    )

    graph.add_node(
        "scenario_onboarding",
        scenario_onboarding,
    )

    graph.add_node(
        "route_request",
        route_request,
    )

    graph.add_node(
        "scenario_agent",
        scenario_agent,
    )

    graph.add_node(
        "process_coach_agent",
        process_coach_agent,
    )

    graph.add_node(
        "ap_plus_navigator_agent",
        ap_plus_navigator_agent,
    )

    graph.add_node(
        "fallback_response",
        fallback_response_node,
    )

    graph.add_node(
        "finalize_response",
        finalize_response,
    )

    graph.add_edge(
        START,
        "prepare_input",
    )

    graph.add_conditional_edges(
        "prepare_input",
        route_after_input_preparation,
        {
            "load_context": ("load_context"),
            "finalize_response": ("finalize_response"),
        },
    )

    graph.add_conditional_edges(
        "load_context",
        route_after_context_load,
        {
            "scenario_onboarding": ("scenario_onboarding"),
            "route_request": ("route_request"),
            "finalize_response": ("finalize_response"),
        },
    )

    graph.add_conditional_edges(
        "route_request",
        route_to_graph_target,
        {
            "scenario_agent": ("scenario_agent"),
            "process_coach_agent": ("process_coach_agent"),
            "ap_plus_navigator_agent": ("ap_plus_navigator_agent"),
            "fallback_response": ("fallback_response"),
        },
    )

    graph.add_edge(
        "scenario_onboarding",
        "finalize_response",
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

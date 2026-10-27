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
    apply_state_updates_node,
    execute_agent_node,
    fallback_response_node,
    finalize_response_node,
    load_context_node,
    prepare_input_node,
    record_response_signal_node,
    route_request_node,
    scenario_onboarding_node,
    update_session_summary_node,
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
from app.services.session_state_update_service import (
    SessionStateUpdateService,
)
from app.services.session_summary_service import (
    SessionSummaryService,
)
from app.services.user_context_service import (
    UserContextService,
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
    session_state_update_service: (SessionStateUpdateService),
    session_summary_service: (SessionSummaryService),
    user_context_service: (UserContextService),
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

    async def apply_state_updates(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await apply_state_updates_node(
            state,
            session_state_update_service=(session_state_update_service),
            user_context_service=(user_context_service),
        )

    async def record_response_signal(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await record_response_signal_node(
            state,
            user_context_service=(user_context_service),
        )

    async def update_session_summary(
        state: OrchestrationState,
    ) -> OrchestrationState:
        return await update_session_summary_node(
            state,
            summary_service=(session_summary_service),
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
        "apply_state_updates",
        apply_state_updates,
    )

    graph.add_node(
        "record_response_signal",
        record_response_signal,
    )

    graph.add_node(
        "update_session_summary",
        update_session_summary,
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
        "apply_state_updates",
    )

    graph.add_edge(
        "process_coach_agent",
        "apply_state_updates",
    )

    graph.add_edge(
        "ap_plus_navigator_agent",
        "apply_state_updates",
    )

    graph.add_edge(
        "apply_state_updates",
        "record_response_signal",
    )

    graph.add_edge(
        "record_response_signal",
        "update_session_summary",
    )

    graph.add_edge(
        "fallback_response",
        "update_session_summary",
    )

    graph.add_edge(
        "update_session_summary",
        "finalize_response",
    )

    graph.add_edge(
        "finalize_response",
        END,
    )

    return graph.compile()

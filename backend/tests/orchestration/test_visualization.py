from __future__ import annotations

from pathlib import Path
from typing import cast

from app.agents.orchestrator import OrchestratorAgent
from app.agents.registry import AgentRegistry
from app.agents.routing import (
    OrchestratorRequest,
    OrchestratorResult,
    RoutingDecision,
)
from app.agents.schemas import (
    AgentName,
    AgentRequest,
    AgentResponse,
)
from app.orchestration.graph import build_orchestration_graph
from app.orchestration.visualization import (
    MERMAID_FRONTMATTER,
    MERMAID_STYLES,
    export_graph_mermaid,
)


class FakeDrawableGraph:
    """Fake drawable graph used for isolated Mermaid export tests."""

    def draw_mermaid(self) -> str:
        return """---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__ --> prepare_input;
    prepare_input --> route_request;
    route_request --> scenario_agent;
    scenario_agent --> finalize_response;
    finalize_response --> __end__;
"""


class FakeCompiledGraph:
    """Fake compiled graph exposing the visualization contract."""

    def get_graph(self) -> FakeDrawableGraph:
        return FakeDrawableGraph()


class FakeAgent:
    """Minimal agent implementation used to compile the real graph."""

    def __init__(
        self,
        name: AgentName,
    ) -> None:
        self._name = name

    @property
    def name(self) -> AgentName:
        return self._name

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        return AgentResponse(
            agent=self.name,
            content=f"{self.name}: {request.user_message}",
        )


class FakeOrchestrator:
    """Minimal semantic orchestrator used to compile the real graph.

    Visualization tests do not execute routing. The fake exists only because
    the Sprint 3 graph now requires an OrchestratorAgent dependency.
    """

    async def route(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResult:
        return OrchestratorResult(
            decision=RoutingDecision(
                route="fallback",
                reason="Visualization test routing decision.",
            ),
            metadata={
                "routing_strategy": "fake_visualization_router",
            },
        )


def create_agent_registry() -> AgentRegistry:
    """Create all specialized agents required by the graph."""

    return AgentRegistry(
        agents=[
            FakeAgent("scenario"),
            FakeAgent("process_coach"),
            FakeAgent("ap_plus_navigator"),
        ]
    )


def create_orchestrator() -> OrchestratorAgent:
    """Create the fake orchestrator dependency required by the graph."""

    return cast(
        OrchestratorAgent,
        FakeOrchestrator(),
    )


def test_export_graph_mermaid_creates_styled_file(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "docs" / "architecture" / "orchestration-graph.mmd"

    result = export_graph_mermaid(
        graph=FakeCompiledGraph(),
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.is_file()

    content = output_path.read_text(
        encoding="utf-8",
    )

    assert "title: FREDi Agent Orchestration Graph" in content

    assert "theme: base" in content
    assert "look: classic" in content
    assert "curve: basis" in content

    assert 'primaryTextColor: "#0f172a"' in content

    assert 'background: "#f8fafc"' in content

    assert "graph TD" in content

    assert "prepare_input" in content
    assert "route_request" in content
    assert "scenario_agent" in content
    assert "finalize_response" in content

    assert "classDef inputNode" in content
    assert "classDef routingNode" in content
    assert "classDef agentNode" in content
    assert "classDef fallbackNode" in content
    assert "classDef outputNode" in content
    assert "classDef terminalNode" in content

    assert "class prepare_input inputNode" in content

    assert "class route_request routingNode" in content

    assert "class scenario_agent,process_coach_agent,ap_plus_navigator_agent agentNode" in content

    assert "class fallback_response fallbackNode" in content

    assert "class finalize_response outputNode" in content

    assert "class __start__,__end__ terminalNode" in content

    assert "linkStyle default" in content


def test_export_graph_mermaid_creates_parent_directories(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "nested" / "architecture" / "graph.mmd"

    assert not output_path.parent.exists()

    export_graph_mermaid(
        graph=FakeCompiledGraph(),
        output_path=output_path,
    )

    assert output_path.parent.exists()
    assert output_path.exists()


def test_export_replaces_existing_frontmatter(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "graph.mmd"

    export_graph_mermaid(
        graph=FakeCompiledGraph(),
        output_path=output_path,
    )

    content = output_path.read_text(
        encoding="utf-8",
    )

    # The fake graph contains its own frontmatter.
    # The exporter must replace it with FREDi's styling.
    assert content.count("title: FREDi Agent Orchestration Graph") == 1

    assert "curve: linear" not in content
    assert "curve: basis" in content


def test_mermaid_configuration_defines_readable_text() -> None:
    assert "theme: base" in MERMAID_FRONTMATTER

    assert 'primaryTextColor: "#0f172a"' in MERMAID_FRONTMATTER

    assert 'secondaryTextColor: "#1e1b4b"' in MERMAID_FRONTMATTER

    assert 'tertiaryTextColor: "#451a03"' in MERMAID_FRONTMATTER

    assert 'textColor: "#0f172a"' in MERMAID_FRONTMATTER

    assert 'edgeLabelBackground: "#ffffff"' in MERMAID_FRONTMATTER


def test_mermaid_styles_define_all_node_classes() -> None:
    expected_class_definitions = {
        "inputNode",
        "routingNode",
        "agentNode",
        "fallbackNode",
        "outputNode",
        "terminalNode",
    }

    for class_name in expected_class_definitions:
        assert f"classDef {class_name}" in MERMAID_STYLES


def test_mermaid_styles_assign_all_current_graph_nodes() -> None:
    expected_nodes = {
        "prepare_input",
        "route_request",
        "scenario_agent",
        "process_coach_agent",
        "ap_plus_navigator_agent",
        "fallback_response",
        "finalize_response",
    }

    for node_name in expected_nodes:
        assert node_name in MERMAID_STYLES


def test_mermaid_styles_do_not_reference_old_sprint_2_nodes() -> None:
    obsolete_nodes = {
        "scenario_path",
        "process_coach_path",
        "ap_plus_navigator_path",
        "fallback_path",
        "model_response",
    }

    for node_name in obsolete_nodes:
        assert node_name not in MERMAID_STYLES


def test_real_graph_mermaid_contains_expected_nodes() -> None:
    registry = create_agent_registry()

    graph = build_orchestration_graph(
        agent_registry=registry,
        orchestrator=create_orchestrator(),
    )

    mermaid_source = graph.get_graph().draw_mermaid()

    expected_nodes = {
        "prepare_input",
        "route_request",
        "scenario_agent",
        "process_coach_agent",
        "ap_plus_navigator_agent",
        "fallback_response",
        "finalize_response",
    }

    for node_name in expected_nodes:
        assert node_name in mermaid_source


def test_real_graph_mermaid_does_not_contain_old_nodes() -> None:
    registry = create_agent_registry()

    graph = build_orchestration_graph(
        agent_registry=registry,
        orchestrator=create_orchestrator(),
    )

    mermaid_source = graph.get_graph().draw_mermaid()

    obsolete_nodes = {
        "scenario_path",
        "process_coach_path",
        "ap_plus_navigator_path",
        "fallback_path",
        "model_response",
    }

    for node_name in obsolete_nodes:
        assert node_name not in mermaid_source


def test_real_graph_mermaid_contains_start_and_end() -> None:
    registry = create_agent_registry()

    graph = build_orchestration_graph(
        agent_registry=registry,
        orchestrator=create_orchestrator(),
    )

    mermaid_source = graph.get_graph().draw_mermaid()

    assert "__start__" in mermaid_source
    assert "__end__" in mermaid_source


def test_real_graph_has_specialized_agent_branches() -> None:
    registry = create_agent_registry()

    graph = build_orchestration_graph(
        agent_registry=registry,
        orchestrator=create_orchestrator(),
    )

    mermaid_source = graph.get_graph().draw_mermaid()

    assert "scenario_agent" in mermaid_source

    assert "process_coach_agent" in mermaid_source

    assert "ap_plus_navigator_agent" in mermaid_source

    assert "fallback_response" in mermaid_source

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from app.orchestration.factory import (
    build_orchestration_runtime,
)
from app.orchestration.visualization import (
    MERMAID_FRONTMATTER,
    MERMAID_STYLES,
    export_graph_mermaid,
)
from app.services.model_service import (
    ModelService,
)


class FakeDrawableGraph:
    def draw_mermaid(
        self,
    ) -> str:
        return """---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__ --> prepare_input;
    prepare_input --> load_context;
    load_context --> scenario_onboarding;
    load_context --> route_request;
    route_request --> process_coach_agent;
    process_coach_agent --> finalize_response;
    finalize_response --> __end__;
"""


class FakeCompiledGraph:
    def get_graph(
        self,
    ) -> FakeDrawableGraph:
        return FakeDrawableGraph()


def create_real_graph():
    model_service = cast(
        ModelService,
        MagicMock(spec=ModelService),
    )

    runtime = build_orchestration_runtime(model_service=(model_service))

    return runtime.graph


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

    content = output_path.read_text(encoding="utf-8")

    assert "title: FREDi Agent Orchestration Graph" in content

    assert "curve: basis" in content

    assert "load_context" in content

    assert "scenario_onboarding" in content


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

    content = output_path.read_text(encoding="utf-8")

    assert content.count("title: FREDi Agent Orchestration Graph") == 1

    assert "curve: linear" not in content

    assert "curve: basis" in content


def test_mermaid_configuration_is_present() -> None:
    assert "theme: base" in MERMAID_FRONTMATTER

    assert 'primaryTextColor: "#0f172a"' in MERMAID_FRONTMATTER

    assert 'edgeLabelBackground: "#ffffff"' in MERMAID_FRONTMATTER


def test_mermaid_styles_define_4b5_classes() -> None:
    expected = {
        "inputNode",
        "contextNode",
        "routingNode",
        "onboardingNode",
        "agentNode",
        "fallbackNode",
        "outputNode",
        "terminalNode",
    }

    for class_name in expected:
        assert f"classDef {class_name}" in MERMAID_STYLES


def test_mermaid_styles_reference_all_4b5_nodes() -> None:
    expected = {
        "prepare_input",
        "load_context",
        "scenario_onboarding",
        "route_request",
        "scenario_agent",
        "process_coach_agent",
        "ap_plus_navigator_agent",
        "fallback_response",
        "finalize_response",
    }

    for node in expected:
        assert node in MERMAID_STYLES


def test_real_graph_contains_all_4b5_nodes() -> None:
    graph = create_real_graph()

    mermaid = graph.get_graph().draw_mermaid()

    expected = {
        "prepare_input",
        "load_context",
        "scenario_onboarding",
        "route_request",
        "scenario_agent",
        "process_coach_agent",
        "ap_plus_navigator_agent",
        "fallback_response",
        "finalize_response",
    }

    for node in expected:
        assert node in mermaid


def test_real_graph_contains_start_and_end() -> None:
    graph = create_real_graph()

    mermaid = graph.get_graph().draw_mermaid()

    assert "__start__" in mermaid
    assert "__end__" in mermaid


def test_real_graph_does_not_contain_old_sprint_nodes() -> None:
    graph = create_real_graph()

    mermaid = graph.get_graph().draw_mermaid()

    obsolete = {
        "scenario_path",
        "process_coach_path",
        "ap_plus_navigator_path",
        "fallback_path",
    }

    for node in obsolete:
        assert node not in mermaid

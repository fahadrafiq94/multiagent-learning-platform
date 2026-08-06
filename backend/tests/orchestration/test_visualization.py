from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from app.orchestration.graph import build_orchestration_graph
from app.orchestration.visualization import (
    MERMAID_FRONTMATTER,
    MERMAID_STYLES,
    _remove_existing_frontmatter,
    export_graph_mermaid,
)
from app.services.model_service import ModelService


class FakeDrawableGraph:
    """Fake drawable graph used to test Mermaid export."""

    def draw_mermaid(self) -> str:
        return """---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__ --> prepare_input;
    prepare_input --> route_request;
    route_request --> finalize_response;
    finalize_response --> __end__;
"""


class FakeCompiledGraph:
    """Fake compiled graph exposing LangGraph's visualization contract."""

    def get_graph(self) -> FakeDrawableGraph:
        return FakeDrawableGraph()


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

    content = output_path.read_text(encoding="utf-8")

    assert "title: FREDi Sprint 2 Orchestration Graph" in content
    assert "theme: base" in content
    assert "look: classic" in content
    assert "curve: basis" in content
    assert 'primaryTextColor: "#0f172a"' in content
    assert 'background: "#f8fafc"' in content

    assert "graph TD" in content
    assert "prepare_input" in content
    assert "route_request" in content
    assert "finalize_response" in content

    assert "classDef inputNode" in content
    assert "classDef routingNode" in content
    assert "classDef routeNode" in content
    assert "classDef modelNode" in content
    assert "classDef outputNode" in content
    assert "classDef terminalNode" in content

    assert "class prepare_input inputNode" in content
    assert "class route_request routingNode" in content
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

    content = output_path.read_text(encoding="utf-8")

    # The original fake graph contains its own frontmatter. The exporter
    # should remove it and write only the FREDi frontmatter.
    assert content.count("title: FREDi Sprint 2 Orchestration Graph") == 1

    assert "curve: linear" not in content
    assert "curve: basis" in content


def test_remove_existing_frontmatter() -> None:
    source = """---
config:
  theme: default
---
graph TD;
    start --> end;
"""

    result = _remove_existing_frontmatter(source)

    assert (
        result
        == """graph TD;
    start --> end;"""
    )


def test_remove_existing_frontmatter_preserves_plain_mermaid() -> None:
    source = """graph TD;
    start --> end;
"""

    result = _remove_existing_frontmatter(source)

    assert result == source


def test_remove_existing_frontmatter_handles_unclosed_frontmatter() -> None:
    source = """---
config:
  theme: default
graph TD;
    start --> end;
"""

    result = _remove_existing_frontmatter(source)

    # Malformed source should be returned unchanged instead of being
    # partially deleted.
    assert result == source


def test_mermaid_configuration_defines_readable_text() -> None:
    assert "theme: base" in MERMAID_FRONTMATTER
    assert 'primaryTextColor: "#0f172a"' in MERMAID_FRONTMATTER
    assert 'secondaryTextColor: "#1e1b4b"' in MERMAID_FRONTMATTER
    assert 'tertiaryTextColor: "#451a03"' in MERMAID_FRONTMATTER
    assert 'textColor: "#0f172a"' in MERMAID_FRONTMATTER
    assert 'edgeLabelBackground: "#ffffff"' in MERMAID_FRONTMATTER


def test_mermaid_styles_define_all_orchestration_node_classes() -> None:
    expected_class_definitions = {
        "inputNode",
        "routingNode",
        "routeNode",
        "modelNode",
        "outputNode",
        "terminalNode",
    }

    for class_name in expected_class_definitions:
        assert f"classDef {class_name}" in MERMAID_STYLES

    expected_node_assignments = {
        "prepare_input",
        "route_request",
        "scenario_path",
        "process_coach_path",
        "ap_plus_navigator_path",
        "fallback_path",
        "model_response",
        "finalize_response",
    }

    for node_name in expected_node_assignments:
        assert node_name in MERMAID_STYLES


def test_real_graph_mermaid_contains_expected_nodes() -> None:
    # The graph is compiled but not invoked, so no real model provider
    # or Ollama connection is required.
    model_service = MagicMock(spec=ModelService)

    graph = build_orchestration_graph(model_service)

    mermaid_source = graph.get_graph().draw_mermaid()

    expected_nodes = {
        "prepare_input",
        "route_request",
        "scenario_path",
        "process_coach_path",
        "ap_plus_navigator_path",
        "fallback_path",
        "model_response",
        "finalize_response",
    }

    for node_name in expected_nodes:
        assert node_name in mermaid_source


def test_real_graph_mermaid_contains_start_and_end() -> None:
    model_service = MagicMock(spec=ModelService)

    graph = build_orchestration_graph(model_service)

    mermaid_source = graph.get_graph().draw_mermaid()

    assert "__start__" in mermaid_source
    assert "__end__" in mermaid_source

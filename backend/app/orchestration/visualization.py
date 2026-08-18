from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DrawableGraph(Protocol):
    """Minimal graph interface required for Mermaid export."""

    def draw_mermaid(self) -> str:
        """Return the graph as Mermaid source."""
        ...


class VisualizableCompiledGraph(Protocol):
    """Minimal compiled graph contract required for visualization."""

    def get_graph(self) -> DrawableGraph:
        """Return a drawable graph representation."""
        ...


MERMAID_FRONTMATTER = """---
title: FREDi Agent Orchestration Graph
config:
  theme: base
  look: classic
  flowchart:
    curve: basis
    nodeSpacing: 45
    rankSpacing: 60
    padding: 18
    htmlLabels: true
  themeVariables:
    background: "#f8fafc"
    primaryColor: "#e0f2fe"
    primaryTextColor: "#0f172a"
    primaryBorderColor: "#0284c7"
    secondaryColor: "#ede9fe"
    secondaryTextColor: "#1e1b4b"
    secondaryBorderColor: "#7c3aed"
    tertiaryColor: "#fef3c7"
    tertiaryTextColor: "#451a03"
    tertiaryBorderColor: "#d97706"
    lineColor: "#475569"
    textColor: "#0f172a"
    mainBkg: "#ffffff"
    nodeBorder: "#334155"
    clusterBkg: "#f1f5f9"
    clusterBorder: "#94a3b8"
    edgeLabelBackground: "#ffffff"
    fontFamily: "Inter, Segoe UI, Arial, sans-serif"
    fontSize: "16px"
---
"""


MERMAID_STYLES = """
classDef inputNode fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#172554;
classDef routingNode fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#2e1065;
classDef agentNode fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d;
classDef fallbackNode fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#431407;
classDef outputNode fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#451a03;
classDef terminalNode fill:#f1f5f9,stroke:#475569,stroke-width:2px,color:#0f172a;

class prepare_input inputNode;
class route_request routingNode;
class scenario_agent,process_coach_agent,ap_plus_navigator_agent agentNode;
class fallback_response fallbackNode;
class finalize_response outputNode;
class __start__,__end__ terminalNode;

linkStyle default stroke:#475569,stroke-width:2px;
"""


def export_graph_mermaid(
    graph: VisualizableCompiledGraph,
    output_path: Path,
) -> Path:
    """Export a styled LangGraph workflow as a Mermaid file.

    The export remains local and does not call an external rendering service.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_mermaid = graph.get_graph().draw_mermaid()
    mermaid_body = _remove_existing_frontmatter(raw_mermaid)

    styled_mermaid = f"{MERMAID_FRONTMATTER}\n{mermaid_body.rstrip()}\n{MERMAID_STYLES.strip()}\n"

    output_path.write_text(
        styled_mermaid,
        encoding="utf-8",
    )

    return output_path


def _remove_existing_frontmatter(source: str) -> str:
    """Remove existing Mermaid YAML frontmatter, if present."""

    stripped_source = source.lstrip()

    if not stripped_source.startswith("---"):
        return source

    lines = stripped_source.splitlines()

    try:
        closing_index = lines.index("---", 1)
    except ValueError:
        return source

    return "\n".join(lines[closing_index + 1 :]).lstrip()

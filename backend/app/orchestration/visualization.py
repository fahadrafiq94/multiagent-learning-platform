from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DrawableGraph(Protocol):
    def draw_mermaid(self) -> str: ...


class VisualizableCompiledGraph(Protocol):
    def get_graph(
        self,
    ) -> DrawableGraph: ...


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
    edgeLabelBackground: "#ffffff"
    fontFamily: "Inter, Segoe UI, Arial, sans-serif"
    fontSize: "16px"
---
"""


MERMAID_STYLES = """
classDef inputNode fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#172554;
classDef contextNode fill:#cffafe,stroke:#0891b2,stroke-width:2px,color:#164e63;
classDef routingNode fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#2e1065;
classDef onboardingNode fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#831843;
classDef agentNode fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d;
classDef fallbackNode fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#431407;
classDef outputNode fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#451a03;
classDef terminalNode fill:#f1f5f9,stroke:#475569,stroke-width:2px,color:#0f172a;

class prepare_input inputNode;
class load_context contextNode;
class route_request routingNode;
class scenario_onboarding onboardingNode;
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
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = graph.get_graph().draw_mermaid()

    body = _remove_existing_frontmatter(raw)

    styled = f"{MERMAID_FRONTMATTER}\n{body.rstrip()}\n{MERMAID_STYLES.strip()}\n"

    output_path.write_text(
        styled,
        encoding="utf-8",
    )

    return output_path


def _remove_existing_frontmatter(
    source: str,
) -> str:
    stripped = source.lstrip()

    if not stripped.startswith("---"):
        return source

    lines = stripped.splitlines()

    try:
        closing_index = lines.index(
            "---",
            1,
        )
    except ValueError:
        return source

    return "\n".join(lines[closing_index + 1 :]).lstrip()

from __future__ import annotations

from pathlib import Path

from app.agents.factory import create_agent_registry
from app.orchestration.graph import build_orchestration_graph
from app.orchestration.visualization import export_graph_mermaid
from app.services.model_service import create_model_service


def main() -> None:
    """Build and export the current FREDi agent orchestration graph."""

    model_service = create_model_service()

    agent_registry = create_agent_registry(
        model_service=model_service,
    )

    graph = build_orchestration_graph(
        agent_registry=agent_registry,
    )

    project_root = Path(__file__).resolve().parents[2]

    output_path = project_root / "docs" / "architecture" / "agent-orchestration-graph.mmd"

    exported_path = export_graph_mermaid(
        graph=graph,
        output_path=output_path,
    )

    print(f"FREDi agent orchestration graph exported to: {exported_path}")


if __name__ == "__main__":
    main()

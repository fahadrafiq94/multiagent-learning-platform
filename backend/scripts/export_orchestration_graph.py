from __future__ import annotations

from pathlib import Path

from app.orchestration.graph import build_orchestration_graph
from app.orchestration.visualization import export_graph_mermaid
from app.services.model_service import create_model_service


def main() -> None:
    """Build and export the current orchestration graph."""

    model_service = create_model_service()
    graph = build_orchestration_graph(model_service)

    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "docs" / "architecture" / "sprint-2-orchestration-graph.mmd"

    exported_path = export_graph_mermaid(
        graph=graph,
        output_path=output_path,
    )

    print(f"Graph exported to: {exported_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

from app.orchestration.factory import (
    build_orchestration_runtime,
)
from app.orchestration.visualization import (
    export_graph_mermaid,
)
from app.services.model_service import (
    create_model_service,
)


def main() -> None:
    """
    Build and export the current Sprint 4B.5 FREDi graph.
    """

    runtime = build_orchestration_runtime(model_service=(create_model_service()))

    project_root = Path(__file__).resolve().parents[2]

    output_path = project_root / "docs" / "architecture" / ("agent-orchestration-graph.mmd")

    exported_path = export_graph_mermaid(
        graph=runtime.graph,
        output_path=(output_path),
    )

    print(f"FREDi agent orchestration graph exported to: {exported_path}")


if __name__ == "__main__":
    main()

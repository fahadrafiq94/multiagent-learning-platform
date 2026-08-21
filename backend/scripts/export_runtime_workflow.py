from __future__ import annotations

from pathlib import Path

RUNTIME_WORKFLOW = """\
flowchart TD

    USER[Student / Client]
    API[FastAPI<br/>POST /orchestration/invoke]
    SERVICE[OrchestrationService]
    GRAPH[LangGraph Orchestration]

    PREPARE[prepare_input]
    ROUTE[route_request]

    ORCHESTRATOR[OrchestratorAgent<br/>semantic intent routing]

    KEYWORD[Deterministic Keyword Fallback]

    SCENARIO[ScenarioAgent]
    PROCESS[ProcessCoachAgent]
    APPLUS[APPlusNavigatorAgent]

    FALLBACK[Clarification Response]

    MODEL_SERVICE[Shared ModelService]
    PROVIDER[ModelProvider]
    OLLAMA[Ollama<br/>current provider]
    VLLM[vLLM<br/>future provider]

    FINALIZE[finalize_response]
    RESPONSE[OrchestrationResponse]
    CLIENT[Student / Client]

    USER --> API
    API --> SERVICE
    SERVICE --> GRAPH

    GRAPH --> PREPARE

    PREPARE -->|valid request| ROUTE
    PREPARE -->|invalid request| FINALIZE

    ROUTE --> ORCHESTRATOR

    ORCHESTRATOR -->|scenario| SCENARIO
    ORCHESTRATOR -->|process_coach| PROCESS
    ORCHESTRATOR -->|ap_plus_navigator| APPLUS

    ORCHESTRATOR -->|fallback| KEYWORD
    ORCHESTRATOR -->|model / routing error| KEYWORD

    KEYWORD -->|scenario| SCENARIO
    KEYWORD -->|process_coach| PROCESS
    KEYWORD -->|ap_plus_navigator| APPLUS
    KEYWORD -->|still unclear| FALLBACK

    ORCHESTRATOR -. structured routing request .-> MODEL_SERVICE
    SCENARIO -. generation request .-> MODEL_SERVICE
    PROCESS -. generation request .-> MODEL_SERVICE
    APPLUS -. generation request .-> MODEL_SERVICE

    MODEL_SERVICE --> PROVIDER

    PROVIDER -->|current| OLLAMA
    PROVIDER -. future .-> VLLM

    SCENARIO --> FINALIZE
    PROCESS --> FINALIZE
    APPLUS --> FINALIZE
    FALLBACK --> FINALIZE

    FINALIZE --> RESPONSE
    RESPONSE --> API
    API --> CLIENT
"""


def main() -> None:
    """Export the complete FREDi runtime workflow as Mermaid."""

    project_root = Path(__file__).resolve().parents[2]

    output_path = project_root / "docs" / "architecture" / "fredi-runtime-workflow.mmd"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        RUNTIME_WORKFLOW,
        encoding="utf-8",
    )

    print("FREDi runtime workflow exported successfully.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

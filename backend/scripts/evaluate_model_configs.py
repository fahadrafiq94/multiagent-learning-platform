from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config.settings import settings
from app.orchestration import (
    OrchestrationRequest,
    create_orchestration_service,
)


@dataclass(frozen=True)
class ModelConfiguration:
    """One model assignment configuration to evaluate."""

    name: str
    description: str

    orchestrator_model: str
    scenario_model: str
    process_coach_model: str
    ap_plus_navigator_model: str


@dataclass(frozen=True)
class EvaluationCase:
    """One representative FREDi request."""

    case_id: str
    category: str
    expected_route: str
    message: str
    review_focus: str


@dataclass
class EvaluationResult:
    """Captured result for one model configuration and case."""

    configuration: str
    case_id: str
    category: str

    expected_route: str
    actual_route: str | None

    routing_passed: bool

    semantic_route: str | None
    semantic_routing_failed: bool | None
    keyword_fallback_used: bool | None

    selected_agent: str | None

    orchestrator_model: str | None
    agent_model: str | None

    orchestrator_latency_ms: float | None
    agent_latency_ms: float | None

    total_wall_time_seconds: float

    final_response: str
    error: str | None

    review_focus: str


MODEL_CONFIGURATIONS = [
    ModelConfiguration(
        name="A_current_qwen3_4b",
        description=("Current baseline: qwen3:4b for the orchestrator and all specialist agents."),
        orchestrator_model="qwen3:4b",
        scenario_model="qwen3:4b",
        process_coach_model="qwen3:4b",
        ap_plus_navigator_model="qwen3:4b",
    ),
    ModelConfiguration(
        name="B_qwen35_4b_all",
        description=(
            "Single-model Qwen3.5 baseline: qwen3.5:4b for routing and all specialist agents."
        ),
        orchestrator_model="qwen3.5:4b",
        scenario_model="qwen3.5:4b",
        process_coach_model="qwen3.5:4b",
        ap_plus_navigator_model="qwen3.5:4b",
    ),
    ModelConfiguration(
        name="C_qwen35_4b_router_9b_specialists",
        description=(
            "Two-model candidate: qwen3.5:4b for semantic routing "
            "and qwen3.5:9b for all specialist agents."
        ),
        orchestrator_model="qwen3.5:4b",
        scenario_model="qwen3.5:9b",
        process_coach_model="qwen3.5:9b",
        ap_plus_navigator_model="qwen3.5:9b",
    ),
]


EVALUATION_CASES = [
    # ---------------------------------------------------------
    # Scenario Agent
    # ---------------------------------------------------------
    EvaluationCase(
        case_id="scenario_01",
        category="scenario",
        expected_route="scenario",
        message=(
            "Before I continue, I need to understand what kind of "
            "organisation I am representing and what situation it "
            "is currently facing."
        ),
        review_focus=(
            "Does the answer clarify the scenario without inventing "
            "company facts that were not supplied?"
        ),
    ),
    EvaluationCase(
        case_id="scenario_02",
        category="scenario",
        expected_route="scenario",
        message=(
            "What should I already know about the company and my role "
            "before making any business decisions in this exercise?"
        ),
        review_focus=(
            "Does the agent distinguish scenario facts from process "
            "instructions and ask for missing information appropriately?"
        ),
    ),
    EvaluationCase(
        case_id="scenario_03",
        category="scenario",
        expected_route="scenario",
        message=(
            "I understand the software, but I am still unsure what "
            "business problem our organisation is supposed to solve."
        ),
        review_focus=(
            "Does the model stay focused on business context rather than AP+ navigation?"
        ),
    ),
    # ---------------------------------------------------------
    # Process Coach
    # ---------------------------------------------------------
    EvaluationCase(
        case_id="process_01",
        category="process_coach",
        expected_route="process_coach",
        message=(
            "I know what was entered in the system, but I do not "
            "understand why the business needs this action before "
            "the next event."
        ),
        review_focus=(
            "Does the answer explain business reasoning rather than "
            "giving software-navigation instructions?"
        ),
    ),
    EvaluationCase(
        case_id="process_02",
        category="process_coach",
        expected_route="process_coach",
        message=(
            "We know that raw materials are required. What should I "
            "think about before deciding what business event happens next?"
        ),
        review_focus=(
            "Does the Process Coach guide reasoning in a Socratic way "
            "rather than simply giving the final answer?"
        ),
    ),
    EvaluationCase(
        case_id="process_03",
        category="process_coach",
        expected_route="process_coach",
        message=(
            "The purchase order has already been created. "
            "What purpose does it serve before the goods arrive?"
        ),
        review_focus=(
            "Does the response correctly reason about the business "
            "process and avoid unnecessary AP+ UI guidance?"
        ),
    ),
    # ---------------------------------------------------------
    # AP+ Navigator
    # ---------------------------------------------------------
    EvaluationCase(
        case_id="applus_01",
        category="ap_plus_navigator",
        expected_route="ap_plus_navigator",
        message=(
            "I know exactly what business information must be recorded, "
            "but I cannot work out how to enter it in the ERP application."
        ),
        review_focus=(
            "Does the answer recognize system-navigation intent and "
            "avoid inventing unsupported AP+ screen or field names?"
        ),
    ),
    EvaluationCase(
        case_id="applus_02",
        category="ap_plus_navigator",
        expected_route="ap_plus_navigator",
        message=(
            "The next business decision is already clear to me. "
            "I only need help performing it inside AP+."
        ),
        review_focus=(
            "Does the response remain procedural and grounded, asking "
            "for screen context when necessary?"
        ),
    ),
    EvaluationCase(
        case_id="applus_03",
        category="ap_plus_navigator",
        expected_route="ap_plus_navigator",
        message=(
            "I understand why the supplier data is needed. "
            "Where should I work with that information in AP+?"
        ),
        review_focus=(
            "Does the agent avoid business-process explanation and focus on AP+ navigation?"
        ),
    ),
    # ---------------------------------------------------------
    # Ambiguous / fallback
    # ---------------------------------------------------------
    EvaluationCase(
        case_id="fallback_01",
        category="fallback",
        expected_route="fallback",
        message="Hello there.",
        review_focus=(
            "Does the system ask for useful clarification instead "
            "of making an arbitrary routing decision?"
        ),
    ),
    # ---------------------------------------------------------
    # Deliberately mixed cases
    # ---------------------------------------------------------
    EvaluationCase(
        case_id="mixed_01",
        category="mixed",
        expected_route="process_coach",
        message=(
            "I already clicked the button and know where the field is, "
            "but I still do not understand why the company must perform "
            "this action."
        ),
        review_focus=(
            "The immediate blocking need is conceptual reasoning, "
            "despite the message containing UI terminology."
        ),
    ),
    EvaluationCase(
        case_id="mixed_02",
        category="mixed",
        expected_route="ap_plus_navigator",
        message=(
            "I understand why the process requires this information "
            "and what should happen next, but I cannot find how to "
            "perform the action in the application."
        ),
        review_focus=(
            "The immediate blocking need is software execution, "
            "despite business-process language in the message."
        ),
    ),
]


def apply_model_configuration(
    configuration: ModelConfiguration,
) -> None:
    """Apply one model configuration to the global application settings."""

    settings.orchestrator_model = configuration.orchestrator_model

    settings.scenario_agent_model = configuration.scenario_model

    settings.process_coach_agent_model = configuration.process_coach_model

    settings.ap_plus_navigator_model = configuration.ap_plus_navigator_model


def get_nested(
    data: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = data.get(key)

    if isinstance(value, dict):
        return value

    return {}


async def run_case(
    *,
    configuration: ModelConfiguration,
    case: EvaluationCase,
    case_index: int,
) -> EvaluationResult:
    """Run one evaluation case with one model configuration."""

    service = create_orchestration_service()

    started = time.perf_counter()

    try:
        response = await service.execute(
            OrchestrationRequest(
                session_id=(f"model-eval-{configuration.name}-{case_index}"),
                student_id="model-evaluation-student",
                user_message=case.message,
                metadata={
                    "source": "model-selection-evaluation",
                    "configuration": configuration.name,
                    "case_id": case.case_id,
                },
            )
        )

        wall_time = time.perf_counter() - started

        metadata = dict(response.metadata)

        orchestrator_metadata = get_nested(
            metadata,
            "orchestrator_metadata",
        )

        agent_metadata = get_nested(
            metadata,
            "agent_metadata",
        )

        actual_route = response.route

        return EvaluationResult(
            configuration=configuration.name,
            case_id=case.case_id,
            category=case.category,
            expected_route=case.expected_route,
            actual_route=actual_route,
            routing_passed=(actual_route == case.expected_route),
            semantic_route=metadata.get("semantic_route"),
            semantic_routing_failed=metadata.get("semantic_routing_failed"),
            keyword_fallback_used=metadata.get("keyword_fallback_used"),
            selected_agent=metadata.get("selected_agent"),
            orchestrator_model=orchestrator_metadata.get("model"),
            agent_model=agent_metadata.get("model"),
            orchestrator_latency_ms=orchestrator_metadata.get("model_latency_ms"),
            agent_latency_ms=agent_metadata.get("model_latency_ms"),
            total_wall_time_seconds=round(
                wall_time,
                3,
            ),
            final_response=response.final_response,
            error=response.error,
            review_focus=case.review_focus,
        )

    except Exception as exc:
        wall_time = time.perf_counter() - started

        return EvaluationResult(
            configuration=configuration.name,
            case_id=case.case_id,
            category=case.category,
            expected_route=case.expected_route,
            actual_route=None,
            routing_passed=False,
            semantic_route=None,
            semantic_routing_failed=True,
            keyword_fallback_used=None,
            selected_agent=None,
            orchestrator_model=None,
            agent_model=None,
            orchestrator_latency_ms=None,
            agent_latency_ms=None,
            total_wall_time_seconds=round(
                wall_time,
                3,
            ),
            final_response="",
            error=str(exc),
            review_focus=case.review_focus,
        )


def print_case_result(
    result: EvaluationResult,
) -> None:
    """Print one readable model-evaluation result."""

    print()
    print("-" * 88)

    print(f"{result.configuration} | {result.case_id}")

    print("-" * 88)

    print(f"Expected route : {result.expected_route}")

    print(f"Actual route   : {result.actual_route}")

    print(f"Routing result : {'PASS' if result.routing_passed else 'FAIL'}")

    print(f"Semantic route : {result.semantic_route}")

    print(f"Semantic failed: {result.semantic_routing_failed}")

    print(f"Fallback used  : {result.keyword_fallback_used}")

    print(f"Router model   : {result.orchestrator_model}")

    print(f"Agent model    : {result.agent_model}")

    print(f"Router latency : {result.orchestrator_latency_ms} ms")

    print(f"Agent latency  : {result.agent_latency_ms} ms")

    print(f"Wall time      : {result.total_wall_time_seconds:.3f}s")

    if result.error:
        print()
        print(f"ERROR: {result.error}")

    print()
    print("REVIEW FOCUS")
    print(result.review_focus)

    print()
    print("FINAL RESPONSE")
    print(result.final_response)


def average(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return round(
        sum(values) / len(values),
        2,
    )


def print_configuration_summary(
    *,
    configuration: ModelConfiguration,
    results: list[EvaluationResult],
) -> None:
    """Print aggregate metrics for one model assignment."""

    routing_passes = sum(result.routing_passed for result in results)

    total = len(results)

    router_latencies = [
        result.orchestrator_latency_ms
        for result in results
        if result.orchestrator_latency_ms is not None
    ]

    agent_latencies = [
        result.agent_latency_ms for result in results if result.agent_latency_ms is not None
    ]

    wall_times = [result.total_wall_time_seconds for result in results]

    fallback_count = sum(result.keyword_fallback_used is True for result in results)

    semantic_failures = sum(result.semantic_routing_failed is True for result in results)

    print()
    print("=" * 88)
    print(f"SUMMARY: {configuration.name}")
    print("=" * 88)

    print(configuration.description)

    print()

    print(f"Routing accuracy     : {routing_passes}/{total} ({routing_passes / total * 100:.1f}%)")

    print(f"Semantic failures    : {semantic_failures}")

    print(f"Keyword fallbacks    : {fallback_count}")

    print(f"Avg router latency   : {average(router_latencies)} ms")

    print(f"Avg agent latency    : {average(agent_latencies)} ms")

    print(f"Avg end-to-end time  : {average(wall_times)} s")


def write_json_report(
    *,
    output_path: Path,
    results: list[EvaluationResult],
) -> None:
    payload = [asdict(result) for result in results]

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_csv_report(
    *,
    output_path: Path,
    results: list[EvaluationResult],
) -> None:
    if not results:
        return

    rows = [asdict(result) for result in results]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    """Evaluate candidate model assignments for FREDi."""

    backend_root = Path(__file__).resolve().parents[1]

    output_directory = backend_root / "reports" / "model_evaluation"

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results: list[EvaluationResult] = []

    print()
    print("=" * 88)
    print("FREDi Pre-Sprint-4 Model Selection Evaluation")
    print("=" * 88)

    print(f"Configurations: {len(MODEL_CONFIGURATIONS)}")

    print(f"Cases per configuration: {len(EVALUATION_CASES)}")

    print(f"Total requests: {len(MODEL_CONFIGURATIONS) * len(EVALUATION_CASES)}")

    for configuration in MODEL_CONFIGURATIONS:
        print()
        print("#" * 88)
        print(f"CONFIGURATION: {configuration.name}")
        print("#" * 88)

        print(configuration.description)

        print()
        print(f"Orchestrator : {configuration.orchestrator_model}")

        print(f"Scenario     : {configuration.scenario_model}")

        print(f"Process Coach: {configuration.process_coach_model}")

        print(f"AP+ Navigator: {configuration.ap_plus_navigator_model}")

        apply_model_configuration(configuration)

        configuration_results: list[EvaluationResult] = []

        for case_index, case in enumerate(
            EVALUATION_CASES,
            start=1,
        ):
            result = await run_case(
                configuration=configuration,
                case=case,
                case_index=case_index,
            )

            configuration_results.append(result)

            all_results.append(result)

            print_case_result(result)

        print_configuration_summary(
            configuration=configuration,
            results=configuration_results,
        )

    json_path = output_directory / "model_evaluation_results.json"

    csv_path = output_directory / "model_evaluation_results.csv"

    write_json_report(
        output_path=json_path,
        results=all_results,
    )

    write_csv_report(
        output_path=csv_path,
        results=all_results,
    )

    print()
    print("=" * 88)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 88)

    print(f"JSON report: {json_path}")

    print(f"CSV report : {csv_path}")

    print()
    print("Routing accuracy and latency are measured automatically.")

    print(
        "Review specialist responses manually using the "
        "'review_focus' field before selecting the final models."
    )

    print()


if __name__ == "__main__":
    asyncio.run(main())

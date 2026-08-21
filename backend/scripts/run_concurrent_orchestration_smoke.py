from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from app.orchestration import (
    OrchestrationRequest,
    create_orchestration_service,
)


@dataclass(frozen=True)
class SmokeCase:
    """One independent simulated student request."""

    name: str
    student_id: str
    expected_route: str
    message: str


@dataclass(frozen=True)
class SmokeResult:
    """Result captured from one concurrent request."""

    case: SmokeCase
    actual_route: str
    selected_agent: str | None
    semantic_route: str | None
    semantic_failed: bool | None
    keyword_fallback_used: bool | None
    response: str
    error: str | None
    elapsed_seconds: float
    metadata: dict[str, Any]

    @property
    def passed(self) -> bool:
        return (
            self.actual_route == self.case.expected_route
            and self.error is None
            and self.semantic_failed is False
        )


TEST_CASES = [
    SmokeCase(
        name="Student A - Scenario",
        student_id="student-a",
        expected_route="scenario",
        message=(
            "Before continuing, I need to understand "
            "what kind of organisation I am representing "
            "and what situation it is currently facing."
        ),
    ),
    SmokeCase(
        name="Student B - Process Coach",
        student_id="student-b",
        expected_route="process_coach",
        message=(
            "The action itself is clear to me, "
            "but I do not understand why the business "
            "needs to perform it at this point."
        ),
    ),
    SmokeCase(
        name="Student C - AP+ Navigator",
        student_id="student-c",
        expected_route="ap_plus_navigator",
        message=(
            "I already understand what the business action is. "
            "I need help carrying it out inside the ERP software."
        ),
    ),
    SmokeCase(
        name="Student D - Scenario",
        student_id="student-d",
        expected_route="scenario",
        message=(
            "Who am I acting as in this exercise, "
            "and what facts about the organisation "
            "should I already know?"
        ),
    ),
    SmokeCase(
        name="Student E - Process Coach",
        student_id="student-e",
        expected_route="process_coach",
        message=(
            "I know what was entered, but what business purpose "
            "does that action serve before the next event happens?"
        ),
    ),
    SmokeCase(
        name="Student F - AP+ Navigator",
        student_id="student-f",
        expected_route="ap_plus_navigator",
        message=(
            "The next business decision is already clear. "
            "I just do not know how to record it "
            "in the application."
        ),
    ),
]


async def execute_case(
    *,
    service: Any,
    index: int,
    case: SmokeCase,
) -> SmokeResult:
    """Execute one student's request through the shared service."""

    session_id = f"concurrency-smoke-{index}"

    started = time.perf_counter()

    response = await service.execute(
        OrchestrationRequest(
            session_id=session_id,
            student_id=case.student_id,
            user_message=case.message,
            metadata={
                "source": ("concurrent-orchestration-smoke"),
                "case_index": index,
                "case_name": case.name,
            },
        )
    )

    elapsed = time.perf_counter() - started

    metadata = dict(response.metadata)

    return SmokeResult(
        case=case,
        actual_route=response.route,
        selected_agent=metadata.get("selected_agent"),
        semantic_route=metadata.get("semantic_route"),
        semantic_failed=metadata.get("semantic_routing_failed"),
        keyword_fallback_used=metadata.get("keyword_fallback_used"),
        response=response.final_response,
        error=response.error,
        elapsed_seconds=elapsed,
        metadata=metadata,
    )


def print_result(
    *,
    index: int,
    result: SmokeResult,
) -> None:
    """Print one student's isolated orchestration result."""

    print()
    print("=" * 88)
    print(f"CASE {index}: {result.case.name}")
    print("=" * 88)

    print(f"Student ID      : {result.case.student_id}")

    print(f"Expected route  : {result.case.expected_route}")

    print(f"Actual route    : {result.actual_route}")

    print(f"Semantic route  : {result.semantic_route}")

    print(f"Selected agent  : {result.selected_agent}")

    print(f"Semantic failed : {result.semantic_failed}")

    print(f"Keyword fallback: {result.keyword_fallback_used}")

    print(f"Elapsed         : {result.elapsed_seconds:.2f}s")

    print(f"Result          : {'PASS' if result.passed else 'FAIL'}")

    if result.error is not None:
        print()
        print("ERROR")
        print("-" * 88)
        print(result.error)

    print()
    print("FINAL RESPONSE")
    print("-" * 88)
    print(result.response)


def print_summary(
    *,
    results: list[SmokeResult],
    total_elapsed: float,
) -> None:
    """Print overall concurrent-run statistics."""

    passed = sum(result.passed for result in results)

    failed = len(results) - passed

    sum_individual_time = sum(result.elapsed_seconds for result in results)

    print()
    print("=" * 88)
    print("CONCURRENT EXECUTION SUMMARY")
    print("=" * 88)

    print(f"Requests               : {len(results)}")

    print(f"Passed                 : {passed}")

    print(f"Failed                 : {failed}")

    print(f"Total wall-clock time  : {total_elapsed:.2f}s")

    print(f"Sum individual times   : {sum_individual_time:.2f}s")

    if total_elapsed > 0:
        overlap_ratio = sum_individual_time / total_elapsed

        print(f"Concurrency overlap    : {overlap_ratio:.2f}x")

    print()

    if failed == 0:
        print("RESULT: ALL CONCURRENT REQUESTS PASSED")
    else:
        print("RESULT: ONE OR MORE REQUESTS FAILED")

    print("=" * 88)


async def main() -> None:
    """Run multiple independent students concurrently."""

    service = create_orchestration_service()

    print()
    print("=" * 88)
    print("FREDi Sprint 3 - Concurrent Multi-User Smoke Test")
    print("=" * 88)

    print(f"Launching {len(TEST_CASES)} student requests concurrently...")

    started = time.perf_counter()

    results = await asyncio.gather(
        *(
            execute_case(
                service=service,
                index=index,
                case=case,
            )
            for index, case in enumerate(
                TEST_CASES,
                start=1,
            )
        )
    )

    total_elapsed = time.perf_counter() - started

    for index, result in enumerate(
        results,
        start=1,
    ):
        print_result(
            index=index,
            result=result,
        )

    print_summary(
        results=list(results),
        total_elapsed=total_elapsed,
    )


if __name__ == "__main__":
    asyncio.run(main())

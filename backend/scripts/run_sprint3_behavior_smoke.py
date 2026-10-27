from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.orchestration import (  # noqa: E402
    OrchestrationRequest,
    OrchestrationServiceError,
    create_orchestration_service,
)
from scripts.smoke_support import (  # noqa: E402
    ensure_students,
    new_session_id,
)

STUDENT_ID = "manual-test-student"

TEST_CASES = [
    (
        "Scenario Agent",
        (
            "Our company manufactures bicycles, but I am not sure "
            "how to define the business problem."
        ),
    ),
    (
        "Process Coach Agent",
        ("We know we need raw materials. Tell me exactly what business step I should do next."),
    ),
    (
        "AP+ Navigator Agent",
        ("Where can I find the supplier field in AP+?"),
    ),
    (
        "Fallback",
        "Hello there.",
    ),
]


async def main() -> None:
    await ensure_students([STUDENT_ID])

    service = create_orchestration_service()

    for index, (name, message) in enumerate(
        TEST_CASES,
        start=1,
    ):
        try:
            response = await service.execute(
                OrchestrationRequest(
                    session_id=new_session_id("sprint3-smoke"),
                    student_id=STUDENT_ID,
                    user_message=message,
                    metadata={
                        "source": "sprint3-behavior-smoke",
                        "case_index": index,
                    },
                )
            )
        except OrchestrationServiceError as exc:
            print()
            print("=" * 72)
            print(name)
            print("=" * 72)
            print(f"ERROR: {exc}")
            continue

        print()
        print("=" * 72)
        print(name)
        print("=" * 72)
        print(f"Route: {response.route}")
        print(f"Agent: {response.metadata.get('selected_agent')}")
        print(f"Guidance: {response.metadata.get('agent_metadata', {}).get('guidance_level')}")
        print()
        print(response.final_response)


if __name__ == "__main__":
    asyncio.run(main())

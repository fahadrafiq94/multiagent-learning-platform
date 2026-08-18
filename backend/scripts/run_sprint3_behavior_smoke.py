from __future__ import annotations

import asyncio

from app.orchestration import (
    OrchestrationRequest,
    create_orchestration_service,
)

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
    service = create_orchestration_service()

    for index, (name, message) in enumerate(
        TEST_CASES,
        start=1,
    ):
        response = await service.execute(
            OrchestrationRequest(
                session_id=f"sprint3-smoke-{index}",
                student_id="manual-test-student",
                user_message=message,
                metadata={
                    "source": "sprint3-behavior-smoke",
                },
            )
        )

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

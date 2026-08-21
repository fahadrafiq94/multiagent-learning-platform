from __future__ import annotations

import asyncio

from app.orchestration import (
    OrchestrationRequest,
    create_orchestration_service,
)

TEST_CASES = [
    (
        "Expected: AP+ Navigator",
        (
            "I already know which supplier and material I need, "
            "but I cannot work out how to record that information "
            "in the system."
        ),
    ),
    (
        "Expected: Process Coach",
        (
            "I have completed the purchase order. "
            "What purpose does it serve before the goods arrive?"
        ),
    ),
    (
        "Expected: Scenario Agent",
        ("What kind of organisation am I supposed to represent in this exercise?"),
    ),
    (
        "Expected: AP+ Navigator",
        (
            "I know the next business action, "
            "but I am unsure how to carry it out "
            "inside the software."
        ),
    ),
    (
        "Expected: Process Coach",
        ("I understand what I clicked, but I do not understand why the company needs this action."),
    ),
    (
        "Expected: Scenario Agent",
        ("What information about the organisation should I already know before continuing?"),
    ),
    (
        "Expected: AP+ Navigator",
        (
            "The business decision is already clear to me. "
            "I just need help performing it in the ERP system."
        ),
    ),
    (
        "Expected: Process Coach",
        (
            "I know what I entered, but I do not understand "
            "the business reason behind doing it at this point."
        ),
    ),
    (
        "Expected: Scenario Agent",
        (
            "Before I continue, I need to understand "
            "who I am acting as and what situation "
            "the organisation is currently facing."
        ),
    ),
    (
        "Expected: Fallback",
        ("Hello there."),
    ),
]


async def main() -> None:
    service = create_orchestration_service()

    print()
    print("=" * 88)
    print("FREDi Sprint 3 - Semantic Routing Smoke Evaluation")
    print("=" * 88)

    for index, (expected, message) in enumerate(
        TEST_CASES,
        start=1,
    ):
        response = await service.execute(
            OrchestrationRequest(
                session_id=f"semantic-smoke-{index}",
                student_id="manual-test-student",
                user_message=message,
                metadata={
                    "source": "semantic-routing-smoke",
                    "test_case": index,
                },
            )
        )

        metadata = response.metadata

        agent_metadata = metadata.get(
            "agent_metadata",
            {},
        )

        orchestrator_metadata = metadata.get(
            "orchestrator_metadata",
            {},
        )

        print()
        print("=" * 88)
        print(f"CASE {index}")
        print("=" * 88)

        print(f"Expected: {expected}")
        print()
        print("User message:")
        print(message)

        print()
        print("-" * 88)
        print("ROUTING RESULT")
        print("-" * 88)

        print(f"Final route: {response.route}")

        print(f"Selected agent: {metadata.get('selected_agent')}")

        print(f"Route reason: {response.route_reason}")

        print()
        print("-" * 88)
        print("SEMANTIC ROUTER")
        print("-" * 88)

        print(f"Semantic routing attempted: {metadata.get('semantic_routing_attempted')}")

        print(f"Semantic routing failed: {metadata.get('semantic_routing_failed')}")

        print(f"Semantic route: {metadata.get('semantic_route')}")

        print(f"Semantic reason: {metadata.get('semantic_route_reason')}")

        print(f"Routing strategy: {metadata.get('routing_strategy')}")

        print(f"Orchestrator model: {orchestrator_metadata.get('model')}")

        print(f"Orchestrator provider: {orchestrator_metadata.get('model_provider')}")

        print(f"Orchestrator latency: {orchestrator_metadata.get('model_latency_ms')} ms")

        print()
        print("-" * 88)
        print("KEYWORD FALLBACK")
        print("-" * 88)

        print(f"Keyword fallback used: {metadata.get('keyword_fallback_used')}")

        print(f"Keyword fallback trigger: {metadata.get('keyword_fallback_trigger')}")

        print(f"Keyword route: {metadata.get('keyword_route')}")

        print(f"Keyword reason: {metadata.get('keyword_route_reason')}")

        print()
        print("-" * 88)
        print("SPECIALIST AGENT")
        print("-" * 88)

        print(f"Agent called: {metadata.get('agent_called')}")

        print(f"Guidance level: {agent_metadata.get('guidance_level')}")

        print(f"Guidance mode: {agent_metadata.get('guidance_mode')}")

        print(f"Agent model: {agent_metadata.get('model')}")

        print(f"Agent provider: {agent_metadata.get('model_provider')}")

        print(f"Agent latency: {agent_metadata.get('model_latency_ms')} ms")

        print()
        print("-" * 88)
        print("FINAL RESPONSE")
        print("-" * 88)

        print(response.final_response)

        if response.error is not None:
            print()
            print("-" * 88)
            print("ERROR")
            print("-" * 88)

            print(response.error)

            if metadata.get("orchestrator_error") is not None:
                print(f"Orchestrator error: {metadata.get('orchestrator_error')}")

            if metadata.get("agent_error") is not None:
                print(f"Agent error: {metadata.get('agent_error')}")

    print()
    print("=" * 88)
    print("Semantic routing smoke evaluation complete.")
    print("=" * 88)
    print()


if __name__ == "__main__":
    asyncio.run(main())

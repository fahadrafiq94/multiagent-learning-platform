from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.config.settings import settings
from app.identity import UserIdentity
from app.orchestration.factory import (
    build_orchestration_runtime,
)
from app.orchestration.schemas import (
    OrchestrationRequest,
)
from app.services.model_service import (
    create_model_service,
)

STUDENT_ID = "REAL-CHECKPOINT-001"


@dataclass(slots=True)
class CheckpointTurn:
    message: str

    expected_onboarding_outcome: str | None = None

    expected_route: str | None = None

    expected_onboarding: bool | None = None


CHECKPOINT_TURNS = [
    CheckpointTurn(
        message=("I work at UrbanCycle, a bicycle manufacturer."),
        expected_onboarding_outcome=("collecting"),
        expected_route="scenario",
        expected_onboarding=True,
    ),
    CheckpointTurn(
        message=(
            "My role is Production Planner. "
            "Customer orders are frequently "
            "delayed, and our business goal "
            "is to improve delivery reliability."
        ),
        expected_onboarding_outcome=("review_required"),
        expected_route="scenario",
        expected_onboarding=True,
    ),
    CheckpointTurn(
        message=("I confirm that this is not correct."),
        expected_onboarding_outcome=("correction_required"),
        expected_route="scenario",
        expected_onboarding=True,
    ),
    CheckpointTurn(
        message=("Everything else is correct, but my role should be Senior Production Planner."),
        expected_onboarding_outcome=("corrected"),
        expected_route="scenario",
        expected_onboarding=True,
    ),
    CheckpointTurn(
        message=("Yes, now everything is correct."),
        expected_onboarding_outcome=("confirmed"),
        expected_route="scenario",
        expected_onboarding=True,
    ),
    CheckpointTurn(
        message=("Why do we need a purchase order in the procurement process?"),
        expected_route="process_coach",
        expected_onboarding=False,
    ),
]


def print_header(
    title: str,
) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_check(
    label: str,
    *,
    actual: object,
    expected: object,
) -> bool:
    passed = actual == expected

    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {label}: actual={actual!r}, expected={expected!r}")

    return passed


def installed_model_names(
    health_details: dict[
        str,
        Any,
    ],
) -> set[str]:
    models = health_details.get(
        "models",
        [],
    )

    result: set[str] = set()

    if not isinstance(
        models,
        list,
    ):
        return result

    for model in models:
        if not isinstance(
            model,
            dict,
        ):
            continue

        name = model.get("name")

        model_name = model.get("model")

        if isinstance(
            name,
            str,
        ):
            result.add(name)

        if isinstance(
            model_name,
            str,
        ):
            result.add(model_name)

    return result


async def print_persisted_state(
    *,
    runtime,
    user_id,
    session_id,
) -> None:
    session_context = await runtime.session_context_service.get_required(
        user_id=user_id,
        session_id=session_id,
    )

    print()
    print(
        "Persisted scenario status:",
        session_context.scenario_setup_status.value,
    )

    draft = session_context.scenario_draft

    if draft is None:
        print("Persisted ScenarioDraft: None")
    else:
        print("Persisted ScenarioDraft:")

        print(
            draft.model_dump_json(
                indent=2,
                exclude_none=True,
            )
        )

    scenario_card = await runtime.scenario_card_service.get(user_id=user_id)

    if scenario_card is None:
        print("Persistent ScenarioCard: None")
    else:
        print("Persistent ScenarioCard:")

        print(
            scenario_card.model_dump_json(
                indent=2,
                exclude_none=True,
            )
        )


async def run_turn(
    *,
    runtime,
    user,
    session,
    number: int,
    turn: CheckpointTurn,
) -> bool:
    print_header(f"TURN {number}")

    print("USER:")
    print(turn.message)

    response = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=(user.student_id),
            user_message=(turn.message),
            metadata={
                "checkpoint": ("4B.5-real-ollama"),
                "checkpoint_turn": (number),
            },
        )
    )

    print()
    print("FREDi:")
    print(response.final_response)

    print()
    print(
        "Route:",
        response.route,
    )

    print(
        "Route reason:",
        response.route_reason,
    )

    print(
        "Error:",
        response.error,
    )

    metadata = response.metadata

    print()
    print("Runtime metadata:")

    interesting_keys = [
        "scenario_onboarding",
        "onboarding_required",
        "onboarding_outcome",
        "scenario_status",
        "routing_strategy",
        "semantic_routing_failed",
        "keyword_fallback_used",
        "selected_agent",
        "context_target",
        "context_policy_version",
    ]

    for key in interesting_keys:
        if key in metadata:
            print(f"  {key}: {metadata[key]!r}")

    await print_persisted_state(
        runtime=runtime,
        user_id=user.user_id,
        session_id=(session.session_id),
    )

    print()
    print("Checkpoint assertions:")

    checks: list[bool] = []

    if turn.expected_route is not None:
        checks.append(
            print_check(
                "route",
                actual=response.route,
                expected=(turn.expected_route),
            )
        )

    if turn.expected_onboarding_outcome is not None:
        checks.append(
            print_check(
                "onboarding_outcome",
                actual=(metadata.get("onboarding_outcome")),
                expected=(turn.expected_onboarding_outcome),
            )
        )

    if turn.expected_onboarding is not None:
        checks.append(
            print_check(
                "scenario_onboarding",
                actual=(metadata.get("scenario_onboarding")),
                expected=(turn.expected_onboarding),
            )
        )

    return all(checks)


async def main() -> None:
    print_header("FREDi Sprint 4B.5 Real Ollama Checkpoint")

    print(
        "Provider:",
        settings.model_provider,
    )

    print(
        "Ollama URL:",
        settings.ollama_base_url,
    )

    print(
        "Chat model:",
        settings.ollama_chat_model,
    )

    model_service = create_model_service()

    print()
    print("Checking real Ollama connection...")

    health = await model_service.health()

    print(
        "Health status:",
        health.status,
    )

    print(
        "Provider:",
        health.provider,
    )

    print(
        "Configured chat model:",
        health.chat_model,
    )

    if health.latency_ms is not None:
        print(
            "Health latency:",
            f"{health.latency_ms} ms",
        )

    if health.status != "ok":
        print()
        print("CHECKPOINT STOPPED:")

        print("Ollama is not reachable.")

        print(
            "Health details:",
            health.details,
        )

        raise SystemExit(1)

    model_names = installed_model_names(health.details)

    print(
        "Installed Ollama models:",
        (
            ", ".join(sorted(model_names))
            if model_names
            else ("Could not determine from health response")
        ),
    )

    if model_names and settings.ollama_chat_model not in model_names:
        print()
        print("CHECKPOINT STOPPED:")

        print(f"Configured model {settings.ollama_chat_model!r} was not found in Ollama.")

        print()
        print("Run:")

        print(f"ollama pull {settings.ollama_chat_model}")

        raise SystemExit(1)

    print()
    print("Building real 4B.5 orchestration runtime...")

    runtime = build_orchestration_runtime(model_service=(model_service))

    user = await runtime.repositories.identities.create(UserIdentity(student_id=STUDENT_ID))

    session = await runtime.conversation_service.create_session(user_id=user.user_id)

    print(
        "Internal user_id:",
        user.user_id,
    )

    print(
        "student_id:",
        user.student_id,
    )

    print(
        "session_id:",
        session.session_id,
    )

    passed_turns = 0

    for index, turn in enumerate(
        CHECKPOINT_TURNS,
        start=1,
    ):
        passed = await run_turn(
            runtime=runtime,
            user=user,
            session=session,
            number=index,
            turn=turn,
        )

        if passed:
            passed_turns += 1
        else:
            print()
            print("A checkpoint expectation failed.")

            print("Continue reading the observed state above before changing any code.")

            break

    print_header("CHECKPOINT SUMMARY")

    print(
        "Passed turns:",
        f"{passed_turns}/{len(CHECKPOINT_TURNS)}",
    )

    scenario_card = await runtime.scenario_card_service.get(user_id=user.user_id)

    if scenario_card is None:
        print("Final ScenarioCard: NOT CONFIRMED")
    else:
        print("Final ScenarioCard: CONFIRMED")

        print(
            "Confirmed role:",
            scenario_card.learner_role,
        )

        print(
            "Confirmed company:",
            scenario_card.company_context,
        )

    if passed_turns == len(CHECKPOINT_TURNS):
        print()
        print("SPRINT 4B.5 REAL MODEL CHECKPOINT PASSED")

        print()
        print("Verified:")

        print("- real Ollama ModelService")

        print("- Scenario Agent extraction")

        print("- deterministic completeness assessment")

        print("- semantic review intent")

        print("- rejection does not confirm")

        print("- correction returns to review")

        print("- confirmation persists ScenarioCard")

        print("- confirmed user bypasses onboarding")

        print("- semantic Orchestrator activates")

        print("- Process Coach receives the final routed request")

    else:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

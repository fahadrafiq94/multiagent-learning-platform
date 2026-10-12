from __future__ import annotations

import json
from typing import cast
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import httpx
import pytest

from app.api.routes import (
    orchestration as orchestration_routes,
)
from app.context import ScenarioDraft
from app.identity import UserIdentity
from app.main import app
from app.orchestration.factory import (
    build_orchestration_runtime,
)
from app.services.model_service import (
    ModelService,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)


def create_model_service() -> tuple[
    ModelService,
    AsyncMock,
    list[ModelChatRequest],
]:
    model_service = cast(
        ModelService,
        MagicMock(spec=ModelService),
    )

    requests: list[ModelChatRequest] = []

    async def generate(
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        requests.append(request)

        task = request.metadata.get("task")

        if task == ("scenario_draft_extraction"):
            content = json.dumps(
                {
                    "company_context": ("UrbanCycle manufactures bicycles."),
                    "learner_role": ("Production Planner"),
                    "business_problem": ("Customer orders are delayed."),
                    "business_goals": [("Improve delivery reliability.")],
                    "constraints": [],
                    "proposed_facts": [],
                }
            )

        elif task == "scenario_review_intent":
            content = json.dumps(
                {
                    "intent": "confirm",
                    "reason": ("The user accepted the draft."),
                    "correction": None,
                }
            )

        elif task == "semantic_routing":
            message = request.messages[-1].content.casefold()

            if "ap+" in message:
                route = "ap_plus_navigator"
            elif "why" in message or "procurement" in message:
                route = "process_coach"
            elif "scenario" in message or "company" in message:
                route = "scenario"
            else:
                route = "fallback"

            content = json.dumps(
                {
                    "route": route,
                    "reason": ("The current request matches this route."),
                }
            )

        else:
            content = "Integrated API specialist response."

        return ModelChatResponse(
            content=content,
            model="fake-model",
            provider="ollama",
            latency_ms=5.0,
        )

    generate_mock = AsyncMock(side_effect=generate)

    model_service.generate = generate_mock  # type: ignore[method-assign]

    return (
        model_service,
        generate_mock,
        requests,
    )


async def create_user_session(
    runtime,
):
    user = await runtime.repositories.identities.create(UserIdentity(student_id="K12345"))

    session = await runtime.conversation_service.create_session(user_id=user.user_id)

    return user, session


async def create_client(
    runtime,
) -> httpx.AsyncClient:
    app.dependency_overrides[orchestration_routes.get_orchestration_service] = lambda: (
        runtime.orchestration_service
    )

    transport = httpx.ASGITransport(app=app)

    return httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    )


@pytest.mark.asyncio
async def test_http_new_user_enters_scenario_onboarding() -> None:
    (
        model_service,
        generate_mock,
        requests,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=(model_service))

    user, session = await create_user_session(runtime)

    client = await create_client(runtime)

    try:
        async with client:
            response = await client.post(
                "/orchestration/invoke",
                json={
                    "session_id": str(session.session_id),
                    "student_id": (user.student_id),
                    "user_message": (
                        "I am a Production "
                        "Planner at "
                        "UrbanCycle. Orders "
                        "are delayed and we "
                        "want reliable "
                        "delivery dates."
                    ),
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["route"] == "scenario"

    assert body["metadata"]["scenario_onboarding"] is True

    assert body["metadata"]["onboarding_outcome"] == "review_required"

    assert generate_mock.await_count == 1

    assert requests[0].metadata["task"] == ("scenario_draft_extraction")


@pytest.mark.asyncio
async def test_http_confirmed_user_uses_normal_routing() -> None:
    (
        model_service,
        generate_mock,
        _,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=(model_service))

    user, session = await create_user_session(runtime)

    await runtime.scenario_card_service.confirm_draft(
        user_id=user.user_id,
        draft=ScenarioDraft(
            user_id=user.user_id,
            session_id=(session.session_id),
            company_context=("UrbanCycle manufactures bicycles."),
            learner_role=("Production Planner"),
            business_problem=("Customer orders are delayed."),
            business_goals=[("Improve delivery reliability.")],
        ),
    )

    client = await create_client(runtime)

    try:
        async with client:
            response = await client.post(
                "/orchestration/invoke",
                json={
                    "session_id": str(session.session_id),
                    "student_id": (user.student_id),
                    "user_message": ("Why is a purchase order needed in procurement?"),
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["route"] == "process_coach"

    assert body["metadata"]["selected_agent"] == "process_coach"

    assert body["final_response"] == ("Integrated API specialist response.")

    assert generate_mock.await_count == 2


@pytest.mark.asyncio
async def test_http_ap_plus_routes_to_navigator_after_onboarding() -> None:
    (
        model_service,
        _,
        _,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=(model_service))

    user, session = await create_user_session(runtime)

    await runtime.scenario_card_service.confirm_draft(
        user_id=user.user_id,
        draft=ScenarioDraft(
            user_id=user.user_id,
            session_id=(session.session_id),
            company_context=("UrbanCycle manufactures bicycles."),
            learner_role=("Production Planner"),
            business_problem=("Orders are delayed."),
            business_goals=[("Improve delivery reliability.")],
        ),
    )

    client = await create_client(runtime)

    try:
        async with client:
            response = await client.post(
                "/orchestration/invoke",
                json={
                    "session_id": str(session.session_id),
                    "student_id": (user.student_id),
                    "user_message": ("Where do I find this field in AP+?"),
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["route"] == "ap_plus_navigator"

    assert body["metadata"]["selected_agent"] == "ap_plus_navigator"


@pytest.mark.asyncio
async def test_http_invalid_request_without_student_id_returns_422() -> None:
    (
        model_service,
        generate_mock,
        _,
    ) = create_model_service()

    runtime = build_orchestration_runtime(model_service=(model_service))

    client = await create_client(runtime)

    try:
        async with client:
            response = await client.post(
                "/orchestration/invoke",
                json={
                    "session_id": ("whatever"),
                    "user_message": ("Hello"),
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422

    generate_mock.assert_not_awaited()

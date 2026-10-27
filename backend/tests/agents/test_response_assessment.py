from __future__ import annotations

import json
from typing import (
    Any,
    cast,
)
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import pytest

from app.agents.ap_plus_navigator import (
    APPlusNavigatorAgent,
)
from app.agents.policies import (
    RESPONSE_ASSESSMENT_INSTRUCTIONS,
)
from app.agents.process_coach import (
    ProcessCoachAgent,
)
from app.agents.scenario import (
    ScenarioAgent,
)
from app.agents.schemas import (
    AgentRequest,
    StudentResponseAssessment,
    UnderstandingLevel,
)
from app.domain.guidance import (
    GuidanceLevel,
)
from app.services.model_service import (
    ModelService,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)

SPECIALISTS = [
    pytest.param(APPlusNavigatorAgent, id=("ap_plus_navigator")),
    pytest.param(ProcessCoachAgent, id=("process_coach")),
    pytest.param(ScenarioAgent, id=("scenario")),
]


def make_agent(
    agent_class: type,
) -> tuple[Any, AsyncMock]:
    model_service = cast(
        ModelService,
        MagicMock(
            spec=(ModelService),
        ),
    )

    generate_mock: AsyncMock = AsyncMock()

    model_service.generate = generate_mock  # type: ignore[method-assign]

    return agent_class(model_service), generate_mock


def get_generated_request(
    generate_mock: AsyncMock,
) -> ModelChatRequest:
    generate_mock.assert_awaited_once()

    call = generate_mock.await_args

    assert call is not None

    request = call.args[0]

    assert isinstance(
        request,
        ModelChatRequest,
    )

    return request


def assessment_payload(
    understanding: str,
    *,
    misconception: bool = False,
    reason: str = ("The student's answer shows the expected understanding."),
) -> dict[str, Any]:
    return {
        "understanding": (understanding),
        "misconception_detected": (misconception),
        "reason": (reason),
    }


def model_response(
    content: str,
    *,
    assessment: (Any | None) = None,
) -> ModelChatResponse:
    payload: dict[str, Any] = {
        "content": (content),
        "state_update_proposal": {
            "operations": [],
        },
    }

    if assessment is not None:
        payload["response_assessment"] = assessment

    return ModelChatResponse(
        content=json.dumps(payload),
        model=("fake-model"),
        provider="ollama",
        latency_ms=10.0,
    )


def create_request() -> AgentRequest:
    return AgentRequest(
        session_id="session-1",
        student_id="student-1",
        user_message=("The invoice is paid after the goods receipt is posted."),
        guidance_level=(GuidanceLevel.GUIDED),
        metadata={
            "source": "test",
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_class", SPECIALISTS)
async def test_specialist_prompt_contains_shared_assessment_rubric(
    agent_class: type,
) -> None:
    agent, generate_mock = make_agent(agent_class)

    generate_mock.return_value = model_response("Let's work through that.")

    response = await agent.execute(create_request())

    prompt = get_generated_request(generate_mock).messages[0].content

    assert RESPONSE_ASSESSMENT_INSTRUCTIONS in prompt

    assert response.agent is not None


@pytest.mark.asyncio
async def test_all_specialists_share_identical_rubric() -> None:
    for agent_class in (APPlusNavigatorAgent, ProcessCoachAgent, ScenarioAgent):
        agent, generate_mock = make_agent(agent_class)

        generate_mock.return_value = model_response("Let's work through that.")

        await agent.execute(create_request())

        prompt = get_generated_request(generate_mock).messages[0].content

        assert RESPONSE_ASSESSMENT_INSTRUCTIONS in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_class", SPECIALISTS)
async def test_specialist_surfaces_valid_assessment(agent_class: type) -> None:
    agent, generate_mock = make_agent(agent_class)

    generate_mock.return_value = model_response(
        ("Let's work through that."),
        assessment=assessment_payload(
            "partially_correct",
            misconception=True,
        ),
    )

    response = await agent.execute(create_request())

    assert isinstance(response.response_assessment, StudentResponseAssessment)

    assert response.response_assessment.understanding is UnderstandingLevel.PARTIALLY_CORRECT

    assert response.response_assessment.misconception_detected is True

    assert response.response_assessment.reason is not None

    assert response.metadata["response_assessed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_class", SPECIALISTS)
async def test_specialist_tolerates_omitted_assessment(agent_class: type) -> None:
    agent, generate_mock = make_agent(agent_class)

    generate_mock.return_value = model_response("Let's work through that.")

    response = await agent.execute(create_request())

    assert response.response_assessment is None

    assert response.content == ("Let's work through that.")

    assert response.metadata["response_assessed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_class", SPECIALISTS)
@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param(
            {
                "understanding": ("plausible"),
                "misconception_detected": (False),
                "reason": ("Bad enum value."),
            },
            id=("wrong_enum"),
        ),
        pytest.param(
            {
                "understanding": ("correct"),
                "misconception_detected": [True],
            },
            id=("non_coercible_bool"),
        ),
        pytest.param(
            "not-an-object",
            id=("scalar_junk"),
        ),
        pytest.param(
            {
                "understanding": {
                    "nested": ("junk"),
                },
                "misconception_detected": [1, 2],
                "reason": 42,
            },
            id=("nested_junk"),
        ),
    ],
)
async def test_specialist_degrades_malformed_assessment_to_none(
    agent_class: type,
    malformed: Any,
) -> None:
    agent, generate_mock = make_agent(agent_class)

    generate_mock.return_value = model_response(
        ("Let's work through that."),
        assessment=(malformed),
    )

    response = await agent.execute(create_request())

    assert response.response_assessment is None

    assert response.content == ("Let's work through that.")

    assert response.state_update_proposal is not None

    assert response.metadata["response_assessed"] is False

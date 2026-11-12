"""Application guards and natural deterministic adaptation with controlled semantic proposals."""

import json

import pytest

from app.agents.assessment_guard import guarded_assessment
from app.agents.citation_support import TEACHING_RULES
from app.agents.process_coach import ProcessCoachAgent
from app.agents.schemas import SpecialistAgentOutput, StudentResponseAssessment, UnderstandingLevel
from app.context.schemas import HintState
from app.observability.events import build_request_event
from app.orchestration.factory import build_orchestration_runtime
from app.orchestration.schemas import OrchestrationRequest
from app.services.model_service.schemas import ModelChatResponse
from tests.agents.test_citation_support import ControlledModel, linked
from tests.agents.test_grounding import request
from tests.orchestration.test_graph import (
    confirm_scenario,
    create_model_service,
    create_user_and_session,
)


def assessment(
    understanding="correct",
    *,
    kind="assessable",
    quote="I think a tag needs an existing asset.",
    misconception=False,
):
    return StudentResponseAssessment(
        assessability=kind,
        student_claim=quote,
        understanding=UnderstandingLevel(understanding),
        misconception_detected=misconception,
    )


@pytest.mark.parametrize(
    "message",
    ["What is the exercise goal?", "Thanks!", "I am now on TASK-G4.", "Can I have one hint?"],
)
def test_non_assessable_messages_cannot_reward_or_punish(message):
    proposal = assessment(kind="not_assessable", quote=message, misconception=True)
    clean, status = guarded_assessment(proposal, message)
    assert status == "not_assessable"
    assert clean.understanding == UnderstandingLevel.NOT_APPLICABLE
    assert clean.misconception_detected is False and clean.student_claim is None


@pytest.mark.parametrize(
    "proposal,status",
    [
        (assessment(kind="uncertain"), "uncertain"),
        (assessment(quote="A historical assistant claim."), "unanchored"),
        (assessment(quote=" "), "unanchored"),
        (assessment(misconception=True), "contradictory"),
        (StudentResponseAssessment(understanding="correct"), "uncertain"),
    ],
)
def test_missing_uncertain_historical_and_contradictory_assessments_are_neutral(proposal, status):
    clean, actual = guarded_assessment(proposal, "I think a tag needs an existing asset.")
    assert clean is None and actual == status


def test_question_mark_does_not_prevent_assessing_an_endorsed_claim():
    clean, status = guarded_assessment(
        assessment(), "I think a tag needs an existing asset. Is that right?"
    )
    assert status == "accepted" and clean.understanding == UnderstandingLevel.CORRECT


def test_generation_requires_student_claim_first_in_each_complete_schema_branch():
    schema = SpecialistAgentOutput.model_json_schema()
    assert next(iter(schema["properties"])) == "response_assessment"
    assert "response_assessment" in schema["required"]
    assert "$ref" in schema["properties"]["response_assessment"]
    for branch in schema["$defs"]["StudentResponseAssessment"]["anyOf"]:
        assert branch["type"] == "object" and branch["additionalProperties"] is False
        assert set(branch["required"]) == {"assessability", "student_claim", "understanding"}
        assert list(branch["properties"])[:2] == ["student_claim", "assessability"]


@pytest.mark.asyncio
@pytest.mark.parametrize("level", list(TEACHING_RULES))
async def test_only_selected_level_policy_is_trusted_in_the_verifier(level):
    req = request().model_copy(update={"guidance_level": level})
    model = ControlledModel(linked("What relationship matters here?"))
    await ProcessCoachAgent(model).execute(req)
    system = model.requests[-1].messages[0].content
    assert "TRUSTED APPLICATION GUIDANCE LEVEL: " + level.value in system
    assert TEACHING_RULES[level] in system
    assert all(rule not in system for other, rule in TEACHING_RULES.items() if other != level)


@pytest.mark.asyncio
@pytest.mark.parametrize("teaching", ["violated", "uncertain", "not_checked"])
async def test_source_supported_answer_can_still_be_withheld_for_teaching(teaching):
    def mutate(response):
        data = json.loads(response.content)
        data["teaching"] = teaching
        return response.model_copy(update={"content": json.dumps(data)})

    model = ControlledModel(
        linked("If delivery is the prerequisite, what comes first?"), mutate=mutate
    )
    result = await ProcessCoachAgent(model).execute(request())
    assert result.content == "Which part of this task would you like to reason through first?"
    assert result.source_references == [] and result.response_assessment is None
    assert result.state_update_proposal.operations == []
    assert result.metadata["grounding"]["support_status"] == "passed"
    assert result.metadata["grounding"]["teaching_abstained"] is True
    assert len(model.requests) == 2
    assert (
        json.loads(model.requests[-1].messages[-1].content)["teaching_policy"]["level"] == "minimal"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        {"question": "No, the batch must exist first. What should you do?"},
        {"question": "First open the record, then select the batch, and save."},
        {"question": "What is missing? What comes next?"},
        {"question": "Which knowledge chunk K1 gives the answer?"},
    ],
)
async def test_obvious_teaching_violations_are_withheld_even_if_judge_would_approve(content):
    model = ControlledModel({"content": content, "citation_ids": [], "claims": []})
    result = await ProcessCoachAgent(model).execute(request())
    assert result.content == "Which part of this task would you like to reason through first?"
    assert not result.source_references and result.response_assessment is None
    assert result.metadata["grounding"]["teaching_surface_violation"] is True
    assert len(model.requests) == 1  # No semantic checker can approve this surface violation.


@pytest.mark.asyncio
async def test_runtime_adapts_without_overrides_and_keeps_neutral_turns_from_resetting_support():
    model, generate, requests = create_model_service()
    original = generate.side_effect
    judgment = "incorrect"

    async def dispatch(req):
        if req.metadata.get("task") == "process_coach_response":
            requests.append(req)
            claim = req.messages[-1].content
            value = assessment(
                judgment,
                kind="not_assessable" if judgment == "not_applicable" else "assessable",
                quote=None if judgment == "not_applicable" else claim,
            )
            return ModelChatResponse(
                content=json.dumps(
                    {
                        "content": "What would that relationship imply?",
                        "response_assessment": value.model_dump(mode="json"),
                    }
                ),
                model="controlled",
                provider="ollama",
                latency_ms=0,
            )
        return await original(req)

    generate.side_effect = dispatch
    runtime = build_orchestration_runtime(model_service=model)
    user, session = await create_user_and_session(runtime)
    await confirm_scenario(runtime, user=user, session=session)
    expected = [
        ("incorrect", "minimal", "minimal"),
        ("incorrect", "minimal", "directed"),
        ("incorrect", "directed", "explicit"),
        ("not_applicable", "explicit", "explicit"),
        ("partially_correct", "explicit", "explicit"),
        ("correct", "explicit", "explicit"),
        ("correct", "explicit", "directed"),
        ("not_applicable", "directed", "directed"),
        ("correct", "directed", "directed"),
        ("correct", "directed", "guided"),
    ]
    for index, (judgment, used, next_level) in enumerate(expected):
        before = await runtime.user_context_service.get_required(user_id=user.user_id)
        result = await runtime.orchestration_service.execute(
            OrchestrationRequest(
                session_id=str(session.session_id),
                student_id=user.student_id,
                user_message="Why is the relationship important?"
                if judgment == "not_applicable"
                else f"I believe this procurement relationship is necessary, attempt {index}.",
            )
        )
        assert result.error is None
        assert result.metadata["guidance_level_used"] == used
        assert result.metadata["guidance_level_next"] == next_level
        if judgment == "not_applicable":
            after = await runtime.user_context_service.get_required(user_id=user.user_id)
            assert after.hint_state.model_dump() == before.hint_state.model_dump()
            assert result.metadata["response_signal_recorded"] is False
    final = await runtime.user_context_service.get_required(user_id=user.user_id)
    assert final.hint_state.successful_independent_steps == 4
    assert final.hint_state.consecutive_productive_attempts == 0
    assert final.hint_state.guidance_level.value == "guided"


def test_old_hint_state_payload_remains_readable_without_reusing_lifetime_successes():
    state = HintState.model_validate(
        {"guidance_level": "explicit", "successful_independent_steps": 25}
    )
    assert state.successful_independent_steps == 25 and state.consecutive_productive_attempts == 0


@pytest.mark.asyncio
async def test_accepted_blockers_visit_all_four_levels_and_clear_does_not_reset_to_minimal():
    model, generate, _ = create_model_service()
    original = generate.side_effect
    operation = {
        "field": "active_blocker",
        "action": "set",
        "value": "Cannot reason about procurement.",
        "reason": "The student reports being stuck.",
    }

    async def dispatch(req):
        response = await original(req)
        if req.metadata.get("task") == "process_coach_response":
            payload = json.loads(response.content)
            payload["state_update_proposal"] = {"operations": [dict(operation)]}
            return response.model_copy(update={"content": json.dumps(payload)})
        return response

    generate.side_effect = dispatch
    runtime = build_orchestration_runtime(model_service=model)
    user, session = await create_user_and_session(runtime)
    await confirm_scenario(runtime, user=user, session=session)
    for index, (used, next_level) in enumerate(
        [
            ("minimal", "guided"),
            ("guided", "directed"),
            ("directed", "explicit"),
        ]
    ):
        operation["value"] = f"Cannot reason about procurement, attempt {index}."
        result = await runtime.orchestration_service.execute(
            OrchestrationRequest(
                session_id=str(session.session_id),
                student_id=user.student_id,
                user_message=f"I am stuck on procurement, attempt {index}.",
            )
        )
        assert result.metadata["guidance_level_used"] == used
        assert result.metadata["guidance_level_next"] == next_level
        assert result.metadata["state_update_applied"] is True
    operation.update(action="clear", value=None)
    result = await runtime.orchestration_service.execute(
        OrchestrationRequest(
            session_id=str(session.session_id),
            student_id=user.student_id,
            user_message="The procurement blocker is resolved.",
        )
    )
    assert (
        result.metadata["guidance_level_used"]
        == result.metadata["guidance_level_next"]
        == "explicit"
    )
    saved = await runtime.user_context_service.get_required(user_id=user.user_id)
    assert saved.hint_state.consecutive_blocked_attempts == 0


def test_safe_teaching_and_assessment_event_fields_do_not_include_student_quotes():
    event = build_request_event(
        {
            "metadata": {
                "assessment_status": "accepted",
                "guidance_level_used": "minimal",
                "guidance_level_next": "directed",
                "student_claim": "PRIVATE STUDENT CLAIM",
                "rag": {
                    "teaching_status": "violated",
                    "teaching_abstained": True,
                    "reason": "PRIVATE",
                },
            }
        },
        duration_ms=2,
    )
    assert event.guidance_level_used == "minimal" and event.guidance_level_next == "directed"
    assert event.teaching_status == "violated" and event.teaching_abstained is True
    assert "PRIVATE" not in json.dumps(event.log_payload())

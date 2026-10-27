"""
Sprint 4B.8 demo — show how the session summary works end to end.

Runs the REAL service stack in memory (real repositories, real
SessionSummaryService, real ContextService) with a fake model provider
that records exactly what it receives.

Usage (from backend/):

    uv run python -m scripts.demo_session_summary
"""

from __future__ import annotations

import json

from app.context.budget import (
    ContextBudgetPolicy,
    ContextBudgetService,
)
from app.context.summary_prompt import (
    SummaryDraft,
)
from app.context.token_estimator import (
    ApproximateTokenEstimator,
)
from app.identity import UserIdentity
from app.repositories import (
    create_in_memory_repositories,
)
from app.services.context_service import (
    ContextService,
)
from app.services.conversation_service import (
    ConversationService,
)
from app.services.model_service.schemas import (
    ModelChatRequest,
    ModelChatResponse,
)
from app.services.session_context_service import (
    SessionContextService,
)
from app.services.session_summary_service import (
    SessionSummaryService,
)
from app.services.user_context_service import (
    UserContextService,
)

RECENT_WINDOW = 4
MIN_NEW_MESSAGES = 2

SEED_MESSAGES = [
    ("user", "I am the production planner at UrbanCycle, a bike manufacturer."),
    ("assistant", "Understood. Let's talk about your delivery problem."),
    (
        "user",
        "Customer orders are delayed two weeks because purchase orders "
        "are created too late in the month.",
    ),
    (
        "assistant",
        "That sounds like a bottleneck in the purchase requisition step.",
    ),
    ("user", "Yes! And I still do it on paper. I want to move it to AP+."),
    (
        "assistant",
        "Good goal. Let's plan the AP+ steps together, one phase at a time.",
    ),
]

FOLLOW_UP_MESSAGES = [
    ("user", "We decided to pilot AP+ only for the purchasing department."),
    (
        "assistant",
        "Noted: pilot scope is purchasing only. Which step do we start with?",
    ),
]


class RecordingFakeModel:
    """Records every request; returns a plausible merged draft."""

    def __init__(self) -> None:
        self.requests: list[ModelChatRequest] = []

    async def generate(
        self,
        request: ModelChatRequest,
    ) -> ModelChatResponse:
        self.requests.append(request)

        # Pretend the model merged what it saw into the summary shape.
        draft = SummaryDraft(
            overview=(
                "Student is UrbanCycle's production planner moving paper "
                "purchase orders into AP+ to fix two-week delivery delays."
            ),
            established_facts=[
                "Company: UrbanCycle, bicycle manufacturer.",
                "Root cause: purchase orders created too late each month.",
            ],
            decisions=["Pilot AP+ in the purchasing department only."],
            blockers=[
                "Delivery delays of two weeks per customer order.",
            ],
            unresolved_questions=["Which AP+ step to start the pilot with."],
            misconception_candidates=[
                "Student assumed delays come from warehouse logistics.",
            ],
            completed_step_ids=[],
        )

        return ModelChatResponse(
            content=draft.model_dump_json(indent=2),
            model="fake-model",
            provider="ollama",
            latency_ms=12.0,
        )


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def main() -> None:
    repositories = create_in_memory_repositories()

    conversation_service = ConversationService(
        session_repository=repositories.sessions,
        session_context_repository=repositories.session_contexts,
        conversation_repository=repositories.conversations,
    )
    user_context_service = UserContextService(
        repository=repositories.user_contexts,
    )
    session_context_service = SessionContextService(
        repository=repositories.session_contexts,
    )
    # Same recent window as the summary policy so the demo is consistent.
    context_service = ContextService(
        user_context_service=user_context_service,
        session_context_service=session_context_service,
        conversation_service=conversation_service,
        budget_service=ContextBudgetService(
            estimator=ApproximateTokenEstimator(),
            policy=ContextBudgetPolicy(
                max_context_tokens=8192,
                reserved_output_tokens=2048,
                max_recent_messages=RECENT_WINDOW,
            ),
        ),
    )

    model = RecordingFakeModel()

    summary_service = SessionSummaryService(
        conversation_service=conversation_service,
        session_context_service=session_context_service,
        model_service=model,  # type: ignore[arg-type]
        recent_window=RECENT_WINDOW,
        min_new_messages=MIN_NEW_MESSAGES,
    )

    user = await repositories.identities.create(
        UserIdentity(student_id="DEMO01"),
    )
    session = await conversation_service.create_session(
        user_id=user.user_id,
    )

    for role, content in SEED_MESSAGES:
        if role == "user":
            await conversation_service.append_user_message(
                user_id=user.user_id,
                session_id=session.session_id,
                content=content,
            )
        else:
            await conversation_service.append_assistant_message(
                user_id=user.user_id,
                session_id=session.session_id,
                content=content,
                agent_name="process_coach",
            )

    section("STEP 1 — First summary update (6 messages, no summary yet)")

    result = await summary_service.maybe_update(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    print(f"policy: total=6, covered=0, window={RECENT_WINDOW}")
    print(
        f"  newly outside window = 6 - 0 - {RECENT_WINDOW} = 2 >= {MIN_NEW_MESSAGES}  -> UPDATE DUE"
    )
    print(f"result: updated={result.updated}, reason={result.reason}")
    print(f"model calls so far: {len(model.requests)}")

    section("STEP 2 — EXACT REQUEST SENT TO THE MODEL (call #1)")

    request = model.requests[0]
    print(f"model: {request.model or '(default chat model)'}   temperature: {request.temperature}")
    print(f"task:  {request.metadata['task']}")
    print("\n--- system prompt (truncated) ---")
    print(request.messages[0].content[:400] + "\n...")
    print("\n--- user prompt (what the model actually saw) ---")
    print(request.messages[1].content)
    print("\n--- response_schema (JSON keys the model must return) ---")
    assert request.response_schema is not None
    print(json.dumps(list(request.response_schema.get("properties", {}).keys())))

    section("STEP 3 — PERSISTED SUMMARY (new 'SessionSummary')")

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )
    assert context.summary is not None
    print(json.dumps(context.summary.model_dump(mode="json"), indent=2))
    print(f"\nsource_message_count = {context.summary.source_message_count}")
    print("-> messages 1-2 are now 'covered'; they will never be sent raw again.")

    section("STEP 4 — Two more messages, summary update due again")

    for role, content in FOLLOW_UP_MESSAGES:
        if role == "user":
            await conversation_service.append_user_message(
                user_id=user.user_id,
                session_id=session.session_id,
                content=content,
            )
        else:
            await conversation_service.append_assistant_message(
                user_id=user.user_id,
                session_id=session.session_id,
                content=content,
                agent_name="process_coach",
            )

    result = await summary_service.maybe_update(
        user_id=user.user_id,
        session_id=session.session_id,
    )
    print(f"policy: total=8, covered=2, window={RECENT_WINDOW}")
    print(
        f"  newly outside window = 8 - 2 - {RECENT_WINDOW} = 2 >= {MIN_NEW_MESSAGES}  -> UPDATE DUE"
    )
    print(f"result: updated={result.updated}   model calls so far: {len(model.requests)}")

    section("STEP 5 — INCREMENTAL UPDATE: WHAT THE MODEL SEES (call #2)")

    request = model.requests[1]
    print("--- user prompt (existing summary + ONLY messages 3-4) ---")
    print(request.messages[1].content)

    context = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )
    assert context.summary is not None
    print(f"\nafter merge: source_message_count = {context.summary.source_message_count}")

    section("STEP 6 — NEXT REQUEST: WHAT AN AGENT ACTUALLY RECEIVES")

    agent_context = await context_service.build(
        user_id=user.user_id,
        session_id=session.session_id,
        current_message="So, which AP+ step do we configure first for the pilot?",
    )

    metadata = agent_context.metadata
    print(f"summary_included        = {metadata.summary_included}")
    print(f"included_sections       = {metadata.included_sections}")
    print(f"recent_message_count    = {metadata.recent_message_count} (of 8 archived)")
    print("messages 1-4 were never fetched at all — they live on only inside the summary")
    print("\nrecent messages actually in the prompt:")
    for message in agent_context.recent_messages:
        speaker = message.agent_name or message.role.value
        print(f"  [{speaker}] {message.content}")

    section("STEP 7 — FAILURE PATH: bad model output keeps the OLD summary")

    before = context.model_copy(deep=True)

    class BrokenModel(RecordingFakeModel):
        async def generate(
            self,
            request: ModelChatRequest,
        ) -> ModelChatResponse:
            return ModelChatResponse(
                content=("Sure! Let me summarize: (no JSON)"),
                model="fake-model",
                provider="ollama",
                latency_ms=1.0,
            )

    # Age two more messages out of the window so the update actually fires.
    for role, content in [
        ("user", "Can you walk me through the requisition object in AP+?"),
        (
            "assistant",
            "I can, but first: what happens to a requisition after approval?",
        ),
    ]:
        if role == "user":
            await conversation_service.append_user_message(
                user_id=user.user_id,
                session_id=session.session_id,
                content=content,
            )
        else:
            await conversation_service.append_assistant_message(
                user_id=user.user_id,
                session_id=session.session_id,
                content=content,
                agent_name="process_coach",
            )

    broken = SessionSummaryService(
        conversation_service=conversation_service,
        session_context_service=session_context_service,
        model_service=BrokenModel(),  # type: ignore[arg-type]
        recent_window=RECENT_WINDOW,
        min_new_messages=2,
    )

    result = await broken.maybe_update(
        user_id=user.user_id,
        session_id=session.session_id,
    )
    print(f"result: updated={result.updated}, reason={result.reason}")
    print(f"error:  {result.error is not None}")

    after = await session_context_service.get_required(
        user_id=user.user_id,
        session_id=session.session_id,
    )
    print(
        f"old summary intact: {before.summary == after.summary and after.version == before.version}"
    )

    total = await conversation_service.count_messages(
        user_id=user.user_id,
        session_id=session.session_id,
    )

    section("DONE")
    print(
        f"Archived messages: {total} (full history untouched).\n"
        "Agent prompt: summary + last "
        f"{RECENT_WINDOW} raw messages + current question.\n"
        "Model calls: 1 per summary update, only when messages age out of the window."
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

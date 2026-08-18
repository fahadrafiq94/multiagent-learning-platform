import asyncio

from app.agents import (
    AgentRequest,
    APPlusNavigatorAgent,
)
from app.services.model_service import create_model_service


async def main() -> None:
    model_service = create_model_service()

    agent = APPlusNavigatorAgent(
        model_service=model_service,
    )

    response = await agent.execute(
        AgentRequest(
            session_id="ap-plus-navigator-test",
            student_id="student-1",
            user_message=(
                "I am working in AP+ and need to enter supplier "
                "information, but I do not know which screen or field "
                "I should use. What should I do?"
            ),
            metadata={
                "source": "manual-test",
            },
        )
    )

    print(response.model_dump_json(indent=2))


asyncio.run(main())

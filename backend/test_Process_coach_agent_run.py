import asyncio

from app.agents import (
    AgentRequest,
    ProcessCoachAgent,
)
from app.services.model_service import create_model_service


async def main() -> None:
    agent = ProcessCoachAgent(create_model_service())

    response = await agent.execute(
        AgentRequest(
            session_id="process-coach-test",
            student_id="student-1",
            user_message=(
                "We know that we need some raw materials. Tell me exactly what I should do next."
            ),
        )
    )

    print(response.model_dump_json(indent=2))


asyncio.run(main())

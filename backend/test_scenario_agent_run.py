import asyncio

from app.agents import AgentRequest, ScenarioAgent
from app.services.model_service import create_model_service


async def main() -> None:
    model_service = create_model_service()

    agent = ScenarioAgent(model_service)

    response = await agent.execute(
        AgentRequest(
            session_id="scenario-manual-test",
            student_id="student-1",
            user_message=(
                "We manufacture bicycles and want to improve our "
                "procurement process, but I am not sure how to define "
                "the business problem."
            ),
        )
    )

    print(response.model_dump_json(indent=2))


asyncio.run(main())

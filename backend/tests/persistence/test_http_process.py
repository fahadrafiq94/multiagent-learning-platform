"""Real HTTP across killed/restarted Python workers; scratch database only."""

import asyncio
import json
import socket
import subprocess
import sys
from uuid import uuid4

import httpx
import pytest

from app.identity import UserIdentity
from app.identity.pilot_tokens import issue_token
from app.orchestration.factory import build_orchestration_runtime
from app.persistence.turn_models import TurnStatus
from tests.integration.test_orchestration_api import confirm_scenario, create_model_service
from tests.persistence.test_http_runtime import SECRET

SERVER = r"""
import asyncio, json, sys
import uvicorn
from app.config.settings import settings
from scripts.application_credentials import local_application_dsn
from pydantic import SecretStr
options = json.loads(sys.stdin.readline())
settings.application_storage = 'postgres'
settings.application_database_url = local_application_dsn(database=options['database'])
settings.application_pilot_auth_secret = SecretStr(options['secret'])
import app.runtime as runtime_module
from tests.integration.test_orchestration_api import create_model_service
runtime_module.create_model_service = lambda: create_model_service()[0]
original_build = runtime_module.build_orchestration_runtime
def build(**kwargs):
    runtime = original_build(**kwargs)
    graph = runtime.graph
    class SlowGraph:
        async def ainvoke(self, state):
            if state['user_message'] == 'crash-me':
                await asyncio.Event().wait()
            return await graph.ainvoke(state)
    runtime.orchestration_service._graph = SlowGraph()
    return runtime
runtime_module.build_orchestration_runtime = build
config = uvicorn.Config(
    'app.main:app', host='127.0.0.1', port=options['port'], log_level='critical')
asyncio.run(uvicorn.Server(config).serve(), loop_factory=asyncio.SelectorEventLoop)
"""


async def start_server(database):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-c", SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    process.stdin.write(
        json.dumps(
            {
                "database": database,
                "port": port,
                "secret": SECRET.get_secret_value(),
            }
        )
        + "\n"
    )
    process.stdin.close()
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            for _ in range(100):
                if process.poll() is not None:
                    raise AssertionError("HTTP test worker exited during startup.")
                try:
                    if (await client.get("/ready")).status_code == 200:
                        return process, port
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.1)
        raise AssertionError("HTTP test worker did not become ready.")
    except BaseException:
        process.kill()
        await asyncio.to_thread(process.wait, timeout=5)
        raise


@pytest.mark.asyncio
async def test_http_process_kill_recovery_history_replay_and_continuation(pg):
    model, _, _ = create_model_service()
    runtime = build_orchestration_runtime(model_service=model, repositories=pg.repositories)
    user = await runtime.repositories.identities.create(UserIdentity(student_id="process-restart"))
    session = await runtime.conversation_service.create_session(user_id=user.user_id)
    await confirm_scenario(runtime, user=user, session=session)
    auth = {"Authorization": "Bearer " + issue_token(user.user_id, SECRET)}
    question = {
        "student_id": user.student_id,
        "session_id": str(session.session_id),
        "request_id": str(uuid4()),
        "user_message": "Why is a purchase order needed?",
    }
    process, port = await start_server(pg.name)
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            first = await client.post("/orchestration/invoke", headers=auth, json=question)
            assert first.status_code == 200
            pending = {**question, "request_id": str(uuid4()), "user_message": "crash-me"}
            task = asyncio.create_task(
                client.post("/orchestration/invoke", headers=auth, json=pending)
            )
            try:
                for _ in range(100):
                    turn = await pg.repositories.turns.get(
                        user_id=user.user_id,
                        session_id=session.session_id,
                        request_id=pending["request_id"],
                    )
                    if turn is not None:
                        break
                    await asyncio.sleep(0.05)
                assert turn is not None and turn.status == TurnStatus.PROCESSING
            finally:
                process.kill()
                await asyncio.to_thread(process.wait, timeout=5)
                await asyncio.gather(task, return_exceptions=True)
        process, port = await start_server(pg.name)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            path = f"/conversations/{session.session_id}/turns/{pending['request_id']}"
            recovered = await client.get(path, headers=auth)
            assert recovered.json()["status"] == "interrupted"
            assert recovered.json()["error_code"] == "worker_restarted"
            replay = await client.post("/orchestration/invoke", headers=auth, json=question)
            assert replay.json() == first.json()
            assert (
                await client.post("/orchestration/invoke", headers=auth, json=pending)
            ).status_code == 409
            resumed = await client.post(
                "/orchestration/invoke", headers=auth, json={**question, "request_id": str(uuid4())}
            )
            assert resumed.status_code == 200
            history = await client.get(
                f"/conversations/{session.session_id}/messages", headers=auth
            )
            assert len(history.json()["items"]) == 5
    finally:
        if process.poll() is None:
            process.kill()
            await asyncio.to_thread(process.wait, timeout=5)

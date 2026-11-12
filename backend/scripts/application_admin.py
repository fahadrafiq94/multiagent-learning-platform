"""Local operator tools. Tokens are printed only by the explicit token command."""

import argparse
import asyncio
import os
import sys
from uuid import UUID

from dotenv import dotenv_values
from pydantic import SecretStr

from app.context import UserContext
from app.identity import UserIdentity
from app.identity.pilot_tokens import issue_token
from app.identity.schemas import UserStatus
from app.persistence.database import ApplicationDatabase, ApplicationDatabaseConfig
from app.persistence.records import insert
from app.persistence.worker import ApplicationWorker
from app.repositories.exceptions import RepositoryError
from app.repositories.postgres import create_postgres_repositories
from scripts.application_credentials import configured_dsn
from scripts.knowledge_infra import ROOT


def pilot_secret() -> SecretStr:
    value = os.environ.get("APPLICATION_PILOT_AUTH_SECRET")
    if not value:
        value = dotenv_values(ROOT / "backend/.env.knowledge").get("APPLICATION_PILOT_AUTH_SECRET")
    if not value or len(value) < 32:
        raise ValueError("Run application bootstrap to provision pilot credentials.")
    return SecretStr(value)


async def execute(args) -> str:
    async with ApplicationDatabase(ApplicationDatabaseConfig(dsn=configured_dsn())) as database:
        repositories = create_postgres_repositories(database)
        if args.command == "recover":
            worker = ApplicationWorker(database)
            try:
                await worker.start()
                return f"Interrupted abandoned turns: {worker.recovered_turns}."
            finally:
                await worker.close()
        if args.command == "create-user":
            candidate = UserIdentity(student_id=args.student_id)
            existing = await repositories.identities.get_by_student_id(
                student_id=candidate.student_id
            )
            if existing is not None:
                return f"Existing user: {existing.user_id}"
            async with database.connection() as connection:
                await insert(connection, candidate)
                await insert(connection, UserContext(user_id=candidate.user_id))
            return f"Created user: {candidate.user_id}"
        user = await repositories.identities.get_by_user_id(args.user_id)
        if user is None or user.status != UserStatus.ACTIVE:
            raise ValueError("An active provisioned user is required.")
        return issue_token(user.user_id, pilot_secret(), lifetime_seconds=args.lifetime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-user", help="Provision a student and initial context.")
    create.add_argument("--student-id", required=True)
    token = commands.add_parser(
        "token", help="Print a signed bearer token; treat output as secret."
    )
    token.add_argument("--user-id", type=UUID, required=True)
    token.add_argument("--lifetime", type=int, default=3600)
    commands.add_parser("recover", help="Requires the backend stopped; interrupt abandoned turns.")
    args = parser.parse_args()
    try:
        print(asyncio.run(execute(args), loop_factory=asyncio.SelectorEventLoop))
        return 0
    except (RepositoryError, OSError, ValueError):
        print(
            "Application operation failed; verify configuration, user and worker state.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

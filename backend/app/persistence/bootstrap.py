"""Privileged, idempotent application-role/schema bootstrap, never called by runtime."""

from typing import Any

from psycopg import AsyncConnection, sql

from app.repositories.exceptions import RepositoryIntegrityError


async def bootstrap(connection: AsyncConnection[dict[str, Any]], passwords: dict[str, str]) -> None:
    await connection.execute("SELECT pg_advisory_xact_lock(562006)")
    for kind in ("migrator", "runtime"):
        if not passwords.get(kind):
            raise RepositoryIntegrityError("Application bootstrap credentials are missing.")
        name = "fredi_app_" + kind
        cursor = await connection.execute(
            "SELECT oid,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls,rolcanlogin "
            "FROM pg_roles WHERE rolname=%s",
            (name,),
        )
        role = await cursor.fetchone()
        if role is None:
            await connection.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                    "NOREPLICATION NOBYPASSRLS NOINHERIT PASSWORD {}"
                ).format(
                    sql.Identifier(name),
                    sql.Literal(passwords[kind]),
                )
            )
        else:
            if not role["rolcanlogin"]:
                raise RepositoryIntegrityError("Existing application role cannot log in.")
            if any(
                role[key]
                for key in (
                    "rolsuper",
                    "rolcreatedb",
                    "rolcreaterole",
                    "rolreplication",
                    "rolbypassrls",
                )
            ):
                raise RepositoryIntegrityError(
                    "Existing application role has unexpected privileges."
                )
            cursor = await connection.execute(
                "SELECT 1 FROM pg_auth_members WHERE member=%s", (role["oid"],)
            )
            if await cursor.fetchone():
                raise RepositoryIntegrityError(
                    "Existing application role has unexpected memberships."
                )
        await connection.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(connection.info.dbname),
                sql.Identifier(name),
            )
        )
    cursor = await connection.execute(
        "SELECT pg_get_userbyid(nspowner) AS owner FROM pg_namespace WHERE nspname='application'"
    )
    schema = await cursor.fetchone()
    if schema is None:
        await connection.execute("CREATE SCHEMA application AUTHORIZATION fredi_app_migrator")
    elif schema["owner"] != "fredi_app_migrator":
        raise RepositoryIntegrityError("Existing application schema has an unexpected owner.")
    await connection.execute("REVOKE ALL ON SCHEMA application FROM PUBLIC")
    await connection.execute("GRANT USAGE ON SCHEMA application TO fredi_app_runtime")

"""Short PostgreSQL transactions, per-collection admission and fenced publication."""

import hashlib
import json
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from app.knowledge.database import KnowledgeDatabase
from app.knowledge.errors import KnowledgeConflict, KnowledgeIntegrityError, KnowledgeNotFound
from app.knowledge.index_validation import validate_indexed_data
from app.knowledge.intake_schemas import IntakeBinding, IntakeJob, IntakePayload
from app.knowledge.postgres import PostgresKnowledgeRepository, decode
from app.knowledge.repository import authorize_write
from app.knowledge.schemas import KnowledgeCollection

JOB_COLUMNS = (
    "job_id",
    "collection_id",
    "status",
    "attempts",
    "base_active_id",
    "base_revision_id",
    "result_revision_id",
    "duplicate",
    "error_code",
    "lease_token",
    "created_at",
    "updated_at",
    "payload",
)


def job_record(row) -> IntakeJob:
    return IntakeJob.model_validate({key: row[key] for key in JOB_COLUMNS})


class IntakeRepository:
    def __init__(self, database: KnowledgeDatabase):
        self.database = database
        self.knowledge = PostgresKnowledgeRepository(database)

    async def _collection(self, connection, binding, *, lock=False):
        row = await (
            await connection.execute(
                "SELECT * FROM knowledge.knowledge_collections WHERE collection_id=%s"
                + (" FOR UPDATE" if lock else " FOR SHARE"),
                (binding.collection_id,),
            )
        ).fetchone()
        if row is None:
            raise KnowledgeNotFound("Collection is unavailable.")
        authorize_write(decode(KnowledgeCollection, "knowledge_collections", row), binding.scope)
        return row

    async def head(self, binding: IntakeBinding) -> tuple[UUID | None, UUID | None]:
        async with self.database.connection() as connection:
            row = await self._collection(connection, binding)
            return await self._head(connection, binding, row)

    async def _head(self, connection, binding, row) -> tuple[UUID | None, UUID | None]:
        active = row["active_revision_id"]
        base = active or binding.initial_revision_id
        if base is None:
            exists = await (
                await connection.execute(
                    "SELECT 1 FROM knowledge.knowledge_revisions WHERE collection_id=%s "
                    "AND indexing_status='ready' LIMIT 1",
                    (binding.collection_id,),
                )
            ).fetchone()
            if exists:
                raise KnowledgeConflict(
                    "An existing unpublished collection needs an explicit initial revision."
                )
        else:
            revision = await (
                await connection.execute(
                    "SELECT 1 FROM knowledge.knowledge_revisions WHERE collection_id=%s "
                    "AND revision_id=%s AND indexing_status='ready'",
                    (binding.collection_id, base),
                )
            ).fetchone()
            if not revision:
                raise KnowledgeConflict("The intake baseline must be a complete ready revision.")
        return active, base

    async def enqueue(self, binding: IntakeBinding, payload: IntakePayload) -> IntakeJob:
        async with self.database.connection() as connection:
            row = await self._collection(connection, binding, lock=True)
            _, base = await self._head(connection, binding, row)
            identity = hashlib.sha256(
                json.dumps(
                    {"base": str(base), "payload": payload.model_dump(mode="json")},
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            record = await (
                await connection.execute(
                    "INSERT INTO knowledge.intake_jobs(job_id,collection_id,identity_hash,payload) "
                    "VALUES (%s,%s,%s,%s) ON CONFLICT(collection_id,identity_hash) DO UPDATE "
                    "SET identity_hash=excluded.identity_hash RETURNING *",
                    (
                        uuid4(),
                        binding.collection_id,
                        identity,
                        Jsonb(payload.model_dump(mode="json")),
                    ),
                )
            ).fetchone()
            return job_record(record)

    async def get(self, binding: IntakeBinding, job_id: UUID) -> IntakeJob:
        async with self.database.connection() as connection:
            await self._collection(connection, binding)
            row = await (
                await connection.execute(
                    "SELECT * FROM knowledge.intake_jobs WHERE collection_id=%s AND job_id=%s",
                    (binding.collection_id, job_id),
                )
            ).fetchone()
            if row is None:
                raise KnowledgeNotFound("Intake job is unavailable.")
            return job_record(row)

    async def retry(self, binding: IntakeBinding, job_id: UUID) -> IntakeJob:
        async with self.database.connection() as connection:
            await self._collection(connection, binding, lock=True)
            row = await (
                await connection.execute(
                    "UPDATE knowledge.intake_jobs SET status='queued',attempts=0,error_code=NULL, "
                    "updated_at=clock_timestamp() WHERE collection_id=%s AND job_id=%s "
                    "AND status='failed' RETURNING *",
                    (binding.collection_id, job_id),
                )
            ).fetchone()
            if row is None:
                raise KnowledgeConflict("Only failed jobs can be retried.")
            return job_record(row)

    async def list(self, binding: IntakeBinding, *, limit: int = 50) -> list[IntakeJob]:
        if isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("Job list limit must be between 1 and 100.")
        async with self.database.connection() as connection:
            await self._collection(connection, binding)
            rows = await (
                await connection.execute(
                    "SELECT * FROM knowledge.intake_jobs WHERE collection_id=%s "
                    "ORDER BY created_at DESC,job_id DESC LIMIT %s",
                    (binding.collection_id, limit),
                )
            ).fetchall()
            return [job_record(row) for row in rows]

    async def claim(self, binding: IntakeBinding, *, lease_seconds: int = 60) -> IntakeJob | None:
        if not 1 <= lease_seconds <= 600:
            raise ValueError("Invalid lease duration.")
        async with self.database.connection() as connection:
            collection = await self._collection(connection, binding, lock=True)
            await connection.execute(
                "UPDATE knowledge.intake_jobs SET status=CASE WHEN attempts>=5 "
                "THEN 'failed' ELSE 'queued' END, "
                "lease_token=NULL,lease_until=NULL,error_code='worker_interrupted', "
                "updated_at=clock_timestamp() "
                "WHERE collection_id=%s AND status='processing' AND lease_until<clock_timestamp()",
                (binding.collection_id,),
            )
            busy = await (
                await connection.execute(
                    "SELECT 1 FROM knowledge.intake_jobs "
                    "WHERE collection_id=%s AND status='processing'",
                    (binding.collection_id,),
                )
            ).fetchone()
            if busy:
                return None
            # Do not inspect a baseline before there is work: an empty collection's
            # first interrupted attempt may already have created staging revisions.
            pending = await (
                await connection.execute(
                    "SELECT * FROM knowledge.intake_jobs "
                    "WHERE collection_id=%s AND status='queued' "
                    "ORDER BY created_at,job_id LIMIT 1 FOR UPDATE",
                    (binding.collection_id,),
                )
            ).fetchone()
            if pending is None:
                return None
            if collection["active_revision_id"] is None and pending["base_captured"]:
                active, base = None, pending["base_revision_id"]
            else:
                active, base = await self._head(connection, binding, collection)
            row = await (
                await connection.execute(
                    "UPDATE knowledge.intake_jobs SET status='processing',attempts=attempts+1, "
                    "lease_token=%s,lease_until=clock_timestamp()+make_interval(secs=>%s), "
                    "base_active_id=%s,base_revision_id=%s,base_captured=true, "
                    "error_code=NULL,updated_at=clock_timestamp() "
                    "WHERE job_id=%s RETURNING *",
                    (uuid4(), lease_seconds, active, base, pending["job_id"]),
                )
            ).fetchone()
            return job_record(row)

    async def heartbeat(self, binding, job, *, lease_seconds=60):
        async with self.database.connection() as connection:
            await self._collection(connection, binding)
            row = await (
                await connection.execute(
                    "UPDATE knowledge.intake_jobs "
                    "SET lease_until=clock_timestamp()+make_interval(secs=>%s), "
                    "updated_at=clock_timestamp() WHERE job_id=%s AND collection_id=%s "
                    "AND status='processing' AND lease_token=%s "
                    "AND lease_until>clock_timestamp() RETURNING job_id",
                    (lease_seconds, job.job_id, binding.collection_id, job.lease_token),
                )
            ).fetchone()
            if row is None:
                raise KnowledgeConflict("Intake worker lease was lost.")

    async def finish(self, binding, job, *, revision_id=None, duplicate=False, error_code=None):
        revision = None
        if revision_id is not None:
            revision = await self.knowledge.get_revision(revision_id, binding.scope)
            data = await self.knowledge.get_revision_data(revision_id, binding.scope)
            if revision is None or data is None or revision.collection_id != binding.collection_id:
                raise KnowledgeIntegrityError("Publication revision is invalid.")
            baseline = (
                await self.knowledge.get_revision(job.base_revision_id, binding.scope)
                if job.base_revision_id
                else None
            )
            expected_manifest = dict(baseline.manifest) if baseline else {}
            expected_manifest[job.payload.document_id] = job.payload.source_hash
            if (
                revision.manifest != expected_manifest
                or revision.embedding != job.payload.index.embedding
            ):
                raise KnowledgeIntegrityError(
                    "Publication does not match the intake document set/profile."
                )
            if revision.indexing_status != "ready" or set(revision.manifest) != {
                d.document_id for d in data.documents
            }:
                raise KnowledgeIntegrityError("Publication requires complete indexed documents.")
            if any(d.extraction_status != "extracted" for d in data.documents):
                raise KnowledgeIntegrityError("Publication cannot omit failed documents.")
            validate_indexed_data(revision, data)
        async with self.database.connection() as connection:
            collection = await self._collection(connection, binding, lock=True)
            row = await (
                await connection.execute(
                    "SELECT * FROM knowledge.intake_jobs WHERE job_id=%s AND collection_id=%s "
                    "AND status='processing' AND lease_token=%s "
                    "AND lease_until>clock_timestamp() FOR UPDATE",
                    (job.job_id, binding.collection_id, job.lease_token),
                )
            ).fetchone()
            if row is None:
                raise KnowledgeConflict("Intake worker lease was lost.")
            if revision is not None:
                current = await (
                    await connection.execute(
                        "SELECT version,indexing_status FROM knowledge.knowledge_revisions "
                        "WHERE revision_id=%s FOR SHARE",
                        (revision.revision_id,),
                    )
                ).fetchone()
                if (
                    not current
                    or current["version"] != revision.version
                    or current["indexing_status"] != "ready"
                ):
                    raise KnowledgeConflict("Publication revision changed.")
                if collection["active_revision_id"] != row["base_active_id"]:
                    raise KnowledgeConflict("The published collection changed during processing.")
                await connection.execute(
                    "UPDATE knowledge.knowledge_collections SET active_revision_id=%s "
                    "WHERE collection_id=%s",
                    (revision.revision_id, binding.collection_id),
                )
            result = await (
                await connection.execute(
                    "UPDATE knowledge.intake_jobs SET status=%s,result_revision_id=%s, "
                    "duplicate=%s,error_code=%s,lease_token=NULL,lease_until=NULL, "
                    "updated_at=clock_timestamp() WHERE job_id=%s RETURNING *",
                    (
                        "succeeded" if revision else "failed",
                        revision_id,
                        duplicate,
                        error_code,
                        job.job_id,
                    ),
                )
            ).fetchone()
            return job_record(result)

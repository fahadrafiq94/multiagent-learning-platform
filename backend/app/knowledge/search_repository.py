"""Read-only scoped PostgreSQL search port. Model inference never holds SQL open."""

import base64
import hashlib
import json
import math
import struct
from functools import partial
from typing import Protocol
from uuid import UUID, uuid5

from pydantic import ValidationError

from app.knowledge.cache import CoalescedCache, KnowledgeCache, NoKnowledgeCache
from app.knowledge.database import KnowledgeDatabase
from app.knowledge.errors import KnowledgeIntegrityError, KnowledgeNotFound
from app.knowledge.postgres import decode
from app.knowledge.retrieval_schemas import (
    EvidenceCitation,
    MetadataRequest,
    RetrievalRequest,
    RetrievedEvidence,
    RevisionPin,
    SearchCandidate,
)
from app.knowledge.schemas import (
    AccessScope,
    FactSource,
    KnowledgeChunk,
    KnowledgeRevision,
    ProcessStep,
)


class SearchRepository(Protocol):
    async def admit(
        self, collection_id: UUID, revision_id: UUID | None, scope: AccessScope
    ) -> RevisionPin | None: ...
    async def check_pin(self, pin: RevisionPin, scope: AccessScope) -> None: ...
    async def candidates(
        self,
        pin: RevisionPin,
        scope: AccessScope,
        request: RetrievalRequest,
        vector: list[float],
        references: list[str],
        limit: int,
    ) -> list[SearchCandidate]: ...
    async def evidence(
        self, pin: RevisionPin, scope: AccessScope, ids: list[UUID]
    ) -> list[RetrievedEvidence]: ...
    async def metadata(
        self, pin: RevisionPin, scope: AccessScope, request: MetadataRequest
    ) -> tuple[list[RetrievedEvidence], str | None]: ...


# Every method rechecks this gate; UUID knowledge never grants course/owner access.
ADMIT = """
SELECT r.*, c.owner_user_id AS cache_owner_user_id,
EXISTS(SELECT 1 FROM knowledge.knowledge_chunks k
    JOIN knowledge.knowledge_sections s
        ON (s.revision_id,s.section_id)=(k.revision_id,k.section_id)
    JOIN knowledge.knowledge_documents d
        ON (d.revision_id,d.document_id)=(k.revision_id,k.document_id)
    WHERE k.revision_id=r.revision_id AND k.kind='evidence' AND s.processing_status='ready'
      AND d.payload->>'extraction_status'='extracted') AS has_evidence
FROM knowledge.knowledge_revisions r JOIN knowledge.knowledge_collections c USING(collection_id)
WHERE c.collection_id=%s AND c.access_domain=%s
  AND (c.sharing_class='shared_instructional' OR c.owner_user_id=%s)
  AND r.revision_id=COALESCE(%s::uuid,c.active_revision_id) AND r.indexing_status='ready'
"""

# MATERIALIZED limits the vector-ranking universe before ordering. This is the
# exact, scoped correctness baseline, independent of approximate-index filtering.
CANDIDATES = """
WITH input AS (SELECT %s::vector AS v,websearch_to_tsquery('simple',%s) AS q,
    plainto_tsquery('simple',%s) AS purpose,%s::text[] AS refs,%s::text AS current_step,
    %s::text AS namespace,%s::text AS phase),
eligible AS MATERIALIZED (
    SELECT k.chunk_id,k.payload->>'reference_id' AS reference_id,k.step_id,
        1-(k.embedding <=> i.v) AS vector_similarity,
        ts_rank_cd(k.search_vector,i.q)::float8 AS lexical_score,
        COALESCE(lower(k.payload->>'reference_id')=ANY(i.refs)
            OR lower(k.step_id)=ANY(i.refs),false) AS exact_id,
        COALESCE((lower(k.step_id)=lower(i.current_step)
            OR lower(k.payload->>'reference_id')=lower(i.current_step))
            AND (i.namespace IS NULL OR k.namespace=i.namespace),false) AS current_step_match,
        COALESCE(lower(p.payload->>'phase')=lower(i.phase),false) AS phase_match,
        ts_rank_cd(k.search_vector,i.purpose)::float8 AS purpose_score
    FROM knowledge.knowledge_chunks k
    JOIN knowledge.knowledge_revisions r ON r.revision_id=k.revision_id
    JOIN knowledge.knowledge_collections c ON c.collection_id=r.collection_id
    JOIN knowledge.knowledge_sections s ON (s.revision_id,s.section_id)=(k.revision_id,k.section_id)
    JOIN knowledge.knowledge_documents d
        ON (d.revision_id,d.document_id)=(k.revision_id,k.document_id)
    LEFT JOIN knowledge.process_steps p
        ON (p.revision_id,p.namespace,p.step_id)=(k.revision_id,k.namespace,k.step_id)
    CROSS JOIN input i
    WHERE r.revision_id=%s AND r.collection_id=%s AND r.version=%s AND r.indexing_status='ready'
      AND c.access_domain=%s AND (c.sharing_class='shared_instructional' OR c.owner_user_id=%s)
      AND k.kind='evidence' AND k.embedding IS NOT NULL AND s.processing_status='ready'
      AND d.payload->>'extraction_status'='extracted'
),
v AS (SELECT chunk_id,row_number() OVER(ORDER BY vector_similarity DESC,chunk_id) AS vector_rank
      FROM eligible ORDER BY vector_similarity DESC,chunk_id LIMIT %s),
l AS (SELECT chunk_id,row_number() OVER(ORDER BY lexical_score DESC,chunk_id) AS lexical_rank
      FROM eligible WHERE lexical_score>0 ORDER BY lexical_score DESC,chunk_id LIMIT %s),
x AS (SELECT chunk_id,row_number() OVER(
        ORDER BY lexical_score DESC,vector_similarity DESC,chunk_id) AS identifier_rank
      FROM eligible WHERE exact_id
        ORDER BY lexical_score DESC,vector_similarity DESC,chunk_id LIMIT %s),
context AS (SELECT chunk_id FROM eligible WHERE current_step_match
      ORDER BY vector_similarity DESC,chunk_id LIMIT %s),
chosen AS (SELECT chunk_id FROM v UNION SELECT chunk_id FROM l
    UNION SELECT chunk_id FROM x UNION SELECT chunk_id FROM context)
SELECT e.chunk_id,e.reference_id,e.step_id,e.vector_similarity,e.lexical_score,
    e.current_step_match,e.phase_match,e.purpose_score,v.vector_rank,l.lexical_rank,x.identifier_rank
FROM eligible e JOIN chosen USING(chunk_id)
LEFT JOIN v USING(chunk_id) LEFT JOIN l USING(chunk_id) LEFT JOIN x USING(chunk_id)
ORDER BY e.chunk_id
"""

# Return only the quoted fragments of source pages, never whole source documents.
PAGE_PROOFS = """
(SELECT jsonb_agg(jsonb_build_object('span',a.span,
    'number',d.payload->'pages'->((a.span->>'page')::int-1)->'number',
    'fragment',substring(d.payload->'pages'->((a.span->>'page')::int-1)->>'text'
        FROM (a.span->>'start')::int+1 FOR (a.span->>'end')::int-(a.span->>'start')::int))
    ORDER BY a.position)
 FROM jsonb_array_elements({spans}) WITH ORDINALITY AS a(span,position)) AS proofs
"""


def _encode_evidence(item: RetrievedEvidence) -> bytes:
    return item.model_dump_json().encode()


def _decode_evidence(blob: bytes) -> RetrievedEvidence:
    return RetrievedEvidence.model_validate_json(blob)


def _encode_candidates(items: list[SearchCandidate]) -> bytes:
    return json.dumps([item.model_dump(mode="json") for item in items]).encode()


def _decode_candidates(blob: bytes) -> list[SearchCandidate]:
    value = json.loads(blob)
    if not isinstance(value, list) or len(value) > 320:
        raise ValueError("Invalid candidate cache shape.")
    candidates = [SearchCandidate.model_validate(item) for item in value]
    if len({c.chunk_id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate cached candidate identities.")
    return candidates


EVIDENCE_QUERY = (
    """SELECT k.*,k.embedding::text AS vector_text,d.source_hash,
    d.payload->>'title' AS title,
    substring(s.payload->>'text'
        FROM (k.payload->>'source_start')::int+1
        FOR (k.payload->>'source_end')::int-(k.payload->>'source_start')::int)
        AS source_fragment,
    """
    + PAGE_PROOFS.format(spans="k.payload->'page_spans'")
    + """
    FROM knowledge.knowledge_chunks k
    JOIN knowledge.knowledge_documents d USING(revision_id,document_id)
    JOIN knowledge.knowledge_sections s USING(revision_id,section_id)
    WHERE k.revision_id=%s AND k.chunk_id=ANY(%s::uuid[]) AND k.kind='evidence'
      AND s.processing_status='ready' AND d.payload->>'extraction_status'='extracted'"""
)


class PostgresSearchRepository:
    def __init__(
        self,
        database: KnowledgeDatabase,
        *,
        cache: KnowledgeCache | None = None,
        cache_ttl_seconds: int = 3600,
        cache_wait_seconds: float = 2.0,
    ) -> None:
        self.database = database
        self.cache = cache if cache is not None else NoKnowledgeCache()
        # Concurrent misses share entries populated in the configured cache.
        self._coalesced = CoalescedCache(
            self.cache, ttl_seconds=cache_ttl_seconds, wait_seconds=cache_wait_seconds
        )

    async def _admit(self, connection, collection_id, revision_id, scope) -> RevisionPin | None:
        cursor = await connection.execute(
            ADMIT, (collection_id, scope.access_domain, scope.user_id, revision_id)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        revision = decode(KnowledgeRevision, "knowledge_revisions", row)
        if not row["has_evidence"] or row["embedding_dimension"] != revision.embedding.dimension:
            raise KnowledgeIntegrityError("Ready knowledge revision is incomplete or incompatible.")
        return RevisionPin(revision=revision, cache_owner_user_id=row["cache_owner_user_id"])

    async def _check(self, connection, pin, scope):
        actual = await self._admit(
            connection, pin.revision.collection_id, pin.revision.revision_id, scope
        )
        if actual is None:
            raise KnowledgeNotFound("Knowledge revision is unavailable.")
        if (
            actual.revision.version != pin.revision.version
            or actual.revision.fingerprint != pin.revision.fingerprint
        ):
            raise KnowledgeIntegrityError("Pinned knowledge revision changed.")
        return actual

    async def admit(
        self, collection_id: UUID, revision_id: UUID | None, scope: AccessScope
    ) -> RevisionPin | None:
        async with self.database.connection() as connection:
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            return await self._admit(connection, collection_id, revision_id, scope)

    async def check_pin(self, pin: RevisionPin, scope: AccessScope) -> None:
        async with self.database.connection() as connection:
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            await self._check(connection, pin, scope)

    async def candidates(
        self,
        pin: RevisionPin,
        scope: AccessScope,
        request: RetrievalRequest,
        vector: list[float],
        references: list[str],
        limit: int,
    ) -> list[SearchCandidate]:
        if not 1 <= limit <= 100:
            raise ValueError("Candidate limit must be between 1 and 100.")
        if (
            len(vector) != pin.revision.embedding.dimension
            or not all(math.isfinite(v) for v in vector)
            or not any(vector)
        ):
            raise KnowledgeIntegrityError("Query vector is invalid.")
        try:
            vector = [struct.unpack("!f", struct.pack("!f", v))[0] for v in vector]
            if not any(vector):
                raise ValueError
        except (ValueError, OverflowError, struct.error):
            raise KnowledgeIntegrityError("Query vector is invalid.") from None
        # Authority: admission is rechecked on every call, even on cache hits;
        # only the (expensive) hybrid ranking below is served from cache.
        async with self.database.connection() as connection:
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            admitted = await self._check(connection, pin, scope)
        key = self._key(
            scope,
            admitted,
            f"search:{self._search_digest(request, references, vector, limit)}",
            personalized=True,
        )
        items, _ = await self._coalesced.get_or_fill(
            key,
            lambda: self._fetch_candidates(pin, scope, request, vector, references, limit),
            encode=_encode_candidates,
            decode=_decode_candidates,
        )
        await self.check_pin(pin, scope)
        return items

    @staticmethod
    def _search_digest(
        request: RetrievalRequest, references: list[str], vector: list[float], limit: int
    ) -> str:
        # The student's query never appears in the key; only its hash does.
        payload = json.dumps(
            {
                "limit": limit,
                "sql_version": hashlib.sha256(CANDIDATES.encode()).hexdigest(),
                "namespace": request.namespace,
                "phase": request.phase,
                "purpose": request.purpose,
                "query": request.query,
                "references": sorted(ref.lower() for ref in references),
                "step": request.current_step_id,
                "vector": base64.b64encode(struct.pack(f">{len(vector)}f", *vector)).decode(
                    "ascii"
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _key(scope: AccessScope, pin: RevisionPin, suffix: str, *, personalized=False) -> str:
        # Complete scope in every key: domain, owner (sentinel for shared
        # collections), pinned revision, then payload kind. A cache hit can
        # never grant access another scope's admission would deny.
        identity = json.dumps(
            {
                "domain": scope.access_domain,
                "owner": str(pin.cache_owner_user_id),
                "user": str(scope.user_id) if personalized else None,
                "collection": str(pin.revision.collection_id),
                "revision": str(pin.revision.revision_id),
                "version": pin.revision.version,
                "fingerprint": pin.revision.fingerprint,
                "operation": suffix,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "fredi:knw:v2:" + hashlib.sha256(identity.encode()).hexdigest()

    async def _fetch_candidates(
        self,
        pin: RevisionPin,
        scope: AccessScope,
        request: RetrievalRequest,
        vector: list[float],
        references: list[str],
        limit: int,
    ) -> list[SearchCandidate]:
        """Hydration only; candidates() must have run _check for this pin and scope."""
        async with self.database.connection() as connection:
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor = await connection.execute(
                CANDIDATES,
                (
                    json.dumps(vector),
                    request.query,
                    request.purpose,
                    [ref.lower() for ref in references],
                    request.current_step_id,
                    request.namespace,
                    request.phase,
                    pin.revision.revision_id,
                    pin.revision.collection_id,
                    pin.revision.version,
                    scope.access_domain,
                    scope.user_id,
                    limit,
                    limit,
                    limit,
                    min(limit, 20),
                ),
            )
            try:
                return [SearchCandidate.model_validate(row) for row in await cursor.fetchall()]
            except ValidationError:
                raise KnowledgeIntegrityError("Stored search candidates are invalid.") from None

    def _proof(self, row, *, content, spans, document_id, pin):
        if (
            row["source_hash"] != pin.revision.manifest.get(document_id)
            or row["source_fragment"] != content
        ):
            raise KnowledgeIntegrityError("Evidence does not match its source.")
        proofs = row["proofs"] or []
        if not spans or len(proofs) != len(spans):
            raise KnowledgeIntegrityError("Evidence page provenance is missing.")
        covered = 0
        previous_page = 0
        for span, proof in zip(spans, proofs, strict=True):
            fragment = proof.get("fragment")
            if (
                proof["span"] != span.model_dump()
                or proof["number"] != span.page
                or not isinstance(fragment, str)
                or len(fragment) != span.end - span.start
                or span.section_start < covered
                or span.page <= previous_page
                or content[covered : span.section_start].strip("\n")
                or content[span.section_start : span.section_start + len(fragment)] != fragment
            ):
                raise KnowledgeIntegrityError("Evidence page provenance is invalid.")
            covered = span.section_start + len(fragment)
            previous_page = span.page
        if content[covered:].strip("\n"):
            raise KnowledgeIntegrityError("Evidence lacks page provenance.")

    async def evidence(
        self, pin: RevisionPin, scope: AccessScope, ids: list[UUID]
    ) -> list[RetrievedEvidence]:
        if not ids:
            return []
        if len(ids) > 320 or len(set(ids)) != len(ids):
            raise ValueError("Evidence identities must be unique and bounded.")
        # Authority: admission is rechecked against PostgreSQL on every call,
        # even when every chunk below is served from cache.
        async with self.database.connection() as connection:
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            admitted = await self._check(connection, pin, scope)
        result = []
        for chunk_id in ids:
            key = self._key(scope, admitted, f"chunk:{chunk_id}")
            evidence, _ = await self._coalesced.get_or_fill(
                key,
                partial(self._fetch_chunk, pin, chunk_id),
                encode=_encode_evidence,
                decode=partial(self._cached_evidence, pin=pin, evidence_id=chunk_id),
            )
            result.append(evidence)
        await self.check_pin(pin, scope)
        return result

    @staticmethod
    def _cached_evidence(
        blob: bytes,
        pin: RevisionPin,
        evidence_id: UUID,
        *,
        step: ProcessStep | None = None,
        field: str | None = None,
        fact: FactSource | None = None,
    ) -> RetrievedEvidence:
        item = _decode_evidence(blob)
        if item.evidence_id != evidence_id or item.kind != ("metadata" if step else "chunk"):
            raise ValueError("Cached evidence identity mismatch.")
        for citation in item.citations:
            if (
                citation.revision_id != pin.revision.revision_id
                or pin.revision.manifest.get(citation.document_id) != citation.source_hash
                or citation.page_end < citation.page_start
                or (not step and citation.chunk_id != evidence_id)
            ):
                raise ValueError("Cached evidence source mismatch.")
        if step is not None and fact is None:
            raise ValueError("Metadata proof is missing.")
        if (
            step is not None
            and fact is not None
            and (
                item.reference_id != step.step_id
                or item.namespace != step.namespace
                or item.field_name != field
                or item.content != fact.text
                or any(
                    c.document_id != step.source_document_id
                    or c.section_id != fact.section_id
                    or c.page_spans != fact.page_spans
                    for c in item.citations
                )
            )
        ):
            raise ValueError("Cached metadata source mismatch.")
        return item

    async def _fetch_chunk(self, pin: RevisionPin, chunk_id: UUID) -> RetrievedEvidence:
        """Hydration only; callers must have run _check for this pin and scope."""
        async with self.database.connection() as connection:
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor = await connection.execute(
                EVIDENCE_QUERY, (pin.revision.revision_id, [chunk_id])
            )
            row = await cursor.fetchone()
        if row is None:
            raise KnowledgeIntegrityError("Selected evidence is missing or no longer eligible.")
        return self._evidence_from_row(row, pin)

    def _evidence_from_row(self, row, pin: RevisionPin) -> RetrievedEvidence:
        chunk = decode(KnowledgeChunk, "knowledge_chunks", row)
        if chunk.embedding is None or len(chunk.embedding) != pin.revision.embedding.dimension:
            raise KnowledgeIntegrityError("Stored evidence vector is incompatible.")
        if pin.revision.embedding.normalized and not math.isclose(
            math.sqrt(math.fsum(v * v for v in chunk.embedding)),
            1,
            abs_tol=1e-5,
        ):
            raise KnowledgeIntegrityError("Stored evidence vector is not normalized.")
        text = pin.revision.embedding.document_prefix + chunk.context_prefix + "\n" + chunk.content
        if chunk.embedding_input_hash != hashlib.sha256(text.encode()).hexdigest():
            raise KnowledgeIntegrityError("Stored embedding provenance is invalid.")
        self._proof(
            row,
            content=chunk.content,
            spans=chunk.page_spans,
            document_id=chunk.document_id,
            pin=pin,
        )
        return RetrievedEvidence(
            evidence_id=chunk.chunk_id,
            kind="chunk",
            content=chunk.content,
            context_prefix=chunk.context_prefix,
            reference_id=chunk.reference_id,
            namespace=chunk.namespace,
            source_start=chunk.source_start,
            source_end=chunk.source_end,
            citations=[
                EvidenceCitation(
                    revision_id=chunk.revision_id,
                    document_id=chunk.document_id,
                    section_id=chunk.section_id,
                    chunk_id=chunk.chunk_id,
                    title=row["title"],
                    source_hash=row["source_hash"],
                    reference_id=chunk.reference_id,
                    page_start=chunk.page_spans[0].page,
                    page_end=chunk.page_spans[-1].page,
                    page_spans=chunk.page_spans,
                )
            ],
        )

    async def metadata(
        self, pin: RevisionPin, scope: AccessScope, request: MetadataRequest
    ) -> tuple[list[RetrievedEvidence], str | None]:
        async with self.database.connection() as connection:
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            pin = await self._check(connection, pin, scope)
            cursor = await connection.execute(
                "SELECT p.* FROM knowledge.process_steps p WHERE revision_id=%s "
                "AND lower(step_id)=lower(%s) "
                "AND (%s::text IS NULL OR namespace=%s) ORDER BY namespace,step_id LIMIT 2",
                (pin.revision.revision_id, request.step_id, request.namespace, request.namespace),
            )
            rows = await cursor.fetchall()
            if not rows:
                return [], "metadata_missing"
            if len(rows) > 1:
                return [], "metadata_ambiguous"
            step = decode(ProcessStep, "process_steps", rows[0])
            if not step.source_fields:
                raise KnowledgeIntegrityError("Ready metadata has no source provenance.")
        # Release the admission connection before cache waits/new hydration reads;
        # a reader pool of one must also work.
        result = []
        for field in dict.fromkeys(request.fields):
            fact = step.source_fields.get(field)
            if fact is None:
                continue
            result.append(await self._fact(pin, scope, step, field, fact))
        await self.check_pin(pin, scope)
        return result, None if result else "metadata_missing"

    async def _fact(
        self, pin: RevisionPin, scope: AccessScope, step: ProcessStep, field: str, fact: FactSource
    ) -> RetrievedEvidence:
        key = self._key(scope, pin, json.dumps(["fact", step.namespace, step.step_id, field]))
        evidence_id = uuid5(
            pin.revision.revision_id, f"fact:{step.namespace}:{step.step_id}:{field}"
        )
        evidence, _ = await self._coalesced.get_or_fill(
            key,
            lambda: self._fetch_fact(pin, step, field, fact),
            encode=_encode_evidence,
            decode=partial(
                self._cached_evidence,
                pin=pin,
                evidence_id=evidence_id,
                step=step,
                field=field,
                fact=fact,
            ),
        )
        return evidence

    async def _fetch_fact(
        self, pin: RevisionPin, step: ProcessStep, field: str, fact: FactSource
    ) -> RetrievedEvidence:
        """Hydration only; metadata() must have run _check for this pin and scope."""
        async with self.database.connection() as connection:
            await connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            query = (
                """SELECT d.source_hash,d.payload->>'title' AS title,
                substring(s.payload->>'text' FROM %s::int+1 FOR %s::int-%s::int) AS source_fragment,
                """
                + PAGE_PROOFS.format(spans="%s::jsonb")
                + """
                FROM knowledge.knowledge_documents d
                    JOIN knowledge.knowledge_sections s USING(revision_id,document_id)
                WHERE d.revision_id=%s AND d.document_id=%s AND s.section_id=%s
                  AND d.workflow_authority AND s.processing_status='ready'
                  AND d.payload->>'extraction_status'='extracted'"""
            )
            cursor = await connection.execute(
                query,
                (
                    fact.start,
                    fact.end,
                    fact.start,
                    json.dumps([s.model_dump() for s in fact.page_spans]),
                    pin.revision.revision_id,
                    step.source_document_id,
                    fact.section_id,
                ),
            )
            row = await cursor.fetchone()
        if row is None:
            raise KnowledgeIntegrityError("Metadata source is missing.")
        self._proof(
            row,
            content=fact.text,
            spans=fact.page_spans,
            document_id=step.source_document_id,
            pin=pin,
        )
        return RetrievedEvidence(
            evidence_id=uuid5(
                pin.revision.revision_id, f"fact:{step.namespace}:{step.step_id}:{field}"
            ),
            kind="metadata",
            content=fact.text,
            reference_id=step.step_id,
            namespace=step.namespace,
            field_name=field,
            citations=[
                EvidenceCitation(
                    revision_id=pin.revision.revision_id,
                    document_id=step.source_document_id,
                    section_id=fact.section_id,
                    field_name=field,
                    title=row["title"],
                    source_hash=row["source_hash"],
                    reference_id=step.step_id,
                    page_start=fact.page_spans[0].page,
                    page_end=fact.page_spans[-1].page,
                    page_spans=fact.page_spans,
                )
            ],
        )

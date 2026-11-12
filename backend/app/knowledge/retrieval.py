"""Uncached evidence retrieval. No agent routing, generation or student-state writes."""

import math
import struct
from collections.abc import Callable

from pydantic import ValidationError

from app.knowledge.errors import (
    KnowledgeIntegrityError,
    KnowledgeNotFound,
    KnowledgePermissionError,
    KnowledgeUnavailable,
)
from app.knowledge.retrieval_ranking import (
    item_chars,
    rank_candidates,
    references,
    select_evidence,
    source_bundles,
)
from app.knowledge.retrieval_schemas import (
    MetadataRequest,
    RetrievalConfig,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStats,
    RetrievalStatus,
    RetrievedEvidence,
    RevisionPin,
)
from app.knowledge.schemas import AccessScope
from app.knowledge.search_repository import SearchRepository
from app.services.model_service import ModelService
from app.services.model_service.exceptions import ModelEmbeddingError, ModelServiceError
from app.services.model_service.schemas import ModelEmbeddingRequest


class RetrievalFailure(Exception):
    def __init__(self, code: str):
        self.code = code


class KnowledgeRetrievalService:
    def __init__(
        self,
        repository: SearchRepository,
        model_service: ModelService | None = None,
        config: RetrievalConfig | None = None,
    ):
        self.repository = repository
        self.model_service = model_service
        self.config = config or RetrievalConfig()

    def _result(self, request, status, reason, pin=None, items=None, stats=None) -> RetrievalResult:
        items = items or []
        return RetrievalResult(
            status=status,
            reason=reason,
            collection_id=request.collection_id,
            revision_id=pin.revision.revision_id if pin else None,
            revision_version=pin.revision.version if pin else None,
            items=items,
            sources=source_bundles(items),
            stats=stats or RetrievalStats(),
        )

    async def _query_vector(
        self, pin: RevisionPin, query: str, on_embed: Callable[[], None]
    ) -> list[float]:
        if self.model_service is None:
            raise RetrievalFailure("embedding_unavailable")
        profile = pin.revision.embedding

        async def inspect():
            info = await self.model_service.inspect_embedding_model(profile.model)
            if (
                info.provider != profile.provider
                or info.revision != profile.revision
                or info.dimension != profile.dimension
            ):
                raise RetrievalFailure("embedding_profile_mismatch")

        await inspect()
        on_embed()
        try:
            response = await self.model_service.embed(
                ModelEmbeddingRequest(
                    model=profile.model,
                    texts=[profile.query_prefix + query],
                    truncate=False,
                    metadata={"task": "knowledge_query_embedding"},
                )
            )
        except ModelEmbeddingError as exc:
            if exc.details and exc.details.get("input_too_long"):
                raise RetrievalFailure("query_too_long") from None
            raise
        if (
            len(response.embeddings) != 1
            or response.provider != profile.provider
            or response.model != profile.model
        ):
            raise RetrievalFailure("invalid_query_embedding")
        vector = response.embeddings[0]
        if (
            len(vector) != profile.dimension
            or not all(math.isfinite(v) for v in vector)
            or not any(vector)
        ):
            raise RetrievalFailure("invalid_query_embedding")
        try:
            norm = math.sqrt(math.fsum(v * v for v in vector))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError
            if profile.normalized:
                vector = [v / norm for v in vector]
            vector = [struct.unpack("!f", struct.pack("!f", v))[0] for v in vector]
            if not any(vector):
                raise ValueError
        except (ValueError, OverflowError, struct.error):
            raise RetrievalFailure("invalid_query_embedding") from None
        await inspect()
        return vector

    async def search(self, request: RetrievalRequest, *, scope: AccessScope) -> RetrievalResult:
        request = RetrievalRequest.model_validate(request.model_dump())
        pin = None
        stats = RetrievalStats()
        try:
            pin = await self.repository.admit(request.collection_id, request.revision_id, scope)
            if pin is None:
                return self._result(request, RetrievalStatus.UNAVAILABLE, "revision_unavailable")

            def count_call():
                nonlocal stats
                stats = stats.model_copy(update={"model_calls": stats.model_calls + 1})

            vector = await self._query_vector(pin, request.query, count_call)
            candidates = await self.repository.candidates(
                pin,
                scope,
                request,
                vector,
                references(request),
                self.config.candidate_limit,
            )
            stats = stats.model_copy(
                update={
                    "candidate_count": len(candidates),
                    "vector_candidates": sum(c.vector_rank is not None for c in candidates),
                    "lexical_candidates": sum(c.lexical_rank is not None for c in candidates),
                    "identifier_candidates": sum(c.identifier_rank is not None for c in candidates),
                }
            )
            ranked = rank_candidates(candidates, request, self.config)
            items = await self.repository.evidence(pin, scope, [c.chunk_id for c, _ in ranked])
            selected, stats = select_evidence(ranked, items, request, self.config, stats)
            # Recheck authorization/readiness after model I/O and evidence reads,
            # without switching an admitted request to a newly active revision.
            await self.repository.check_pin(pin, scope)
            reason = (
                "matches"
                if selected
                else ("budget_exhausted" if stats.budget_omissions else "no_matches")
            )
            return self._result(
                request,
                RetrievalStatus.FOUND if selected else RetrievalStatus.NO_EVIDENCE,
                reason,
                pin,
                selected,
                stats,
            )
        except KnowledgeNotFound:
            return self._result(
                request, RetrievalStatus.UNAVAILABLE, "revision_unavailable", stats=stats
            )
        except (
            RetrievalFailure,
            ModelServiceError,
            KnowledgeUnavailable,
            KnowledgePermissionError,
            KnowledgeIntegrityError,
            ValidationError,
        ) as exc:
            if isinstance(exc, RetrievalFailure):
                reason = exc.code
            elif isinstance(exc, ModelServiceError):
                reason = "embedding_unavailable"
            elif isinstance(exc, (KnowledgeIntegrityError, ValidationError)):
                reason = "invalid_stored_evidence"
            else:
                reason = "knowledge_unavailable"
            return self._result(request, RetrievalStatus.ERROR, reason, pin, stats=stats)

    async def lookup(self, request: MetadataRequest, *, scope: AccessScope) -> RetrievalResult:
        request = MetadataRequest.model_validate(request.model_dump())
        pin = None
        try:
            pin = await self.repository.admit(request.collection_id, request.revision_id, scope)
            if pin is None:
                return self._result(request, RetrievalStatus.UNAVAILABLE, "revision_unavailable")
            items, reason = await self.repository.metadata(pin, scope, request)
            selected: list[RetrievedEvidence] = []
            omitted = []
            used = 0
            for item in items:
                cost = item_chars(item)
                if used + cost > request.max_chars:
                    omitted.append(item.field_name or "")
                    continue
                selected.append(item)
                used += cost
            await self.repository.check_pin(pin, scope)
            stats = RetrievalStats(
                candidate_count=len(items),
                returned_count=len(selected),
                budget_omissions=len(omitted),
                omitted_fields=omitted,
            )
            reason = "matches" if selected else (reason or "budget_exhausted")
            return self._result(
                request,
                RetrievalStatus.FOUND if selected else RetrievalStatus.NO_EVIDENCE,
                reason,
                pin,
                selected,
                stats,
            )
        except KnowledgeNotFound:
            return self._result(request, RetrievalStatus.UNAVAILABLE, "revision_unavailable")
        except (
            KnowledgeUnavailable,
            KnowledgePermissionError,
            KnowledgeIntegrityError,
            ValidationError,
        ) as exc:
            reason = (
                "invalid_stored_evidence"
                if isinstance(exc, (KnowledgeIntegrityError, ValidationError))
                else "knowledge_unavailable"
            )
            return self._result(request, RetrievalStatus.ERROR, reason, pin)

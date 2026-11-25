"""Authorized intake shared by HTTP and operators; no model work inside transactions."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid5

from app.knowledge.errors import (
    KnowledgeConflict,
    KnowledgeIntegrityError,
    KnowledgeNotFound,
    KnowledgePermissionError,
)
from app.knowledge.extraction import MAX_SOURCE_BYTES, DocumentExtractor
from app.knowledge.importer import ImportRegistration, RegisteredSource, import_registered
from app.knowledge.indexer import clone_extraction, index_revision
from app.knowledge.intake_repository import IntakeRepository
from app.knowledge.intake_schemas import IntakeBinding, IntakeConfig, IntakePayload
from app.knowledge.schemas import KnowledgeRevision, PipelineVersions, RevisionData
from app.knowledge.source_store import SourceStore
from app.services.model_service import ModelService
from app.services.model_service.service import close_model_services


class IntakeService:
    def __init__(
        self,
        jobs: IntakeRepository,
        sources: SourceStore,
        extractor: DocumentExtractor,
        model: ModelService,
        config: IntakeConfig,
    ):
        self.jobs, self.sources, self.extractor, self.model = jobs, sources, extractor, model
        self.bindings = {b.collection_id: b for b in config.bindings}

    async def aclose(self) -> None:
        await close_model_services(self.model)

    def binding(self, collection_id: UUID, manager_id: UUID | None = None) -> IntakeBinding:
        binding = self.bindings.get(collection_id)
        if binding is None or (
            manager_id is not None and manager_id not in binding.manager_user_ids
        ):
            raise KnowledgePermissionError("Knowledge management is not permitted.")
        return binding

    async def submit(
        self,
        collection_id: UUID,
        content: bytes,
        *,
        title: str,
        replace_document_id: UUID | None = None,
        manager_id: UUID | None = None,
    ):
        binding = self.binding(collection_id, manager_id)
        title = title.strip()
        if not title or len(title) > 255:
            raise KnowledgeIntegrityError(
                "A nonblank document title of at most 255 characters is required."
            )
        # Authorize storage access before storing bytes; PDF parsing is deferred to the worker.
        _, base_id = await self.jobs.head(binding)
        if (
            not content
            or len(content) > MAX_SOURCE_BYTES
            or not content.lstrip()[:1024].startswith(b"%PDF-")
        ):
            raise KnowledgeIntegrityError("A nonempty supported-size PDF is required.")
        reference = await asyncio.to_thread(self.sources.put, content)
        data = (
            await self.jobs.knowledge.get_revision_data(base_id, binding.scope)
            if base_id
            else RevisionData()
        )
        if data is None:
            raise KnowledgeNotFound("Collection snapshot is unavailable.")
        documents = {d.document_id: d for d in data.documents}
        previous = documents.get(replace_document_id) if replace_document_id else None
        if replace_document_id is not None and previous is None:
            raise KnowledgeNotFound("Replacement document is not in the current collection.")
        match = next((d for d in data.documents if d.source_hash == reference.source_hash), None)
        if match is not None and previous is not None and match.document_id != previous.document_id:
            raise KnowledgeConflict(
                "These bytes already belong to another document in this collection."
            )
        document_id = (
            previous.document_id
            if previous
            else (match.document_id if match else uuid5(collection_id, reference.source_hash))
        )
        payload = IntakePayload(
            document_id=document_id,
            source_hash=reference.source_hash,
            title=title,
            replace=previous is not None,
            expected_source_hash=previous.source_hash if previous else None,
            workflow_authority=previous.workflow_authority if previous else False,
            index=binding.index,
        )
        return await self.jobs.enqueue(binding, payload)

    async def _process(self, binding, job):
        if binding.index != job.payload.index:
            raise KnowledgeConflict(
                "The trusted indexing configuration changed; resubmit the document."
            )
        repo, payload = self.jobs.knowledge, job.payload
        collection = await repo.get_collection(binding.collection_id, binding.scope)
        if collection is None:
            raise KnowledgeNotFound("Collection is unavailable.")
        original = (
            await repo.get_revision_data(job.base_revision_id, binding.scope)
            if job.base_revision_id
            else RevisionData()
        )
        if original is None:
            raise KnowledgeNotFound("Baseline snapshot is unavailable.")
        previous = next(
            (d for d in original.documents if d.document_id == payload.document_id), None
        )
        if previous and previous.source_hash == payload.source_hash:
            return await self.jobs.finish(
                binding, job, revision_id=job.base_revision_id, duplicate=True
            )
        if payload.replace and (
            previous is None or previous.source_hash != payload.expected_source_hash
        ):
            raise KnowledgeConflict(
                "Replacement target changed after upload; resubmit against the current document."
            )
        if previous is not None and not payload.replace:
            raise KnowledgeConflict("Document identity already exists.")
        if any(d.source_hash == payload.source_hash for d in original.documents):
            raise KnowledgeConflict("Content is already stored under another document identity.")
        if previous is None and len(original.documents) >= 100:
            raise KnowledgeIntegrityError("The collection document limit was reached.")
        content = await asyncio.to_thread(self.sources.get, payload.source_hash)
        with TemporaryDirectory(prefix="fredi-intake-") as directory:
            path = Path(directory) / "source.pdf"
            await asyncio.to_thread(path.write_bytes, content)
            imported = await import_registered(
                ImportRegistration(
                    collection=collection,
                    embedding=payload.index.embedding,
                    sources=[
                        RegisteredSource(
                            document_id=payload.document_id,
                            path="source.pdf",
                            title=payload.title,
                            workflow_authority=payload.workflow_authority,
                            profile=payload.profile,
                        ),
                    ],
                ),
                source_root=Path(directory),
                repository=repo,
                source_store=self.sources,
                extractor=self.extractor,
                scope=binding.scope,
                retry_failed=True,
            )
        if imported.failed_documents:
            raise KnowledgeIntegrityError("PDF extraction failed.")
        extracted = await repo.get_revision_data(imported.revision_id, binding.scope)
        if extracted is None:
            raise KnowledgeNotFound("Extracted snapshot is unavailable.")
        remaining = RevisionData(
            documents=[d for d in original.documents if d.document_id != payload.document_id],
            sections=[s for s in original.sections if s.document_id != payload.document_id],
        )
        # Combine extracted snapshots, preserving unchanged page text/provenance.
        # Canonical structural metadata is rebuilt by the existing general indexer.
        manifest = {
            d.document_id: d.source_hash for d in [*remaining.documents, *extracted.documents]
        }
        revision = KnowledgeRevision(
            collection_id=binding.collection_id,
            embedding=payload.index.embedding,
            manifest=manifest,
            pipeline=PipelineVersions(
                extractor="intake-union-v1:"
                + str(job.base_revision_id)
                + ":"
                + str(imported.revision_id),
                chunker="not-run",
                policy="general-indexing-v2",
            ),
        )
        revision = revision.model_copy(
            update={"revision_id": uuid5(binding.collection_id, revision.fingerprint)}
        )
        one, two = (
            clone_extraction(remaining, revision.revision_id),
            clone_extraction(extracted, revision.revision_id),
        )
        snapshot = RevisionData(
            documents=[*one.documents, *two.documents], sections=[*one.sections, *two.sections]
        )
        try:
            current = await repo.create_revision(revision, scope=binding.scope)
        except KnowledgeConflict:
            current = await repo.get_revision(revision.revision_id, binding.scope)
            if current is None:
                raise
        if current.version == 0:
            await repo.write_revision_data(
                revision.revision_id, snapshot, scope=binding.scope, expected_version=0
            )
        result = await index_revision(
            revision.revision_id,
            config=payload.index,
            repository=repo,
            model_service=self.model,
            scope=binding.scope,
        )
        if result.status != "ready":
            return await self.jobs.finish(
                binding, job, error_code=result.error_code or "indexing_failed"
            )
        return await self.jobs.finish(binding, job, revision_id=result.revision_id)

    async def run_once(self, collection_id: UUID):
        binding = self.binding(collection_id)
        job = await self.jobs.claim(binding)
        if job is None:
            return None

        async def heartbeat():
            while True:
                await asyncio.sleep(10)
                await self.jobs.heartbeat(binding, job)

        work = asyncio.create_task(self._process(binding, job))
        pulse = asyncio.create_task(heartbeat())
        try:
            done, _ = await asyncio.wait({work, pulse}, return_when=asyncio.FIRST_COMPLETED)
            if work in done:
                return await work
            if pulse in done:
                await pulse
            return await work
        except asyncio.CancelledError:
            # Lease expiry allows a new process to resume; the old token cannot publish.
            raise
        except Exception as exc:
            code = (
                "intake_conflict"
                if isinstance(exc, KnowledgeConflict)
                else (
                    "intake_invalid_source"
                    if isinstance(exc, KnowledgeIntegrityError)
                    else "intake_processing_failed"
                )
            )
            return await self.jobs.finish(binding, job, error_code=code)
        finally:
            for task in (work, pulse):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, pulse, return_exceptions=True)

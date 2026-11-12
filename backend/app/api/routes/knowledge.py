"""Management-only binary PDF intake; students cannot self-grant collection access."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.dependencies import CurrentUserDep
from app.knowledge.errors import (
    KnowledgeConflict,
    KnowledgeError,
    KnowledgeIntegrityError,
    KnowledgeNotFound,
    KnowledgePermissionError,
)
from app.knowledge.extraction import MAX_SOURCE_BYTES
from app.knowledge.intake import IntakeService

router = APIRouter(prefix="/knowledge", tags=["knowledge management"])


def get_intake(request: Request) -> IntakeService:
    service = getattr(request.app.state, "knowledge_intake", None)
    if service is None:
        raise HTTPException(503, "Knowledge intake is not enabled.")
    return cast(IntakeService, service)


IntakeDep = Annotated[IntakeService, Depends(get_intake)]


def public_error(exc: KnowledgeError):
    code = (
        403
        if isinstance(exc, KnowledgePermissionError)
        else (
            404
            if isinstance(exc, KnowledgeNotFound)
            else (
                409
                if isinstance(exc, KnowledgeConflict)
                else 422
                if isinstance(exc, KnowledgeIntegrityError)
                else 503
            )
        )
    )
    raise HTTPException(code, "Knowledge intake request could not be completed.") from None


@router.post("/collections/{collection_id}/documents", status_code=202)
async def upload_pdf(
    collection_id: UUID,
    request: Request,
    user: CurrentUserDep,
    intake: IntakeDep,
    title: Annotated[str, Query(min_length=1, max_length=255)],
    replace_document_id: UUID | None = None,
):
    try:
        intake.binding(collection_id, user.user_id)
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/pdf":
            raise HTTPException(415, "Upload the PDF as application/pdf.")
        content = bytearray()
        async for part in request.stream():
            if len(content) + len(part) > MAX_SOURCE_BYTES:
                raise HTTPException(413, "PDF exceeds the supported size limit.")
            content.extend(part)
        result = await intake.submit(
            collection_id,
            bytes(content),
            title=title,
            replace_document_id=replace_document_id,
            manager_id=user.user_id,
        )
        return result.public()
    except KnowledgeError as exc:
        public_error(exc)


@router.get("/collections/{collection_id}/jobs/{job_id}")
async def job_status(collection_id: UUID, job_id: UUID, user: CurrentUserDep, intake: IntakeDep):
    try:
        binding = intake.binding(collection_id, user.user_id)
        return (await intake.jobs.get(binding, job_id)).public()
    except KnowledgeError as exc:
        public_error(exc)


@router.get("/collections/{collection_id}/documents")
async def list_documents(collection_id: UUID, user: CurrentUserDep, intake: IntakeDep):
    try:
        binding = intake.binding(collection_id, user.user_id)
        active, baseline = await intake.jobs.head(binding)
        data = (
            await intake.jobs.knowledge.get_revision_data(baseline, binding.scope)
            if baseline
            else None
        )
        return {
            "active_revision_id": str(active) if active else None,
            "documents": [
                {"document_id": str(d.document_id), "title": d.title, "source_hash": d.source_hash}
                for d in data.documents
            ]
            if data
            else [],
        }
    except KnowledgeError as exc:
        public_error(exc)


@router.post("/collections/{collection_id}/jobs/{job_id}/retry", status_code=202)
async def retry_job(collection_id: UUID, job_id: UUID, user: CurrentUserDep, intake: IntakeDep):
    try:
        binding = intake.binding(collection_id, user.user_id)
        return (await intake.jobs.retry(binding, job_id)).public()
    except KnowledgeError as exc:
        public_error(exc)

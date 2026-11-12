from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.knowledge.errors import (
    KnowledgeConflict,
    KnowledgeIntegrityError,
    KnowledgeNotFound,
    KnowledgePermissionError,
)
from app.knowledge.repository import KnowledgeRepository, validate_data
from app.knowledge.schemas import (
    AccessScope,
    KnowledgeCollection,
    KnowledgeRevision,
    KnowledgeWriteScope,
    ProcessingStatus,
    RevisionData,
    SharingClass,
)

WRITE_SCOPE = KnowledgeWriteScope(access_domain="course-test", can_manage_knowledge=True)


pytestmark = pytest.mark.asyncio
SCOPE = AccessScope(access_domain="course-test")


async def seed(
    repository: KnowledgeRepository,
    sample: tuple[KnowledgeCollection, KnowledgeRevision, RevisionData],
) -> KnowledgeRevision:
    collection, revision, data = sample
    await repository.create_collection(collection, scope=WRITE_SCOPE)
    await repository.create_revision(revision, scope=WRITE_SCOPE)
    return await repository.write_revision_data(
        revision.revision_id, data, expected_version=0, scope=WRITE_SCOPE
    )


async def test_full_roundtrip_and_deep_copy(repository: KnowledgeRepository, sample: tuple) -> None:
    _, revision, data = sample
    saved = await seed(repository, sample)
    assert saved.version == 1
    result = await repository.get_revision_data(revision.revision_id, SCOPE)
    assert result == validate_data(revision, data)
    assert result is not None
    result.sections.clear()
    reread = await repository.get_revision_data(revision.revision_id, SCOPE)
    assert reread is not None and len(reread.sections) == 1
    assert await repository.get_revision(revision.revision_id, SCOPE) == saved


async def test_duplicate_ids_and_fingerprints(
    repository: KnowledgeRepository, sample: tuple
) -> None:
    collection, revision, _ = sample
    await repository.create_collection(collection, scope=WRITE_SCOPE)
    with pytest.raises(KnowledgeConflict):
        await repository.create_collection(collection, scope=WRITE_SCOPE)
    await repository.create_revision(revision, scope=WRITE_SCOPE)
    with pytest.raises(KnowledgeConflict):
        await repository.create_revision(
            revision.model_copy(update={"revision_id": uuid4()}), scope=WRITE_SCOPE
        )


async def test_missing_parent_and_wrong_dimension(
    repository: KnowledgeRepository, sample: tuple
) -> None:
    collection, revision, _ = sample
    with pytest.raises(KnowledgeNotFound):
        await repository.create_revision(revision, scope=WRITE_SCOPE)
    await repository.create_collection(collection, scope=WRITE_SCOPE)
    wrong = revision.model_copy(
        update={"embedding": revision.embedding.model_copy(update={"dimension": 2})}
    )
    with pytest.raises(KnowledgeIntegrityError):
        await repository.create_revision(wrong, scope=WRITE_SCOPE)


async def test_access_domain_cannot_read_other_collection(
    repository: KnowledgeRepository, sample: tuple
) -> None:
    collection, revision, _ = sample
    await seed(repository, sample)
    wrong = AccessScope(access_domain="other-course")
    assert await repository.get_collection(collection.collection_id, wrong) is None
    assert await repository.get_revision(revision.revision_id, wrong) is None
    assert await repository.get_revision_data(revision.revision_id, wrong) is None
    assert await repository.get_revision_data(uuid4(), SCOPE) is None


async def test_private_collection_requires_owner(
    repository: KnowledgeRepository, sample: tuple
) -> None:
    original, revision, _ = sample
    owner = uuid4()
    collection = original.model_copy(
        update={"sharing_class": SharingClass.PRIVATE, "owner_user_id": owner}
    )
    await repository.create_collection(
        collection, scope=WRITE_SCOPE.model_copy(update={"user_id": owner})
    )
    await repository.create_revision(
        revision, scope=WRITE_SCOPE.model_copy(update={"user_id": owner})
    )
    assert await repository.get_revision(revision.revision_id, SCOPE) is None
    assert (
        await repository.get_revision(
            revision.revision_id, AccessScope(access_domain="course-test", user_id=uuid4())
        )
        is None
    )
    assert (
        await repository.get_revision(
            revision.revision_id, AccessScope(access_domain="course-test", user_id=owner)
        )
        == revision
    )


@pytest.mark.parametrize(
    "failure",
    ["cross_revision", "pending", "dimension", "authority", "support", "duplicate", "manifest"],
)
async def test_rejected_batch_preserves_previous_snapshot(
    repository: KnowledgeRepository,
    sample: tuple,
    failure: str,
) -> None:
    _, revision, original = sample
    await seed(repository, sample)
    bad = original.model_copy(deep=True)
    if failure == "cross_revision":
        bad.sections[0] = bad.sections[0].model_copy(update={"revision_id": uuid4()})
    elif failure == "pending":
        bad.sections[0] = bad.sections[0].model_copy(
            update={"processing_status": ProcessingStatus.PENDING}
        )
    elif failure == "dimension":
        evidence = next(c for c in bad.chunks if c.embedding is not None)
        bad.chunks[bad.chunks.index(evidence)] = evidence.model_copy(
            update={"embedding": [1.0, 0.0]}
        )
    elif failure == "authority":
        bad.documents[0] = bad.documents[0].model_copy(update={"workflow_authority": False})
    elif failure == "support":
        bad.chunks[0] = bad.chunks[0].model_copy(update={"support_ids": [uuid4()]})
    elif failure == "duplicate":
        bad.documents.append(bad.documents[0])
    else:
        bad.documents[0] = bad.documents[0].model_copy(update={"source_hash": "0" * 64})
    with pytest.raises(KnowledgeIntegrityError):
        await repository.write_revision_data(
            revision.revision_id, bad, expected_version=1, scope=WRITE_SCOPE
        )
    assert await repository.get_revision_data(revision.revision_id, SCOPE) == validate_data(
        revision, original
    )
    unchanged = await repository.get_revision(revision.revision_id, SCOPE)
    assert unchanged is not None and unchanged.version == 1


async def test_concurrent_writers_have_one_winner(
    repository: KnowledgeRepository, sample: tuple
) -> None:
    collection, revision, data = sample
    await repository.create_collection(collection, scope=WRITE_SCOPE)
    await repository.create_revision(revision, scope=WRITE_SCOPE)
    outcomes = await asyncio.gather(
        repository.write_revision_data(
            revision.revision_id, data, expected_version=0, scope=WRITE_SCOPE
        ),
        repository.write_revision_data(
            revision.revision_id, data, expected_version=0, scope=WRITE_SCOPE
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(result, KnowledgeRevision) for result in outcomes) == 1
    assert sum(isinstance(result, KnowledgeConflict) for result in outcomes) == 1


async def test_nonexact_single_precision_roundtrip(
    repository: KnowledgeRepository, sample: tuple
) -> None:
    collection, revision, original = sample
    data = original.model_copy(deep=True)
    for index, chunk in enumerate(data.chunks):
        if chunk.embedding is not None:
            values = chunk.model_dump()
            values["embedding"] = [0.1, 0.2, 0.3]
            data.chunks[index] = type(chunk).model_validate(values)
    await repository.create_collection(collection, scope=WRITE_SCOPE)
    await repository.create_revision(revision, scope=WRITE_SCOPE)
    await repository.write_revision_data(
        revision.revision_id, data, expected_version=0, scope=WRITE_SCOPE
    )
    assert await repository.get_revision_data(revision.revision_id, SCOPE) == validate_data(
        revision, data
    )


@pytest.mark.parametrize(
    "denied_scope",
    [
        AccessScope(access_domain="course-test"),
        KnowledgeWriteScope(access_domain="course-test"),
        KnowledgeWriteScope(access_domain="other-course", can_manage_knowledge=True),
    ],
)
async def test_all_mutations_require_scoped_editing_grant(repository, sample, denied_scope):
    collection, revision, data = sample
    await seed(repository, sample)
    with pytest.raises(KnowledgePermissionError):
        await repository.create_collection(
            collection.model_copy(update={"collection_id": uuid4()}), scope=denied_scope
        )
    with pytest.raises(KnowledgePermissionError):
        await repository.create_revision(
            revision.model_copy(update={"revision_id": uuid4()}), scope=denied_scope
        )
    with pytest.raises(KnowledgePermissionError):
        await repository.write_revision_data(
            revision.revision_id, RevisionData(), expected_version=1, scope=denied_scope
        )
    assert await repository.get_revision_data(revision.revision_id, SCOPE) == validate_data(
        revision, data
    )
    assert (await repository.get_revision(revision.revision_id, SCOPE)).version == 1


async def test_private_writes_require_owner_and_editing_grant(repository, sample):
    original, revision, data = sample
    owner = uuid4()
    owner_scope = WRITE_SCOPE.model_copy(update={"user_id": owner})
    collection = original.model_copy(
        update={
            "sharing_class": SharingClass.PRIVATE,
            "owner_user_id": owner,
        }
    )
    await repository.create_collection(collection, scope=owner_scope)
    await repository.create_revision(revision, scope=owner_scope)
    await repository.write_revision_data(
        revision.revision_id, data, expected_version=0, scope=owner_scope
    )
    for denied in [
        WRITE_SCOPE.model_copy(update={"user_id": uuid4()}),
        owner_scope.model_copy(update={"can_manage_knowledge": False}),
    ]:
        with pytest.raises(KnowledgePermissionError):
            await repository.create_collection(
                collection.model_copy(update={"collection_id": uuid4()}), scope=denied
            )
        with pytest.raises(KnowledgePermissionError):
            await repository.create_revision(revision, scope=denied)
        with pytest.raises(KnowledgePermissionError):
            await repository.write_revision_data(
                revision.revision_id, RevisionData(), expected_version=1, scope=denied
            )
    assert await repository.get_revision_data(revision.revision_id, owner_scope) == validate_data(
        revision, data
    )

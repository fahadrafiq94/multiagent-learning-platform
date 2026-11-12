"""Terminal workflow over the existing intake service; mapping is explicit."""

import asyncio
from pathlib import Path

from app.knowledge.errors import KnowledgeConflict, KnowledgeNotFound
from app.knowledge.extraction import MAX_SOURCE_BYTES
from scripts.knowledge_operator_setup import OperatorError


class Console:
    def say(self, text: str):
        print(text, flush=True)

    def ask(self, text: str) -> str:
        return input(text).strip()

    def choose(self, text, values, label=str):
        if not values:
            raise OperatorError("There are no available choices.")
        self.say(text)
        for index, value in enumerate(values, 1):
            self.say(f"  {index}. {label(value)}")
        while True:
            answer = self.ask("Choose a number (0 cancels): ")
            if answer == "0":
                return None
            if answer.isdecimal() and 1 <= int(answer) <= len(values):
                return values[int(answer) - 1]
            self.say("Enter one of the listed numbers.")


def pdf_paths(raw: str, inbox: Path) -> list[Path]:
    path = Path(raw.strip().strip('"') or str(inbox)).expanduser()
    if path.is_symlink():
        raise OperatorError("Choose a regular PDF file or directory, not a symlink.")
    if path.is_dir():
        paths = sorted(
            (p for p in path.iterdir() if p.suffix.lower() == ".pdf"), key=lambda p: p.name.lower()
        )
    else:
        paths = [path]
    if not paths or len(paths) > 100:
        raise OperatorError("Choose a folder containing between 1 and 100 PDFs.")
    for item in paths:
        if item.is_symlink() or not item.is_file() or item.suffix.lower() != ".pdf":
            raise OperatorError("Every selected input must be a regular PDF file.")
        if not 0 < item.stat().st_size <= MAX_SOURCE_BYTES:
            raise OperatorError("Every PDF must be nonempty and at most 25 MiB.")
    return paths


async def drain(service, binding, target_ids, console, *, wait_seconds=180):
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while True:
        jobs = [await service.jobs.get(binding, job_id) for job_id in target_ids]
        pending = [j for j in jobs if j.status in {"queued", "processing"}]
        if not pending:
            for job in jobs:
                console.say(
                    f"{job.job_id}: {job.status}"
                    + (" (duplicate; reused existing index)" if job.duplicate else "")
                    + (f" [{job.error_code}]" if job.error_code else "")
                )
            return
        console.say(
            f"{len(pending)} selected job(s) pending. Processing this collection's queue..."
        )
        result = await service.run_once(binding.collection_id)
        if result:
            console.say(f"Processed {result.job_id}: {result.status}")
            deadline = asyncio.get_running_loop().time() + wait_seconds
        else:
            if asyncio.get_running_loop().time() >= deadline:
                console.say(
                    "Another worker still owns this work. Jobs remain saved; check option 3 later."
                )
                return
            await asyncio.sleep(2)


async def upload_flow(service, binding, operator_id, inbox, console, *, replace=False):
    raw = console.ask(f"PDF file or folder [Enter uses {inbox}]: ")
    paths = pdf_paths(raw, inbox)
    _, baseline = await service.jobs.head(binding)
    snapshot = (
        await service.jobs.knowledge.get_revision_data(baseline, binding.scope)
        if baseline
        else None
    )
    documents = sorted(snapshot.documents, key=lambda d: d.title) if snapshot else []
    if replace and not documents:
        raise KnowledgeNotFound("This collection has no existing documents to replace.")
    plan, selected = [], set()
    for path in paths:
        document = None
        if replace:
            document = console.choose(
                f"Which document does {path.name} replace?",
                documents,
                lambda d: f"{d.title} ({d.document_id})",
            )
            if document is None:
                console.say("Cancelled; no jobs were submitted.")
                return
            if document.document_id in selected:
                raise OperatorError("Two files cannot replace the same document in one selection.")
            selected.add(document.document_id)
        plan.append((path, document))
    console.say("Selected operations:")
    for path, doc in plan:
        console.say(
            f"  {path} -> "
            + (f"replace {doc.title} ({doc.document_id})" if doc else "add/reuse document")
        )
    console.say(
        "Completed jobs publish individually. "
        "Existing queued jobs in this collection may run first."
    )
    if console.ask("Submit these files and process the queue? [y/N]: ").lower() != "y":
        console.say("Cancelled; no jobs were submitted.")
        return
    if (await service.jobs.head(binding))[1] != baseline:
        raise KnowledgeConflict(
            "The collection changed while selecting files; review the mapping again."
        )
    targets = []
    for path, doc in plan:
        with path.open("rb") as stream:
            content = stream.read(MAX_SOURCE_BYTES + 1)
        job = await service.submit(
            binding.collection_id,
            content,
            title=doc.title if doc else path.stem,
            replace_document_id=doc.document_id if doc else None,
            manager_id=operator_id,
        )
        targets.append(job.job_id)
        console.say(f"Saved job {job.job_id}: {job.status}")
    await drain(service, binding, targets, console)


async def jobs_flow(service, binding, console):
    jobs = await service.jobs.list(binding)
    if not jobs:
        console.say("No processing jobs in this collection.")
        return
    for index, job in enumerate(jobs, 1):
        console.say(
            f"{index}. {job.job_id}: {job.status}"
            + (f" [{job.error_code}]" if job.error_code else "")
        )
    action = console.ask(
        "Enter a job number to inspect/retry, 'p' to process queued jobs, or Enter to return: "
    )
    if action.lower() == "p":
        pending = [j.job_id for j in jobs if j.status in {"queued", "processing"}]
        if pending:
            await drain(service, binding, pending, console)
        return
    if not action:
        return
    if not action.isdecimal() or not 1 <= int(action) <= len(jobs):
        raise OperatorError("Choose a listed job number.")
    job = jobs[int(action) - 1]
    console.say(f"Document: {job.payload.title}; status: {job.status}; attempts: {job.attempts}")
    if job.result_revision_id:
        console.say(f"Published revision: {job.result_revision_id}")
    if job.status == "failed" and console.ask("Retry this failed job? [y/N]: ").lower() == "y":
        await service.jobs.retry(binding, job.job_id)
        await drain(service, binding, [job.job_id], console)

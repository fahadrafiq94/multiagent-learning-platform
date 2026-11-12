# Local knowledge management wizard

From a VS Code PowerShell terminal, run:

```powershell
cd C:\Users\rafique\Desktop\multiagent-learning-platform\backend
.venv/Scripts/python.exe -m scripts.manage_knowledge
```

Docker PostgreSQL and Ollama must already be running. This native local operator
command uses the existing protected credentials. It does not start containers,
download models, enable the HTTP management API or enable student-facing RAG.

## Automatic preparation

The wizard creates or completes the repository-root `.knowledge-intake.json` and
validates it before showing the menu. Existing valid collection bindings, manager
grants and index settings are preserved. Recognized REPLACE_ management UUID
placeholders are removed and the local operator UUID is added. Invalid JSON, invalid
scope, malformed UUIDs and mismatched model profiles fail rather than being replaced
with defaults. Unknown configuration fields still follow the strict existing schema.

It discovers existing collections and technically ready revisions from PostgreSQL.
When more than one is eligible, it asks you to select one. An unpublished collection
uses an explicitly recorded ready baseline so earlier documents are preserved.
Index settings come from existing configuration, a matching trusted indexing preset,
or the ready revision's stored embedding profile and existing structural namespace.
Creating new empty collections is still an existing registration/operator task.

For serving, it checks the configured Ollama endpoint and can fall back to the local
127.0.0.1:11434 endpoint when the configured service is unavailable. It verifies the
exact pinned provider/model digest and dimension. Use --model-url for an explicit
endpoint. A mismatched digest is not silently adopted. Process-local ModelService
settings remove the need for manual environment commands; your shell, backend .env
and running app model settings are not changed. Existing additive knowledge migrations
are checked/applied using the configured dimension; applied checksums remain immutable.

The dedicated management identity is created or reused through an OS-bound external
identity link in the existing application tables. Windows uses the actual account SID;
POSIX uses UID and hostname. It never chooses the first student or adopts an account
merely because its student_id is knowledge-operator. The account and initial context
are created atomically, and concurrent launches reuse the same linked identity.
Disabled operators are not reactivated or replaced automatically. No bearer token is
printed, and knowing the management UUID is not sufficient to authenticate.

Configuration writes use a protected temporary file, then atomic replacement with
checks for intervening edits. Existing valid content is not rewritten on repeated
setup. Windows ACLs allow the current OS user, SYSTEM and administrators; POSIX uses
mode 0600. Credentials keep their existing protection. Other ordinary OS accounts and
students using the application do not receive these file permissions. Users sharing
your OS login, or administrators with privileged access, remain inside this trust
boundary. Run this wizard as your local application-file owner, not inside a student
environment or the backend container.

To run preparation without opening the menu:

```powershell
.venv/Scripts/python.exe -m scripts.manage_knowledge --setup-only
```

This can provision the operator and update configuration, but never submits PDFs.

## Menu and file selection

```text
1. Add new PDFs
2. Replace existing documents
3. View/retry processing jobs
0. Exit
```

For either upload option, enter a PDF path or folder path. A folder scan includes
only immediate .pdf/.PDF files, not subdirectories. Pressing Enter uses the
repository-root `.knowledge-inbox` folder, which the wizard creates and Git ignores.
Put only the PDFs you want to submit into that folder. You may also select files
directly from another folder; the wizard does not move or delete your originals.
Inputs retain the existing 25 MiB per-file and 100-file selection limits. Symlink
inputs are rejected.

With option 1, files become new documents or reuse an existing identical source.
Duplicate handling is automatic. With option 2, the wizard shows current document
titles and IDs from PostgreSQL and asks which document each file replaces. It does
not infer replacement identity from filenames or semantic similarity. Two selected
files cannot replace the same document in one selection.

Review the complete mapping and answer y to submit and process it; Enter cancels
without queuing anything. The wizard verifies that the baseline did not change while
you were selecting files. The existing intake service then queues durable jobs,
extracts/indexes the selected PDFs and publishes each completed revision. Existing
queued jobs in the same collection may process first. Updates publish individually,
not as one atomic multi-file batch. Unchanged bytes reuse the existing ready index.

Progress shows saved job IDs, processing status, duplicate reuse and sanitized failure
codes. If another worker owns the collection, the wizard waits for a bounded period
and leaves the durable jobs available for later inspection. Ctrl+C stops the local
wizard; submitted jobs survive and existing lease recovery applies. It is not a
guaranteed transaction rollback for a partially submitted batch.

Option 3 lists the latest 50 jobs for the selected collection with numbered choices.
Select a job to inspect its status/result and retry it if failed, or enter p to process
queued jobs. The same repository scope checks and fenced worker logic apply. Old jobs
outside that list can still be queried using the existing lower-level status command
and their recorded job ID.

## Paths and current state

Defaults are independent of your current working directory:

| Path | Purpose |
|---|---|
| .knowledge-intake.json at repository root | Protected management bindings/index settings |
| .knowledge-inbox at repository root | Optional local PDF selection folder |
| backend/.knowledge-sources | Durable source blobs used by this native wizard |

The script also supports --config, --source-store and --inbox. Keep the same source
store for submission and processing. Native storage is distinct from the existing
Docker knowledge-sources volume; avoid mixing native and Docker workers against
different source stores. See [the intake runbook](knowledge-intake.md) for the lower-level
commands and Docker deployment.

The actual local setup has completed successfully and filled the user's existing
placeholder file. The dedicated operator is provisioned; no PDFs were submitted by
the setup/menu smoke checks. Subsequent operator PDF updates have published the updated
collection. HTTP intake and student RAG remain disabled. Confirmed uploads process and
publish their jobs. Semantic duplicate detection, OCR and general RAG quality calibration
are not added by this usability improvement. Optional Redis retrieval caching is
provided separately by [Checkpoint 10](knowledge-cache.md).

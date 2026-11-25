# Student browser access (Sprint 6 Checkpoints 3-4)

The user selected operator-issued pilot access on 2026-11-17. Each student receives
an expiring signed credential for an already provisioned identity, enters it once
at `/student`, and then uses an HttpOnly browser cookie. There is no public registration
or password account system in this checkpoint. The supplied prototype remains available
as local design source; its simulated conversation is not the authenticated product.

## Run the native pilot entry

Use the existing initialized PostgreSQL infrastructure and a single authoritative
backend worker. If another backend owns the application-worker lease, stop that
backend through its normal operational workflow before starting a native worker.
These commands do not enable RAG, intake or Redis, migrate data or restart vLLM.

From PowerShell in the repository root:

```powershell
cd backend
$env:APPLICATION_STORAGE = 'postgres'
$env:APPLICATION_DATABASE_URL = & .venv/Scripts/python.exe -c "from scripts.application_credentials import configured_dsn; print(configured_dsn().get_secret_value())"
$env:APPLICATION_PILOT_AUTH_SECRET = & .venv/Scripts/python.exe -c "from scripts.application_admin import pilot_secret; print(pilot_secret().get_secret_value())"
$env:STUDENT_BROWSER_ORIGIN = 'http://127.0.0.1:8000'
$env:STUDENT_BROWSER_SECURE_COOKIE = 'false'

.venv/Scripts/python.exe -m scripts.application_admin create-user --student-id P01
```

The operator command returns the internal user UUID, reusing an existing identity
for the same student ID. Use that UUID in the next command:

```powershell
.venv/Scripts/python.exe -m scripts.application_admin token --user-id YOUR_USER_UUID --lifetime 3600
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --loop asyncio:SelectorEventLoop
```

Give the resulting per-student credential privately. It is not the signing secret.
The explicit Selector event loop is required for native Windows psycopg connections.
The installed Uvicorn defaults to Proactor for a single Windows process, which psycopg
rejects; the application wraps that failure as `Application worker could not start`.
Keep this loop option when launching the native PostgreSQL backend.
Open **http://127.0.0.1:8000/student**, enter the credential and sign in. The entry
shows the identity returned by the server and supports reload and sign-out.
Choose New chat to start onboarding. Review the Company panel, correct details in
chat if needed, then explicitly confirm. Chats and history reload from PostgreSQL.
See [the connected workspace guide](../checkpoints/sprint-6-checkpoint-4.md).

Browser settings can instead be placed in the private backend environment file.
`STUDENT_BROWSER_ORIGIN` defaults to unset, so browser-cookie authentication remains
disabled until configured. Existing bearer APIs continue to work. This implementation
did not enable browser access or edit the private environment file.

## Deployment origin

The student HTML, JavaScript and APIs share one externally visible origin. FastAPI
serves `/student` and its `/student/css` and `/student/js` assets directly. There is
no Node production server, frontend build step or permissive CORS configuration.
The browser never connects to vLLM, Redis or PostgreSQL.

For HTTPS deployments set, for example:

```dotenv
STUDENT_BROWSER_ORIGIN=https://your-application.example
STUDENT_BROWSER_SECURE_COOKIE=true
```

Use the exact public origin, without a URL path. A same-origin HTTPS reverse proxy
must forward `/student`, `/auth/*`, `/me` and the existing application APIs. Production
browser authentication refuses HTTP origins or insecure-cookie configuration. Plain
HTTP with `false` is for local development. Spark deployment/TLS remains Checkpoint 6.

Only the public entry and CSS/JS asset directories are mounted. Environment files,
source PDFs, generated contracts, tests, reports and package manifests are not served.
The image copies static assets explicitly; development Compose mounts the frontend
read-only. The original design-review page is not mounted as the student application.

## Authentication contract

| Endpoint | Behavior |
|---|---|
| `POST /auth/session` | Same-origin JSON `{credential}`; verify signature/expiry and active repository identity; set HttpOnly cookie and return `{user, csrf_token}`. |
| `GET /auth/session` | Restore cookie identity and CSRF token; does not refresh the cookie or extend credential expiry. |
| `GET /me` | Verified `{user_id, student_id}` only, through existing bearer auth or browser cookie. No external-provider links or private context. |
| `POST /auth/logout` | Same-origin plus session CSRF; delete the browser cookie. |

The cookie is host-only, Path `/`, SameSite `Strict`, HttpOnly and normally Secure.
HTTPS uses `__Host-fredi_pilot`; local insecure development uses `fredi_pilot_dev`.
It expires no later than the original credential. Every cookie-authenticated write
through the existing identity dependency requires the configured Origin and matching
`X-FREDI-CSRF`. The frontend holds that CSRF value only in memory. Login itself requires
the configured Origin and a bounded JSON body. Existing bearer clients remain compatible;
they do not acquire a new CSRF requirement or silently fall back from invalid auth headers.

The browser does not put access credentials into local/session storage, URLs or console
logs. Form fields clear after submission; identity is rendered as text. Session/API data
is returned with `no-store`. The entry also sets a CSP, no-referrer and nosniff policy.
The supplied prototype's localStorage design preferences do not hold real credentials.

Logout clears this browser's cookie. The original operator-issued bearer credential
remains valid until its expiry; this pilot design does not add a server-side revocation
table. Disabling the identity blocks subsequent requests because the repository status
is checked on every authentication. Provisioning, signing and management permissions
stay in trusted operator tools; a participant code or user-entered student ID is not
authentication. Instructor involvement in RAG content decisions is not introduced.

## Validation and next work

Four Chromium tests exercise real HTTP cookie login, HttpOnly/CSRF storage boundaries,
reload, expiry/logout, two independent student browsers and mobile layout. They also
capture the unchanged prototype at desktop/mobile sizes. Python tests cover CSRF on
existing conversation writes, origin/payload rejection, production-cookie rules,
inactive identities, public identity projection and bearer compatibility. A real
PostgreSQL test verifies identity and conversation persistence after runtime restart.
CI runs schema drift checks, generated types, type checking, client and browser tests.

See [Checkpoint 3](../checkpoints/sprint-6-checkpoint-3.md) for access validation and
[Checkpoint 4](../checkpoints/sprint-6-checkpoint-4.md) for workspace validation and
limitations. Ten Chromium tests now include onboarding, confirmation, history,
citation display and retry reconciliation; separate SQL tests verify durable replay.
Next is prefix reuse and concurrency measurement, not automatic student release.

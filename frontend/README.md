# FREDi frontend: student access and supplied prototype

`student.html` is the real same-origin pilot sign-in entry added in Sprint 6
Checkpoint 3. FastAPI serves it at `/student`; it validates operator-issued access,
restores authenticated identity on reload and supports logout. The original
`index.html`, `css/fredi.css`, `js/fredi.js` and vendor asset remain unchanged.
`index.html` is still the clickable design prototype: its chat answers are simulated.
Checkpoint 4 connects onboarding, explicit scenario confirmation, saved chats/history,
source display and durable request recovery. Mobile Chat/Company tabs use the same
backend state. See [Checkpoint 4](../docs/checkpoints/sprint-6-checkpoint-4.md).

See [student access setup](../docs/operations/student-browser-access.md) for the native
backend configuration and per-student credential commands. Opening `student.html`
as a file is not supported: cookies and APIs require the same HTTP(S) origin.

## Development checks

Node 26 is used in CI. There is no bundler or frontend runtime dependency.

```powershell
cd backend
.venv/Scripts/python.exe -m scripts.export_frontend_contract
cd ../frontend
npm ci --ignore-scripts
npm run types
npm run typecheck
npm test
npx playwright install chromium
npm run test:browser
```

The lockfile pins development tools. `generated/openapi.json` comes from FastAPI;
`generated/api.d.ts` comes from that contract, not handwritten API approximations.
CI checks both for drift. Exported title/version use repository defaults so private
deployment metadata does not make the schema non-reproducible.

Browser tests start and stop their own isolated server on `127.0.0.1:8766`, use
synthetic identities and never call Spark or the retained database. The test-only
credential endpoints are absent from the production app/image. Screenshots and
reports are retained locally under ignored `test-results/` and `playwright-report/`.

The notes below describe only the unchanged design prototype.

## Run it

Open `index.html` in a browser. There is no build step and nothing to install. Fonts load from Google Fonts. Without internet, the browser's fallback font is used.

Student test version: in `index.html`, uncomment the line `window.FREDI_STUDENT = true`. It then starts with a random design variant, asks for a participant code and shows the required check-ins.

## Files

| File | Content |
|---|---|
| `index.html` | Page skeleton plus static markup for the review panel and dialogs |
| `css/fredi.css` | All styles. Design tokens (colours, type, spacing, radii) are in `:root` at the top, dark mode right below |
| `js/fredi.js` | All logic, vanilla JavaScript in one IIFE |
| `vendor/html2canvas.min.js` | html2canvas 1.4.1 (MIT). Only used for screenshots in test feedback |

## How the code is organised (`js/fredi.js`)

- **State:** one object `S` (created by `fresh()`). Chats are in `S.chats`, and `S.messages` always points to the open chat. The company profile is `S.card` and is shared by all chats.
- **Rendering:** `render()` rebuilds the current screen from `S` via template strings: `wsHTML()`, `sideHTML()` (chat list), `panelHTML()` (company profile) and `msgHTML()` (one message).
- **Chats:** `newChat()`, `openChat()`, rename and delete live in the click handler (`data-act="chat-rename"` / `"chat-delete"`). `chatWelcome()` makes sure a new chat never asks for the profile again.
- **Simulated FREDi, the part an agent backend replaces:**
  - `send()` and `respond()` take the student's message and add the reply after a short delay. This is the hook for a real API call.
  - `fredi(text, intent)` decides the reply. Today it runs as a fixed script:
    - company profile setup: `STAGES` with checks, then `answerReview()` and `confirmCard()`
    - APplus coaching: `answerStep()` with `STEPS`, hint ladder and click path
  - Example values are in `EX`. Fallback answers are in `PROTO_REPLIES`.
  - Suggested replies are `chips: [{label, intent}]` on a FREDi message. The intent is sent back with the click.
- **40-step ERP game:** `STEPS` and `PHASES` are still in the code but not shown in the design (decision of 27 Oct 2026).

## Only for the prototype or the student test (not needed in the product)

- Review panel (`#review`): A/B variants, jump to screen states, design notes (`NOTES`), open questions, design settings (`applyDesign()`)
- Logo vote, start dialog with participant code, required check-ins, closing questionnaire
- Results file (`resultsData()`), interaction log (`logEv()`), screenshots (html2canvas)
- Phone frame and simulated keyboard (phone preview on a laptop)

## Data model suggestion for the real system

```json
{
  "student": { "code": "P07" },
  "companyProfile": {
    "version": 1,
    "company": "", "product": "", "customer": "",
    "materials": [{ "name": "", "vendor": "" }],
    "workCenter": "", "operations": [""]
  },
  "chats": [
    { "id": "c1", "title": "Company profile setup", "createdAt": "", "messages": [
      { "role": "user|fredi", "agent": "coach|navigator|scenario", "text": "", "chips": [{ "label": "", "intent": "" }] }
    ]}
  ]
}
```

In the prototype, chats and the profile exist only while the page is open. The real system should store them per student.

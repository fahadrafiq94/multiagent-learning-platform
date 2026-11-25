// @ts-check
import {APIError} from './api.js';
/** @typedef {import('../generated/api.js').components['schemas']} Schemas */
/** @param {unknown} value */
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c] ?? c));
const dayFormat = new Intl.DateTimeFormat(undefined, {dateStyle: 'long'});
const timeFormat = new Intl.DateTimeFormat(undefined, {hour: 'numeric', minute: '2-digit'});

const DRAFT_ID = 'draft'; // unsaved chat: the backend session is created with the first message
export class Workspace {
  /** @param {HTMLElement} root @param {import('./api.js').StudentAPI} api @param {()=>void} expired @param {()=>void} [signedOut] */
  constructor(root, api, expired, signedOut) {
    this.root = root; this.api = api; this.expired = expired; this.signedOut = signedOut;
    /** @type {Schemas['CurrentUserView'] | null} */ this.user = null;
    /** @type {Schemas['ConversationSession'][]} */ this.chats = [];
    /** @type {Schemas['ConversationMessage'][]} */ this.messages = [];
    /** @type {Schemas['WorkspaceView'] | null} */ this.view = null;
    /** @type {Schemas['OrchestrationRequest'] | null} */ this.pending = null;
    /** @type {Map<string, string>} */ this.drafts = new Map();
    this.current = ''; this.error = ''; this.busy = false; this.generation = 0; this.streamTimer = null; this.forceBottom = false;
    this.sideOpen = false; this.userMenu = false;
    /** @type {Schemas['ScenarioView'] | null} */ this.userScenario = null;
    /** @type {{top: number, left: number} | null} */ this.menuPos = null;
    /** @type {'' | 'chats' | 'profile'} */ this.modal = '';
    this.chatQuery = ''; this.chatMenu = ''; this.renaming = ''; this.editing = '';
    root.addEventListener('click', event => {
      if (event.target instanceof Element && event.target.classList.contains('modal-backdrop')) { this.modal = ''; this.render(); return; }
      const target = event.target instanceof Element ? event.target : null;
      let closedPopup = false;
      if (this.userMenu && target && !target.closest('.user-menu') && !target.closest('[data-action="user"]')) { this.userMenu = false; closedPopup = true; }
      if (this.chatMenu && target && !target.closest('.ci-menu') && (target.closest('button[data-action="chat-menu"]')?.getAttribute('data-id') ?? '') !== this.chatMenu) { this.chatMenu = ''; closedPopup = true; }
      const button = target ? target.closest('button') : null;
      if (button?.dataset.chat) { this.modal = ''; void this.open(button.dataset.chat); }
      if (button?.dataset.action === 'side') { this.sideOpen = !this.sideOpen; this.userMenu = false; this.render(); }
      if (button?.dataset.action === 'user') { this.userMenu = !this.userMenu; this.modal = ''; this.render(); if (this.userMenu) this.root.querySelector('.user-menu .btn')?.closest('button')?.focus(); }
      if (button?.dataset.action === 'profile') { this.userMenu = false; this.modal = 'profile'; this.render(); if (!this.view?.scenario) void this.loadScenario(); }
      if (button?.dataset.action === 'sign-out') void this.signOut();
      if (button?.dataset.action === 'modal-new') { this.userMenu = false; this.newDraft(); }
      if (button?.dataset.action === 'modal-chats') { this.userMenu = false; this.chatQuery = ''; this.modal = 'chats'; this.render(); }
      if (button?.dataset.action === 'modal-close') { this.modal = ''; this.render(); }
      if (button?.dataset.action === 'retry') void this.reconcile();
      if (button?.dataset.action === 'refresh') void this.load();
      if (button?.dataset.action === 'confirm') void this.send('I confirm that the scenario shown is correct.');
      if (button?.dataset.action === 'review') { this.userMenu = false; this.modal = 'profile'; this.render(); }
      if (button?.dataset.action === 'chat-menu') {
        if (this.chatMenu === button.dataset.id) { this.chatMenu = ''; this.menuPos = null; this.render(); }
        else {
          const rect = button.getBoundingClientRect();
          const width = 150, height = 112; // ponytail: estimated menu box, only used to flip/clamp at the viewport edge
          const top = rect.bottom + height + 8 > window.innerHeight ? Math.max(8, rect.top - height - 8) : rect.bottom + 4;
          this.chatMenu = button.dataset.id ?? ''; this.menuPos = {top, left: Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8))};
          this.render(); this.root.querySelector('.ci-menu .btn')?.closest('button')?.focus();
        }
      }
      if (button?.dataset.action === 'chat-rename') { this.chatMenu = ''; this.menuPos = null; this.renaming = button.dataset.id ?? ''; this.render(); }
      if (button?.dataset.action === 'chat-delete') { this.chatMenu = ''; this.menuPos = null; void this.deleteChat(button.dataset.id ?? ''); }
      if (button?.dataset.action === 'copy') { const msg = button.closest('.msg'); void this.copyText(button, msg?.querySelector('.message-text')?.textContent ?? ''); }
      if (button?.dataset.action === 'edit') {
        this.editing = button.dataset.id ?? '';
        this.render();
        const textarea = this.root.querySelector('.msg-edit-form textarea');
        if (textarea instanceof HTMLTextAreaElement) {
          textarea.focus();
          textarea.setSelectionRange(textarea.value.length, textarea.value.length);
          textarea.scrollTop = textarea.scrollHeight;
        }
      }
      if (button?.dataset.action === 'edit-cancel') { this.editing = ''; this.render(); }
      if (closedPopup) this.render();
    });
    root.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        if (this.renaming) { const id = this.renaming; this.renaming = ''; this.render(); this.root.querySelector(`.ci-more[data-id="${id}"]`)?.closest('button')?.focus(); return; }
        if (this.chatMenu) { const id = this.chatMenu; this.chatMenu = ''; this.render(); this.root.querySelector(`.ci-more[data-id="${id}"]`)?.closest('button')?.focus(); return; }
        if (this.modal) { this.modal = ''; this.render(); this.root.querySelector('.rail-user')?.closest('button')?.focus(); return; }
        if (this.userMenu) { this.userMenu = false; this.render(); this.root.querySelector('.rail-user')?.closest('button')?.focus(); return; }
        if (this.sideOpen) { this.sideOpen = false; this.render(); this.root.querySelector('.rail-logo')?.closest('button')?.focus(); return; }
        return;
      }
      if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
      if (!(event.target instanceof HTMLTextAreaElement) || event.target.id !== 'message' || matchMedia('(pointer: coarse)').matches) return;
      event.preventDefault();
      void this.send(event.target.value);
    });
    root.addEventListener('input', event => {
      const target = event.target;
      if (target instanceof HTMLTextAreaElement) {
        if (target.id === 'message') this.drafts.set(this.current, target.value);
      }
      else if (target instanceof HTMLInputElement) {
        if (target.id === 'chat-search') {
          this.chatQuery = target.value; this.render();
          const el = this.root.querySelector('#chat-search');
          if (el instanceof HTMLInputElement) { el.focus(); el.setSelectionRange(el.value.length, el.value.length); }
        }
      }
    });
    root.addEventListener('submit', event => {
      event.preventDefault();
      const form = event.target;
      if (form instanceof HTMLFormElement && form.classList.contains('ci-rename')) { void this.commitRename(form); return; }
      if (form instanceof HTMLFormElement && form.classList.contains('msg-edit-form')) { void this.commitEdit(form); return; }
      const input = form instanceof HTMLFormElement ? form.querySelector('textarea') : null;
      if (input) void this.send(input.value);
    });
  }
  stop() { this.clearStream(); this.drafts.clear(); ++this.generation; this.user = null; this.pending = null; this.current = ''; this.messages = []; this.view = null; this.chats = []; this.sideOpen = false; this.userMenu = false; this.menuPos = null; this.modal = ''; this.chatQuery = ''; this.chatMenu = ''; this.renaming = ''; this.editing = ''; this.userScenario = null; this.root.replaceChildren(); }
  async signOut() {
    this.userMenu = false; this.modal = ''; this.render();
    try { await this.api.logout(); } catch (error) {
      if (!(error instanceof APIError) || error.status !== 401) { this.failed(error); return; }
    }
    this.clearPrivateState();
    (this.signedOut ?? this.expired)();
  }
  /** @param {string} id */
  async deleteChat(id) {
    const chat = this.chats.find(c => c.session_id === id);
    if (!chat || !window.confirm('Delete this chat? This cannot be undone.')) return;
    const generation = this.generation;
    try {
      await this.api.deleteConversation(id);
    } catch {
      if (generation === this.generation) { this.error = 'Could not delete this chat.'; this.render(); }
      return;
    }
    if (generation !== this.generation) return;
    this.chats = this.chats.filter(c => c.session_id !== id);
    this.drafts.delete(id);
    if (this.current === id) { this.current = ''; this.messages = []; this.view = null; }
    this.render();
    if (!this.current && this.chats.length) await this.open(this.chats[0].session_id ?? '');
  }
  /** @param {HTMLFormElement} form */
  async commitRename(form) {
    const id = form.dataset.id ?? '';
    if (!id || this.renaming !== id) { this.renaming = ''; this.render(); return; }
    const value = (form.querySelector('input')?.value ?? '').trim();
    this.renaming = '';
    const generation = this.generation;
    try {
      const updated = await this.api.renameConversation(id, value || null);
      if (generation !== this.generation) return;
      const index = this.chats.findIndex(c => c.session_id === id);
      if (index >= 0) this.chats[index] = updated;
    } catch {
      if (generation === this.generation) this.error = 'Could not rename this chat.';
    }
    this.render();
    this.root.querySelector(`.ci-open[data-chat="${id}"]`)?.closest('button')?.focus();
  }
  /** Open an unsaved draft chat; the backend session is created with the first message. */
  newDraft() {
    if (this.busy || this.pending) return;
    ++this.generation;
    this.clearStream();
    if (window.innerWidth < 761) this.sideOpen = false; // ponytail: window width as container-width proxy
    if (this.view?.scenario.status === 'confirmed') this.userScenario = this.view.scenario;
    this.current = DRAFT_ID; this.messages = []; this.view = null; this.error = ''; this.editing = '';
    this.render();
    this.root.querySelector('textarea#message')?.closest('textarea')?.focus();
  }
  savePending() {
    if (!this.user) return;
    try {
      const key = 'fredi-pending-' + this.user.user_id;
      if (this.pending) sessionStorage.setItem(key, JSON.stringify(this.pending));
      else sessionStorage.removeItem(key);
    } catch { /* In-memory retry still works when browser storage is unavailable. */ }
  }
  clearPrivateState() {
    if (this.user) { this.pending = null; this.savePending(); }
    this.stop();
  }
  /** @param {Schemas['CurrentUserView']} user */
  async start(user) {
    if (this.user?.user_id === user.user_id) return;
    this.stop(); this.user = user; this.error = ''; this.busy = false;
    try {
      const raw = sessionStorage.getItem('fredi-pending-' + user.user_id);
      if (raw) {
        const value = JSON.parse(raw);
        const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
        if (value.student_id === user.student_id && uuid.test(value.session_id) && uuid.test(value.request_id) && typeof value.user_message === 'string' && value.user_message.length <= 4000) {
          this.pending = {student_id:user.student_id,session_id:value.session_id,request_id:value.request_id,user_message:value.user_message};
          if (typeof value.edit_message_id === 'string' && uuid.test(value.edit_message_id)) {
            this.pending.edit_message_id = value.edit_message_id;
          }
          this.current = value.session_id;
        }
      }
    } catch { /* Ignore malformed tab-local recovery data. */ }
    await this.load();
    if (this.pending) await this.reconcile();
  }
  /** @param {unknown} error */
  failed(error) {
    if (error instanceof APIError && error.status === 401) { this.stop(); this.expired(); return; }
    this.error = 'Unable to finish this request. Check the connection and try again.';
    this.render();
  }
  async load() {
    const generation = this.generation;
    try {
      /** @type {Schemas['ConversationSession'][]} */ const chats = [];
      for (let offset = 0; ; offset += 100) {
        const page = await this.api.conversations(offset); chats.push(...page.items);
        if (generation !== this.generation) return;
        if (page.items.length < 100) break;
      }
      if (generation !== this.generation) return;
      this.chats = chats;
      if (!this.current && this.chats.length) await this.open(this.chats[0].session_id ?? '');
      else if (!this.current && !this.pending) {
        this.newDraft();
        await this.loadScenario();
      }
      else if (this.current && this.current !== DRAFT_ID && (!this.pending || this.pending.edit_message_id)) { this.forceBottom = true; await this.refresh(generation); this.error = ''; this.render(); }
      else this.render();
    } catch(error) { if (generation === this.generation) this.failed(error); }
  }
  /** Fetch the user-level saved scenario so the profile dialog works without an open chat. */
  async loadScenario() {
    const generation = this.generation;
    try {
      const scenario = await this.api.scenario();
      if (generation === this.generation) {
        this.userScenario = scenario;
        if (this.modal === 'profile' || (this.current === DRAFT_ID && scenario.status === 'confirmed')) this.render();
      }
    } catch { /* No saved card yet: the dialog keeps its "being set up" state. */ }
  }
  /** @param {string} id */
  async open(id) {
    this.clearStream();
    if (this.busy || this.pending) return;
    if (window.innerWidth < 761) this.sideOpen = false; // ponytail: window width as container-width proxy
    const index = this.chats.findIndex(c => c.session_id === id);
    if (index > 0) { const [chat] = this.chats.splice(index, 1); this.chats.unshift(chat); }
    const generation = ++this.generation; this.forceBottom = true; this.current = id; this.messages = []; this.view = null;
    this.busy = true; this.editing = ''; this.error = ''; this.render();
    try { await this.refresh(generation); }
    catch(error) { if (generation === this.generation) this.failed(error); }
    finally { if (generation === this.generation) { this.busy = false; this.render(); } }
  }
  /** @param {number} generation */
  async refresh(generation) {
    const id = this.current;
    const view = await this.api.workspace(id);
    /** @type {Schemas['ConversationMessage'][]} */ const messages = [];
    for (let offset = 0; ; offset += 100) {
      const page = await this.api.history(id, offset); messages.push(...page.items);
      if (generation !== this.generation) return;
      if (page.items.length < 100) break;
    }
    if (generation === this.generation) { this.view = view; this.messages = messages; }
  }
  /** @param {string} text */
  async send(text) {
    if (this.busy || this.pending || !this.user || !this.current || !text.trim() || text.trim().length > 4000) return;
    const generation = this.generation;
    let sessionId = this.current;
    if (sessionId === DRAFT_ID) {
      this.busy = true;
      this.error = '';
      this.render();
      try {
        const words = Array.from(text.trim().replace(/\s+/g, ' '));
        const title = words.slice(0, 48).join('') + (words.length > 48 ? '…' : '');
        const chat = await this.api.createConversation(title);
        if (generation !== this.generation) return;
        this.chats.unshift(chat);
        this.drafts.delete(DRAFT_ID);
        sessionId = chat.session_id ?? '';
        this.current = sessionId;
      } catch(error) {
        if (generation === this.generation) { this.busy = false; this.failed(error); }
        return;
      }
    }
    this.pending = {request_id: crypto.randomUUID(), session_id: sessionId, student_id: this.user.student_id, user_message: text.trim()};
    this.savePending();
    this.drafts.delete(sessionId);
    await this.submit();
  }
  async submit() {
    if (!this.pending) return;
    const generation = this.generation; let stream = false; this.busy = true; this.error = ''; this.render();
    try {
      const reply = await this.api.invoke(this.pending);
      if (generation !== this.generation) return;
      await this.refresh(generation);
      if (generation !== this.generation) return;
      this.pending = null;
      this.savePending();
      if (reply.error) this.error = 'FREDi could not complete the reply. Your saved messages are shown below.';
      else stream = true;
    } catch(error) {
      if (generation !== this.generation) return;
      if (this.pending?.edit_message_id && error instanceof APIError && error.status === 409) {
        this.busy = false;
        await this.reconcile(false);
      } else this.failed(error);
    } finally { if (generation === this.generation) { this.busy = false; this.render(); if (stream) this.playStreaming(); } }
  }
  async reconcile(resendMissing = true) {
    if (!this.pending || this.busy) return;
    const generation = this.generation; let stream = false; this.busy = true; this.error = ''; this.render();
    try {
      const turn = await this.api.turn(this.current, String(this.pending.request_id));
      if (generation !== this.generation) return;
      if (turn.status === 'processing') this.error = 'FREDi is still working. Check again shortly.';
      else {
        await this.refresh(generation);
        if (generation !== this.generation) return;
        this.pending = null;
        this.savePending();
        if (turn.status !== 'completed') this.error = 'The previous attempt stopped. Review the saved history before sending another message.';
        else stream = true;
      }
    } catch(error) {
      if (generation !== this.generation) return;
      if (error instanceof APIError && error.status === 404) {
        if (resendMissing) {
          this.busy = false;
          await this.submit();
        } else {
          this.pending = null;
          this.savePending();
          try {
            await this.refresh(generation);
            if (generation === this.generation) this.error = 'This edit could not be saved. Check the latest exchange and try again.';
          } catch (refreshError) {
            if (generation === this.generation) this.failed(refreshError);
          }
        }
      } else this.failed(error);
    } finally { if (generation === this.generation) { this.busy = false; this.render(); if (stream) this.playStreaming(); } }
  }
  /** Reveal the last assistant reply typewriter-style without re-rendering, so the thread keeps its scroll. */
  playStreaming() {
    this.clearStream();
    const nodes = this.root.querySelectorAll('.msg.fredi .message-text');
    const last = nodes[nodes.length - 1];
    if (!last) return;
    const full = last.textContent ?? '';
    if (full.length < 2) return;
    // ponytail: constant ~2s reveal (120 ticks x 18ms); swap for real SSE streaming when the backend gains an endpoint
    const perTick = Math.max(2, Math.ceil(full.length / 120));
    let i = 0;
    last.textContent = '';
    const step = () => {
      if (!last.isConnected) { this.clearStream(); return; }
      i = Math.min(full.length, i + perTick);
      last.textContent = full.slice(0, i);
      const thread = this.root.querySelector('.thread');
      if (thread) thread.scrollTo({top: thread.scrollHeight, behavior: 'instant'});
      if (i < full.length) this.streamTimer = setTimeout(step, 18);
    };
    step();
  }
  clearStream() { if (this.streamTimer) { clearTimeout(this.streamTimer); this.streamTimer = null; } }
  /** @returns {string} The account popover, rendered next to whichever chip opened it. */
  userMenuMarkup() {
    /** @param {string} action @param {string} label */
    const item = (action, label, extra = '') => `<button class="btn ghost ${extra}" role="menuitem" data-action="${action}">${label}</button>`;
    return `<div class="menu user-menu" role="menu">${item('profile', 'Profile')}${item('sign-out', 'Sign out', 'danger-text')}</div>`;
  }
  /** @param {Schemas['ConversationMessage']} m @param {string} who @param {string} sources @param {string} timestamp */
  messageArticle(m, who, sources, timestamp) {
    const isUser = who === 'You';
    const canEdit = isUser && this.editableMessageId() === m.message_id && !this.busy && !this.pending;
    const editing = this.editing === m.message_id;
    const tools = editing ? '' : `<div class="msg-tools">${timestamp}<button class="msg-tool" data-action="copy" title="Copy" aria-label="Copy message">⧉</button>${canEdit ? `<button class="msg-tool" data-action="edit" title="Edit" data-id="${esc(m.message_id)}" aria-label="Edit message">✎</button>` : ''}</div>`;
    const body = editing
      ? `<form class="msg-edit-form" data-id="${esc(m.message_id)}" novalidate>
          <label class="sr-only" for="me-${esc(m.message_id)}">Edit message</label>
          <textarea class="input" id="me-${esc(m.message_id)}" rows="3" maxlength="4000">${esc(m.content)}</textarea>
          <div class="msg-edit-actions">
            <button class="btn outline" type="button" data-action="edit-cancel">Cancel</button>
            <button class="btn primary" type="submit">Send</button>
          </div>
        </form>`
      : `<p class="message-text">${esc(m.content)}</p>`;
    return `<article class="msg ${isUser ? 'user' : 'fredi'}${editing ? ' editing' : ''}"><div class="bubble">${isUser ? '' : '<strong>FREDi</strong>'}${body}${sources}</div>${tools}</article>`;
  }
  editableMessageId() {
    const [user, assistant] = this.messages.slice(-2);
    return this.view?.can_regenerate && user?.role === 'user' && assistant?.role === 'assistant'
      ? user.message_id : undefined;
  }
  displayMessages() {
    const edit = this.pending;
    const messages = this.messages.filter(m => m.role === 'user' || m.role === 'assistant');
    if (!edit?.edit_message_id) return messages;
    const index = messages.findIndex(m => m.message_id === edit.edit_message_id);
    if (index < 0) return messages;
    const original = messages[index];
    const afterReply = messages[index + 1]?.role === 'assistant' ? index + 2 : index + 1;
    return [
      ...messages.slice(0, index),
      {...original, content: edit.user_message, created_at: undefined},
      ...messages.slice(afterReply),
    ];
  }
  /** @param {HTMLFormElement} form */
  async commitEdit(form) {
    const id = form.dataset.id;
    const value = form.querySelector('textarea')?.value.trim();
    if (!this.user || this.busy || this.pending || !id || id !== this.editableMessageId() || !value || value.length > 4000) return;
    this.editing = '';
    this.forceBottom = true;
    this.pending = {
      edit_message_id: id,
      request_id: crypto.randomUUID(),
      session_id: this.current,
      student_id: this.user.student_id,
      user_message: value,
    };
    this.savePending();
    await this.submit();
  }
  /** @param {HTMLButtonElement} button @param {string} text */
  async copyText(button, text) {
    try { await navigator.clipboard.writeText(text); }
    catch { const area = document.createElement('textarea'); area.value = text; document.body.append(area); area.select(); document.execCommand('copy'); area.remove(); }
    button.textContent = '✓';
    setTimeout(() => { if (button.isConnected) button.textContent = '⧉'; }, 1200);
  }
  /** @param {Schemas['ConversationSession']} c */
  chatRow(c, snippet = '') {
    const id = c.session_id;
    const cur = id === this.current;
    const title = c.title || (snippet ? (snippet.length >= 48 ? snippet + '…' : snippet) : 'Chat');
    if (this.renaming === id) {
      return `<li class="ci ${cur ? 'cur' : ''}"><form class="ci-rename" data-id="${esc(id)}" novalidate><label class="sr-only" for="rn-${esc(id)}">Chat name</label><input class="input" id="rn-${esc(id)}" value="${esc(c.title ?? '')}" maxlength="60" autocomplete="off"></form></li>`;
    }
    return `<li class="ci ${cur ? 'cur' : ''}"><button class="ci-open" data-chat="${esc(id)}" ${cur ? 'aria-current="page"' : ''}>${esc(title)}</button><button class="ci-more" data-action="chat-menu" data-id="${esc(id)}" title="Options" aria-label="Options for ${esc(title)}" aria-haspopup="menu" aria-expanded="${this.chatMenu === id}">⋯</button></li>`;
  }
  profileBody() {
    const scenario = this.view?.scenario ?? this.userScenario;
    const fields = [['Company',scenario?.company_context],['Your role',scenario?.learner_role],['Business problem',scenario?.business_problem],['Goals',scenario?.business_goals?.join('\n')],['Constraints',scenario?.constraints?.join('\n')]];
    const status = scenario?.status==='confirmed'?'Saved':scenario?.status==='reviewing'?'Please review':'Being set up';
    return `<div class="card-head"><span class="status">${status}</span></div><dl class="facts card-facts">${fields.map(([name,value])=>`<div><dt>${esc(name)}</dt><dd>${esc(value || 'Not defined yet')}</dd></div>`).join('')}</dl>${scenario?.status==='reviewing'?`<button class="btn primary" data-action="confirm" ${this.busy||this.pending?'disabled':''}>Confirm scenario</button><p>To change a detail, tell FREDi in the chat before confirming.</p>`:''}<p class="shared">${scenario?.status==='confirmed'?'Saved to your profile and available in your other chats.':'Review and confirm your scenario before it is saved to your profile.'}</p>${this.view?.current_step_id?`<p>Current step: ${esc(this.view.current_step_id)}</p>`:''}`;
  }
  modalContent() {
    if (this.modal === 'chats') {
      const query = this.chatQuery.toLowerCase();
      const rows = this.chats.filter(c => !query || (c.title ?? '').toLowerCase().includes(query));
      return `<div class="modal" role="dialog" aria-modal="true" aria-label="Chats"><div class="modal-head"><h2>Chats</h2><button class="btn ghost" data-action="modal-close" aria-label="Close">✕</button></div><div class="modal-body"><input class="input" id="chat-search" placeholder="Search chats…" value="${esc(this.chatQuery)}" autocomplete="off"><ul class="chat-list modal-list">${rows.length?rows.map(c=>this.chatRow(c)).join(''):'<p class="empty">No chats match.</p>'}</ul></div></div>`;
    }
    if (this.modal === 'profile') {
      return `<div class="modal" role="dialog" aria-modal="true" aria-label="Company profile"><div class="modal-head"><h2>Company profile</h2><button class="btn ghost" data-action="modal-close" aria-label="Close">✕</button></div><div class="modal-body profile-body">${this.profileBody()}</div></div>`;
    }
    return '';
  }
  render() {
    if (!this.user) return;
    const disabled = this.busy || !!this.pending;
    const scenario = this.view?.scenario ?? this.userScenario;
    const firstUser = this.messages.find(m => m.role === 'user');
    const snippet = firstUser ? firstUser.content.replace(/\s+/g, ' ').trim().slice(0, 48) : '';
    const rows = this.chats.map(c => this.chatRow(c, c.session_id === this.current ? snippet : '')).join('');
    const studentId = this.user.student_id;
    /** @param {string} cls @param {boolean} expanded */
    const userButton = (cls, expanded) => `<button class="${cls}" data-action="user" title="Account menu" aria-haspopup="menu" aria-expanded="${expanded}" aria-label="Account menu">${esc(studentId)}</button>`;
    const rail = `<div class="rail"><button class="rail-logo" data-action="side" title="Chats" aria-expanded="${this.sideOpen}" aria-label="${this.sideOpen ? 'Hide chat list' : 'Show chat list'}">F</button><button class="rail-btn" data-action="modal-new" ${disabled ? 'disabled' : ''} title="New chat" aria-label="New chat">+</button><button class="rail-btn" data-action="modal-chats" title="Search chats" aria-label="Search chats">⌕</button><span class="rail-user-anchor">${userButton('rail-user', this.userMenu)}${this.userMenu ? this.userMenuMarkup() : ''}</span></div>`;
    const side = `<nav class="side" aria-label="Your chats"><div class="side-head"><strong>FREDi</strong><button class="btn ghost" data-action="side" aria-label="Close chat list">Close</button></div><button class="btn outline new-chat" data-action="modal-new" title="New chat" ${disabled ? 'disabled' : ''}>New chat</button><h2 class="side-h">Chats</h2><ul class="chat-list">${rows}</ul></nav>`;
    const greeting = !this.current ? '<p>Create a new chat to begin.</p>'
      : !this.messages.length && !this.busy
        ? (scenario?.status === 'confirmed' ? '<p>Your company profile is saved. Ask FREDi anything about your learning.</p>'
          : '<article class="msg fredi"><div class="bubble"><strong>FREDi</strong><p class="message-text">Hi! I’m FREDi. Let’s explore a company you’d enjoy working with. Do you have an idea, or would you like a few suggestions?</p></div></article>')
        : '';
    let previousDay = '';
    const messages = this.displayMessages().map(m => {
      let divider = '';
      let timestamp = '';
      if (m.created_at) {
        const date = new Date(m.created_at);
        if (Number.isFinite(date.getTime())) {
          const day = dayFormat.format(date);
          if (day !== previousDay) {
            divider = `<div class="msg-day">${esc(day)}</div>`;
            previousDay = day;
          }
          const time = timeFormat.format(date);
          timestamp = `<time class="msg-time" datetime="${esc(m.created_at)}" tabindex="0" aria-label="${esc(`${day}, ${time}`)}">${esc(time)}</time>`;
        }
      }
      const who = m.role === 'user' ? 'You' : 'FREDi';
      const sources = m.source_references?.length
        ? `<details><summary>Sources</summary><ul>${m.source_references.map(r => `<li>${esc(r.title || 'Source document')}${r.page_start ? ` / page ${r.page_start}${r.page_end && r.page_end !== r.page_start ? `-${r.page_end}` : ''}` : ''}</li>`).join('')}</ul></details>`
        : '';
      return divider + this.messageArticle(m, who, sources, timestamp);
    }).join('');
    const pendingMsg = this.pending && !this.pending.edit_message_id ? `<article class="msg user"><div class="bubble"><p class="message-text">${esc(this.pending.user_message)}</p></div><div class="msg-tools"><button class="msg-tool" data-action="copy" title="Copy" aria-label="Copy message">⧉</button></div></article>` : '';
    const working = this.busy ? '<p role="status">FREDi is working...</p>' : '';
    const reviewCta = scenario?.status === 'reviewing' ? `<div class="review-cta"><span>Your company profile is ready. Review the details, then confirm it.</span><button class="btn primary" data-action="review">Review profile</button></div>` : '';
    const errText = this.error ? `<p class="err" role="alert">${esc(this.error)}</p>` : '';
    const retry = this.pending && !this.busy ? '<button class="btn outline" data-action="retry">Check request / retry</button>' : '';
    const refresh = this.error && !this.pending ? '<button class="btn outline" data-action="refresh">Refresh chats</button>' : '';
    const composer = `<form class="composer"><label class="sr-only" for="message">Message to FREDi</label><div class="composer-row"><textarea id="message" maxlength="4000" rows="1" placeholder="Message FREDi..." ${disabled || !this.current ? 'disabled' : ''}>${esc(this.drafts.get(this.current) ?? '')}</textarea><button class="btn primary" type="submit" ${disabled || !this.current ? 'disabled' : ''}>Send</button></div></form>`;
    const chat = `<section class="chat" aria-label="Conversation with FREDi"><div class="thread" role="log" aria-live="polite"><div class="msgs">${greeting}${messages}${pendingMsg}${working}</div></div><div class="composer-wrap">${reviewCta}${errText}${retry}${refresh}${composer}</div></section>`;
    const currentChat = this.chats.find(c => c.session_id === this.current);
    const topTitle = !this.current ? 'FREDi' : this.current === DRAFT_ID ? 'New chat' : currentChat?.title || (snippet ? (snippet.length >= 48 ? snippet + '…' : snippet) : 'Chat');
    const workspace = `<div class="ws ctx-A tab-chat panel-off"><header class="ws-head"><div class="ws-title"><strong>${esc(topTitle)}</strong></div></header><div class="ws-body">${chat}</div></div>`;
    const prevThread = this.root.querySelector('.thread');
    const prevScrollTop = prevThread ? prevThread.scrollTop : 0;
    const prevScrollHeight = prevThread ? prevThread.scrollHeight : 0;
    const chatMenu = this.chatMenu && this.menuPos
      ? `<div class="menu ci-menu" role="menu"><button class="btn ghost" role="menuitem" data-action="chat-rename" data-id="${esc(this.chatMenu)}">Rename</button><button class="btn ghost danger-text" role="menuitem" data-action="chat-delete" data-id="${esc(this.chatMenu)}">Delete</button></div>`
      : '';
    const backdrop = chatMenu + (this.modal ?`<div class="modal-backdrop">${this.modalContent()}</div>` : '');
    this.root.innerHTML = `<div class="shell${this.sideOpen ? ' side-open' : ''}">${rail}${side}${workspace}</div>${backdrop}`;
    const thread = this.root.querySelector('.thread');
    // ponytail: within 80px of the bottom counts as "staying at the latest"; a real scroll-follow control replaces this if finer control is wanted
    const stick = this.forceBottom || (prevThread ? prevThread.scrollHeight - prevThread.scrollTop - prevThread.clientHeight <= 80 : false);
    if (thread) thread.scrollTo({top: stick ? thread.scrollHeight : prevScrollTop, behavior: 'instant'});
    this.forceBottom = false;
    const menu = this.root.querySelector('.ci-menu');
    if (menu instanceof HTMLElement && this.menuPos) {
      menu.style.top = `${this.menuPos.top}px`;
      menu.style.left = `${this.menuPos.left}px`;
    }
    const modalFocus = this.root.querySelector('#chat-search') ?? this.root.querySelector('.modal-head button');
    if (this.modal && modalFocus instanceof HTMLElement) modalFocus.focus();
    const renameInput = this.root.querySelector('.ci-rename input');
    if (renameInput instanceof HTMLInputElement && renameInput.form) {
      const form = renameInput.form;
      setTimeout(() => { renameInput.focus(); renameInput.select(); }, 0);
      let done = false;
      const finish = () => { if (done) return; done = true; if (form.isConnected) void this.commitRename(form); };
      renameInput.addEventListener('blur', () => setTimeout(finish, 0));
    }
  }
}

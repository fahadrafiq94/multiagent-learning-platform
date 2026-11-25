// @ts-check
/** @typedef {import('../generated/api.js').components['schemas']['BrowserSessionView']} BrowserSession */
/** @typedef {import('../generated/api.js').components['schemas']['CurrentUserView']} CurrentUser */

export class APIError extends Error {
  /** @param {number} status */
  constructor(status) {
    super(status === 401 ? 'Your access credential is invalid or has expired.'
      : status === 403 ? 'Please refresh the page and try again.'
      : status === 503 ? 'Student access is currently unavailable. Please contact your instructor.'
      : 'Unable to connect. Please try again.');
    this.status = status;
  }
}

/** Same-origin requests only. Credentials never enter storage, URLs or console output. */
export class StudentAPI {
  /** @type {string | null} */
  #csrf = null;
  /** @param {typeof fetch} [transport] */
  constructor(transport = globalThis.fetch.bind(globalThis)) { this.transport = transport; }

  /** @template T @param {string} path @param {RequestInit} [options] @returns {Promise<T>} */
  async request(path, options = {}) {
    if (!path.startsWith('/') || path.startsWith('//')) throw new Error('Same-origin path required.');
    const headers = new Headers(options.headers);
    if (options.body) headers.set('Content-Type', 'application/json');
    if (options.method && options.method !== 'GET' && this.#csrf) headers.set('X-FREDI-CSRF', this.#csrf);
    let response;
    try {
      response = await this.transport(path, {
        ...options, headers, credentials: 'same-origin', cache: 'no-store', redirect: 'error',
        signal: options.signal ?? AbortSignal.timeout(15000),
      });
    } catch { throw new APIError(0); }
    if (!response.ok) {
      if (response.status === 401) this.#csrf = null;
      throw new APIError(response.status);
    }
    if (response.status === 204) return /** @type {T} */ (undefined);
    try { return await response.json(); } catch { throw new APIError(502); }
  }

  /** @param {string} credential @returns {Promise<BrowserSession>} */
  async login(credential) {
    const session = /** @type {BrowserSession} */ (await this.request('/auth/session', {
      method: 'POST', body: JSON.stringify({credential}),
    }));
    this.#csrf = session.csrf_token;
    return session;
  }
  /** @returns {Promise<BrowserSession>} */
  async session() {
    const session = /** @type {BrowserSession} */ (await this.request('/auth/session'));
    this.#csrf = session.csrf_token;
    return session;
  }
  /** @returns {Promise<CurrentUser>} */
  me() { return this.request('/me'); }
  /** @param {number} [offset] @returns {Promise<import('../generated/api.js').components['schemas']['ConversationPage']>} */
  conversations(offset = 0) { return this.request(`/conversations?limit=100&offset=${offset}`); }
  /** @param {string | null} [title] @returns {Promise<import('../generated/api.js').components['schemas']['ConversationSession']>} */
  createConversation(title = null) { return this.request('/conversations', {method: 'POST', body: JSON.stringify(title ? {title} : {})}); }
  /** @param {string} id @returns {Promise<import('../generated/api.js').components['schemas']['WorkspaceView']>} */
  workspace(id) { return this.request(`/student-api/conversations/${encodeURIComponent(id)}`); }
  /** @returns {Promise<import('../generated/api.js').components['schemas']['ScenarioView']>} */
  scenario() { return this.request('/student-api/scenario'); }
  /** @param {string} id @param {string | null} title @returns {Promise<import('../generated/api.js').components['schemas']['ConversationSession']>} */
  renameConversation(id, title) { return this.request(`/conversations/${encodeURIComponent(id)}`, {method: 'PATCH', body: JSON.stringify({title})}); }
  /** @param {string} id @returns {Promise<void>} */
  deleteConversation(id) { return this.request(`/conversations/${encodeURIComponent(id)}`, {method: 'DELETE'}); }
  /** @param {string} id @param {string} messageId @param {string} content @returns {Promise<import('../generated/api.js').components['schemas']['ConversationMessage']>} */
  updateMessage(id, messageId, content) { return this.request(`/conversations/${encodeURIComponent(id)}/messages/${encodeURIComponent(messageId)}`, {method: 'PATCH', body: JSON.stringify({content})}); }
  /** @param {string} id @param {number} offset @returns {Promise<import('../generated/api.js').components['schemas']['HistoryPage']>} */
  history(id, offset = 0) { return this.request(`/conversations/${encodeURIComponent(id)}/messages?limit=100&offset=${offset}`); }
  /** @param {import('../generated/api.js').components['schemas']['OrchestrationRequest']} body @returns {Promise<import('../generated/api.js').components['schemas']['OrchestrationResponse']>} */
  invoke(body) { return this.request('/orchestration/invoke', {method: 'POST', body: JSON.stringify(body), signal: AbortSignal.timeout(180000)}); }
  /** @param {string} id @param {string} request @returns {Promise<import('../generated/api.js').components['schemas']['TurnView']>} */
  turn(id, request) { return this.request(`/conversations/${encodeURIComponent(id)}/turns/${encodeURIComponent(request)}`); }
  async logout() {
    await this.request('/auth/logout', {method: 'POST'});
    this.#csrf = null;
  }
}

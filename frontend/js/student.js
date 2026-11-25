// @ts-check
import {StudentAPI, APIError} from './api.js';
import {Workspace} from './workspace.js';

const api = new StudentAPI();
/** @template {HTMLElement} T @param {string} id @returns {T} */
function element(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error('Missing student entry element.');
  return /** @type {T} */ (node);
}
const loading = element('loading');
const signIn = element('sign-in');
const signedIn = element('signed-in');
/** @type {HTMLInputElement} */
const credential = element('credential');
/** @type {HTMLButtonElement} */
const signInButton = element('sign-in-button');
/** @type {HTMLButtonElement} */
const signOut = element('sign-out');
let generation = 0;
const workspace = new Workspace(element('workspace'), api,
  () => showLogin('Your session expired. Please sign in again.'),
  () => showLogin());

/** @param {string} [message] */
function showLogin(message = '') {
  workspace.stop(); element('workspace').hidden = true;
  document.body.classList.remove('workspace-active');
  loading.hidden = signedIn.hidden = true;
  signIn.hidden = false;
  element('student-id').textContent = '';
  element('access-error').textContent = message;
  credential.value = '';
}
/** @param {import('../generated/api.js').components['schemas']['BrowserSessionView']} session */
function showSession(session) {
  loading.hidden = signIn.hidden = true;
  signedIn.hidden = false;
  credential.value = '';
  element('student-id').textContent = session.user.student_id;
  element('session-error').textContent = '';
  element('workspace').hidden = false;
  document.body.classList.add('workspace-active');
  void workspace.start(session.user);
}
/** @param {unknown} error */
const errorMessage = error => error instanceof APIError ? error.message : 'Unable to connect. Please try again.';

async function resume() {
  const current = ++generation;
  try {
    const session = await api.session();
    if (current === generation) showSession(session);
  } catch (error) {
    if (current === generation) showLogin(error instanceof APIError && error.status === 401 ? '' : errorMessage(error));
  }
}
element('access-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (signInButton.disabled) return;
  const current = ++generation;
  const value = credential.value.trim();
  credential.value = '';
  signInButton.disabled = true;
  element('access-error').textContent = '';
  try {
    const session = await api.login(value);
    if (current === generation) showSession(session);
  } catch (error) {
    if (current === generation) showLogin(errorMessage(error));
  } finally { credential.value = ''; signInButton.disabled = false; }
});
signOut.addEventListener('click', async () => {
  if (signOut.disabled) return;
  const current = ++generation;
  signOut.disabled = true;
  try {
    await api.logout();
    workspace.clearPrivateState();
    if (current === generation) { showLogin(); credential.focus(); }
  } catch (error) {
    if (current === generation) {
      if (error instanceof APIError && error.status === 401) showLogin();
      else element('session-error').textContent = errorMessage(error);
    }
  } finally { signOut.disabled = false; }
});
addEventListener('pageshow', event => { if (event.persisted) void resume(); });
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && !signedIn.hidden) void resume(); });
void resume();

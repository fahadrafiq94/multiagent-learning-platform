import test from 'node:test';
import assert from 'node:assert/strict';
import {StudentAPI, APIError} from '../js/api.js';

test('login uses a same-origin body, cookie requests and ephemeral CSRF', async () => {
  const calls = [];
  const api = new StudentAPI(async (url, options) => {
    calls.push({url, options});
    return url === '/auth/logout' ? new Response(null, {status: 204})
      : Response.json({user: {user_id: 'id', student_id: 'student'}, csrf_token: 'csrf'});
  });
  await api.login('synthetic-secret');
  await api.logout();
  assert.equal(calls[0].url, '/auth/session');
  assert.deepEqual(JSON.parse(calls[0].options.body), {credential: 'synthetic-secret'});
  assert.equal(calls[0].options.credentials, 'same-origin');
  assert.equal(calls[0].options.redirect, 'error');
  assert.equal(calls[0].options.cache, 'no-store');
  assert.equal(calls[1].options.headers.get('X-FREDI-CSRF'), 'csrf');
  assert.equal(calls[0].options.headers.has('Authorization'), false);
  await assert.rejects(api.request('https://other.example/me'), /Same-origin/);
  await assert.rejects(api.request('//other.example/me'), /Same-origin/);
});

test('provider errors and malformed response bodies are not reflected to students', async () => {
  const api = new StudentAPI(async () => new Response('private-provider-body', {status: 503}));
  await assert.rejects(api.me(), error => error instanceof APIError && error.status === 503 && !error.message.includes('private'));
  const broken = new StudentAPI(async () => new Response('not-json'));
  await assert.rejects(broken.me(), error => error instanceof APIError && error.status === 502);
});

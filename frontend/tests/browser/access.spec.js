import {test, expect} from '@playwright/test';

test.beforeEach(async ({context}) => {
  await context.route('https://fonts.googleapis.com/**', route => route.abort());
  await context.route('https://fonts.gstatic.com/**', route => route.abort());
});

test('login, HttpOnly isolation, reload and logout use the actual backend', async ({page, request}, testInfo) => {
  const credentials = await (await request.get('/__test__/credentials')).json();
  await page.goto('/student');
  await expect(page.getByRole('button', {name: 'Sign in', exact: true})).toBeVisible();
  await page.screenshot({path: testInfo.outputPath('student-access-desktop.png'), fullPage: true});
  await page.getByLabel('Access credential').fill(credentials['student-one']);
  await page.getByRole('button', {name: 'Sign in', exact: true}).click();
  await expect(page.locator('#student-id')).toHaveText('student-one');
  await expect(page.locator('#credential')).toHaveValue('');
  const cookie = (await page.context().cookies()).find(c => c.name === 'fredi_pilot_dev');
  expect(cookie.httpOnly).toBe(true);
  expect(cookie.sameSite).toBe('Strict');
  const browserData = await page.evaluate(() => ({cookie: document.cookie, local: {...localStorage}, session: {...sessionStorage}, url: location.href}));
  expect(JSON.stringify(browserData)).not.toContain(credentials['student-one']);
  await page.reload();
  await expect(page.locator('#student-id')).toHaveText('student-one');
  await page.getByRole('button', {name: 'Account menu'}).first().click();
  await page.locator('.user-menu').getByRole('menuitem', {name: 'Sign out', exact: true}).click();
  await expect(page.getByRole('button', {name: 'Sign in', exact: true})).toBeVisible();
  expect((await page.request.get('/me')).status()).toBe(401);
});

test('two independent student browsers retain their own verified identity', async ({browser, request}) => {
  const credentials = await (await request.get('/__test__/credentials')).json();
  const contexts = [await browser.newContext(), await browser.newContext()];
  try {
    for (const [index, context] of contexts.entries()) {
      const page = await context.newPage();
      await page.goto('http://127.0.0.1:8766/student');
      await page.getByLabel('Access credential').fill(credentials[index ? 'student-two' : 'student-one']);
      await page.getByRole('button', {name: 'Sign in', exact: true}).click();
      await expect(page.locator('#student-id')).toHaveText(index ? 'student-two' : 'student-one');
    }
    const one = await contexts[0].request.get('http://127.0.0.1:8766/me');
    const two = await contexts[1].request.get('http://127.0.0.1:8766/me');
    expect((await one.json()).student_id).toBe('student-one');
    expect((await two.json()).student_id).toBe('student-two');
  } finally { for (const context of contexts) await context.close(); }
});

test('expiry requires reauthentication and invalid credentials show an accessible error', async ({page, request}) => {
  await page.goto('/student');
  await page.getByLabel('Access credential').fill('invalid');
  await page.getByRole('button', {name: 'Sign in', exact: true}).click();
  await expect(page.locator('#access-error')).toContainText('invalid or has expired');
  const {credential} = await (await request.get('/__test__/short-credential')).json();
  await page.getByLabel('Access credential').fill(credential);
  await page.getByRole('button', {name: 'Sign in', exact: true}).click();
  await expect(page.locator('#student-id')).toHaveText('student-one');
  await page.waitForTimeout(3300);
  await page.reload();
  await expect(page.getByRole('button', {name: 'Sign in', exact: true})).toBeVisible();
});

test('mobile entry fits the viewport and original prototype remains reviewable', async ({page}, testInfo) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.goto('/student');
  await expect(page.getByRole('button', {name: 'Sign in', exact: true})).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path: testInfo.outputPath('student-access-mobile.png'), fullPage: true});
  await page.addInitScript(() => localStorage.setItem('fredi-intro', '1'));
  await page.goto('/__prototype/index.html');
  await expect(page).toHaveTitle('FREDi Prototype');
  await expect(page.getByRole('button', {name: 'Review panel', exact: true})).toBeVisible();
  await page.screenshot({path: testInfo.outputPath('supplied-prototype-mobile.png'), fullPage: true});
  await page.setViewportSize({width: 1440, height: 1000});
  await page.screenshot({path: testInfo.outputPath('supplied-prototype-desktop.png'), fullPage: true});
});

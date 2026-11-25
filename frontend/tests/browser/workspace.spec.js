import {test, expect} from '@playwright/test';

async function login(page, request, name = 'student-one') {
  const credentials = await (await request.get('/__test__/credentials')).json();
  await page.goto('/student');
  await page.getByLabel('Access credential').fill(credentials[name]);
  await page.getByRole('button', {name: 'Sign in', exact: true}).click();
  await expect(page.locator('#student-id')).toHaveText(name);
  await expect(page.locator('#workspace .shell')).toBeVisible();
}
async function newChat(page) {
  await page.getByRole('button', {name:'New chat',exact:true}).click();
  await expect(page.getByLabel('Message to FREDi')).toBeEnabled();
}
async function openProfile(page) {
  await page.getByRole('button', {name:'Account menu'}).first().click();
  await page.locator('.user-menu').getByRole('menuitem', {name:'Profile'}).click();
  await expect(page.getByRole('dialog', {name:'Company profile'})).toBeVisible();
}
test.beforeEach(async ({context}) => {
  await context.route('https://fonts.googleapis.com/**', route => route.abort());
  await context.route('https://fonts.gstatic.com/**', route => route.abort());
});

test('onboarding review, explicit confirmation, saved chat reload and shared scenario', async ({page, request}, testInfo) => {
  let releaseProfile;
  const holdProfile = new Promise(resolve => { releaseProfile = resolve; });
  await page.route('**/student-api/scenario', async route => {
    await holdProfile;
    await route.continue();
  });
  await login(page, request);
  try {
    await expect(page.getByLabel('Message to FREDi')).toBeEnabled();
    await expect(page.locator('.ci-open')).toHaveCount(0);
    const saved = await page.request.get('/conversations');
    expect(saved.ok()).toBe(true);
    expect((await saved.json()).items).toHaveLength(0);
  } finally {
    releaseProfile();
  }
  await page.getByLabel('Message to FREDi').fill('I am a Production Planner at UrbanCycle. Orders are delayed and I want reliable delivery.');
  await page.getByLabel('Message to FREDi').press('Enter');
  await expect(page.getByRole('button', {name:'Review profile',exact:true})).toBeVisible();
  await expect(page.locator('.side .ci-open')).toHaveCount(1);
  await page.getByRole('button', {name:'Review profile',exact:true}).click();
  const profile = page.getByRole('dialog', {name:'Company profile'});
  await expect(profile).toBeVisible();
  await expect(profile).toContainText('Production Planner');
  await page.screenshot({path:testInfo.outputPath('workspace-profile-modal-desktop.png'),fullPage:true});
  await page.getByRole('button', {name:'Confirm scenario'}).click();
  await expect(profile).toContainText('Saved');
  await profile.getByRole('button', {name:'Close',exact:true}).click();
  await expect(profile).toHaveCount(0);
  await expect(page.getByLabel('Message to FREDi')).toBeEnabled();
  const before = await page.locator('.msg').count();
  expect(before).toBe(4);
  await page.reload();
  await expect(page.locator('.msg')).toHaveCount(before);
  await newChat(page);
  await expect(page.getByText('Your company profile is saved.', {exact: false})).toBeVisible();
  await expect(page.locator('.msg')).toHaveCount(0);
  await expect(profile).toBeHidden();
  await openProfile(page);
  await expect(profile).toContainText('Saved');
  await profile.getByRole('button', {name:'Close',exact:true}).click();
  await page.getByRole('button', {name:'Account menu'}).first().click();
  await page.locator('.user-menu').getByRole('menuitem', {name:'Sign out',exact:true}).click();
  await expect(page.locator('#workspace')).toBeHidden();
});

test('latest exchange editor keeps its size and sends an edit for regeneration', async ({page, request}) => {
  await page.route('**/student-api/conversations/*', async route => {
    const response = await route.fetch();
    await route.fulfill({response, json: {...await response.json(), can_regenerate: true}});
  });
  let edit;
  let releaseEdit;
  const holdEdit = new Promise(resolve => { releaseEdit = resolve; });
  // The browser fixture uses memory storage. The SQL suite exercises regeneration itself.
  await page.route('**/orchestration/invoke', async route => {
    const payload = route.request().postDataJSON();
    if (!payload.edit_message_id) return route.continue();
    edit = payload;
    await holdEdit;
    const response = await route.fetch({postData: JSON.stringify({...payload, edit_message_id: null})});
    await route.fulfill({response});
  });
  await page.route('**/messages?*', async route => {
    const response = await route.fetch();
    const history = await response.json();
    if (edit && history.items.length === 4) history.items = history.items.slice(-2);
    await route.fulfill({response, json: history});
  });
  await login(page, request);
  await newChat(page);
  const composer = page.getByLabel('Message to FREDi');
  await composer.fill('Why is a purchase order needed?');
  await composer.press('Enter');
  const openEditor = page.getByRole('button', {name: 'Edit message'});
  await openEditor.click();
  const area = page.getByRole('textbox', {name: 'Edit message'});
  expect(await area.evaluate(node => [node.selectionStart, node.selectionEnd])).toEqual([31, 31]);
  await expect(area).toHaveCSS('background-color', 'rgba(0, 0, 0, 0)');
  const size = await area.boundingBox();
  await area.fill('A longer question about purchase orders. '.repeat(45));
  const expanded = await area.boundingBox();
  expect(expanded.height).toBe(size.height);
  expect(expanded.width).toBe(size.width);
  const cancel = page.getByRole('button', {name: 'Cancel', exact: true});
  const save = page.locator('.msg-edit-form').getByRole('button', {name: 'Send', exact: true});
  expect((await cancel.boundingBox()).y).toBe((await save.boundingBox()).y);
  await cancel.click();
  await expect(page.locator('.msg.user .message-text')).toHaveText('Why is a purchase order needed?');
  await openEditor.click();
  await area.fill('Why is a purchase order actually needed?');
  await save.click();
  try {
    await expect(page.locator('.msg.user .message-text')).toHaveText('Why is a purchase order actually needed?');
    await expect(page.locator('.msg.fredi')).toHaveCount(0);
  } finally {
    releaseEdit();
  }
  await expect(page.locator('.msg.user .message-text')).toHaveText('Why is a purchase order actually needed?');
  await expect(page.locator('.msg.fredi')).toHaveCount(1);
  expect(edit.edit_message_id).toMatch(/^[0-9a-f-]{36}$/);
  await page.reload();
  await expect(page.locator('.msg.user .message-text')).toHaveText(edit.user_message);
});

test('chat options menu shows rename and delete and closes on outside click', async ({page, request}, testInfo) => {
  await login(page, request);
  await newChat(page);
  await page.getByLabel('Message to FREDi').fill('Why is a purchase order needed?');
  await page.getByLabel('Message to FREDi').press('Enter');
  await expect(page.locator('.msg')).toHaveCount(2);
  await page.getByRole('button', {name: 'Show chat list'}).click();
  await expect(page.locator('.ci.cur .ci-open')).toHaveText('Why is a purchase order needed?');
  await page.getByRole('button', {name: /Options for/}).first().click();
  const menu = page.locator('.ci-menu');
  await expect(menu).toBeVisible();
  await expect(menu.getByRole('menuitem', {name: 'Rename'})).toBeVisible();
  await expect(menu.getByRole('menuitem', {name: 'Delete'})).toBeVisible();
  const bounds = await menu.boundingBox();
  const viewport = page.viewportSize();
  expect(bounds.x).toBeGreaterThanOrEqual(0);
  expect(bounds.y).toBeGreaterThanOrEqual(0);
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width);
  expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height);
  await page.screenshot({path: testInfo.outputPath('chat-menu.png')});
  await page.locator('.rail-logo').click();
  await expect(menu).toHaveCount(0);
});

test('sending the first message locks the draft while its conversation is created', async ({page, request}) => {
  await login(page, request);
  await newChat(page);
  let release;
  const hold = new Promise(resolve => { release = resolve; });
  let creates = 0;
  await page.route('**/conversations', async route => {
    creates++;
    await hold;
    await route.continue();
  });
  const composer = page.getByLabel('Message to FREDi');
  await composer.fill('Why is a purchase order needed?');
  await composer.press('Enter');
  try {
    await expect(composer).toBeDisabled();
  } finally {
    release();
  }
  await expect(page.locator('.msg')).toHaveCount(2);
  expect(creates).toBe(1);
});

test('lost invoke response is reconciled after reload without a second invocation', async ({page, request}) => {
  await login(page, request);
  await newChat(page);
  let calls = 0;
  let payload;
  await page.route('**/orchestration/invoke', async route => {
    calls++; payload = route.request().postDataJSON();
    await route.fetch();
    await route.abort('failed');
  });
  await page.getByLabel('Message to FREDi').fill('Why is a purchase order needed?');
  await page.getByLabel('Message to FREDi').press('Enter');
  await expect(page.getByRole('button', {name:'Check request / retry'})).toBeVisible();
  await page.route('**/turns/*', route => route.fulfill({json:{request_id:payload.request_id,status:'completed',error_code:null,user_message_id:'00000000-0000-0000-0000-000000000001',assistant_message_id:'00000000-0000-0000-0000-000000000002'}}));
  await page.reload();
  await expect(page.locator('.msg')).toHaveCount(2);
  await expect(page.getByLabel('Message to FREDi')).toBeEnabled();
  expect(calls).toBe(1);
});

test('unreceived request retries the exact request id and text', async ({page, request}) => {
  await login(page, request);
  await newChat(page);
  const bodies=[];
  await page.route('**/orchestration/invoke', async route => {
    bodies.push(route.request().postDataJSON());
    if(bodies.length===1) await route.abort('failed'); else await route.continue();
  });
  await page.route('**/turns/*', route=>route.fulfill({status:404,json:{detail:'Turn not found.'}}));
  await page.getByLabel('Message to FREDi').fill('Why do delivery checks matter?');
  await page.getByRole('button', {name:'Send',exact:true}).click();
  await page.getByRole('button', {name:'Check request / retry'}).click();
  await expect(page.locator('.msg')).toHaveCount(2);
  expect(bodies).toHaveLength(2);
  expect(bodies[0]).toEqual(bodies[1]);
});

test('mobile rail, drawer and modals fit the viewport', async ({page, request}, testInfo) => {
  await page.setViewportSize({width:390,height:844});
  await login(page, request);
  await page.getByRole('button', {name:'Show chat list'}).click();
  await expect(page.locator('.side')).toBeVisible();
  await page.locator('.side').getByRole('button', {name:'New chat',exact:true}).click();
  await expect(page.locator('.side')).toBeHidden();
  await expect(page.getByLabel('Message to FREDi')).toBeEnabled();
  await page.getByLabel('Message to FREDi').fill('My unfinished question');
  await page.getByRole('button', {name:'Account menu'}).first().click();
  await page.locator('.user-menu').getByRole('menuitem', {name:'Profile'}).click();
  await expect(page.getByRole('dialog', {name:'Company profile'})).toBeVisible();
  await page.screenshot({path:testInfo.outputPath('workspace-profile-mobile.png'),fullPage:true});
  await page.getByRole('dialog', {name:'Company profile'}).getByRole('button', {name:'Close',exact:true}).click();
  await expect(page.getByLabel('Message to FREDi')).toHaveValue('My unfinished question');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});

test('archived citations and untrusted text are rendered as text, not HTML', async ({page, request}) => {
  await login(page, request);
  await page.route('**/messages?*', route => route.fulfill({json:{items:[{
    message_id:'00000000-0000-0000-0000-000000000001', role:'assistant',
    content:'<img src=x onerror="window.injected=true"> Source-backed reply.',
    source_references:[{title:'Guide <script>unsafe</script>',page_start:2,page_end:4,kind:'chunk'}]
  }],limit:100,offset:0}}));
  await newChat(page);
  await page.getByLabel('Message to FREDi').fill('Why is a purchase order needed?');
  await page.getByLabel('Message to FREDi').press('Enter');
  await expect(page.locator('.message-text')).toContainText('<img');
  await page.getByText('Sources',{exact:true}).click();
  await expect(page.locator('.bubble details')).toContainText('Guide <script>unsafe</script> / page 2-4');
  expect(await page.locator('.bubble img, .bubble script').count()).toBe(0);
});

test('saved messages show one date divider per day and reveal their saved time on focus', async ({page, request}) => {
  await login(page, request);
  const timestamps = ['2026-11-17T10:30:00Z', '2026-11-17T10:31:00Z', '2026-11-18T10:30:00Z'];
  await page.route('**/messages?*', route => route.fulfill({json: {
    items: timestamps.map((created_at, index) => ({
      message_id: String(index), role: index === 1 ? 'assistant' : 'user',
      content: `Message ${index + 1}`, created_at,
    })),
    limit: 100, offset: 0,
  }}));
  await newChat(page);
  await page.getByLabel('Message to FREDi').fill('Why is a purchase order needed?');
  await page.getByLabel('Message to FREDi').press('Enter');
  await expect(page.locator('.msg-day')).toHaveCount(2);
  const time = page.locator('.msg-time').first();
  await expect(time).toHaveAttribute('datetime', timestamps[0]);
  await time.focus();
  await expect(time).toHaveCSS('opacity', '1');
});

test('second student cannot see another student conversation or profile', async ({browser, request}) => {
  const contexts=[await browser.newContext(),await browser.newContext()];
  try {
    const first=await contexts[0].newPage();
    const second=await contexts[1].newPage();
    await login(first,request,'student-one');
    await newChat(first);
    await first.getByLabel('Message to FREDi').fill('Why is a purchase order needed?');
    await first.getByLabel('Message to FREDi').press('Enter');
    await expect(first.locator('.msg')).toHaveCount(2);
    await first.getByRole('button',{name:'Show chat list'}).click();
    const id=await first.locator('.ci.cur .ci-open').getAttribute('data-chat');
    await login(second,request,'student-two');
    await second.getByRole('button',{name:'Show chat list'}).click();
    expect(await second.locator(`[data-chat="${id}"]`).count()).toBe(0);
    expect((await contexts[1].request.get(`http://127.0.0.1:8766/student-api/conversations/${id}`)).status()).toBe(404);
    expect((await contexts[1].request.get(`http://127.0.0.1:8766/conversations/${id}/messages`)).status()).toBe(404);
  } finally {for(const context of contexts) await context.close();}
});

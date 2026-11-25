import {defineConfig, devices} from '@playwright/test';

const python = process.env.FREDI_TEST_PYTHON ?? (process.platform === 'win32'
  ? '../backend/.venv/Scripts/python.exe' : '../backend/.venv/bin/python');

export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: false,
  workers: 1,
  reporter: [['list'], ['html', {open: 'never'}]],
  use: {baseURL: 'http://127.0.0.1:8766', trace: 'retain-on-failure'},
  projects: [{name: 'chromium', use: {...devices['Desktop Chrome']}}],
  webServer: {
    command: `"${python}" -m uvicorn tests.browser_server:app --app-dir ../backend --host 127.0.0.1 --port 8766 --no-access-log`,
    url: 'http://127.0.0.1:8766/health',
    reuseExistingServer: false,
    timeout: 30000,
  },
});

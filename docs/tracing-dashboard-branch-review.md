# Tracing Dashboard Branch Review

Review date: 2026-05-06

Scope: `tracing_dashboard` branch, especially the new local debugging dashboard under `dashboard/` and its backend trace API.

Assumption: this dashboard is intended for local debugging, not as a production observability product. The recommendations below optimize for fast feedback, reliable live updates, clean packaging, and avoiding accidental data exposure when a local Hayhooks server is reachable from outside the developer's machine.

## Findings

### Blocker: clean checkout cannot build the dashboard

`dashboard/src/components/Header.tsx` imports `@/lib/utils` and `@/assets/haystack-icon.png`, and `dashboard/index.html` also references the icon. Those files exist locally as ignored files, but they are not tracked by git and are not included in the package data.

Impact: `npm run build` can pass on the current machine while failing in CI, a fresh clone, or a wheel/source distribution.

Recommended fix:

- Track `dashboard/src/lib/utils.ts` and the required assets, or remove those imports.
- Add dashboard CI checks from a clean checkout: `npm ci`, `npm run test`, and `npm run build`.

### High: cursor-based polling can permanently skip traces

`src/hayhooks/server/routers/dashboard.py` returns `X-Hayhooks-Trace-Cursor` as the current buffer head before applying `limit`, while `dashboard/src/hooks/useTraces.ts` advances `afterSeq` to that value.

Impact: if more than `fetchLimit` traces are created between polls, the API returns only the first page but the frontend advances past the rest. Those traces are never fetched. This is a reliability issue even for a local dashboard because bursty debugging sessions are common.

Recommended fix:

- Return the cursor for the last trace included in the response, not the buffer head.
- Alternatively return `{ traces, next_after_seq, has_more }` in the JSON payload and keep fetching until `has_more` is false.
- Add a frontend/backend test where more than `fetchLimit` trace updates arrive between polls.

### High: raw pipeline payload values are exposed in trace tags

`src/hayhooks/server/utils/deploy_utils.py` records full `key=value` request payload values under `hayhooks.payload.values`, and `src/hayhooks/server/routers/dashboard.py` returns those tags from `/dashboard/api/traces`.

For local debugging, seeing payloads can be useful. The risk is that Hayhooks can bind to non-localhost hosts, screenshots/logs can be shared, and the dashboard API is included even when the static dashboard UI is disabled.

Impact: prompts, documents, API keys, file metadata, or PII can appear in the dashboard API and UI.

Recommended fix:

- Default to payload keys/types/sizes rather than raw values.
- Add an explicit opt-in setting for raw payload values, for example `HAYHOOKS_DASHBOARD_TRACE_INCLUDE_PAYLOAD_VALUES=false`.
- Redact common sensitive keys even in raw mode: `api_key`, `token`, `authorization`, `password`, `secret`, `key`.
- Consider only including dashboard routes when dashboard capture or UI is enabled.

### High: custom dashboard path breaks API routing

The static UI is mounted at `settings.dashboard_path`, but the API routes are hardcoded under `/dashboard/api`.

Impact: `HAYHOOKS_DASHBOARD_PATH=/observability` serves the frontend from `/observability`, but the frontend resolves API calls to `/observability/api`, which does not exist.

Recommended fix:

- Mount the API router under the configured dashboard path.
- Or expose a config value that tells the frontend the fixed API base.
- Add a test for a non-default dashboard path.

### Important: live trace buffer can become slow or memory-heavy

`src/hayhooks/server/utils/live_trace_buffer.py` limits the number of traces, but not the number of spans per trace or the size of stored tag values. It also normalizes traces while holding the writer lock, and span tree construction scans all spans for each node.

Impact: one large trace can make polling slow, block trace writers, or consume excessive memory. This directly affects local debugging responsiveness.

Recommended fix:

- Cap spans per trace and truncate tag values before storing them.
- Build child indexes once per trace normalization instead of repeatedly scanning all spans.
- Copy trace state under lock, then normalize outside the lock.
- Add a stress test with many spans and large tag values.

### Important: runtime dashboard build can fail with `NODE_ENV=production`

`src/hayhooks/cli/base.py` runs `npm ci` before `npm run build`. If `NODE_ENV=production` is set, npm omits dev dependencies, but the build requires `typescript`, `vite`, and `@vitejs/plugin-react`.

Impact: `hayhooks run --with-tracing-dashboard` can fail in developer or container environments that set `NODE_ENV=production`.

Recommended fix:

- Use `npm ci --include=dev` for runtime dashboard builds.
- Add a CLI test that sets `NODE_ENV=production` and verifies the dashboard build path still installs build tooling.

### Medium: Vite dev mode cannot reliably read the cursor header cross-origin

`dashboard/src/api.ts` points local Vite dev ports to `http://localhost:1416/dashboard/api`. Browser CORS hides custom response headers unless they are exposed, and `settings.cors_expose_headers` defaults to `[]`.

Impact: during frontend development, the app may not see `X-Hayhooks-Trace-Cursor` and can fall back to inefficient full fetches.

Recommended fix:

- Expose `X-Hayhooks-Trace-Cursor` by default when dashboard APIs are enabled.
- Or move the cursor into the JSON payload.

### Medium: documented env-var build command uses the wrong path

`docs/reference/tracing.md` shows:

```bash
cd dashboard
npm install
npm run build
export HAYHOOKS_DASHBOARD_DIST_DIR=./dashboard/dist
hayhooks run
```

After `cd dashboard`, `./dashboard/dist` resolves to `dashboard/dashboard/dist`.

Recommended fix:

- Use `export HAYHOOKS_DASHBOARD_DIST_DIR="$(pwd)/dist"` after `cd dashboard`.
- Or run `hayhooks run` from the repo root with `HAYHOOKS_DASHBOARD_DIST_DIR=dashboard/dist`.

### Medium: demo script can report failures but still exit successfully

`scripts/demo_dashboard_traces.sh` has a `check()` helper that prints failed checks but does not return non-zero.

Impact: the demo can finish with a success exit code even when deploys or runs failed, hiding broken trace generation.

Recommended fix:

- Make `check()` return `1` on mismatch.
- Add a lightweight smoke test or mocked failure check for the script.

### Medium: custom prebuilt dashboard dist override is ambiguous

`HAYHOOKS_DASHBOARD_DIST_DIR` is documented as a way to provide prebuilt assets, but `hayhooks run --with-tracing-dashboard` attempts to find and build dashboard source first, then overwrites `settings.dashboard_dist_dir` with the built dist when source is available.

Impact: a caller that intentionally points to custom prebuilt assets may not get those assets in the common CLI path.

Recommended fix:

- Define precedence explicitly: either custom dist always wins, or `--with-tracing-dashboard` always rebuilds packaged source.
- Add a CLI test for `HAYHOOKS_DASHBOARD_DIST_DIR` plus `--with-tracing-dashboard`.

## Suggested Fix Order

1. Track the missing frontend files/assets and add clean dashboard build/test CI.
2. Fix the polling cursor contract to prevent skipped traces.
3. Add payload redaction or an explicit raw-payload opt-in.
4. Make dashboard API routing respect `HAYHOOKS_DASHBOARD_PATH`.
5. Add span/tag caps and normalize traces outside the writer lock.
6. Harden runtime build, docs, CORS, and demo script behavior.

## Recommended Verification

Run these after fixes:

```bash
cd dashboard
npm ci
npm run test
npm run build
```

```bash
pytest tests/test_dashboard_mount.py tests/test_it_dashboard.py tests/test_live_trace_buffer.py tests/test_tracing.py
```

Add targeted tests for:

- clean dashboard build from tracked files only;
- cursor pagination with more updates than `fetchLimit`;
- non-default `HAYHOOKS_DASHBOARD_PATH`;
- payload redaction defaults and explicit raw-payload opt-in;
- large trace/span/tag stress behavior;
- `NODE_ENV=production hayhooks run --with-tracing-dashboard` build behavior.

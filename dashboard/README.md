# Hayhooks Tracing Dashboard Frontend

This directory contains the React + TypeScript frontend for the Hayhooks tracing dashboard.

When you run:

```bash
hayhooks run --with-tracing-dashboard
```

Hayhooks serves this app at `/dashboard` and uses it to visualize live trace activity.
The dashboard API reads traces from Hayhooks' in-process live trace buffer.

## What This Dashboard Shows

- Live trace feed with freshness highlights
- Ongoing vs completed traces
- Failure highlighting
- Entrypoint filtering
- Span waterfall details and trace tags
- Basic trace stats (count, average duration, last trace)

## Prerequisites

- Node.js + npm (required for local frontend development/build)
- A running Hayhooks backend (default: `http://localhost:1416`)

## Local Development

1. Start Hayhooks in another terminal:

   ```bash
   hayhooks run --with-tracing-dashboard
   ```

2. Install frontend dependencies:

   ```bash
   cd dashboard
   npm install
   ```

3. Start Vite dev server:

   ```bash
   npm run dev
   ```

4. Open the local frontend (usually `http://localhost:5173`).

By default, local dev assumes the backend API at `http://localhost:1416/dashboard/api`.
If your backend runs elsewhere, set `VITE_HAYHOOKS_DASHBOARD_API_BASE`:

```bash
VITE_HAYHOOKS_DASHBOARD_API_BASE="http://localhost:1416/dashboard/api" npm run dev
```

## Useful Commands

Run these from `dashboard/`:

```bash
# Start dev server (HMR)
npm run dev

# Run tests once
npm run test

# Run tests in watch mode
npm run test:watch

# Lint
npm run lint

# Build production assets to dist/
npm run build

# Preview production build
npm run preview
```

Common pre-PR check:

```bash
npm run lint && npm run test && npm run build
```

## Production/Runtime Notes

- `hayhooks run --with-tracing-dashboard` builds and serves static assets from `dashboard/dist`.
- You can override where Hayhooks reads built assets with `HAYHOOKS_DASHBOARD_DIST_DIR`.
- The dashboard trace API (`/dashboard/api/traces`) is local-buffer only (no direct Jaeger/SigNoz fetch mode).

## Related Documentation

- [`docs/reference/tracing.md`](../docs/reference/tracing.md)
- [`docs/reference/environment-variables.md`](../docs/reference/environment-variables.md)

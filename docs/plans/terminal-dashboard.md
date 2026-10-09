# Hayhooks terminal dashboard rebuild plan

Build an opt-in, browsable, read-only terminal dashboard with Textual. It should help Hayhooks developers and operators inspect deployments, distinguish Agents from Pipelines, and investigate durable executions without leaving the terminal. Navigation and presentation should feel polished while runtime state remains accurate and clearly scoped.

This carries forward the decisions agreed in the dashboard conversation: Textual, an optional `hayhooks[tui]` extra, a separate `hayhooks dashboard` command, keyboard and mouse browsing, deployment and execution views, and progressive detail disclosure. A passive Rich display is no longer the target.

## Historical baseline

Verified on 7 October 2026 against `ec83936c`. This baseline is historical;
the 8 October migration below supersedes the separate browser tracing dashboard:

- The existing React tracing dashboard lives under `dashboard/`. It is a separate feature and remains available.
- There is no Textual dependency, `tui` extra, or terminal dashboard command in the current checkout. Rebuilding does not assume that earlier implementation files can be recovered.
- [`GET /status`](../../src/hayhooks/server/routers/status.py) returns deployment names, durable mode, and cached durable health. It can return HTTP 200 with `status: Degraded`.
- [`DurableDeployment.health()`](../../src/hayhooks/durable/runtime.py) exposes local worker state and bounded store counts.
- [`create_durable_router()`](../../src/hayhooks/durable/fastapi.py) already provides inspect, cancel, resume, and stream routes for known execution IDs. There is no execution collection route.
- [`ExecutionStore`](../../src/hayhooks/durable/store.py) has control reads and operational counts, but no listing method. Both Memory and Redis stores need execution discovery support.
- [`Registry metadata`](../../src/hayhooks/server/pipelines/models.py) and [`wrapper method flags`](../../src/hayhooks/server/utils/base_pipeline_wrapper.py) provide descriptions and API capabilities, but ordinary wrapper metadata does not identify agent type reliably.

## Scope and opt in

The first complete release includes deployment browsing, durable health, paginated execution browsing, search and filters, execution details, and resilient live refresh. The client performs GET requests only. It does not deploy, undeploy, submit, cancel, resume, clear traces, or modify Redis execution state directly.

Installation and commands below are proposed interfaces:

```bash
pip install 'hayhooks[tui]'

# Connect to the default HTTP server.
hayhooks dashboard

# Connect through a reverse proxy with a path prefix.
hayhooks dashboard --url https://example.com/hayhooks

# Start with a deployment selected.
hayhooks dashboard --pipeline research_agent

# Export a single snapshot without starting the interactive UI.
hayhooks dashboard --once --json
```

Keep Textual imports inside the interactive command path. Ordinary CLI commands and server startup must work without the extra. Verify the chosen Textual release against Hayhooks' supported Python versions and package constraints during implementation. Do not add Node.js, a second HTTP library, or a custom widget framework.

Add a separate server setting, provisionally `HAYHOOKS_MONITORING_ENABLED` and `hayhooks run --with-monitoring`, for execution indexing and enumeration. It defaults to false. Deployment overview works without it; the Executions view explains how to enable monitoring when the server does not advertise enumeration support. This flag does not start a terminal UI or enable the browser tracing dashboard.

## Browsing and presentation

Use two main views, **Deployments** and **Executions**, with a compact connection header and a contextual shortcut footer. Details sit beside the selected row on wide terminals and open as a separate screen on narrow terminals. The data below is illustrative:

```text
Hayhooks   localhost:1416   Healthy   Updated 2s ago
Deployments | Executions

Name             Type       Execution       State
research_agent   Agent      Durable         Healthy
rag_query        Pipeline   Request-bound   Loaded
custom_chat      Unknown    Request-bound   Loaded

Selected: research_agent
Revision v3   Local active 2   Worker slots 4/4
Store nonterminal 12   Revision runnable 5   Accepting yes

Tab Focus   Enter Details   / Search   Space Pause   ? Help
```

| View | List information | Selected item details |
| --- | --- | --- |
| Deployments | Name, type, execution mode, state | Description, API capabilities, revision, durable admission, local workers and draining work, store counts and errors |
| Executions | ID, deployment, type, revision, status, attempt, last update | Full ID, timestamps, progress, waiting information, cancellation pending, bounded error details, retry timing where available |

Initially browse executions for one selected durable deployment. Keep its selection when switching views. Users can select a different deployment, including one chosen through `--pipeline`; do not introduce a global cross-deployment merge service for the first release.

Support deployment/type filters and execution status filters. Search deployment names and execution IDs. Execution queries must search the indexed collection, not silently search only the visible page. If a particular filter is local to loaded rows, label that limit explicitly. Offer explicit next/previous page navigation and restore filters when returning from details.

| Input | Behavior |
| --- | --- |
| Arrow keys | Move within a list or scroll focused details |
| Tab and Shift+Tab | Move focus between controls and panes |
| Enter | Open the selected item's details |
| Esc | Close help/details or leave search and restore prior focus |
| `/` | Focus search |
| Space | Pause/resume refresh outside text inputs |
| `r` | Request a refresh outside text inputs |
| `?` | Show shortcut help outside text inputs |
| `q` or Ctrl+C | Exit and restore the terminal; typing `q` in search remains text |
| Mouse click and wheel | Select rows and scroll without requiring the mouse |

Start with Textual's built-in [DataTable](https://textual.textualize.io/widgets/data_table/), tabs, inputs, selects, scrolling containers, and footer. Use stable row keys so updates preserve selection. Updating health or adding an execution must not reset search focus, scroll position, or the item being read.

Make the dashboard attractive through clear hierarchy, aligned columns, deliberate spacing, readable detail grouping, and restrained use of Hayhooks' existing [palette](../../src/hayhooks/colors.py). Use color for selection and state, always paired with text. Provide readable light and dark themes. Avoid flashing rows, decorative animation, oversized metric cards, and excessive borders. Pause and stale indicators should be easy to spot without dominating the screen.

## Honest type and health information

Keep these fields independent:

- **Type:** agent, pipeline, or unknown. Durable deployments reuse their existing `ExecutionKind`. Recognized ordinary wrapper targets can be classified conservatively; custom wrappers may supply an explicit validated metadata hint. Do not recursively inspect arbitrary object graphs or infer agent type from names, chat endpoints, or A2A exposure.
- **Execution mode:** durable or request-bound. A server in durable mode can still contain ordinary wrappers; derive this per deployment.
- **Capabilities:** run API, chat completion, Responses, and file upload, based on existing wrapper flags. MCP/A2A eligibility is distinct from proving that a separate protocol server is running.
- **Deployment state:** ordinary wrappers are `Loaded`; durable health can be `Healthy` or `Degraded`. Loaded does not certify external LLM, database, or tool health.

Worker slots, active executions, maintenance, and draining work describe the responding process. Redis-backed counts describe the shared store and sometimes only the served revision; Memory counts are process-local. Label these scopes. A load-balanced URL does not provide a fleet-wide worker total.

Do not rename `revision_runnable` to running or ready: it includes queued delayed retries. Do not derive exact waiting, failed, or completed totals from one page of results. Retain the engine's statuses: queued, running, waiting, completed, failed, and canceled. Delayed retry and cancellation pending are annotations, not new lifecycle states. Distinguish elapsed age from actual runtime duration when start/finish timestamps are unavailable.

## API and store changes

Extend `/status` additively with typed deployment summaries and a small capability indicator for execution enumeration. Preserve the existing `pipelines`, `durable_mode`, and `durable` fields and the current health cache/timeout behavior. Build summaries from the app's registry rather than creating a parallel deployment registry.

Add `GET /{pipeline_name}/executions` alongside the existing durable routes only when monitoring is enabled. Start with a default page size of 50 and a maximum of 100. Accept a validated cursor and supported status/revision/ID filters. Return `executions`, `next_cursor`, and `has_more`; do not promise a costly exact filtered total.

Execution summaries contain control metadata: execution ID, deployment, stored kind and revision, status, attempt, creation/update times, cancellation pending, and retry availability where relevant. Never serialize raw `ExecutionControl`; it also contains owner, lease, fence, and idempotency material. Listing must not fetch inputs, checkpoints, results, progress payloads, or chunk histories.

Details reuse the existing authorized inspect endpoint. It includes public result data; the client should not render or retain results automatically. Waiting/progress/error content may also contain application data, so keep it behind item selection, bound presentation, and avoid logging response bodies. Treat all server text as untrusted literal text and neutralize terminal control sequences.

### Execution index

Extend the existing store contract with a bounded control-list operation. Implement equivalent semantics in Memory and Redis. Keep index entries in the deployment's existing Redis hash slot, use stable creation ordering with an execution ID tie-breaker, and return cursors that do not depend on mutable update times.

Index new submissions atomically with the existing submission write, including correct behavior for idempotent replays. Retain discoverability for all indexed live executions, especially old waiting executions, and retain terminal entries only while their controls remain inspectable. TTL-aware, incremental cleanup must prevent expired references from accumulating indefinitely. Bound candidate reads and cleanup work even when a page contains expired or corrupt controls or filters are selective; advance the cursor honestly if a bounded page is sparse.

Do not use Redis `KEYS`, scan the whole keyspace on refresh, or infer the collection from runnable/lease indexes: those omit waiting and terminal executions. Do not trim old live executions merely to cap recent history. Monitoring reads must not change execution status, capacity, leases, or retry budgets.

All replicas writing the monitored deployment must enable indexing. Earlier submissions remain inspectable by ID but are not automatically backfilled. Document the start of monitoring coverage and keep startup free of an unbounded migration. Test index key validation before extending Lua writes because script failures do not roll back prior writes.

### Authorization

Reuse each wrapper's durable owner dependency for owner-scoped collection access. Filter before returning IDs or summaries; cursors and pagination metadata must not expose another owner's records. Keep the existing inspect/resume/cancel authorization behavior intact.

An unscoped deployment currently uses possession of a random execution ID as access. Enumeration changes that boundary. Require an explicit authenticated monitoring/operator dependency for unscoped collection access and fail closed when it is absent; do not equate the monitoring enable flag with authentication. Reuse the host's authentication middleware/dependencies rather than adding a new account system. Broader access across owners requires an explicit operator policy; it must never silently bypass owner scoping.

The client accepts existing bearer/API-key credentials through environment/configuration, keeps TLS verification enabled, and does not print credentials. Document one supported authentication integration before release. The exact host dependency wiring must be settled before implementing the collection route.

## Refresh and failure behavior

Poll every two seconds initially, with a configurable positive interval and explicit connection/read timeouts. Use Textual's [thread workers](https://textual.textualize.io/guide/workers/) with Requests so network I/O does not freeze navigation. Keep at most one refresh cycle in flight, fetch only the active view/page and selected detail, and discard responses from obsolete selections or filters. Applying a thread result must happen on the UI thread.

The existing CLI `make_request()` prints errors and aborts the command, so it cannot be called unchanged inside the TUI. Reuse URL/TLS conventions and add the smallest non-printing request path that returns data or an error for display. Preserve reverse-proxy path prefixes when joining URLs. Avoid changing unrelated CLI error behavior.

On timeouts or disconnections, retain the last good snapshot, show its age, mark it stale, and retry with bounded backoff. Distinguish authentication failure, unsupported API/disabled monitoring, server degradation, no executions, no filter matches, expired execution, and an unavailable store. Never present an unavailable service as an empty collection.

Pausing freezes visible data and stops automatic polling; resuming refreshes it. Refresh responses cannot overwrite a paused view. Leaving the app closes client resources and restores the terminal. `--once --json` prints only structured JSON on stdout, uses stderr for errors, and reports unavailable snapshots with a nonzero exit code.

## Implementation milestones

### 1 Deployment browsing

Add the optional extra and lazy CLI command, typed deployment metadata, connection handling, the Textual shell, browsable deployment table/details, responsive layout, refresh/pause, and stale states. Include missing-extra guidance and the snapshot command. Ordinary and durable deployments must both work against a single HTTP server.

This is an independently useful milestone, but it does not complete execution browsing. Before starting new work, inspect the checkout for matching functionality and reuse any recovered files.

### 2 Execution discovery and inspection

Implement and test the authorization contract, monitoring setting, store listing/index, collection route, bounded pagination, and retention cleanup. Then connect the execution view, server filters, search, details, revision visibility, and disabled-monitoring guidance. Read existing retained executions from their pinned revision; never substitute the deployment's current kind/revision for stored values.

### 3 Visual and behavioral qualification

Polish focus, density, themes, narrow screens, and calm updates. Run headless keyboard/mouse/resize checks, inspect rendered screenshots, and launch the app in a real terminal against representative Hayhooks deployments. Validate wheel installation with and without the extra, document authentication and index coverage, and complete the focused CI checks.

The first complete release requires all three milestones. Do not claim a cool dashboard solely because it launches or a screenshot looks good.

## Acceptance checks

Use the repository's existing pytest infrastructure and Textual's [headless testing and Pilot](https://textual.textualize.io/guide/testing/). Keep tests focused on behavior and failure boundaries.

| Area | Required evidence |
| --- | --- |
| Packaging | Core CLI/server work without Textual; missing extra has clear guidance; built wheel includes required styles/assets; supported Python environments install successfully |
| Browsing | Keyboard-only path from deployment to execution details and back; mouse selection; search/filter/page navigation; text entry does not trigger global shortcuts |
| Updates | Selection, focus, filters, and scrolling survive refresh; slow/obsolete replies cannot replace a newer selection; pause/resume behaves as described |
| Layout | Review Deployments, Executions, details, empty, stale, and error states at 80x24, 120x36, and 160x48; IDs and important errors remain reachable; resize restores the selected item |
| Appearance | Light/dark contrast, readable focus and status text, aligned columns, no clipping of controls, and calm updates; save representative screenshots for review |
| Truthfulness | Degraded body with HTTP 200; unknown wrapper types; loaded versus healthy; process/store/revision scopes; runnable is not running; unavailable is not empty |
| Store | Memory/Redis contract parity; submissions and idempotent replay; timestamp ties and concurrent insertion; waiting and old revisions; terminal TTL and bounded cleanup; corrupt controls and malformed cursors |
| Authorization | Cross-owner IDs never leak; anonymous unscoped enumeration fails; authorized operator access is explicit; detail access stays scoped; secrets/markup/control sequences are not emitted unsafely |
| Integration | Real request-bound Pipeline, Agent, and custom wrapper; Redis durable executions through queued/running/waiting/completed/failed/canceled; restart/reconnect and expiry |
| Read only | Recorded client traffic contains GET only; refresh leaves lifecycle, admission capacity, checkpoints, leases, and retry counters unchanged |

After those checks pass, review the actual terminal interface for discoverability, information density, and whether an execution can be investigated without losing the operator's place. Fix concrete findings before calling the dashboard finished.

## File locations and recovery record

Start with `src/hayhooks/cli/dashboard.py` for command wiring, a small `src/hayhooks/tui/` package for the app and styles, and minimal edits to the existing status, metadata, durable router, store, settings, and server setup files. Reuse the current test modules for backend contracts and add a focused TUI test module. Do not create a generic SDK, registry, state manager, or rendering abstraction for this feature.

Update `docs/features/cli-commands.md`, the durable operations/API/environment references, and installation guidance when each behavior exists. Keep this plan outside generated documentation navigation until it becomes user documentation.

Record completed milestones here with commit hashes and check results, and retain visual review artifacts at a documented location. Make small reviewable commits at milestone boundaries. A later implementation session can resume from the latest recorded commit instead of reconstructing progress from chat history. This plan does not itself recover lost edits or implement the dashboard.

## Completed implementation — 7 October 2026

All three milestones are complete. The implementation is additive, monitoring
defaults to disabled, and the terminal client sends GET requests only.

| Milestone | Commits | Delivered behavior |
| --- | --- | --- |
| Deployment browsing | `ec9bea41`, `be71bd2d` | Typed status summaries, optional Textual extra, lazy CLI command, JSON snapshot, responsive deployment browsing, pause and stale handling |
| Execution discovery and inspection | `ec9bea41`, `be71bd2d` | Owner-scoped Memory/Redis indexes, explicit operator authorization, bounded collection API, server search/filters, cursor pagination and selected execution details |
| Visual and behavioral qualification | `c5798fd5`, `3e99fd0a`, `d5e58f22`, `d564dd06` | Shutdown race guard, theme recoloring, invalid health validation, failure checks, reviewed screenshots and complete installation/API/authentication documentation |

The latest implementation/documentation commit is **`d564dd06`**. The commit
containing this recovery record follows it; no implementation changes are pending.

### Validation evidence

- Full unit run: `.venv/bin/pytest -q -m 'not integration'` — **1,330 passed,
  4 skipped, 129 deselected, 3 expected failures**. This environment used Haystack
  3.3.0 and Textual 8.2.8.
- Default Hatch environment with Haystack 2.31.0:
  `hatch run test:unit tests/test_terminal_dashboard.py tests/test_durable_monitoring.py tests/test_it_status.py tests/test_cli.py`
  — **56 passed, 8 integration cases deselected**.
- Real Redis suite: `tests/test_it_durable_redis.py`,
  `tests/test_it_durable_redis_process.py`, `tests/test_it_durable_mode.py`, and
  `tests/test_durable_monitoring.py` with `HAYHOOKS_TEST_REDIS_URL` pointing at a
  dedicated local Redis — **129 passed**. Includes lifecycle, restart/recovery,
  index parity, replay, owner isolation, sparse pages, corruption and expiry.
- The focused TUI module has **21 passing cases**, including keyboard/mouse
  browsing, resize, filtering/paging, selection/focus/scroll preservation,
  obsolete replies, single-worker polling, pause, reconnect, expired selected
  details, theme contrast, shutdown and safe connection failures.
- Built-wheel installations succeeded on **Python 3.10, 3.11, 3.12, 3.13 and
  3.14**, both core-only and with `[tui]`. Core CLI/server startup, missing-extra
  guidance and JSON snapshots worked without Textual. Installed extras included
  styles and passed an 80x24 headless app/theme smoke check.
- A real terminal connected to one HTTP server containing a request-bound
  Pipeline, a real Haystack Agent using a local mock generator, a custom wrapper,
  and a Redis durable deployment. Execution browsing, details and pause worked;
  quitting returned exit code 0 and restored the terminal. Fixture setup submitted
  executions separately; recorded dashboard requests were GET only. Automated
  runtime checks cover queued/running/waiting/completed/failed/canceled and verify
  that enumeration leaves lifecycle, capacity and private execution data unchanged.
- `ruff check src/hayhooks`, `ruff format --check src/hayhooks`, the focused type
  check of changed dashboard/store/router modules, and `git diff --check` passed.
  Documentation built with `hatch run docs:build`.

Full `ty check src/hayhooks` still reports **10 existing diagnostics**, reproduced
against a clean archive of baseline `ec83936c`; they concern existing engine,
registry and A2A types. The final unit run used protobuf 6.33.6 because a2a-sdk
1.1.2 accesses `FieldDescriptor.label`, removed in protobuf 7.36.2. No unrelated
dependency pin or A2A change was introduced. Documentation build warnings concern
this plan's pre-existing links to source files outside the documentation tree;
the plan remains outside navigation.

### Visual review artifacts

Eighteen SVG captures are retained in `docs/assets/terminal-dashboard/`.
Each state was reviewed at 80x24, 120x36 and 160x48: deployment overview, execution
list, selected details, light theme, empty filter results, and a stale connection
error retaining the last snapshot. The review led to compact detail rows,
contextual focus hints, preserving details outside the current page, and
recoloring unchanged rows when switching themes.

| State | 80x24 | 120x36 | 160x48 |
| --- | --- | --- | --- |
| Deployments | [Capture](../assets/terminal-dashboard/deployments-80x24.svg) | [Capture](../assets/terminal-dashboard/deployments-120x36.svg) | [Capture](../assets/terminal-dashboard/deployments-160x48.svg) |
| Executions | [Capture](../assets/terminal-dashboard/executions-80x24.svg) | [Capture](../assets/terminal-dashboard/executions-120x36.svg) | [Capture](../assets/terminal-dashboard/executions-160x48.svg) |
| Details | [Capture](../assets/terminal-dashboard/details-80x24.svg) | [Capture](../assets/terminal-dashboard/details-120x36.svg) | [Capture](../assets/terminal-dashboard/details-160x48.svg) |
| Light theme | [Capture](../assets/terminal-dashboard/light-80x24.svg) | [Capture](../assets/terminal-dashboard/light-120x36.svg) | [Capture](../assets/terminal-dashboard/light-160x48.svg) |
| Empty | [Capture](../assets/terminal-dashboard/empty-80x24.svg) | [Capture](../assets/terminal-dashboard/empty-120x36.svg) | [Capture](../assets/terminal-dashboard/empty-160x48.svg) |
| Stale/error | [Capture](../assets/terminal-dashboard/stale-80x24.svg) | [Capture](../assets/terminal-dashboard/stale-120x36.svg) | [Capture](../assets/terminal-dashboard/stale-160x48.svg) |

These are synthetic qualification fixtures, not production data. Temporary
qualification servers and browser tabs were stopped after review.

## Visual refinement — 7 October 2026

Commit **`cfbfc5a9`** refines the terminal interface using the requested
high-end visual design guidance within Textual's native widgets. A quieter
charcoal blue / warm light palette, inset masthead, compact controls, subdued
selection and grouped inspector establish clearer hierarchy. Short terminals
condense spacing; full execution IDs and timestamps remain in details while
table IDs and revisions show explicit ellipses. Theme switching is visible in
the footer. No project dependencies were added.

- Focused dashboard, CLI and status checks: **41 passed**, including **24 TUI
  cases**. Compact selector navigation and full-ID inspection pass at 80x24,
  120x36 and 160x48.
- TUI Ruff, formatting, type checks and `git diff --check` passed. Documentation
  built successfully with the same pre-existing source-link warnings.
- The final wheel passed core-only CLI/server, missing-extra, JSON snapshot,
  bundled styles and headless theme checks on **Python 3.10 and 3.14**.
- A real terminal exercised execution browsing, details, theme switching and
  pause. Exit code was 0, with the alternate terminal screen restored.
- The eighteen state captures above were refreshed and reviewed. Six additional
  realistic synthetic overview/execution captures cover all three sizes. Raster
  previews are retained for convenient review:
  [Deployment overview](../assets/terminal-dashboard/overview-120x36.png) and
  [Execution inspector](../assets/terminal-dashboard/browse-160x48.png).

The earlier full backend/Redis qualification remains recorded above; this
follow-up changes terminal presentation and its focused checks.

## Deferred features

Defer execution controls, live token streams, charts, cross-server/fleet aggregation, cross-deployment execution merging, persistent dashboard preferences, and automatic historical backfill. Add selected-execution SSE only if polling proves insufficient; reuse the existing endpoint and cursor behavior when that need arises. Request-bound execution history also needs its own observation source, so do not invent it from durable records or present process-local traces as a complete history.


## Terminal tracing migration — 8 October 2026

Commits **`dd6211e7`** (browser removal and authenticated trace API) and
**`fccf4b44`** (terminal views, documentation and reviewed artifacts) complete
this migration. The recovery record follows those implementation commits.

The requested migration replaces the React tracing dashboard with a third native
**Traces** tab in `hayhooks dashboard`. The browser sources, npm build/test job,
static asset packaging/mounting, web-only settings and old CLI flags are removed.
The existing bounded trace buffer, Haystack component capture, durable attempt
traces and optional OTLP export remain in use. No project dependencies were added.

The terminal now offers entrypoint/kind/state filters, trace/tag search,
newest/slowest sorting, failure counts, finished-duration averages, streaming
indicators, tags/errors and a native expandable span tree with timing offsets,
waterfall bars and slow-component highlighting. Keyboard navigation, light/dark
themes, compact layouts, pause, local clearing and stale snapshots work across
80x24, 120x36 and 160x48. Expanded branches, selected spans and scroll positions
survive trace updates; inspecting an evicted/filtered trace explicitly labels
its retained snapshot. Returning from help restores inspector focus.

Enable the API with `hayhooks run --with-tracing` and open it using
`hayhooks dashboard --traces`. See [operator setup and migration](../reference/tracing.md).
All `/traces` routes require a private bearer token or a host-provided verified
operator dependency. These traces can include every user served by one process;
owner-scoped durable authorization alone does not grant access. The terminal
performs GET requests only, fetching at most 100 recent traces per refresh on
its existing background worker. Clearing hides traces locally. This bounded
process sample is distinct from durable execution history or persistent OTLP data.

### Qualification

- Final unit run: **1,346 passed, 4 skipped, 129 deselected, 3 expected failures**.
  Command: `PYTHONPATH=/tmp/hayhooks-tracing-protobuf6:.:src .venv/bin/pytest -q -m 'not integration'`.
  The temporary target supplies protobuf 6.33.6 for the previously documented
  A2A/protobuf 7 incompatibility. Project dependency constraints and the existing
  environment were not changed. This run used Haystack 3.3.0 and Textual 8.2.8.
- The **38 terminal cases** passed, including 14 trace cases covering all three
  sizes, search/filter/sort, navigation, live root semantics, malformed metadata,
  local clearing, pause, eviction, themes, help/focus and JSON snapshots.
  Trace API checks cover disabled mounting, removed browser paths, default denial,
  bearer credentials, operator overrides and every protected route.
- Final wheel smoke checks on **Python 3.10 and 3.14** cover core-only CLI/server
  startup, missing-extra guidance, JSON trace snapshots without Textual, bundled
  styles and an installed native trace inspector/tree/theme. The wheel contains
  no React dashboard files.
- A real local HTTP server captured successful and failed real Haystack Pipeline
  runs with component spans. Unauthorized access returned 401. A real terminal
  browsed the span tree, changed theme, cleared locally and exited with code 0;
  the alternate screen was restored. A six-trace fixture had a median snapshot
  round trip of **1.74 ms** across ten local requests; this is a small fixture
  observation, not a production performance claim.
- Source Ruff/format checks, focused type checks of TUI/CLI/settings/trace router,
  `git diff --check` and the documentation build passed. The previously recorded
  full-source type diagnostics and documentation source-link warnings remain.
  Redis store/runtime behavior is unchanged; the earlier Redis qualification
  above remains the recorded evidence, rather than a new Redis run.

### Trace review artifacts

Twenty-one additional SVG captures in `docs/assets/terminal-dashboard/` cover
traces, light traces, spans, light spans, errors, empty filters and stale snapshots
at all three sizes. They use synthetic fixtures, not production data.

| State | 80x24 | 120x36 | 160x48 |
| --- | --- | --- | --- |
| Traces | [Capture](../assets/terminal-dashboard/traces-80x24.svg) | [Capture](../assets/terminal-dashboard/traces-120x36.svg) | [Capture](../assets/terminal-dashboard/traces-160x48.svg) |
| Light traces | [Capture](../assets/terminal-dashboard/traces-light-80x24.svg) | [Capture](../assets/terminal-dashboard/traces-light-120x36.svg) | [Capture](../assets/terminal-dashboard/traces-light-160x48.svg) |
| Span inspector | [Capture](../assets/terminal-dashboard/spans-80x24.svg) | [Capture](../assets/terminal-dashboard/spans-120x36.svg) | [Capture](../assets/terminal-dashboard/spans-160x48.svg) |
| Light spans | [Capture](../assets/terminal-dashboard/spans-light-80x24.svg) | [Capture](../assets/terminal-dashboard/spans-light-120x36.svg) | [Capture](../assets/terminal-dashboard/spans-light-160x48.svg) |
| Errors | [Capture](../assets/terminal-dashboard/traces-error-80x24.svg) | [Capture](../assets/terminal-dashboard/traces-error-120x36.svg) | [Capture](../assets/terminal-dashboard/traces-error-160x48.svg) |
| Empty | [Capture](../assets/terminal-dashboard/traces-empty-80x24.svg) | [Capture](../assets/terminal-dashboard/traces-empty-120x36.svg) | [Capture](../assets/terminal-dashboard/traces-empty-160x48.svg) |
| Stale | [Capture](../assets/terminal-dashboard/traces-stale-80x24.svg) | [Capture](../assets/terminal-dashboard/traces-stale-120x36.svg) | [Capture](../assets/terminal-dashboard/traces-stale-160x48.svg) |

Raster previews: [Trace list](../assets/terminal-dashboard/traces-120x36.png)
and [Span inspector](../assets/terminal-dashboard/spans-160x48.png).
Temporary qualification servers and browser tabs were stopped after review.

## Tracing simplification — 9 October 2026

Local tracing now exposes one authenticated `GET /traces` snapshot, with current
inspection settings included. SSE, cursor replay, the separate config/entrypoint
routes, server clearing, stream settings and broadcaster lifecycle wiring are
removed. Terminal filters, local clearing, live spans and OTLP remain available.
Operation traces reuse the existing span-capture helper. This removes **400
implementation lines**, without dependencies or changes to the terminal layout.

Qualification: **1,349 unit tests passed, 4 skipped, 129 deselected and 3 expected
failures**, using the same isolated protobuf 6.33.6 target documented above.
Source lint/format and focused type checks passed. The documentation build
retains its existing source-link warnings. A real local HTTP server verified
operator authentication, nested component capture and CLI JSON export; it was
stopped afterwards. Redis and the Python-version wheel matrix were not rerun.
External trace consumers must migrate to snapshots; see the
[API migration](../reference/tracing.md#trace-snapshot-api).

### Final review fixes

The agent review found two refresh failures: deeply nested span trees exceeded
client validation, and sanitized IDs broke row sorting. Trees now stop at depth
64 while retaining total span counts and disclosing omitted spans; sorting uses
stored cell identity independently of displayed labels. Regression tests failed
before the fixes. All **71 focused cases** and **1,354 unit tests** passed with
the isolated protobuf target (4 skipped, 129 deselected, 3 expected failures).
Source lint/format, focused types and docs passed; the agent's follow-up found
both issues resolved with no new actionable finding. No Redis or wheel matrix rerun.

## Default capture and mixed executions — 9 October 2026

This supersedes the earlier opt-in and durable-only execution scope. The main
host now enables local trace inspection and durable indexing by default, with
explicit CLI/environment opt-outs. The Executions tab combines recent ordinary
REST/OpenAI/MCP/A2A runs with indexed durable history across deployments.
Durable attempts are not duplicated. Deployment, ID and status filters work for
both sources; revision filters and subsequent pages apply to durable history.
Each durable deployment contributes at most 50 summaries per page. Recent
ordinary runs are a bounded responding-process sample, lost on eviction/restart.

Local POSIX servers and same-user clients automatically share a private operator
credential. Explicit credentials, verified middleware and FastAPI dependency
overrides remain supported; owner-scoped collections retain their owner policy.
Automatic credentials are never sent to remote URLs or environment proxies.
Unsafe/unavailable local credentials do not prevent server startup and leave
operator access denied. Each unavailable collection retains its own previous
rows while other collections and the trace sample continue to refresh.

Qualification: **1,372 unit tests passed, 4 skipped, 129 deselected, 3 expected
failures**, with the isolated protobuf 6.33.6 target described above. Source
lint/format, focused types and docs passed (existing source-link warnings remain).
The agent review's authentication, proxy and source-isolation findings were
fixed and verified. A temporary real HTTP server exercised automatic credentials,
ordinary and durable runs, and the combined terminal view at 80×24 and 120×36;
it was stopped afterward. SVG captures remain in
`/tmp/hayhooks-default-dashboard-captures/`. The user's running server on port
1416 was not restarted. Redis integration and the wheel/Python matrix were not rerun.

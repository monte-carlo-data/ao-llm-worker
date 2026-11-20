# llm-worker

Polling service that reads LLM evaluation batches from ClickHouse, invokes the
configured LLM provider (AWS Bedrock, Vertex AI, or Microsoft Foundry) in
parallel, and writes results back.

## How it works

The worker runs a continuous loop:

1. **Poll** -- query ClickHouse for batches in `pending` status.
2. **Process** -- for each batch, fetch unprocessed input rows in pages and
   invoke the configured LLM provider in parallel (`ThreadPoolExecutor`, `MAX_WORKERS`).
3. **Write** -- stream results back to ClickHouse in chunks of 50 rows, then
   mark the batch `complete`. A batch with throttled rows stays `pending` until
   a later pass finishes them (see [Throttling](#throttling)).

Reads are idempotent: rows already in `llm_results` are skipped, so the worker
can safely recover from crashes mid-batch. An auth/permission failure from the
provider triggers a batch-level abort to avoid burning retries on every remaining
row.

On startup the worker publishes its `(cloud, provider, capabilities)` to the
`otel_traces.llm_worker_info` table. This is a **cross-service contract**:

- The monolith reads the cloud to resolve which cloud-native Claude model pool a
  deployment uses (`aws`→bedrock, `gcp`→vertex, `azure`→foundry) when a batch's
  rows carry no explicit model.
- The monolith reads `capabilities` to decide which models it can offer (see
  [Request profiles](#request-profiles)). Every provider lists `output_format`,
  `thinking` and `effort`.

The worker republishes every 10 minutes, checked between batches and between pages
of a batch. On an older chart without the `capabilities` column, it writes the
cloud and provider only, so the capabilities land once the chart's migration adds
the column. The write is best-effort — a failure is logged and never blocks batch
processing.

### Request profiles

Some models, such as Claude Sonnet 5.5 and Opus 5.5, reject a forced tool call.
For them the monolith sends a different request:

| Column        | Key                 | Meaning                                                                 |
| ------------- | ------------------- | ----------------------------------------------------------------------- |
| `tool_config` | `output_format`     | `{"type": "json_schema", "name", "description", "schema"}`, instead of `tools` and `tool_choice` |
| `params`      | `thinking`          | Thinking type, such as `adaptive`                                       |
| `params`      | `effort`            | Optional effort level                                                   |

The worker maps them to each backend:

- **Vertex, Foundry:** `output_config.format` (schema only), `thinking`, and
  `output_config.effort`.
- **Bedrock:** Converse `outputConfig.textFormat`, with the schema as a JSON
  string. Thinking and effort go in `additionalModelRequestFields`.

A request with `thinking` or `output_format` never sends a temperature. A row
that combines `output_format` with `tools`, or `thinking` with a forced tool
choice, fails with a message. The JSON reply is stored as
`tool_uses[0]`, where a forced tool call's input goes. A refusal, a token-cap stop
or a reply that isn't a JSON object still completes the row, without `tool_uses`.
So does a JSON reply that didn't end with `end_turn` or `stop_sequence`. The
response records `stop_reason`, plus `refusal_category` or `result_error` when
they apply. Only API errors fail a row. On Bedrock, a refusal arrives as
`stopReason: content_filtered`; the worker asks for Claude's `stop_details` to
record it as a refusal with its category.

The Bedrock client waits up to 120 seconds for a reply and keeps botocore's
default retries, which absorb short throttles before the worker sees them. Its
connection pool holds `MAX_WORKERS` connections, one per worker thread.

On the `bedrock` provider, a `converse()` call resolves through a
cost-attribution inference profile ARN when `BEDROCK_INFERENCE_PROFILES` maps
its model id to one, falling back to the bare model id otherwise. If Bedrock
rejects that ARN as not found, the call retries once with the bare model id,
and the ARN is skipped for 300 seconds before being tried again.

### Throttling

A throttled call doesn't fail its row. Throttles are Bedrock's
`ThrottlingException` and, on Vertex and Foundry, 429 (rate limit) and 529
(overloaded).

- **The row is deferred.** No result is written, so the batch stays `pending`
  and a later poll fetches the row again. A throttle doesn't use up the row's
  `RETRY_MAX_ATTEMPTS`.
- **The model pauses.** Every worker thread stops calling that model for
  `THROTTLE_PAUSE_MIN_SECONDS`, and each new pause doubles, up to
  `THROTTLE_PAUSE_MAX_SECONDS`. A burst of throttles grows the pause once. A
  success resets the next pause to the minimum. Other models aren't paused:
  rows for other models in the same page keep running. Batches are handled one
  at a time, though, so a batch waiting on a paused model delays the batches
  after it.
- **The turn ends.** After a page with deferred rows, the worker moves on to the
  next batch instead of fetching the same rows again. When every batch in a
  pass deferred rows, it waits `POLL_INTERVAL` before polling again.
- **Old rows fail.** A throttled row fails with `throttled and older than N
  seconds` once its age plus `THROTTLE_PAUSE_MAX_SECONDS` reaches
  `THROTTLE_MAX_ROW_AGE_SECONDS`. This keeps a batch from staying pending
  indefinitely. A deferred row is tried again on its batch's next turn, one
  pass over pending batches later, so adding the max pause is a heuristic that
  leaves room for one more pause before the limit, not a bound on when the next
  attempt comes. ClickHouse measures the age from the row's `created_at`, and
  the worker adds the time since it fetched the row. A row past the limit still
  waits out its model's pause and gets one call; it fails only if that call is
  throttled too. The limit must be greater than `THROTTLE_PAUSE_MAX_SECONDS`.
- **Late results under sustained throttling.** The worker doesn't know how
  long the reader of a batch waits for it. Batches are handled one at a time,
  and each turn on a batch throttled on a paused model waits out the pause, so
  under sustained throttling a pass can take several pauses. A deferred row can
  then come back after the reader has stopped waiting, and the reader times out
  instead of seeing per-row errors.

Other transient errors, such as 5xx responses (including Vertex 503 and 504),
timeouts and connection errors, are retried up to `RETRY_MAX_ATTEMPTS` and then
fail the row.

The worker logs `llm_throttle_pause` when a model pauses and `row_deferred` for
each deferred row. A turn that deferred rows logs `batch_deferred` instead of
`batch_completed`, and a turn cut short by shutdown logs `batch_interrupted`.
These logs carry error codes, never prompt or response content.

### Shutdown

On SIGTERM or SIGINT the worker stops starting new provider calls, waits for
the calls already in flight, writes their results and exits. Rows in the page
that haven't started, and rows waiting out a throttle pause or a retry backoff,
are deferred as `stopping`, not failed. They and the batch's remaining pages
stay `pending` for the next worker.

Each in-flight call can take up to the client timeout (120 seconds) on each of
the SDK's own attempts: botocore makes 5 attempts by default, and the Anthropic
SDK retries twice. Size the termination grace period with that in mind. If the
worker is killed before it finishes, results it hasn't written yet (they are
written in chunks of 50) are lost, and the next worker processes those rows
again.

## Quick start

```bash
# install
uv sync

# run (needs ClickHouse + the selected provider's credentials —
# see the LLM provider section under Configuration)
uv run llm-worker
```

## Configuration

All settings come from environment variables.

### ClickHouse

| Variable       | Default     | Description               |
| -------------- | ----------- | ------------------------- |
| `CH_HOST`      | `localhost` | ClickHouse host           |
| `CH_PORT`      | `8123`      | ClickHouse HTTP port      |
| `CH_USER`      | `default`   | ClickHouse user           |
| `CH_PASSWORD`  | *(empty)*   | ClickHouse password       |
| `CH_DATABASE`  | `default`   | ClickHouse database       |

### LLM provider

The worker talks to one LLM backend per deployment, selected by `LLM_PROVIDER`.
Each cloud ships as its own image variant (built with `--build-arg CLOUD=<aws|gcp|azure>`),
carrying only that backend's SDK.

| Variable             | Default   | Description                                                  |
| -------------------- | --------- | ------------------------------------------------------------ |
| `LLM_PROVIDER`       | `bedrock` | `bedrock` (AWS), `vertex` (GCP), or `foundry` (Azure)        |
| `MAX_WORKERS`        | `20`      | Concurrent provider calls; also the Bedrock connection pool size |
| `RETRY_MAX_ATTEMPTS` | `5`       | Max attempts per call for transient errors; throttles don't count |
| `RETRY_MAX_BACKOFF`  | `30`      | Max backoff in seconds between retries                       |

Provider-specific settings — only the selected provider's vars are read:

**`bedrock`** (aws image) — auth via the pod's AWS credentials (IRSA):

| Variable                     | Default     | Description                                                                   |
| ---------------------------- | ----------- | ------------------------------------------------------------------------------ |
| `AWS_REGION`                  | `us-east-1` | AWS region for Bedrock                                                        |
| `BEDROCK_INFERENCE_PROFILES` | `{}`        | JSON object mapping a model id to its application-inference-profile ARN, for cost attribution. Unmapped models invoke as-is. |

Map keys go through the same `mc:`/`provider:` prefix stripping as an invoked
model id, so a key can be given either as the bare model id or with the
prefix — `"us.anthropic.claude-sonnet-5"` and `"provider:us.anthropic.claude-sonnet-5"`
both match a row with `model_id: "provider:us.anthropic.claude-sonnet-5"`.

**`vertex`** (gcp image) — Claude on Vertex AI, auth via GKE Workload Identity (ADC, no key):

| Variable                      | Default  | Description                                             |
| ----------------------------- | -------- | -------------------------------------------------------- |
| `ANTHROPIC_VERTEX_PROJECT_ID` | —        | GCP project id                                          |
| `CLOUD_ML_REGION`             | `global` | Vertex region                                           |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `120`    | Per-request timeout (seconds) for the Anthropic client  |

Model selection is per-request via each input row's `model_id`, not env-configured.

**`foundry`** (azure image) — Claude on Microsoft Foundry, auth via Entra ID (`DefaultAzureCredential`, no static secret):

| Variable                      | Default | Description                                             |
| ----------------------------- | ------- | -------------------------------------------------------- |
| `ANTHROPIC_FOUNDRY_RESOURCE`  | —       | Microsoft Foundry resource name                         |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `120`   | Per-request timeout (seconds) for the Anthropic client  |

### Service

| Variable              | Default | Description                                    |
| --------------------- | ------- | ---------------------------------------------- |
| `POLL_INTERVAL`       | `10`    | Seconds between polling cycles when idle or when every batch was throttled |
| `PENDING_BATCH_LIMIT` | `100`   | Max pending batches to fetch per poll          |
| `BATCH_PAGE_SIZE`     | `100`   | Input rows fetched per page within a batch     |
| `THROTTLE_PAUSE_MIN_SECONDS` | `1` | First pause for a throttled model, in seconds |
| `THROTTLE_PAUSE_MAX_SECONDS` | `60` | Longest pause for a throttled model, in seconds |
| `THROTTLE_MAX_ROW_AGE_SECONDS` | `600` | A throttled row fails instead of being deferred once its age (seconds since insert) plus `THROTTLE_PAUSE_MAX_SECONDS` reaches this (540s of age with the defaults). Must be greater than `THROTTLE_PAUSE_MAX_SECONDS`; see [Throttling](#throttling) |

## Testing

```bash
# unit tests (no external deps)
uv run pytest

# integration tests (requires ClickHouse)
docker compose up -d
uv run pytest -m clickhouse
```

## Verification

```bash
uv run ruff check src/ tests/
uv run pyright src/
```

## Docker

Each image variant bundles one cloud's LLM SDK; select it with `--build-arg CLOUD`:

```bash
docker build --build-arg CLOUD=aws -t llm-worker:aws .
docker run -e CH_HOST=clickhouse -e LLM_PROVIDER=bedrock -e AWS_REGION=us-east-1 llm-worker:aws
```

## Licensing

Each image bundles per-cloud third-party attribution under `licensing/<cloud>/`
(`NOTICE` + verbatim `LICENSES/`), generated from the built image. After changing
dependencies you must regenerate and commit those artifacts — CI enforces it via
`gen_licenses.py --check`. See [`licensing/README.md`](licensing/README.md) for the
regeneration workflow.

## Project structure

```
src/llm_worker/
  config.py       -- environment-based configuration + provider selection
  main.py         -- polling loop, batch orchestration, graceful shutdown
  executor.py     -- provider-agnostic thread pool, retry, throttle deferral, and result mapping
  throttle.py     -- per-model pause shared by worker threads when a backend throttles
  contract.py     -- MC eval contract v1 types + input normalization
  providers/      -- pluggable LLM backends (bedrock, vertex, foundry)
  clickhouse.py   -- ClickHouse reads/writes for inputs, results, and batch status
```

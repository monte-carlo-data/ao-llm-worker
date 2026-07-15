# llm-worker

Polling service that reads LLM evaluation batches from ClickHouse, invokes the
configured LLM provider (AWS Bedrock, Azure OpenAI, or Vertex AI) in parallel,
and writes results back.

## How it works

The worker runs a continuous loop:

1. **Poll** -- query ClickHouse for batches in `pending` status.
2. **Process** -- for each batch, fetch unprocessed input rows in pages and
   invoke the configured LLM provider in parallel (`ThreadPoolExecutor`, `MAX_WORKERS`).
3. **Write** -- stream results back to ClickHouse in chunks of 50 rows, then
   mark the batch `complete`.

Reads are idempotent: rows already in `llm_results` are skipped, so the worker
can safely recover from crashes mid-batch. An auth/permission failure from the
provider triggers a batch-level abort to avoid burning retries on every remaining
row.

## Quick start

```bash
# install
uv sync

# run (needs CH + AWS credentials)
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
Each cloud ships as its own image variant (built with `--build-arg CLOUD=<aws|azure|gcp>`),
carrying only that backend's SDK.

| Variable             | Default   | Description                                                  |
| -------------------- | --------- | ------------------------------------------------------------ |
| `LLM_PROVIDER`       | `bedrock` | `bedrock` (AWS), `azure_openai` (Azure), or `vertex` (GCP)   |
| `MAX_WORKERS`        | `20`      | Concurrent provider calls                                    |
| `RETRY_MAX_ATTEMPTS` | `5`       | Max retries per call                                         |
| `RETRY_MAX_BACKOFF`  | `30`      | Max backoff in seconds between retries                       |

Provider-specific settings — only the selected provider's vars are read:

**`bedrock`** (aws image) — auth via the pod's AWS credentials (IRSA):

| Variable     | Default     | Description            |
| ------------ | ----------- | ---------------------- |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |

**`azure_openai`** (azure image) — auth via Entra workload identity (no key):

| Variable                   | Default | Description                    |
| -------------------------- | ------- | ------------------------------ |
| `AZURE_OPENAI_ENDPOINT`    | —       | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_VERSION` | —       | REST API version               |
| `AZURE_OPENAI_DEPLOYMENT`  | —       | Deployment name to invoke      |

**`vertex`** (gcp image) — Claude on Vertex AI, auth via GKE Workload Identity (ADC, no key):

| Variable                      | Default  | Description               |
| ----------------------------- | -------- | ------------------------- |
| `ANTHROPIC_VERTEX_PROJECT_ID` | —        | GCP project id            |
| `CLOUD_ML_REGION`             | `global` | Vertex region             |
| `VERTEX_MODEL`                | —        | Vertex model id to invoke |

### Service

| Variable              | Default | Description                                    |
| --------------------- | ------- | ---------------------------------------------- |
| `POLL_INTERVAL`       | `10`    | Seconds between polling cycles when idle       |
| `PENDING_BATCH_LIMIT` | `100`   | Max pending batches to fetch per poll          |
| `BATCH_PAGE_SIZE`     | `100`   | Input rows fetched per page within a batch     |

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

## Project structure

```
src/llm_worker/
  config.py       -- environment-based configuration + provider selection
  main.py         -- polling loop, batch orchestration, graceful shutdown
  executor.py     -- provider-agnostic thread pool, retry, and result mapping
  contract.py     -- MC eval contract v1 types + input normalization
  providers/      -- pluggable LLM backends (bedrock, azure_openai, vertex)
  clickhouse.py   -- ClickHouse reads/writes for inputs, results, and batch status
```

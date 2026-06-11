# llm-worker

Polling service that reads LLM evaluation batches from ClickHouse, calls
AWS Bedrock Converse API in parallel, and writes results back.

## How it works

The worker runs a continuous loop:

1. **Poll** -- query ClickHouse for batches in `pending` status.
2. **Process** -- for each batch, fetch unprocessed input rows in pages and
   invoke Bedrock Converse in parallel (`ThreadPoolExecutor`, default 20 workers).
3. **Write** -- stream results back to ClickHouse in chunks of 50 rows, then
   mark the batch `complete`.

Reads are idempotent: rows already in `llm_results` are skipped, so the worker
can safely recover from crashes mid-batch. An `AccessDeniedException` from
Bedrock triggers a batch-level abort to avoid burning retries on every remaining
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

### Bedrock

| Variable              | Default     | Description                               |
| --------------------- | ----------- | ----------------------------------------- |
| `AWS_REGION`          | `us-east-1` | AWS region for Bedrock                    |
| `MAX_WORKERS`         | `20`        | Concurrent Bedrock calls                  |
| `RETRY_MAX_ATTEMPTS`  | `5`         | Max retries per Bedrock call              |
| `RETRY_MAX_BACKOFF`   | `30`        | Max backoff in seconds between retries    |

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

```bash
docker build -t llm-worker .
docker run -e CH_HOST=clickhouse -e AWS_REGION=us-east-1 llm-worker
```

## Project structure

```
src/llm_worker/
  config.py       -- environment-based configuration
  main.py         -- polling loop, batch orchestration, graceful shutdown
  bedrock.py      -- Bedrock Converse client with retry and parallel execution
  clickhouse.py   -- ClickHouse reads/writes for inputs, results, and batch status
```

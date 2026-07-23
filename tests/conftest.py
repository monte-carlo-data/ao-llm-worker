import os

import pytest

from llm_worker.clickhouse import ClickHouseClient

CH_HOST = os.getenv("CLICKHOUSE_HOST", "127.0.0.1")
CH_HTTP_PORT = int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123"))
CH_USER = os.getenv("CLICKHOUSE_USER", "default")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
CH_DATABASE = "llm_worker_test"

CREATE_INPUTS_TABLE = """
CREATE TABLE IF NOT EXISTS llm_inputs
(
    batch_id        UUID,
    row_id          UUID,
    model_id        LowCardinality(String),
    prompt          String,
    params          String DEFAULT '{}',
    tool_config     String DEFAULT '',
    created_at      DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (batch_id, row_id)
TTL created_at + INTERVAL 30 DAY DELETE
"""

CREATE_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS llm_results
(
    batch_id        UUID,
    row_id          UUID,
    response        String,
    status          Enum8('complete' = 1, 'failed' = 2),
    error           String DEFAULT '',
    created_at      DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (batch_id, row_id)
TTL created_at + INTERVAL 30 DAY DELETE
"""

CREATE_BATCHES_TABLE = """
CREATE TABLE IF NOT EXISTS llm_batches
(
    batch_id        UUID,
    status          Enum8('pending' = 1, 'complete' = 2),
    total_rows      UInt32 DEFAULT 0,
    completed_rows  UInt32 DEFAULT 0,
    failed_rows     UInt32 DEFAULT 0,
    created_at      DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (batch_id, created_at)
TTL created_at + INTERVAL 30 DAY DELETE
"""

# Mirrors the helm sql/ migration for otel_traces.llm_worker_info. The worker
# publishes its (cloud, provider); ReplacingMergeTree(updated_at) keeps a single
# current row per (cloud, provider) so re-publishing on each startup is idempotent.
CREATE_WORKER_INFO_TABLE = """
CREATE TABLE IF NOT EXISTS llm_worker_info
(
    cloud       LowCardinality(String),
    provider    LowCardinality(String),
    updated_at  DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (cloud, provider)
"""


@pytest.fixture(scope="session")
def ch_client():
    """Session-scoped ClickHouseClient backed by a test database."""
    import clickhouse_connect

    raw_client = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_HTTP_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
    )
    raw_client.command(f"CREATE DATABASE IF NOT EXISTS {CH_DATABASE}")
    raw_client.database = CH_DATABASE
    raw_client.command(CREATE_INPUTS_TABLE)
    raw_client.command(CREATE_RESULTS_TABLE)
    raw_client.command(CREATE_BATCHES_TABLE)
    raw_client.command(CREATE_WORKER_INFO_TABLE)

    client = ClickHouseClient(client=raw_client)
    yield client

    raw_client.command(f"DROP DATABASE IF EXISTS {CH_DATABASE}")


@pytest.fixture(autouse=True)
def clean_tables(request):
    """Truncate tables before each clickhouse-marked test."""
    if "clickhouse" not in [m.name for m in request.node.iter_markers()]:
        yield
        return
    ch_client = request.getfixturevalue("ch_client")
    ch_client._client.command("TRUNCATE TABLE llm_inputs")
    ch_client._client.command("TRUNCATE TABLE llm_results")
    ch_client._client.command("TRUNCATE TABLE llm_batches")
    ch_client._client.command("TRUNCATE TABLE llm_worker_info")
    yield

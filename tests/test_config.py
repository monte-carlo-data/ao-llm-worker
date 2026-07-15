import pytest

from llm_worker.config import load_config


def test_load_config_defaults():
    config = load_config()
    assert config.clickhouse.host == "localhost"
    assert config.clickhouse.port == 8123
    assert config.clickhouse.user == "default"
    assert config.clickhouse.password == ""
    assert config.clickhouse.database == "default"
    assert config.provider == "bedrock"
    assert config.bedrock is not None
    assert config.bedrock.region == "us-east-1"
    assert config.vertex is None
    assert config.max_workers == 20
    assert config.retry_max_attempts == 5
    assert config.retry_max_backoff == 30
    assert config.poll_interval == 10
    assert config.pending_batch_limit == 100


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("CH_HOST", "ch.example.com")
    monkeypatch.setenv("CH_PORT", "9000")
    monkeypatch.setenv("CH_USER", "admin")
    monkeypatch.setenv("CH_PASSWORD", "secret")
    monkeypatch.setenv("CH_DATABASE", "mydb")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("MAX_WORKERS", "50")
    monkeypatch.setenv("RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("RETRY_MAX_BACKOFF", "60")
    monkeypatch.setenv("POLL_INTERVAL", "5")
    monkeypatch.setenv("PENDING_BATCH_LIMIT", "25")

    config = load_config()
    assert config.clickhouse.host == "ch.example.com"
    assert config.clickhouse.port == 9000
    assert config.clickhouse.user == "admin"
    assert config.clickhouse.password == "secret"
    assert config.clickhouse.database == "mydb"
    assert config.provider == "bedrock"
    assert config.bedrock is not None
    assert config.bedrock.region == "us-west-2"
    assert config.max_workers == 50
    assert config.retry_max_attempts == 3
    assert config.retry_max_backoff == 60
    assert config.poll_interval == 5
    assert config.pending_batch_limit == 25


def test_load_vertex_config(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "mc-proj")
    monkeypatch.setenv("CLOUD_ML_REGION", "us-east5")
    monkeypatch.setenv("VERTEX_MODEL", "claude-sonnet-4-5@20250929")

    config = load_config()
    assert config.provider == "vertex"
    assert config.bedrock is None
    assert config.vertex is not None
    assert config.vertex.project == "mc-proj"
    assert config.vertex.region == "us-east5"
    assert config.vertex.model == "claude-sonnet-4-5@20250929"


def test_vertex_region_defaults_to_global(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "mc-proj")
    monkeypatch.setenv("VERTEX_MODEL", "claude-sonnet-4-5@20250929")
    # CLOUD_ML_REGION not set → defaults to "global"

    config = load_config()
    assert config.vertex is not None
    assert config.vertex.region == "global"


def test_vertex_missing_env_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    # ANTHROPIC_VERTEX_PROJECT_ID / VERTEX_MODEL not set
    with pytest.raises(ValueError, match="ANTHROPIC_VERTEX_PROJECT_ID is required"):
        load_config()


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nonexistent_provider")
    with pytest.raises(ValueError, match="not supported"):
        load_config()


def test_config_is_frozen():
    config = load_config()
    with pytest.raises(AttributeError):
        config.poll_interval = 99  # type: ignore[reportAttributeAccessIssue]

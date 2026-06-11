import pytest

from llm_worker.config import load_config


def test_load_config_defaults():
    config = load_config()
    assert config.clickhouse.host == "localhost"
    assert config.clickhouse.port == 8123
    assert config.clickhouse.user == "default"
    assert config.clickhouse.password == ""
    assert config.clickhouse.database == "default"
    assert config.bedrock.region == "us-east-1"
    assert config.bedrock.max_workers == 20
    assert config.bedrock.retry_max_attempts == 5
    assert config.bedrock.retry_max_backoff == 30
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
    assert config.bedrock.region == "us-west-2"
    assert config.bedrock.max_workers == 50
    assert config.bedrock.retry_max_attempts == 3
    assert config.bedrock.retry_max_backoff == 60
    assert config.poll_interval == 5
    assert config.pending_batch_limit == 25


def test_config_is_frozen():
    config = load_config()
    with pytest.raises(AttributeError):
        config.poll_interval = 99  # type: ignore[reportAttributeAccessIssue]

import pytest

from llm_worker.config import load_config

# load_config is provider-agnostic (config is a leaf module); provider-config
# resolution is tested against load_provider_config in test_providers.py.


def test_load_config_defaults():
    config = load_config()
    assert config.clickhouse.host == "localhost"
    assert config.clickhouse.port == 8123
    assert config.clickhouse.user == "default"
    assert config.clickhouse.password == ""
    assert config.clickhouse.database == "default"
    assert config.max_workers == 20
    assert config.retry_max_attempts == 5
    assert config.retry_max_backoff == 30
    assert config.poll_interval == 10
    assert config.pending_batch_limit == 100
    assert config.throttle_pause_min_seconds == 1
    assert config.throttle_pause_max_seconds == 60
    assert config.throttle_max_row_age_seconds == 600


def test_throttle_settings_from_env(monkeypatch):
    monkeypatch.setenv("THROTTLE_PAUSE_MIN_SECONDS", "0.5")
    monkeypatch.setenv("THROTTLE_PAUSE_MAX_SECONDS", "30")
    monkeypatch.setenv("THROTTLE_MAX_ROW_AGE_SECONDS", "300")

    config = load_config()
    assert config.throttle_pause_min_seconds == 0.5
    assert config.throttle_pause_max_seconds == 30
    assert config.throttle_max_row_age_seconds == 300


def test_throttle_pause_max_below_min_raises(monkeypatch):
    monkeypatch.setenv("THROTTLE_PAUSE_MIN_SECONDS", "10")
    monkeypatch.setenv("THROTTLE_PAUSE_MAX_SECONDS", "5")
    with pytest.raises(ValueError, match="THROTTLE_PAUSE_MAX_SECONDS"):
        load_config()


@pytest.mark.parametrize("max_row_age", ["30", "20"])
def test_throttle_max_row_age_not_above_pause_max_raises(monkeypatch, max_row_age):
    # A row throttled once would always fail: its age plus the max pause would
    # already reach the limit.
    monkeypatch.setenv("THROTTLE_PAUSE_MAX_SECONDS", "30")
    monkeypatch.setenv("THROTTLE_MAX_ROW_AGE_SECONDS", max_row_age)
    with pytest.raises(ValueError, match="THROTTLE_MAX_ROW_AGE_SECONDS"):
        load_config()


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("CH_HOST", "ch.example.com")
    monkeypatch.setenv("CH_PORT", "9000")
    monkeypatch.setenv("CH_USER", "admin")
    monkeypatch.setenv("CH_PASSWORD", "secret")
    monkeypatch.setenv("CH_DATABASE", "mydb")
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
    assert config.max_workers == 50
    assert config.retry_max_attempts == 3
    assert config.retry_max_backoff == 60
    assert config.poll_interval == 5
    assert config.pending_batch_limit == 25


def test_config_is_frozen():
    config = load_config()
    with pytest.raises(AttributeError):
        config.poll_interval = 99  # type: ignore[reportAttributeAccessIssue]

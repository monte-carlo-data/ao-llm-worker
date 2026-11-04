"""``write_worker_info`` against a mocked ClickHouse client.

``test_clickhouse.py`` covers the same against a real server, but CI doesn't run it.
"""

from unittest.mock import MagicMock

from llm_worker.clickhouse import WORKER_CAPABILITIES, ClickHouseClient


def _client(has_capabilities_column: bool) -> tuple[ClickHouseClient, MagicMock]:
    raw = MagicMock()
    raw.query.return_value.result_rows = [[1 if has_capabilities_column else 0]]
    return ClickHouseClient(client=raw), raw


def test_writes_the_capabilities_when_the_column_exists():
    ch, raw = _client(has_capabilities_column=True)

    assert ch.write_worker_info("azure", "foundry") is True

    raw.insert.assert_called_once()
    args, kwargs = raw.insert.call_args
    assert args[1] == [["azure", "foundry", list(WORKER_CAPABILITIES)]]
    assert kwargs["column_names"] == ["cloud", "provider", "capabilities"]


def test_writes_cloud_and_provider_only_on_an_older_table():
    ch, raw = _client(has_capabilities_column=False)

    assert ch.write_worker_info("azure", "foundry") is False

    raw.insert.assert_called_once()
    args, kwargs = raw.insert.call_args
    assert args[1] == [["azure", "foundry"]]
    assert kwargs["column_names"] == ["cloud", "provider"]


def test_capabilities_cover_the_request_profile_fields():
    # The monolith offers models with a request profile only when all three are listed.
    assert set(WORKER_CAPABILITIES) == {"output_format", "thinking", "effort"}

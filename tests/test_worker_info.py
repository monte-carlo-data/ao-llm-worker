"""``write_worker_info`` against a mocked ClickHouse client.

``test_clickhouse.py`` covers the same against a real server, but CI doesn't run it.
"""

import logging
from unittest.mock import MagicMock

from llm_worker.clickhouse import ClickHouseClient

CAPABILITIES = ("output_format", "thinking")


def _client(has_capabilities_column: bool) -> tuple[ClickHouseClient, MagicMock]:
    raw = MagicMock()
    raw.query.return_value.result_rows = [[1 if has_capabilities_column else 0]]
    return ClickHouseClient(client=raw), raw


def test_writes_the_capabilities_when_the_column_exists():
    ch, raw = _client(has_capabilities_column=True)

    ch.write_worker_info("azure", "foundry", CAPABILITIES)

    raw.insert.assert_called_once()
    args, kwargs = raw.insert.call_args
    assert args[1] == [["azure", "foundry", list(CAPABILITIES)]]
    assert kwargs["column_names"] == ["cloud", "provider", "capabilities"]


def test_writes_cloud_and_provider_only_on_an_older_table():
    ch, raw = _client(has_capabilities_column=False)

    ch.write_worker_info("azure", "foundry", CAPABILITIES)

    raw.insert.assert_called_once()
    args, kwargs = raw.insert.call_args
    assert args[1] == [["azure", "foundry"]]
    assert kwargs["column_names"] == ["cloud", "provider"]


def test_warns_once_about_a_missing_column_then_logs_at_debug(caplog):
    ch, _ = _client(has_capabilities_column=False)
    caplog.set_level(logging.DEBUG, logger="llm_worker.clickhouse")

    for _ in range(3):
        ch.write_worker_info("azure", "foundry", CAPABILITIES)

    levels = [
        r.levelno
        for r in caplog.records
        if r.message == "worker_info_written_without_capabilities"
    ]
    assert levels == [logging.WARNING, logging.DEBUG, logging.DEBUG]


def test_every_write_sends_a_new_deduplication_token():
    # A replicated table drops a repeated insert block, so an identical republish
    # couldn't replace a newer row without a fresh token.
    for has_column in (True, False):
        ch, raw = _client(has_capabilities_column=has_column)

        ch.write_worker_info("azure", "foundry", CAPABILITIES)
        ch.write_worker_info("azure", "foundry", CAPABILITIES)

        tokens = [
            call.kwargs["settings"]["insert_deduplication_token"]
            for call in raw.insert.call_args_list
        ]
        assert len(set(tokens)) == 2
        assert all(tokens)

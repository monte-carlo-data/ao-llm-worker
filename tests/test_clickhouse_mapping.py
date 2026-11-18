"""Row-mapping tests for ClickHouseClient that need no ClickHouse server.

tests/test_clickhouse.py is marked ``clickhouse`` and doesn't run by default,
so the column-to-field mapping is checked here against a mocked client.
"""

from uuid import UUID

from llm_worker.clickhouse import ClickHouseClient, LLMInput

BATCH_1 = UUID("00000000-0000-0000-0000-000000000001")
ROW_1 = UUID("00000000-0000-0000-0000-000000000011")


def test_get_batch_rows_maps_every_column(mocker):
    mock_client = mocker.Mock()
    mock_client.query.return_value.result_rows = [
        (BATCH_1, ROW_1, "model-1", "hello", '{"t": 1}', '{"tools": []}', 42),
    ]

    rows = ClickHouseClient(client=mock_client).get_batch_rows(BATCH_1)

    assert rows == [
        LLMInput(
            batch_id=BATCH_1,
            row_id=ROW_1,
            model_id="model-1",
            prompt="hello",
            params='{"t": 1}',
            tool_config='{"tools": []}',
            age_seconds=42,
        )
    ]


def test_get_batch_rows_selects_age_from_clickhouse_clock(mocker):
    mock_client = mocker.Mock()
    mock_client.query.return_value.result_rows = []

    ClickHouseClient(client=mock_client).get_batch_rows(BATCH_1)

    sql = mock_client.query.call_args.args[0]
    assert "dateDiff('second', created_at, now()) AS age_seconds" in sql

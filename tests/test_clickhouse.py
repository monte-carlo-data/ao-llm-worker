from uuid import UUID, uuid4

import pytest

from llm_worker.clickhouse import ClickHouseClient, LLMResult

pytestmark = [pytest.mark.clickhouse]

# Fixed UUIDs for deterministic tests.
BATCH_1 = UUID("00000000-0000-0000-0000-000000000001")
BATCH_2 = UUID("00000000-0000-0000-0000-000000000002")
ROW_1 = UUID("00000000-0000-0000-0000-000000000011")
ROW_2 = UUID("00000000-0000-0000-0000-000000000012")


def _insert_input(
    ch_client, batch_id, row_id, model_id="test-model", prompt="test prompt"
):
    ch_client._client.insert(
        "llm_inputs",
        [[batch_id, row_id, model_id, prompt, "{}", ""]],
        column_names=[
            "batch_id",
            "row_id",
            "model_id",
            "prompt",
            "params",
            "tool_config",
        ],
    )


def _insert_result(ch_client, batch_id, row_id, status="complete"):
    ch_client._client.insert(
        "llm_results",
        [[batch_id, row_id, '{"output_text": "done"}', status, ""]],
        column_names=["batch_id", "row_id", "response", "status", "error"],
    )


def _insert_pending_batch(ch_client, batch_id):
    ch_client.write_batch_status(batch_id, "pending")


class TestWriteWorkerInfo:
    def test_writes_row(self, ch_client):
        ch_client.write_worker_info("aws", "bedrock")
        result = ch_client._client.query(
            "SELECT cloud, provider FROM llm_worker_info FINAL ORDER BY updated_at DESC LIMIT 1"
        )
        assert result.result_rows[0] == ("aws", "bedrock")

    def test_republish_keeps_single_row(self, ch_client):
        ch_client.write_worker_info("gcp", "vertex")
        ch_client.write_worker_info("gcp", "vertex")
        result = ch_client._client.query("SELECT count() FROM llm_worker_info FINAL")
        assert result.result_rows[0][0] == 1


class TestGetPendingBatches:
    def test_no_pending_batches(self, ch_client):
        assert ch_client.get_pending_batches() == []

    def test_one_pending_batch(self, ch_client):
        _insert_pending_batch(ch_client, BATCH_1)
        assert ch_client.get_pending_batches() == [BATCH_1]

    def test_multiple_pending_batches(self, ch_client):
        _insert_pending_batch(ch_client, BATCH_1)
        _insert_pending_batch(ch_client, BATCH_2)
        result = ch_client.get_pending_batches()
        assert sorted(result) == sorted([BATCH_1, BATCH_2])

    def test_completed_batch_not_returned(self, ch_client):
        _insert_pending_batch(ch_client, BATCH_1)
        ch_client.write_batch_status(BATCH_1, "complete", completed_rows=1)
        assert ch_client.get_pending_batches() == []

    def test_batch_without_pending_status_not_returned(self, ch_client):
        _insert_input(ch_client, BATCH_1, ROW_1)
        assert ch_client.get_pending_batches() == []


class TestGetBatchRows:
    def test_returns_all_rows(self, ch_client):
        _insert_input(ch_client, BATCH_1, ROW_1, "model-a", "prompt 1")
        _insert_input(ch_client, BATCH_1, ROW_2, "model-a", "prompt 2")
        rows = ch_client.get_batch_rows(BATCH_1)
        assert len(rows) == 2
        assert {r.row_id for r in rows} == {ROW_1, ROW_2}

    def test_skips_already_processed_rows(self, ch_client):
        _insert_input(ch_client, BATCH_1, ROW_1)
        _insert_input(ch_client, BATCH_1, ROW_2)
        _insert_result(ch_client, BATCH_1, ROW_1)
        rows = ch_client.get_batch_rows(BATCH_1)
        assert len(rows) == 1
        assert rows[0].row_id == ROW_2

    def test_does_not_return_other_batches(self, ch_client):
        _insert_input(ch_client, BATCH_1, ROW_1)
        _insert_input(ch_client, BATCH_2, ROW_1)
        rows = ch_client.get_batch_rows(BATCH_1)
        assert len(rows) == 1
        assert rows[0].batch_id == BATCH_1

    def test_returns_all_fields(self, ch_client):
        b1 = uuid4()
        r1 = uuid4()
        ch_client._client.insert(
            "llm_inputs",
            [
                [
                    b1,
                    r1,
                    "claude-3",
                    "analyze this",
                    '{"maxTokens": 100}',
                    '{"tools": []}',
                ]
            ],
            column_names=[
                "batch_id",
                "row_id",
                "model_id",
                "prompt",
                "params",
                "tool_config",
            ],
        )
        rows = ch_client.get_batch_rows(b1)
        assert len(rows) == 1
        row = rows[0]
        assert row.batch_id == b1
        assert row.row_id == r1
        assert row.model_id == "claude-3"
        assert row.prompt == "analyze this"
        assert row.params == '{"maxTokens": 100}'
        assert row.tool_config == '{"tools": []}'

    def test_empty_batch(self, ch_client):
        rows = ch_client.get_batch_rows(uuid4())
        assert rows == []


class TestWriteResults:
    def test_write_single_result(self, ch_client):
        b1 = uuid4()
        r1 = uuid4()
        results = [
            LLMResult(
                batch_id=b1,
                row_id=r1,
                response='{"output_text": "hello"}',
                status="complete",
                error="",
            )
        ]
        ch_client.write_results(results)
        rows = ch_client._client.query("SELECT * FROM llm_results").result_rows
        assert len(rows) == 1
        assert rows[0][0] == b1
        assert rows[0][1] == r1
        assert rows[0][3] == "complete"

    def test_write_multiple_results(self, ch_client):
        b1 = uuid4()
        r1 = uuid4()
        r2 = uuid4()
        results = [
            LLMResult(b1, r1, '{"output_text": "a"}', "complete", ""),
            LLMResult(b1, r2, "", "failed", "model error"),
        ]
        ch_client.write_results(results)
        rows = ch_client._client.query(
            "SELECT * FROM llm_results ORDER BY row_id"
        ).result_rows
        assert len(rows) == 2
        statuses = {r[3] for r in rows}
        assert statuses == {"complete", "failed"}
        failed_row = next(r for r in rows if r[3] == "failed")
        assert failed_row[4] == "model error"

    def test_write_empty_list(self, ch_client):
        ch_client.write_results([])
        rows = ch_client._client.query("SELECT * FROM llm_results").result_rows
        assert len(rows) == 0

    def test_write_results_waits_for_async_visibility(self, mocker):
        mock_client = mocker.Mock()
        ch = ClickHouseClient(client=mock_client)
        result = LLMResult(uuid4(), uuid4(), '{"output_text": "hello"}', "complete", "")

        ch.write_results([result])

        assert mock_client.insert.call_count == 1
        assert mock_client.insert.call_args.kwargs["settings"] == {
            "async_insert": 1,
            "wait_for_async_insert": 1,
        }


class TestBatchCounts:
    def test_get_batch_counts(self, ch_client):
        _insert_input(ch_client, BATCH_1, ROW_1)
        _insert_input(ch_client, BATCH_1, ROW_2)
        _insert_input(ch_client, BATCH_1, uuid4())
        _insert_result(ch_client, BATCH_1, ROW_1, status="complete")
        _insert_result(ch_client, BATCH_1, ROW_2, status="failed")

        total_rows, completed_rows, failed_rows = ch_client.get_batch_counts(BATCH_1)

        assert total_rows == 3
        assert completed_rows == 1
        assert failed_rows == 1


class TestWriteBatchStatus:
    def test_write_pending_status(self, ch_client):
        ch_client.write_batch_status(BATCH_1, "pending", total_rows=5)
        rows = ch_client._client.query(
            """SELECT
                batch_id,
                status,
                total_rows,
                completed_rows,
                failed_rows
            FROM llm_batches"""
        ).result_rows
        assert len(rows) == 1
        assert rows[0][0] == BATCH_1
        assert rows[0][1] == "pending"
        assert rows[0][2] == 5
        assert rows[0][3] == 0
        assert rows[0][4] == 0

    def test_write_complete_status(self, ch_client):
        ch_client.write_batch_status(
            BATCH_1, "complete", total_rows=10, completed_rows=8, failed_rows=2
        )
        rows = ch_client._client.query(
            """SELECT
                batch_id,
                status,
                total_rows,
                completed_rows,
                failed_rows
            FROM llm_batches"""
        ).result_rows
        assert len(rows) == 1
        assert rows[0][1] == "complete"
        assert rows[0][2] == 10
        assert rows[0][3] == 8
        assert rows[0][4] == 2

    def test_append_only_multiple_events(self, ch_client):
        ch_client.write_batch_status(BATCH_1, "pending")
        ch_client.write_batch_status(BATCH_1, "complete", completed_rows=5)
        rows = ch_client._client.query(
            "SELECT status FROM llm_batches WHERE batch_id = {batch_id:UUID}",
            parameters={"batch_id": BATCH_1},
        ).result_rows
        assert len(rows) == 2
        statuses = {r[0] for r in rows}
        assert statuses == {"pending", "complete"}

import time
from uuid import UUID

import pytest
from botocore.exceptions import ClientError

from llm_worker.config import BedrockConfig
from llm_worker.executor import BatchExecutor
from llm_worker.main import BatchTurn, LLMWorkerService, WorkerInfo
from llm_worker.providers.bedrock import BedrockProvider
from llm_worker.throttle import ThrottleGate

pytestmark = [pytest.mark.clickhouse]

BEDROCK_CONFIG = BedrockConfig(region="us-east-1")

BATCH_1 = UUID("00000000-0000-0000-0000-000000000001")
BATCH_2 = UUID("00000000-0000-0000-0000-000000000002")
ROW_1 = UUID("00000000-0000-0000-0000-000000000011")
ROW_2 = UUID("00000000-0000-0000-0000-000000000012")
ROW_3 = UUID("00000000-0000-0000-0000-000000000013")


def _text_response(text="ok"):
    return {"output": {"message": {"content": [{"text": text}]}}}


def _insert_input(ch_client, batch_id, row_id, prompt="test prompt"):
    ch_client._client.insert(
        "llm_inputs",
        [[batch_id, row_id, "test-model", prompt, "{}", ""]],
        column_names=[
            "batch_id",
            "row_id",
            "model_id",
            "prompt",
            "params",
            "tool_config",
        ],
    )


def _insert_aged_input(ch_client, batch_id, row_id, minutes_old):
    # created_at comes from ClickHouse's own clock, as it does in production.
    ch_client._client.command(
        """INSERT INTO llm_inputs
            (batch_id, row_id, model_id, prompt, params, tool_config, created_at)
        SELECT {batch_id:UUID}, {row_id:UUID}, 'test-model', 'old prompt', '{}', '',
            now() - toIntervalMinute({minutes:UInt32})""",
        parameters={"batch_id": batch_id, "row_id": row_id, "minutes": minutes_old},
    )


def _throttling_error():
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "Converse",
    )


def _get_results(ch_client, batch_id):
    return ch_client._client.query(
        """SELECT
            batch_id,
            row_id,
            response,
            status,
            error
        FROM llm_results
        WHERE batch_id = {batch_id:UUID}
        ORDER BY row_id""",
        parameters={"batch_id": batch_id},
    ).result_rows


def _get_batch_status(ch_client, batch_id):
    rows = ch_client._client.query(
        """SELECT
            status,
            total_rows,
            completed_rows,
            failed_rows
        FROM llm_batches
        WHERE batch_id = {batch_id:UUID}
        ORDER BY created_at DESC
        LIMIT 1""",
        parameters={"batch_id": batch_id},
    ).result_rows
    return rows[0] if rows else None


def _make_service(ch_client, mock_boto, max_row_age_seconds=600):
    provider = BedrockProvider(BEDROCK_CONFIG, boto_client=mock_boto)
    executor = BatchExecutor(
        provider,
        max_workers=2,
        retry_max_attempts=1,
        retry_max_backoff=0,
        # Short pauses so throttled tests don't wait out the default.
        throttle_gate=ThrottleGate(0.01, 0.01),
        max_row_age_seconds=max_row_age_seconds,
    )
    return LLMWorkerService(
        ch_client,
        executor,
        poll_interval=1,
        worker_info=WorkerInfo("aws", "bedrock", BedrockProvider.capabilities),
    )


ROW_4 = UUID("00000000-0000-0000-0000-000000000014")
ROW_5 = UUID("00000000-0000-0000-0000-000000000015")


class TestProcessSingleBatch:
    def test_end_to_end_success(self, ch_client, mocker):
        _insert_input(ch_client, BATCH_1, ROW_1, "hello")
        _insert_input(ch_client, BATCH_1, ROW_2, "world")

        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("done")
        service = _make_service(ch_client, mock_boto)

        service.process_single_batch(BATCH_1)

        results = _get_results(ch_client, BATCH_1)
        assert len(results) == 2
        assert all(r[3] == "complete" for r in results)

        batch_status = _get_batch_status(ch_client, BATCH_1)
        assert batch_status is not None
        assert batch_status[0] == "complete"
        assert batch_status[1] == 2  # total_rows
        assert batch_status[2] == 2  # completed_rows
        assert batch_status[3] == 0  # failed_rows

    def test_end_to_end_with_failures(self, ch_client, mocker):
        _insert_input(ch_client, BATCH_1, ROW_1)
        _insert_input(ch_client, BATCH_1, ROW_2)

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _text_response("ok")
            raise Exception("bedrock error")

        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = side_effect
        service = _make_service(ch_client, mock_boto)

        service.process_single_batch(BATCH_1)

        results = _get_results(ch_client, BATCH_1)
        assert len(results) == 2
        statuses = {r[3] for r in results}
        assert "complete" in statuses
        assert "failed" in statuses

        batch_status = _get_batch_status(ch_client, BATCH_1)
        assert batch_status is not None
        assert batch_status[0] == "complete"
        assert batch_status[1] == 2  # total_rows
        assert batch_status[2] == 1  # completed_rows
        assert batch_status[3] == 1  # failed_rows

    def test_idempotent_rerun(self, ch_client, mocker):
        _insert_input(ch_client, BATCH_1, ROW_1)
        _insert_input(ch_client, BATCH_1, ROW_2)

        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("done")
        service = _make_service(ch_client, mock_boto)

        service.process_single_batch(BATCH_1)
        assert mock_boto.converse.call_count == 2

        mock_boto.reset_mock()
        service.process_single_batch(BATCH_1)
        assert mock_boto.converse.call_count == 0

        results = _get_results(ch_client, BATCH_1)
        assert len(results) == 2

    def test_resumes_after_partial_completion(self, ch_client, mocker):
        _insert_input(ch_client, BATCH_1, ROW_1)
        _insert_input(ch_client, BATCH_1, ROW_2)
        _insert_input(ch_client, BATCH_1, ROW_3)

        ch_client._client.insert(
            "llm_results",
            [[BATCH_1, ROW_1, '{"output_text": "previous"}', "complete", ""]],
            column_names=["batch_id", "row_id", "response", "status", "error"],
        )

        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("recovered")
        service = _make_service(ch_client, mock_boto)

        service.process_single_batch(BATCH_1)

        assert mock_boto.converse.call_count == 2
        results = _get_results(ch_client, BATCH_1)
        assert len(results) == 3
        batch_status = _get_batch_status(ch_client, BATCH_1)
        assert batch_status is not None
        assert batch_status[0] == "complete"
        assert batch_status[1] == 3  # total_rows
        assert batch_status[2] == 3  # completed_rows
        assert batch_status[3] == 0  # failed_rows

    def test_empty_batch(self, ch_client, mocker):
        mock_boto = mocker.Mock()
        service = _make_service(ch_client, mock_boto)
        nonexistent = UUID("00000000-0000-0000-0000-ffffffffffff")

        service.process_single_batch(nonexistent)

        mock_boto.converse.assert_not_called()
        batch_status = _get_batch_status(ch_client, nonexistent)
        assert batch_status is not None
        assert batch_status[0] == "complete"
        assert batch_status[1] == 0  # total_rows
        assert batch_status[2] == 0  # completed_rows
        assert batch_status[3] == 0  # failed_rows

    def test_does_not_process_other_batches(self, ch_client, mocker):
        _insert_input(ch_client, BATCH_1, ROW_1)
        _insert_input(ch_client, BATCH_2, ROW_1)

        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("done")
        service = _make_service(ch_client, mock_boto)

        service.process_single_batch(BATCH_1)

        assert len(_get_results(ch_client, BATCH_1)) == 1
        assert len(_get_results(ch_client, BATCH_2)) == 0

    def test_pages_through_large_batch(self, ch_client, mocker):
        """Insert 5 rows with page_size=2, verify all are processed across pages."""
        from llm_worker.clickhouse import ClickHouseClient

        small_page_client = ClickHouseClient(
            client=ch_client._client, batch_page_size=2
        )
        for row_id in [ROW_1, ROW_2, ROW_3, ROW_4, ROW_5]:
            _insert_input(ch_client, BATCH_1, row_id)

        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("done")
        service = _make_service(small_page_client, mock_boto)

        service.process_single_batch(BATCH_1)

        results = _get_results(ch_client, BATCH_1)
        assert len(results) == 5
        assert all(r[3] == "complete" for r in results)
        assert mock_boto.converse.call_count == 5

        batch_status = _get_batch_status(ch_client, BATCH_1)
        assert batch_status is not None
        assert batch_status[0] == "complete"
        assert batch_status[1] == 5  # total_rows
        assert batch_status[2] == 5  # completed_rows
        assert batch_status[3] == 0  # failed_rows


class TestThrottledBatches:
    def test_throttled_batch_stays_pending_then_completes(self, ch_client, mocker):
        _insert_input(ch_client, BATCH_1, ROW_1, "hello")
        _insert_input(ch_client, BATCH_1, ROW_2, "world")
        ch_client.write_batch_status(BATCH_1, "pending", total_rows=2)
        throttled = True

        def converse(**kwargs):
            if throttled:
                raise _throttling_error()
            return _text_response("done")

        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = converse
        service = _make_service(ch_client, mock_boto)

        # Throttled: nothing is written and the batch is still pending.
        assert service.process_single_batch(BATCH_1) == BatchTurn(deferred_rows=2)
        assert _get_results(ch_client, BATCH_1) == []
        assert BATCH_1 in ch_client.get_pending_batches()

        # The backend recovers: the next turn completes every row.
        throttled = False
        time.sleep(0.05)  # past the short pause
        assert service.process_single_batch(BATCH_1) == BatchTurn(deferred_rows=0)

        results = _get_results(ch_client, BATCH_1)
        assert [r[3] for r in results] == ["complete", "complete"]
        assert _get_batch_status(ch_client, BATCH_1) == ("complete", 2, 2, 0)
        assert BATCH_1 not in ch_client.get_pending_batches()

    def test_throttled_row_past_the_age_limit_fails(self, ch_client, mocker):
        _insert_aged_input(ch_client, BATCH_1, ROW_1, minutes_old=20)
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _throttling_error()
        service = _make_service(ch_client, mock_boto, max_row_age_seconds=600)

        assert service.process_single_batch(BATCH_1) == BatchTurn(deferred_rows=0)

        results = _get_results(ch_client, BATCH_1)
        assert len(results) == 1
        assert results[0][3] == "failed"
        assert results[0][4] == "throttled and older than 600 seconds"
        assert _get_batch_status(ch_client, BATCH_1) == ("complete", 1, 0, 1)

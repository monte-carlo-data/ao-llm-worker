import signal
import time
from unittest.mock import ANY, MagicMock
from uuid import UUID

import pytest
from clickhouse_connect.driver.exceptions import ClickHouseError

from llm_worker.clickhouse import LLMResult
from llm_worker.main import LLMWorkerService, run

BATCH_1 = UUID("00000000-0000-0000-0000-000000000001")
BATCH_2 = UUID("00000000-0000-0000-0000-000000000002")
BATCH_3 = UUID("00000000-0000-0000-0000-000000000003")


def _make_service(ch_mock=None, bedrock_mock=None, poll_interval: float = 1):
    ch = ch_mock or MagicMock()
    bedrock = bedrock_mock or MagicMock()
    if not hasattr(bedrock, "process_batch_iter"):
        bedrock.process_batch_iter = MagicMock(return_value=iter(()))
    return LLMWorkerService(ch, bedrock, poll_interval)


class TestSignalHandling:
    def test_initial_state(self):
        service = _make_service()
        assert service._should_stop is False

    def test_stop(self):
        service = _make_service()
        service.stop()
        assert service._should_stop is True

    def test_signal_handler(self):
        service = _make_service()
        service._handle_signal(signal.SIGTERM, None)
        assert service._should_stop is True


class TestPollingLoop:
    def test_processes_pending_batch(self):
        ch = MagicMock()
        call_count = 0

        def get_pending():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [BATCH_1]
            signal.raise_signal(signal.SIGTERM)
            return []

        ch.get_pending_batches.side_effect = get_pending
        ch.get_batch_rows.return_value = []
        ch.get_batch_counts.return_value = (0, 0, 0)
        service = _make_service(ch_mock=ch)

        service.run()

        assert ch.get_batch_rows.call_count == 1
        assert ch.get_batch_rows.call_args[0][0] == BATCH_1

    def test_sleeps_when_no_batches(self):
        ch = MagicMock()
        call_count = 0

        def get_pending():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                signal.raise_signal(signal.SIGTERM)
            return []

        ch.get_pending_batches.side_effect = get_pending
        service = _make_service(ch_mock=ch, poll_interval=0.1)

        start = time.time()
        service.run()
        elapsed = time.time() - start

        assert elapsed >= 0.15

    def test_processes_multiple_batches(self):
        ch = MagicMock()
        call_count = 0

        def get_pending():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [BATCH_1, BATCH_2]
            signal.raise_signal(signal.SIGTERM)
            return []

        ch.get_pending_batches.side_effect = get_pending
        ch.get_batch_rows.return_value = []
        ch.get_batch_counts.return_value = (0, 0, 0)
        service = _make_service(ch_mock=ch)

        service.run()

        assert ch.get_batch_rows.call_count == 2
        batch_ids = [call[0][0] for call in ch.get_batch_rows.call_args_list]
        assert batch_ids == [BATCH_1, BATCH_2]

    def test_continues_after_error(self):
        ch = MagicMock()
        call_count = 0

        def get_pending():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ClickHouseError("connection lost")
            if call_count >= 3:
                signal.raise_signal(signal.SIGTERM)
            return []

        ch.get_pending_batches.side_effect = get_pending
        service = _make_service(ch_mock=ch, poll_interval=0.1)

        service.run()

        assert call_count >= 3

    def test_reraises_unexpected_errors(self):
        ch = MagicMock()
        ch.get_pending_batches.side_effect = RuntimeError("bug")
        service = _make_service(ch_mock=ch, poll_interval=0.1)

        with pytest.raises(RuntimeError, match="bug"):
            service.run()

    def test_continues_after_error_during_batch(self):
        ch = MagicMock()
        bedrock = MagicMock()
        call_count = 0

        def get_pending():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [BATCH_1]
            if call_count >= 3:
                signal.raise_signal(signal.SIGTERM)
            return []

        ch.get_pending_batches.side_effect = get_pending
        ch.get_batch_rows.side_effect = ClickHouseError("connection lost")
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock, poll_interval=0.1)

        service.run()

        assert call_count >= 3

    def test_sigterm_stops_between_batches(self):
        ch = MagicMock()
        ch.get_pending_batches.return_value = [BATCH_1, BATCH_2, BATCH_3]

        def get_rows(batch_id):
            signal.raise_signal(signal.SIGTERM)
            return []

        ch.get_batch_rows.side_effect = get_rows
        ch.get_batch_counts.return_value = (0, 0, 0)
        service = _make_service(ch_mock=ch)

        service.run()

        assert ch.get_batch_rows.call_count == 1


class TestBatchWrites:
    def test_process_single_batch_writes_completed_results(self):
        ch = MagicMock()
        bedrock = MagicMock()
        result = LLMResult(BATCH_1, UUID(int=1), '{"output_text":"ok"}', "complete", "")
        ch.get_batch_rows.side_effect = [[MagicMock()], []]
        ch.get_batch_counts.return_value = (1, 1, 0)
        bedrock.process_batch_iter.return_value = iter([result])
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock)

        service.process_single_batch(BATCH_1)

        ch.write_results.assert_called_once_with([result])

    def test_retries_result_write_on_clickhouse_error(self):
        ch = MagicMock()
        bedrock = MagicMock()
        result = LLMResult(BATCH_1, UUID(int=1), '{"output_text":"ok"}', "complete", "")
        ch.get_batch_rows.side_effect = [[MagicMock()], []]
        ch.get_batch_counts.return_value = (1, 1, 0)
        bedrock.process_batch_iter.return_value = iter([result])
        ch.write_results.side_effect = [ClickHouseError("connection lost"), None]
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock, poll_interval=0.01)

        service.process_single_batch(BATCH_1)

        assert ch.write_results.call_count == 2

    def test_no_retry_on_non_clickhouse_error(self):
        ch = MagicMock()
        bedrock = MagicMock()
        result = LLMResult(BATCH_1, UUID(int=1), '{"output_text":"ok"}', "complete", "")
        ch.get_batch_rows.side_effect = [[MagicMock()], []]
        ch.get_batch_counts.return_value = (1, 1, 0)
        bedrock.process_batch_iter.return_value = iter([result])
        ch.write_results.side_effect = RuntimeError("bug")
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock, poll_interval=0.01)

        with pytest.raises(RuntimeError, match="bug"):
            service.process_single_batch(BATCH_1)

        assert ch.write_results.call_count == 1


class TestBatchPaging:
    def test_processes_multiple_pages(self):
        ch = MagicMock()
        bedrock = MagicMock()
        page1_rows = [MagicMock(), MagicMock()]
        page2_rows = [MagicMock()]
        ch.get_batch_rows.side_effect = [page1_rows, page2_rows, []]

        page1_results = [
            LLMResult(BATCH_1, UUID(int=1), '{"output_text":"a"}', "complete", ""),
            LLMResult(BATCH_1, UUID(int=2), '{"output_text":"b"}', "complete", ""),
        ]
        page2_results = [
            LLMResult(BATCH_1, UUID(int=3), '{"output_text":"c"}', "failed", "err"),
        ]
        bedrock.process_batch_iter.side_effect = [
            iter(page1_results),
            iter(page2_results),
        ]
        ch.get_batch_counts.return_value = (3, 2, 1)
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock)

        service.process_single_batch(BATCH_1)

        assert ch.get_batch_rows.call_count == 3
        assert bedrock.process_batch_iter.call_count == 2
        assert ch.write_results.call_count == 2

    def test_empty_batch_does_not_call_bedrock(self):
        ch = MagicMock()
        bedrock = MagicMock()
        ch.get_batch_rows.return_value = []
        ch.get_batch_counts.return_value = (0, 0, 0)
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock)

        service.process_single_batch(BATCH_1)

        bedrock.process_batch_iter.assert_not_called()
        ch.write_results.assert_not_called()

    def test_single_page_batch(self):
        ch = MagicMock()
        bedrock = MagicMock()
        ch.get_batch_rows.side_effect = [[MagicMock()], []]
        result = LLMResult(BATCH_1, UUID(int=1), '{"output_text":"ok"}', "complete", "")
        ch.get_batch_counts.return_value = (1, 1, 0)
        bedrock.process_batch_iter.return_value = iter([result])
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock)

        service.process_single_batch(BATCH_1)

        assert ch.get_batch_rows.call_count == 2
        assert bedrock.process_batch_iter.call_count == 1


class TestBatchStatusWrite:
    def test_writes_complete_status_all_success(self):
        ch = MagicMock()
        bedrock = MagicMock()
        results = [
            LLMResult(BATCH_1, UUID(int=1), '{"output_text":"a"}', "complete", ""),
            LLMResult(BATCH_1, UUID(int=2), '{"output_text":"b"}', "complete", ""),
        ]
        ch.get_batch_rows.side_effect = [[MagicMock(), MagicMock()], []]
        ch.get_batch_counts.return_value = (2, 2, 0)
        bedrock.process_batch_iter.return_value = iter(results)
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock)

        service.process_single_batch(BATCH_1)

        ch.write_batch_status.assert_called_once_with(BATCH_1, "complete", 2, 2, 0)

    def test_writes_complete_status_with_failures(self):
        ch = MagicMock()
        bedrock = MagicMock()
        results = [
            LLMResult(BATCH_1, UUID(int=1), '{"output_text":"a"}', "complete", ""),
            LLMResult(BATCH_1, UUID(int=2), "", "failed", "err"),
            LLMResult(BATCH_1, UUID(int=3), '{"output_text":"c"}', "complete", ""),
        ]
        ch.get_batch_rows.side_effect = [[MagicMock(), MagicMock(), MagicMock()], []]
        ch.get_batch_counts.return_value = (3, 2, 1)
        bedrock.process_batch_iter.return_value = iter(results)
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock)

        service.process_single_batch(BATCH_1)

        ch.write_batch_status.assert_called_once_with(BATCH_1, "complete", 3, 2, 1)

    def test_empty_batch_with_no_results_writes_complete_status(self):
        ch = MagicMock()
        bedrock = MagicMock()
        ch.get_batch_rows.return_value = []
        ch.get_batch_counts.return_value = (0, 0, 0)
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock)

        service.process_single_batch(BATCH_1)

        ch.write_batch_status.assert_called_once_with(BATCH_1, "complete", 0, 0, 0)

    def test_recovers_batch_with_all_results_already_written(self):
        ch = MagicMock()
        bedrock = MagicMock()
        ch.get_batch_rows.return_value = []
        ch.get_batch_counts.return_value = (3, 2, 1)
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock)

        service.process_single_batch(BATCH_1)

        bedrock.process_batch_iter.assert_not_called()
        ch.write_batch_status.assert_called_once_with(BATCH_1, "complete", 3, 2, 1)

    def test_retries_on_clickhouse_error(self):
        ch = MagicMock()
        bedrock = MagicMock()
        result = LLMResult(BATCH_1, UUID(int=1), '{"output_text":"ok"}', "complete", "")
        ch.get_batch_rows.side_effect = [[MagicMock()], []]
        ch.get_batch_counts.return_value = (1, 1, 0)
        bedrock.process_batch_iter.return_value = iter([result])
        ch.write_batch_status.side_effect = [
            ClickHouseError("connection lost"),
            None,
        ]
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock, poll_interval=0.01)

        service.process_single_batch(BATCH_1)

        assert ch.write_batch_status.call_count == 2

    def test_propagates_non_clickhouse_error(self):
        ch = MagicMock()
        bedrock = MagicMock()
        result = LLMResult(BATCH_1, UUID(int=1), '{"output_text":"ok"}', "complete", "")
        ch.get_batch_rows.side_effect = [[MagicMock()], []]
        ch.get_batch_counts.return_value = (1, 1, 0)
        bedrock.process_batch_iter.return_value = iter([result])
        ch.write_batch_status.side_effect = RuntimeError("bug")
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock, poll_interval=0.01)

        with pytest.raises(RuntimeError, match="bug"):
            service.process_single_batch(BATCH_1)

    def test_multi_page_writes_single_status(self):
        ch = MagicMock()
        bedrock = MagicMock()
        page1_results = [
            LLMResult(BATCH_1, UUID(int=1), '{"output_text":"a"}', "complete", ""),
            LLMResult(BATCH_1, UUID(int=2), '{"output_text":"b"}', "complete", ""),
        ]
        page2_results = [
            LLMResult(BATCH_1, UUID(int=3), "", "failed", "err"),
        ]
        ch.get_batch_rows.side_effect = [[MagicMock(), MagicMock()], [MagicMock()], []]
        bedrock.process_batch_iter.side_effect = [
            iter(page1_results),
            iter(page2_results),
        ]
        ch.get_batch_counts.return_value = (3, 2, 1)
        service = _make_service(ch_mock=ch, bedrock_mock=bedrock)

        service.process_single_batch(BATCH_1)

        ch.write_batch_status.assert_called_once_with(BATCH_1, "complete", 3, 2, 1)


class TestRunStartup:
    def test_run_survives_non_clickhouse_error_from_write_worker_info(self, mocker):
        mock_config = mocker.MagicMock()
        mocker.patch("llm_worker.main.load_config", return_value=mock_config)

        mock_ch = mocker.MagicMock()
        mock_ch.write_worker_info.side_effect = RuntimeError("boom")
        mocker.patch("llm_worker.main.ClickHouseClient", return_value=mock_ch)

        mocker.patch("llm_worker.main.create_provider", return_value=mocker.MagicMock())

        mock_service = mocker.MagicMock()
        mock_service_cls = mocker.patch(
            "llm_worker.main.LLMWorkerService", return_value=mock_service
        )

        # write_worker_info is best-effort: a non-ClickHouseError bug in it must
        # not prevent the worker from starting the actual service loop.
        run()

        mock_service_cls.assert_called_once_with(
            mock_ch, ANY, mock_config.poll_interval
        )
        mock_service.run.assert_called_once()

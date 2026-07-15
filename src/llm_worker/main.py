"""Entry point and polling loop."""

import logging
import signal
import time
from collections.abc import Iterable
from types import FrameType
from uuid import UUID

from clickhouse_connect.driver.exceptions import ClickHouseError
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_worker.clickhouse import ClickHouseClient, LLMResult
from llm_worker.config import load_config
from llm_worker.executor import BatchExecutor
from llm_worker.providers import create_provider
from llm_worker.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

RESULT_WRITE_BATCH_SIZE = 50
RESULT_WRITE_MAX_ATTEMPTS = 3
BATCH_STATUS_WRITE_MAX_ATTEMPTS = 3


class LLMWorkerService:
    def __init__(
        self, ch: ClickHouseClient, executor: BatchExecutor, poll_interval: float
    ):
        self._ch = ch
        self._executor = executor
        self._poll_interval = poll_interval
        self._should_stop = False

    def run(self) -> None:
        self._install_signal_handlers()
        logger.info("worker_started", extra={"poll_interval": self._poll_interval})

        while not self._should_stop:
            try:
                batch_ids = self._ch.get_pending_batches()
                if batch_ids:
                    for batch_id in batch_ids:
                        if self._should_stop:
                            break
                        self.process_single_batch(batch_id)
                else:
                    time.sleep(self._poll_interval)
            except ClickHouseError as exc:
                logger.error("polling_error", extra={"error": str(exc)})
                if not self._should_stop:
                    time.sleep(self._poll_interval)

        logger.info("worker_stopped")

    def process_single_batch(self, batch_id: UUID) -> None:
        saw_rows = False

        while True:
            rows = self._ch.get_batch_rows(batch_id)
            if not rows:
                break

            if not saw_rows:
                logger.info("batch_started", extra={"batch_id": str(batch_id)})
                saw_rows = True

            logger.info(
                "batch_page_fetched",
                extra={"batch_id": str(batch_id), "rows": len(rows)},
            )
            self._write_results_in_chunks(self._executor.process_batch_iter(rows))

        if not saw_rows:
            total_rows, total_completed, total_failed = self._ch.get_batch_counts(
                batch_id
            )
            if total_completed + total_failed == 0:
                logger.info("batch_empty", extra={"batch_id": str(batch_id)})
            else:
                logger.info(
                    "batch_recovering",
                    extra={
                        "batch_id": str(batch_id),
                        "completed": total_completed,
                        "failed": total_failed,
                    },
                )
        else:
            total_rows, total_completed, total_failed = self._ch.get_batch_counts(
                batch_id
            )
            logger.info(
                "batch_completed",
                extra={
                    "batch_id": str(batch_id),
                    "completed": total_completed,
                    "failed": total_failed,
                },
            )

        self._write_batch_status_with_retry(
            batch_id, "complete", total_rows, total_completed, total_failed
        )
        logger.info(
            "batch_status_written",
            extra={"batch_id": str(batch_id), "status": "complete"},
        )

    def _write_results_in_chunks(self, results: Iterable[LLMResult]) -> None:
        pending: list[LLMResult] = []

        for result in results:
            pending.append(result)
            if len(pending) >= RESULT_WRITE_BATCH_SIZE:
                self._write_results_with_retry(pending)
                pending = []

        if pending:
            self._write_results_with_retry(pending)

    def _write_results_with_retry(self, results: list[LLMResult]) -> None:
        retrying = Retrying(
            retry=retry_if_exception_type(ClickHouseError),
            stop=stop_after_attempt(RESULT_WRITE_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=1, max=self._poll_interval),
            reraise=True,
            before_sleep=lambda state: logger.warning(
                "results_write_retry",
                extra={
                    "attempt": state.attempt_number,
                    "max_attempts": RESULT_WRITE_MAX_ATTEMPTS,
                    "results": len(results),
                    "error": str(state.outcome.exception())
                    if state.outcome
                    else "unknown",
                },
            ),
        )
        for attempt in retrying:
            with attempt:
                self._ch.write_results(results)

    def _write_batch_status_with_retry(
        self,
        batch_id: UUID,
        status: str,
        total_rows: int,
        completed_rows: int,
        failed_rows: int,
    ) -> None:
        retrying = Retrying(
            retry=retry_if_exception_type(ClickHouseError),
            stop=stop_after_attempt(BATCH_STATUS_WRITE_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=1, max=self._poll_interval),
            reraise=True,
            before_sleep=lambda state: logger.warning(
                "batch_status_write_retry",
                extra={
                    "batch_id": str(batch_id),
                    "attempt": state.attempt_number,
                    "max_attempts": BATCH_STATUS_WRITE_MAX_ATTEMPTS,
                    "error": str(state.outcome.exception())
                    if state.outcome
                    else "unknown",
                },
            ),
        )
        for attempt in retrying:
            with attempt:
                self._ch.write_batch_status(
                    batch_id, status, total_rows, completed_rows, failed_rows
                )

    def stop(self) -> None:
        self._should_stop = True

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("signal_received", extra={"signal": sig_name})
        self._should_stop = True


def run():
    config = load_config()
    ch = ClickHouseClient(
        config.clickhouse,
        pending_batch_limit=config.pending_batch_limit,
        batch_page_size=config.batch_page_size,
    )
    provider = create_provider(config)
    executor = BatchExecutor(
        provider,
        config.max_workers,
        config.retry_max_attempts,
        config.retry_max_backoff,
    )
    # Best-effort: publish this worker's cloud/provider for the monolith's
    # cloud-native model resolution. Non-critical metadata — a failure here
    # (e.g. the llm_worker_info table not yet migrated) must not stop batch
    # processing, so log and continue.
    try:
        ch.write_worker_info(config.cloud, config.provider)
    except ClickHouseError as exc:
        logger.warning("worker_info_write_failed", extra={"error": str(exc)})

    service = LLMWorkerService(ch, executor, config.poll_interval)
    service.run()


if __name__ == "__main__":
    run()

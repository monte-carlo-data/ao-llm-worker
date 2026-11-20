"""Entry point and polling loop."""

import logging
import os
import signal
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
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
from llm_worker.executor import BatchExecutor, Deferred, RowOutcome
from llm_worker.providers import create_provider, load_provider_config
from llm_worker.logging_setup import configure_logging
from llm_worker.throttle import ThrottleGate

configure_logging()
logger = logging.getLogger(__name__)

RESULT_WRITE_BATCH_SIZE = 50
RESULT_WRITE_MAX_ATTEMPTS = 3
BATCH_STATUS_WRITE_MAX_ATTEMPTS = 3
# How often the worker republishes its cloud, provider and capabilities. The chart's
# schema Job and the worker roll out together, so the capabilities column can appear
# after startup; republishing picks it up without a restart.
WORKER_INFO_REPUBLISH_SECONDS = 600
# A failed write, such as at startup before ClickHouse is reachable, retries sooner.
WORKER_INFO_RETRY_SECONDS = 30


class TurnStatus(Enum):
    """How a turn on a batch ended."""

    # Every row has a result and the batch is marked complete.
    COMPLETED = "completed"
    # The backend throttled some rows; the batch stays pending for a later turn.
    DEFERRED = "deferred"
    # The worker is stopping; the batch stays pending for the next worker.
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class BatchTurn:
    """What one turn on a batch did, for the caller's scheduling.

    ``deferred_rows`` counts rows left without a result in this turn (throttled,
    or the worker stopped); the batch stays pending and a later turn retries
    them.
    """

    status: TurnStatus
    deferred_rows: int = 0


@dataclass(frozen=True)
class WorkerInfo:
    """What the worker publishes to llm_worker_info."""

    cloud: str
    provider: str
    capabilities: tuple[str, ...]


class LLMWorkerService:
    def __init__(
        self,
        ch: ClickHouseClient,
        executor: BatchExecutor,
        poll_interval: float,
        *,
        worker_info: WorkerInfo,
        stop_event: threading.Event,
    ):
        self._ch = ch
        self._executor = executor
        self._poll_interval = poll_interval
        # Set on SIGTERM/SIGINT or stop(). An Event rather than a bool so threads
        # waiting out a throttle pause can share it and wake up promptly.
        self._stop_event = stop_event
        self._worker_info = worker_info
        self._worker_info_next_at: float | None = None

    @property
    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        self._install_signal_handlers()
        logger.info("worker_started", extra={"poll_interval": self._poll_interval})

        while not self._should_stop:
            self._maybe_publish_worker_info()
            try:
                progressed = False
                for batch_id in self._ch.get_pending_batches():
                    if self._should_stop:
                        break
                    turn = self.process_single_batch(batch_id)
                    if turn.status is TurnStatus.COMPLETED:
                        progressed = True
                # Wait when there was nothing to do, or when every batch only
                # deferred rows: polling again at once would refetch rows the
                # backend is still throttling.
                if not progressed:
                    self._idle_wait()
            except ClickHouseError as exc:
                logger.error("polling_error", extra={"error": str(exc)})
                self._idle_wait()

        logger.info("worker_stopped")

    def _idle_wait(self) -> None:
        """Wait one poll interval, returning early if the worker is stopping."""
        self._stop_event.wait(self._poll_interval)

    def _maybe_publish_worker_info(self) -> None:
        """Publish the worker info at startup, then every WORKER_INFO_REPUBLISH_SECONDS.

        Called from the poll loop and between pages of a batch, so a long backlog
        can't delay it. Best-effort: a failure is logged, retried after
        WORKER_INFO_RETRY_SECONDS, and never stops batch processing. A stopping
        worker doesn't publish.
        """
        # After a rollback, the replacement worker's startup row must stay the
        # newest, so a worker finishing its page during shutdown doesn't write.
        if self._should_stop:
            return
        now = time.monotonic()
        if self._worker_info_next_at is not None and now < self._worker_info_next_at:
            return
        info = self._worker_info
        try:
            self._ch.write_worker_info(info.cloud, info.provider, info.capabilities)
        except Exception as exc:  # noqa: BLE001 — best-effort metadata write
            logger.warning("worker_info_write_failed", extra={"error": str(exc)})
            self._worker_info_next_at = now + WORKER_INFO_RETRY_SECONDS
            return
        self._worker_info_next_at = now + WORKER_INFO_REPUBLISH_SECONDS

    def process_single_batch(self, batch_id: UUID) -> BatchTurn:
        saw_rows = False
        deferred_rows = 0

        while True:
            # A stopping worker doesn't start another page. The batch keeps its
            # remaining rows and stays pending for the next worker.
            if self._should_stop:
                return self._interrupted(batch_id, deferred_rows)

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
            deferred_rows += self._write_results_in_chunks(
                self._executor.process_batch_iter(rows)
            )
            self._maybe_publish_worker_info()
            # Deferred rows have no result, so the next page would fetch them
            # again straight away. End the turn and let a later poll retry them.
            if deferred_rows:
                break

        if deferred_rows:
            # Rows deferred because the worker is stopping, possibly alongside
            # throttled ones: the batch is left for the next worker.
            if self._should_stop:
                return self._interrupted(batch_id, deferred_rows)
            total_rows, total_completed, total_failed = self._ch.get_batch_counts(
                batch_id
            )
            logger.info(
                "batch_deferred",
                extra={
                    "batch_id": str(batch_id),
                    "deferred_rows": deferred_rows,
                    "completed": total_completed,
                    "failed": total_failed,
                },
            )
            return BatchTurn(TurnStatus.DEFERRED, deferred_rows)

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
        return BatchTurn(TurnStatus.COMPLETED)

    @staticmethod
    def _interrupted(batch_id: UUID, deferred_rows: int) -> BatchTurn:
        logger.info(
            "batch_interrupted",
            extra={"batch_id": str(batch_id), "deferred_rows": deferred_rows},
        )
        return BatchTurn(TurnStatus.INTERRUPTED, deferred_rows)

    def _write_results_in_chunks(self, results: Iterable[RowOutcome]) -> int:
        """Write results in chunks and return how many rows were deferred."""
        pending: list[LLMResult] = []
        deferred = 0

        for result in results:
            if isinstance(result, Deferred):
                deferred += 1
                continue
            pending.append(result)
            if len(pending) >= RESULT_WRITE_BATCH_SIZE:
                self._write_results_with_retry(pending)
                pending = []

        if pending:
            self._write_results_with_retry(pending)
        return deferred

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
        self._stop_event.set()

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("signal_received", extra={"signal": sig_name})
        self._stop_event.set()


def run():
    config = load_config()
    provider_config = load_provider_config(os.environ.get("LLM_PROVIDER", "bedrock"))
    ch = ClickHouseClient(
        config.clickhouse,
        pending_batch_limit=config.pending_batch_limit,
        batch_page_size=config.batch_page_size,
    )
    provider = create_provider(provider_config, max_concurrency=config.max_workers)
    # One stop signal for the whole worker: the service sets it on SIGTERM/SIGINT.
    stop_event = threading.Event()
    executor = BatchExecutor(
        provider,
        config.max_workers,
        config.retry_max_attempts,
        config.retry_max_backoff,
        throttle_gate=ThrottleGate(
            config.throttle_pause_min_seconds, config.throttle_pause_max_seconds
        ),
        stop_event=stop_event,
        max_row_age_seconds=config.throttle_max_row_age_seconds,
    )
    # The service publishes the worker info at startup and then periodically.
    service = LLMWorkerService(
        ch,
        executor,
        config.poll_interval,
        worker_info=WorkerInfo(
            cloud=provider_config.cloud,
            provider=provider_config.provider,
            capabilities=provider.capabilities,
        ),
        stop_event=stop_event,
    )
    service.run()


if __name__ == "__main__":
    run()

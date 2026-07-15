"""Provider-agnostic batch execution: concurrency, retry, abort, result mapping.

Owns everything that is not backend-specific — the thread pool, per-call retry,
batch-abort handling, JSON parsing of the input row, and mapping a provider
response (or error) into an ``LLMResult``. The backend itself is an injected
:class:`~llm_worker.providers.base.LLMProvider`.
"""

import json
import logging
import threading
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from llm_worker.clickhouse import LLMInput, LLMResult
from llm_worker.contract import ContractRequest, build_request
from llm_worker.providers.base import ErrorDisposition, LLMProvider

logger = logging.getLogger(__name__)


class BatchExecutor:
    def __init__(
        self,
        provider: LLMProvider,
        max_workers: int,
        retry_max_attempts: int,
        retry_max_backoff: int,
    ):
        self._provider = provider
        self._max_workers = max_workers
        self._retry_max_attempts = retry_max_attempts
        self._retry_max_backoff = retry_max_backoff

    def _complete_with_retry(self, request: ContractRequest):
        retrying = Retrying(
            retry=retry_if_exception(
                lambda exc: self._provider.classify_error(exc) == ErrorDisposition.RETRY
            ),
            stop=stop_after_attempt(self._retry_max_attempts),
            wait=wait_exponential(multiplier=1, max=self._retry_max_backoff),
            reraise=True,
            before_sleep=lambda state: logger.warning(
                "llm_retry",
                extra={
                    "attempt": state.attempt_number,
                    "model_id": request.model_id,
                    "error": str(state.outcome.exception())
                    if state.outcome
                    else "unknown",
                },
            ),
        )
        for attempt in retrying:
            with attempt:
                return self._provider.complete(request)
        raise RuntimeError("Retrying exhausted without result")

    @staticmethod
    def _aborted_result(input_row: LLMInput) -> LLMResult:
        return LLMResult(
            batch_id=input_row.batch_id,
            row_id=input_row.row_id,
            response="",
            status="failed",
            error="Batch aborted",
        )

    def invoke(
        self,
        input_row: LLMInput,
        abort_event: threading.Event | None = None,
    ) -> LLMResult:
        if abort_event and abort_event.is_set():
            logger.info(
                "row_skipped_abort",
                extra={
                    "row_id": str(input_row.row_id),
                    "batch_id": str(input_row.batch_id),
                },
            )
            return self._aborted_result(input_row)

        try:
            params = json.loads(input_row.params) if input_row.params else {}
            tool_config = (
                json.loads(input_row.tool_config) if input_row.tool_config else {}
            )
        except json.JSONDecodeError as e:
            logger.warning(
                "row_invalid_json",
                extra={"row_id": str(input_row.row_id), "error": str(e)},
            )
            return LLMResult(
                batch_id=input_row.batch_id,
                row_id=input_row.row_id,
                response="",
                status="failed",
                error=f"Invalid JSON in params or tool_config: {e}",
            )

        request = build_request(
            input_row.model_id, input_row.prompt, params, tool_config
        )

        try:
            response = self._complete_with_retry(request)
            logger.info(
                "row_complete",
                extra={
                    "row_id": str(input_row.row_id),
                    "batch_id": str(input_row.batch_id),
                    "model_id": input_row.model_id,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
            return LLMResult(
                batch_id=input_row.batch_id,
                row_id=input_row.row_id,
                response=json.dumps(response.output),
                status="complete",
                error="",
            )
        except Exception as e:
            disposition = self._provider.classify_error(e)
            if disposition == ErrorDisposition.ABORT_BATCH and abort_event:
                logger.error(
                    "batch_aborted",
                    extra={
                        "row_id": str(input_row.row_id),
                        "batch_id": str(input_row.batch_id),
                        "model_id": input_row.model_id,
                        "error": str(e),
                    },
                )
                abort_event.set()
            else:
                logger.error(
                    "row_failed",
                    extra={"row_id": str(input_row.row_id), "error": str(e)},
                )
            return LLMResult(
                batch_id=input_row.batch_id,
                row_id=input_row.row_id,
                response="",
                status="failed",
                error=str(e),
            )

    def process_batch_iter(self, inputs: list[LLMInput]) -> Iterator[LLMResult]:
        abort_event = threading.Event()
        pending_inputs = iter(inputs)
        aborted_rows: list[LLMInput] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            in_flight: set[Future[LLMResult]] = set()

            while len(in_flight) < self._max_workers:
                try:
                    row = next(pending_inputs)
                except StopIteration:
                    break
                in_flight.add(executor.submit(self.invoke, row, abort_event))

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()

                if abort_event.is_set():
                    aborted_rows.extend(list(pending_inputs))
                    break

                while len(in_flight) < self._max_workers:
                    try:
                        row = next(pending_inputs)
                    except StopIteration:
                        break
                    in_flight.add(executor.submit(self.invoke, row, abort_event))

            for future in in_flight:
                future.cancel()

            for future in in_flight:
                if future.cancelled():
                    continue
                yield future.result()

        for row in aborted_rows:
            logger.info(
                "row_skipped_unsubmitted",
                extra={"row_id": str(row.row_id), "batch_id": str(row.batch_id)},
            )
            yield self._aborted_result(row)

    def process_batch(self, inputs: list[LLMInput]) -> list[LLMResult]:
        return list(self.process_batch_iter(inputs))

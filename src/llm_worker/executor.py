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
from dataclasses import dataclass
from uuid import UUID

from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from llm_worker.clickhouse import LLMInput, LLMResult
from llm_worker.contract import ContractRequest, build_request
from llm_worker.providers.base import ErrorDisposition, LLMProvider
from llm_worker.throttle import ThrottleGate

logger = logging.getLogger(__name__)

# Used when no gate is injected; main.run() passes one built from settings.
DEFAULT_THROTTLE_PAUSE_MIN_SECONDS = 1.0
DEFAULT_THROTTLE_PAUSE_MAX_SECONDS = 60.0
DEFAULT_THROTTLE_MAX_ROW_AGE_SECONDS = 600

# The main thread waits on row futures in slices of this length. Python runs
# signal handlers on the main thread, and an untimed wait may not wake for a
# signal on every platform, so SIGTERM could otherwise go unhandled until a row
# finishes, which can be a whole throttle pause.
_FUTURES_WAIT_SLICE_SECONDS = 0.5


class _Stopped(Exception):
    """Raised inside a call attempt when the worker is stopping."""


def _error_code(exc: BaseException) -> str:
    """A short, content-free name for an error, for logs: the botocore error
    code when there is one, otherwise the exception class name."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code:
            return str(code)
    return type(exc).__name__


@dataclass(frozen=True)
class Deferred:
    """A row the backend throttled. Nothing is written for it, so the batch
    stays pending and a later turn fetches the row again."""

    batch_id: UUID
    row_id: UUID
    model_id: str


# What the executor yields for each row: a result to write, or a deferral.
RowOutcome = LLMResult | Deferred


class BatchExecutor:
    def __init__(
        self,
        provider: LLMProvider,
        max_workers: int,
        retry_max_attempts: int,
        retry_max_backoff: int,
        *,
        throttle_gate: ThrottleGate | None = None,
        stop_event: threading.Event | None = None,
        max_row_age_seconds: int = DEFAULT_THROTTLE_MAX_ROW_AGE_SECONDS,
    ):
        self._provider = provider
        # A throttled row this old (seconds since insert) fails instead of being
        # deferred again, so its batch can't stay pending indefinitely.
        self._max_row_age_seconds = max_row_age_seconds
        self._max_workers = max_workers
        self._retry_max_attempts = retry_max_attempts
        self._retry_max_backoff = retry_max_backoff
        self._gate = throttle_gate or ThrottleGate(
            DEFAULT_THROTTLE_PAUSE_MIN_SECONDS, DEFAULT_THROTTLE_PAUSE_MAX_SECONDS
        )
        # Set when the worker is stopping; wakes threads waiting out a pause or a
        # retry backoff.
        self._stop_event = stop_event or threading.Event()

    def _interruptible_sleep(self, seconds: float) -> None:
        self._stop_event.wait(seconds)

    def _call_through_gate(self, request: ContractRequest):
        """One call attempt: wait out the model's pause, call the provider, and
        report the outcome to the gate."""
        model_id = request.model_id
        # A stopping worker starts no new calls; the row is left for the next one.
        if self._stop_event.is_set() or not self._gate.wait(model_id, self._stop_event):
            raise _Stopped()
        started_at = self._gate.now()
        try:
            response = self._provider.complete(request)
        except Exception as exc:
            if self._provider.classify_error(exc) == ErrorDisposition.THROTTLED:
                pause = self._gate.on_throttle(model_id, started_at)
                if pause is not None:
                    logger.warning(
                        "llm_throttle_pause",
                        extra={
                            "model_id": model_id,
                            "pause_seconds": pause,
                            "error_code": _error_code(exc),
                        },
                    )
            raise
        self._gate.on_success(model_id, started_at)
        return response

    def _complete_with_retry(self, request: ContractRequest):
        retrying = Retrying(
            # Throttles are not retried here: they defer the row instead, so they
            # never use up its retry budget.
            retry=retry_if_exception(
                lambda exc: (
                    not isinstance(exc, _Stopped)
                    and self._provider.classify_error(exc) == ErrorDisposition.RETRY
                )
            ),
            stop=stop_after_attempt(self._retry_max_attempts),
            wait=wait_exponential(multiplier=1, max=self._retry_max_backoff),
            # Interruptible, so a stopping worker doesn't sit out a backoff.
            sleep=self._interruptible_sleep,
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
                return self._call_through_gate(request)
        raise RuntimeError("Retrying exhausted without result")

    @staticmethod
    def _deferred(input_row: LLMInput, reason: str) -> Deferred:
        logger.info(
            "row_deferred",
            extra={
                "row_id": str(input_row.row_id),
                "batch_id": str(input_row.batch_id),
                "model_id": input_row.model_id,
                "reason": reason,
                "age_seconds": input_row.age_seconds,
            },
        )
        return Deferred(input_row.batch_id, input_row.row_id, input_row.model_id)

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
    ) -> RowOutcome:
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
            request = build_request(
                input_row.model_id, input_row.prompt, params, tool_config
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
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            logger.warning(
                "row_invalid_contract",
                extra={"row_id": str(input_row.row_id), "error": str(e)},
            )
            return LLMResult(
                batch_id=input_row.batch_id,
                row_id=input_row.row_id,
                response="",
                status="failed",
                error=f"Malformed params or tool_config: {e}",
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
        except _Stopped:
            # The worker is stopping while this row waited on a pause. Leave it
            # for the next worker rather than failing it.
            return self._deferred(input_row, reason="stopping")
        except Exception as e:
            disposition = self._provider.classify_error(e)
            if disposition == ErrorDisposition.THROTTLED:
                # An aborted batch completes this turn, so the abort wins.
                if abort_event and abort_event.is_set():
                    return self._aborted_result(input_row)
                if input_row.age_seconds >= self._max_row_age_seconds:
                    error = (
                        f"throttled and older than {self._max_row_age_seconds} seconds"
                    )
                    logger.error(
                        "row_failed",
                        extra={
                            "row_id": str(input_row.row_id),
                            "error": error,
                            "error_code": _error_code(e),
                            "age_seconds": input_row.age_seconds,
                        },
                    )
                    return LLMResult(
                        batch_id=input_row.batch_id,
                        row_id=input_row.row_id,
                        response="",
                        status="failed",
                        error=error,
                    )
                return self._deferred(input_row, reason=_error_code(e))
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

    def process_batch_iter(self, inputs: list[LLMInput]) -> Iterator[RowOutcome]:
        abort_event = threading.Event()
        pending_inputs = iter(inputs)
        aborted_rows: list[LLMInput] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Map each in-flight future to its input row so a future cancelled during
            # an abort can still be emitted as an aborted result rather than silently
            # dropped — every row must be yielded exactly once.
            in_flight: dict[Future[RowOutcome], LLMInput] = {}

            def submit(row: LLMInput) -> None:
                in_flight[executor.submit(self.invoke, row, abort_event)] = row

            while len(in_flight) < self._max_workers:
                try:
                    submit(next(pending_inputs))
                except StopIteration:
                    break

            while in_flight:
                done, _ = wait(
                    set(in_flight),
                    timeout=_FUTURES_WAIT_SLICE_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    del in_flight[future]
                    yield future.result()

                if abort_event.is_set():
                    aborted_rows.extend(list(pending_inputs))
                    break

                while len(in_flight) < self._max_workers:
                    try:
                        submit(next(pending_inputs))
                    except StopIteration:
                        break

            for future in in_flight:
                future.cancel()

            for future, row in in_flight.items():
                if future.cancelled():
                    # Submitted but never started before the abort — report it aborted
                    # (via aborted_rows below) instead of dropping it silently.
                    aborted_rows.append(row)
                else:
                    yield future.result()

        for row in aborted_rows:
            logger.info(
                "row_skipped_unsubmitted",
                extra={"row_id": str(row.row_id), "batch_id": str(row.batch_id)},
            )
            yield self._aborted_result(row)

    def process_batch(self, inputs: list[LLMInput]) -> list[RowOutcome]:
        return list(self.process_batch_iter(inputs))

"""Provider-agnostic interface for LLM backends."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from llm_worker.contract import ContractRequest

logger = logging.getLogger(__name__)

T = TypeVar("T")
ExcT = TypeVar("ExcT", bound=BaseException)


@dataclass
class LLMResponse:
    """Canonical result of a single completion, independent of the backend.

    ``output`` is the envelope written to ``llm_results.response``:
    ``{"output_text": str}`` or ``{"output_text": str, "tool_uses": [...]}``
    (``tool_uses`` omitted when empty). Every adapter must produce this same
    shape so downstream consumers are unaffected by the backend.
    """

    output: dict
    input_tokens: int
    output_tokens: int


class ErrorDisposition(Enum):
    """How the executor should treat a backend error."""

    RETRY = "retry"  # transient — retry the call
    ABORT_BATCH = "abort_batch"  # e.g. auth failure — abort the whole batch
    FAIL_ROW = "fail_row"  # permanent for this row — fail it, keep going


class LLMProvider(ABC):
    """A pluggable LLM backend.

    A provider translates a single request to its backend and back, and
    classifies its native exceptions. Retry, concurrency, abort handling, and
    result mapping are owned by the executor — not the provider.
    """

    def __init__(self) -> None:
        # Resolved model refs learned to reject a custom temperature. Per-model,
        # not process-wide: a single provider instance serves many models (it
        # honors request.model_id), and only some models deprecate temperature —
        # a shared flag would needlessly strip determinism from the rest.
        self._omit_temperature: set[str] = set()

    @abstractmethod
    def complete(self, request: ContractRequest) -> LLMResponse:
        """Translate a normalized v1 request to the backend and return the
        canonical response.

        Raises the backend's native exception on failure; does not retry (the
        executor wraps this call with retry driven by :meth:`classify_error`).
        """

    @abstractmethod
    def classify_error(self, exc: BaseException) -> ErrorDisposition:
        """Map a backend-native exception to an :class:`ErrorDisposition`."""

    def _complete_with_temperature_fallback(
        self,
        model: str,
        *,
        call: Callable[[bool], T],
        exception_type: type[ExcT],
        is_rejection: Callable[[ExcT], bool],
        log_event: str,
    ) -> T:
        """Shared "try with temperature, learn and retry once without it" state
        machine used by both the Bedrock and Anthropic Messages adapters.

        ``call`` is invoked with a single ``include_temperature`` bool and must
        perform the backend request. On ``exception_type`` raised by ``call``,
        the exception is re-raised unless temperature was included in this
        attempt *and* ``is_rejection`` confirms it was a genuine
        model-does-not-support-temperature rejection (as opposed to e.g. an
        out-of-range validation error) — in which case the model is learned
        into ``self._omit_temperature`` and ``call`` is retried once, without
        temperature.
        """
        include_temperature = model not in self._omit_temperature
        try:
            return call(include_temperature)
        except exception_type as exc:
            if not include_temperature or not is_rejection(exc):
                raise
            # Learn that this model rejects a custom temperature and retry once
            # without it. Determinism degrades to the model default here.
            logger.warning(log_event, extra={"model": model})
            self._omit_temperature.add(model)
            return call(False)

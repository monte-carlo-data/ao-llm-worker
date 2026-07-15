"""Provider-agnostic interface for LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from llm_worker.contract import ContractRequest


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

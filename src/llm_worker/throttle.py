"""Per-model pause shared by every worker thread when a backend throttles.

A provider throttle (Bedrock ``ThrottlingException``, Anthropic 429/529) means
the quota or capacity behind one model is exhausted. Rather than each thread
backing off on its own, the executor reports throttles here and every thread
waits on the same pause before calling that model again. Other models are not
affected, because Bedrock quotas are per model.

Rules:

* A throttle starts a pause for its model. The pause lasts ``next_delay``,
  which starts at ``min_pause`` and doubles with each new pause, up to
  ``max_pause``.
* A pause grows only once per burst. Throttles from calls that were already in
  flight when the latest pause began, or that arrive while it is still running,
  don't start or extend a pause.
* A success from a call started after the latest pause began resets
  ``next_delay`` to ``min_pause`` for the next pause. It doesn't shorten a pause
  already running.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _ModelState:
    next_delay: float
    paused_until: float = float("-inf")
    pause_started_at: float = float("-inf")


class ThrottleGate:
    def __init__(
        self,
        min_pause: float,
        max_pause: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if min_pause <= 0:
            raise ValueError(f"min_pause={min_pause} must be > 0")
        if max_pause < min_pause:
            raise ValueError(f"max_pause={max_pause} must be >= min_pause={min_pause}")
        self._min_pause = min_pause
        self._max_pause = max_pause
        self._clock = clock
        self._lock = threading.Lock()
        self._models: dict[str, _ModelState] = {}

    def now(self) -> float:
        """The gate's clock. Callers pass this as ``call_started_at``."""
        return self._clock()

    def wait(self, model_id: str, stop_event: threading.Event) -> bool:
        """Block until ``model_id`` isn't paused.

        Returns True when the caller may proceed, or False if ``stop_event`` was
        set while waiting. Re-checks after each wake, because another thread may
        have started a new pause in the meantime.
        """
        while True:
            with self._lock:
                state = self._models.get(model_id)
                remaining = state.paused_until - self._clock() if state else 0.0
            if remaining <= 0:
                return True
            if stop_event.wait(remaining):
                return False

    def on_throttle(self, model_id: str, call_started_at: float) -> float | None:
        """Report a throttled call. Returns the new pause in seconds, or None
        when the throttle belongs to a pause already started."""
        with self._lock:
            now = self._clock()
            state = self._state(model_id)
            if call_started_at < state.pause_started_at or now < state.paused_until:
                return None
            delay = state.next_delay
            state.pause_started_at = now
            state.paused_until = now + delay
            state.next_delay = min(delay * 2, self._max_pause)
            return delay

    def on_success(self, model_id: str, call_started_at: float) -> None:
        """Report a successful call, resetting the next pause to the minimum."""
        with self._lock:
            state = self._models.get(model_id)
            if state is None or call_started_at < state.pause_started_at:
                return
            state.next_delay = self._min_pause

    def _state(self, model_id: str) -> _ModelState:
        state = self._models.get(model_id)
        if state is None:
            state = _ModelState(next_delay=self._min_pause)
            self._models[model_id] = state
        return state

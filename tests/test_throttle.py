import threading
import time

import pytest

from llm_worker.throttle import ThrottleGate


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _gate(min_pause=1.0, max_pause=8.0):
    clock = FakeClock()
    return ThrottleGate(min_pause, max_pause, clock=clock), clock


class TestPauseGrowth:
    def test_first_throttle_pauses_for_min(self):
        gate, clock = _gate()
        assert gate.on_throttle("m", call_started_at=clock.now) == 1.0

    def test_delay_doubles_per_pause_up_to_max(self):
        gate, clock = _gate(min_pause=1.0, max_pause=5.0)
        delays = []
        for _ in range(5):
            delays.append(gate.on_throttle("m", call_started_at=clock.now))
            clock.advance(10)  # past the pause; the next call starts after it
        assert delays == [1.0, 2.0, 4.0, 5.0, 5.0]

    def test_throttle_during_active_pause_does_not_extend_it(self):
        gate, clock = _gate()
        gate.on_throttle("m", call_started_at=clock.now)
        clock.advance(0.5)

        assert gate.on_throttle("m", call_started_at=clock.now) is None
        clock.advance(0.6)  # 1.1s after the pause began
        assert gate.on_throttle("m", call_started_at=clock.now) == 2.0

    def test_throttle_from_call_in_flight_before_pause_is_ignored(self):
        gate, clock = _gate()
        in_flight_started = clock.now
        clock.advance(0.1)
        gate.on_throttle("m", call_started_at=clock.now)
        clock.advance(5)  # pause over, but the slow call began before it

        assert gate.on_throttle("m", call_started_at=in_flight_started) is None

    def test_concurrent_throttles_grow_the_delay_once(self):
        gate, clock = _gate()
        started = clock.now
        clock.advance(0.1)
        results = []
        barrier = threading.Barrier(20)

        def report():
            barrier.wait()
            results.append(gate.on_throttle("m", call_started_at=started))

        threads = [threading.Thread(target=report) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert sorted(r for r in results if r is not None) == [1.0]
        clock.advance(10)
        assert gate.on_throttle("m", call_started_at=clock.now) == 2.0


class TestSuccess:
    def test_success_resets_next_pause_to_min(self):
        gate, clock = _gate()
        gate.on_throttle("m", call_started_at=clock.now)
        clock.advance(10)
        gate.on_throttle("m", call_started_at=clock.now)  # 2s
        clock.advance(10)

        gate.on_success("m", call_started_at=clock.now)

        assert gate.on_throttle("m", call_started_at=clock.now) == 1.0

    def test_success_does_not_shorten_active_pause(self):
        gate, clock = _gate(min_pause=10.0, max_pause=10.0)
        gate.on_throttle("m", call_started_at=clock.now)
        clock.advance(1)

        gate.on_success("m", call_started_at=clock.now)

        # Still paused: a new throttle is part of the running pause.
        assert gate.on_throttle("m", call_started_at=clock.now) is None

    def test_success_from_call_in_flight_before_pause_is_ignored(self):
        gate, clock = _gate()
        in_flight_started = clock.now
        clock.advance(0.1)
        gate.on_throttle("m", call_started_at=clock.now)
        clock.advance(10)
        gate.on_throttle("m", call_started_at=clock.now)  # next delay is now 4s
        clock.advance(10)

        gate.on_success("m", call_started_at=in_flight_started)

        assert gate.on_throttle("m", call_started_at=clock.now) == 4.0

    def test_success_for_unknown_model_is_a_noop(self):
        gate, clock = _gate()
        gate.on_success("never-throttled", call_started_at=clock.now)


class TestPerModel:
    def test_pause_on_one_model_does_not_affect_another(self):
        gate, clock = _gate()
        gate.on_throttle("a", call_started_at=clock.now)

        assert gate.on_throttle("b", call_started_at=clock.now) == 1.0


class TestMaxPause:
    def test_exposes_the_configured_max(self):
        assert ThrottleGate(1.0, 8.0).max_pause == 8.0


class TestWait:
    def test_returns_immediately_without_a_pause(self):
        gate = ThrottleGate(1.0, 8.0)
        started = time.monotonic()

        assert gate.wait("m", threading.Event()) is True
        assert time.monotonic() - started < 0.05

    def test_blocks_until_pause_ends(self):
        gate = ThrottleGate(0.2, 0.2)
        gate.on_throttle("m", call_started_at=gate.now())
        started = time.monotonic()

        assert gate.wait("m", threading.Event()) is True
        assert time.monotonic() - started >= 0.15

    def test_other_model_does_not_wait(self):
        gate = ThrottleGate(30.0, 30.0)
        gate.on_throttle("a", call_started_at=gate.now())
        started = time.monotonic()

        assert gate.wait("b", threading.Event()) is True
        assert time.monotonic() - started < 0.05

    def test_stop_event_wakes_every_waiter(self):
        gate = ThrottleGate(30.0, 30.0)
        gate.on_throttle("m", call_started_at=gate.now())
        stop = threading.Event()
        results = []

        threads = [
            threading.Thread(target=lambda: results.append(gate.wait("m", stop)))
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        stop.set()
        for t in threads:
            t.join(timeout=1)

        assert not any(t.is_alive() for t in threads)
        assert results == [False] * 5

    def test_any_wake_event_ends_the_wait(self):
        gate = ThrottleGate(30.0, 30.0)
        gate.on_throttle("m", call_started_at=gate.now())
        stop, abort = threading.Event(), threading.Event()
        threading.Timer(0.05, abort.set).start()
        started = time.monotonic()

        assert gate.wait("m", stop, abort) is False
        assert time.monotonic() - started < 1


class TestValidation:
    @pytest.mark.parametrize(("min_pause", "max_pause"), [(0, 1), (-1, 1), (2, 1)])
    def test_rejects_invalid_bounds(self, min_pause, max_pause):
        with pytest.raises(ValueError):
            ThrottleGate(min_pause, max_pause)

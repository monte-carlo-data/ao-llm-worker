import json
import threading
import time
from uuid import UUID

from botocore.exceptions import ClientError

from llm_worker.clickhouse import LLMInput
from llm_worker.config import BedrockConfig
from llm_worker.executor import BatchExecutor, Deferred, DeferralReason
from llm_worker.providers.base import ErrorDisposition, LLMProvider, LLMResponse
from llm_worker.providers.bedrock import BedrockProvider
from llm_worker.throttle import ThrottleGate

BATCH_1 = UUID("00000000-0000-0000-0000-000000000001")
ROW_1 = UUID("00000000-0000-0000-0000-000000000011")


def _make_input(
    batch_id=BATCH_1,
    row_id=ROW_1,
    model_id="model-1",
    prompt="hello",
    params="{}",
    tool_config="",
    age_seconds=0,
):
    return LLMInput(
        batch_id, row_id, model_id, prompt, params, tool_config, age_seconds
    )


class _Retryable(Exception):
    pass


class _Fatal(Exception):
    pass


class _Auth(Exception):
    pass


class _Throttled(Exception):
    pass


_DISPOSITIONS = {
    _Retryable: ErrorDisposition.RETRY,
    _Throttled: ErrorDisposition.THROTTLED,
    _Auth: ErrorDisposition.ABORT_BATCH,
    _Fatal: ErrorDisposition.FAIL_ROW,
}


class FakeProvider(LLMProvider):
    """A provider whose behavior is driven entirely by the test, so the executor
    is exercised independently of any real backend."""

    def __init__(self, complete_fn=None):
        self.calls = 0
        self._complete_fn = complete_fn or (
            lambda request: LLMResponse({"output_text": "ok"}, 0, 0)
        )

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        return self._complete_fn(request)

    def classify_error(self, exc: BaseException) -> ErrorDisposition:
        return _DISPOSITIONS.get(type(exc), ErrorDisposition.FAIL_ROW)


def _executor(
    provider,
    max_workers=3,
    retry_max_attempts=1,
    retry_max_backoff=0,
    throttle_gate=None,
    stop_event=None,
    max_row_age_seconds=600,
):
    return BatchExecutor(
        provider,
        max_workers,
        retry_max_attempts,
        retry_max_backoff,
        # Short pauses so throttled tests don't sit out the production default.
        throttle_gate=throttle_gate or ThrottleGate(0.01, 0.01),
        stop_event=stop_event or threading.Event(),
        max_row_age_seconds=max_row_age_seconds,
    )


class TestInvoke:
    def test_successful_call(self):
        result = _executor(FakeProvider()).invoke(_make_input())

        assert result.status == "complete"
        assert result.error == ""
        assert json.loads(result.response) == {"output_text": "ok"}

    def test_serializes_provider_output(self):
        provider = FakeProvider(
            lambda *a: LLMResponse(
                {"output_text": "x", "tool_uses": [{"cat": "err"}]}, 5, 7
            )
        )

        result = _executor(provider).invoke(_make_input())

        assert json.loads(result.response) == {
            "output_text": "x",
            "tool_uses": [{"cat": "err"}],
        }

    def test_invalid_params_json(self):
        provider = FakeProvider()

        result = _executor(provider).invoke(_make_input(params="not json"))

        assert result.status == "failed"
        assert "Invalid JSON" in result.error
        assert provider.calls == 0

    def test_invalid_tool_config_json(self):
        provider = FakeProvider()

        result = _executor(provider).invoke(_make_input(tool_config="not json"))

        assert result.status == "failed"
        assert "Invalid JSON" in result.error
        assert provider.calls == 0

    def test_malformed_tool_config_fails_row_not_loop(self):
        # Valid JSON, but the toolSpec is missing its required "name" key.
        # build_request -> _normalize_tools -> _normalize_tool raises ValueError
        # via spec.get("name"); that must not escape invoke() -- the row should
        # come back failed instead.
        provider = FakeProvider()
        tool_config = json.dumps({"tools": [{"toolSpec": {"description": "x"}}]})

        result = _executor(provider).invoke(_make_input(tool_config=tool_config))

        assert result.status == "failed"
        assert provider.calls == 0

    def test_preserves_batch_and_row_id(self):
        b = UUID("00000000-0000-0000-0000-000000000099")
        r = UUID("00000000-0000-0000-0000-000000000042")

        result = _executor(FakeProvider()).invoke(_make_input(batch_id=b, row_id=r))

        assert result.batch_id == b
        assert result.row_id == r

    def test_aborted_when_event_set(self):
        provider = FakeProvider()
        abort_event = threading.Event()
        abort_event.set()

        result = _executor(provider).invoke(_make_input(), abort_event=abort_event)

        assert result.status == "failed"
        assert result.error == "Batch aborted"
        assert provider.calls == 0

    def test_provider_failure_fails_row(self):
        def boom(*a):
            raise _Fatal("backend down")

        result = _executor(FakeProvider(boom)).invoke(_make_input())

        assert result.status == "failed"
        assert "backend down" in result.error


class TestRetry:
    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky(*a):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _Retryable("transient")
            return LLMResponse({"output_text": "done"}, 0, 0)

        provider = FakeProvider(flaky)
        result = _executor(provider, retry_max_attempts=5).invoke(_make_input())

        assert result.status == "complete"
        assert provider.calls == 3

    def test_fails_after_max_retries(self):
        def always(*a):
            raise _Retryable("transient")

        provider = FakeProvider(always)
        result = _executor(provider, retry_max_attempts=3).invoke(_make_input())

        assert result.status == "failed"
        assert provider.calls == 3

    def test_throttled_row_is_deferred_without_using_retries(self):
        def always(*a):
            raise _Throttled("slow down")

        provider = FakeProvider(always)
        result = _executor(provider, retry_max_attempts=5).invoke(_make_input())

        assert result == Deferred(BATCH_1, ROW_1, "model-1")
        assert provider.calls == 1

    def test_bedrock_throttling_defers_row(self, mocker):
        boto = mocker.Mock()
        boto.converse.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
            "Converse",
        )
        provider = BedrockProvider(BedrockConfig(region="us-east-1"), boto_client=boto)

        result = _executor(provider, retry_max_attempts=5).invoke(_make_input())

        assert result == Deferred(BATCH_1, ROW_1, "model-1")
        assert boto.converse.call_count == 1

    def test_no_retry_on_fail_row(self):
        def always(*a):
            raise _Fatal("permanent")

        provider = FakeProvider(always)
        result = _executor(provider, retry_max_attempts=3).invoke(_make_input())

        assert result.status == "failed"
        assert provider.calls == 1

    def test_no_retry_on_abort(self):
        def always(*a):
            raise _Auth("denied")

        provider = FakeProvider(always)
        result = _executor(provider, retry_max_attempts=3).invoke(
            _make_input(), abort_event=threading.Event()
        )

        assert result.status == "failed"
        assert provider.calls == 1


class TestAbort:
    def test_abort_disposition_sets_event(self):
        def always(*a):
            raise _Auth("denied")

        abort_event = threading.Event()
        _executor(FakeProvider(always)).invoke(_make_input(), abort_event=abort_event)

        assert abort_event.is_set()

    def test_abort_disposition_without_event_does_not_crash(self):
        def always(*a):
            raise _Auth("denied")

        result = _executor(FakeProvider(always)).invoke(_make_input())

        assert result.status == "failed"

    def test_non_abort_failure_does_not_set_event(self):
        def always(*a):
            raise _Fatal("permanent")

        abort_event = threading.Event()
        _executor(FakeProvider(always), retry_max_attempts=1).invoke(
            _make_input(), abort_event=abort_event
        )

        assert not abort_event.is_set()

    def test_batch_aborts_on_abort_disposition(self):
        call_count = {"n": 0}

        def side_effect(*a):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _Auth("denied")
            return LLMResponse({"output_text": "ok"}, 0, 0)

        provider = FakeProvider(side_effect)
        inputs = [_make_input(row_id=UUID(int=i)) for i in range(5)]

        results = _executor(provider, max_workers=1).process_batch(inputs)

        assert len(results) == 5
        failed = [r for r in results if r.status == "failed"]
        assert len(failed) >= 2
        assert provider.calls < len(inputs)

    def test_partitions_rows_when_abort_happens_with_concurrent_in_flight_calls(self):
        # max_workers=2: two calls are genuinely in flight together. Both must
        # clear invoke()'s abort_event check (and reach the provider) before
        # either resolves, so this proves the abort disposition doesn't race
        # ahead of a call that was already in flight when it happened.
        #
        # Row 1 is held until row 0's invoke() has returned, i.e. until abort_event
        # is set. Without that hold, row 1 can finish while row 0 is still between
        # raising and abort_event.set(); the loop then submits rows 2-4, which run
        # to completion instead of being reported aborted.
        lock = threading.Lock()
        both_started = threading.Event()
        abort_signalled = threading.Event()
        started = {"n": 0}

        def side_effect(request):
            with lock:
                started["n"] += 1
                if started["n"] == 2:
                    both_started.set()
            both_started.wait(timeout=5)
            if request.model_id == "abort-me":
                raise _Auth("denied")
            abort_signalled.wait(timeout=5)
            return LLMResponse({"output_text": "ok"}, 0, 0)

        provider = FakeProvider(side_effect)
        executor = _executor(provider, max_workers=2)
        real_invoke = executor.invoke

        def invoke(input_row, abort_event=None, **kwargs):
            try:
                return real_invoke(input_row, abort_event, **kwargs)
            finally:
                if input_row.model_id == "abort-me":
                    abort_signalled.set()

        executor.invoke = invoke
        inputs = [
            _make_input(row_id=UUID(int=0), model_id="abort-me"),
            _make_input(row_id=UUID(int=1), model_id="ok"),
            _make_input(row_id=UUID(int=2), model_id="ok"),
            _make_input(row_id=UUID(int=3), model_id="ok"),
            _make_input(row_id=UUID(int=4), model_id="ok"),
        ]

        results: list = []
        worker = threading.Thread(
            target=lambda: results.extend(executor.process_batch_iter(inputs))
        )
        worker.start()
        assert both_started.wait(timeout=5)
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert len(results) == len(inputs)
        row_ids = [r.row_id for r in results]
        assert len(set(row_ids)) == len(inputs)  # every row yielded exactly once
        assert set(row_ids) == {i.row_id for i in inputs}  # none lost

        by_row = {r.row_id: r for r in results}
        assert by_row[UUID(int=0)].status == "failed"
        assert "denied" in by_row[UUID(int=0)].error
        assert by_row[UUID(int=1)].status == "complete"
        # Every row reached a terminal status (the exactly-once checks above already
        # guarantee none were lost or duplicated).
        assert all(r.status in ("complete", "failed") for r in results)
        # Row 1 can't finish before abort_event is set, so the first completion the
        # loop sees is row 0 with the abort already signalled: rows 2-4 are never
        # submitted and are always reported aborted.
        for i in (2, 3, 4):
            assert by_row[UUID(int=i)].status == "failed"
            assert by_row[UUID(int=i)].error == "Batch aborted"


class TestProcessBatch:
    def test_processes_all_rows(self):
        provider = FakeProvider()
        inputs = [_make_input(row_id=UUID(int=i)) for i in range(5)]

        results = _executor(provider).process_batch(inputs)

        assert len(results) == 5
        assert all(r.status == "complete" for r in results)

    def test_mixed_success_and_failure(self):
        call_count = {"n": 0}

        def side_effect(*a):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise _Fatal("fail")
            return LLMResponse({"output_text": "ok"}, 0, 0)

        provider = FakeProvider(side_effect)
        inputs = [_make_input(row_id=UUID(int=i)) for i in range(4)]

        results = _executor(provider, max_workers=1).process_batch(inputs)

        assert len(results) == 4
        assert len([r for r in results if r.status == "complete"]) == 2
        assert len([r for r in results if r.status == "failed"]) == 2

    def test_empty_input(self):
        provider = FakeProvider()

        assert _executor(provider).process_batch([]) == []
        assert provider.calls == 0

    def test_limits_in_flight_work_to_max_workers(self):
        started = 0
        peak = 0
        lock = threading.Lock()
        release = threading.Event()

        def side_effect(*a):
            nonlocal started, peak
            with lock:
                started += 1
                peak = max(peak, started)
            release.wait(timeout=0.1)
            with lock:
                started -= 1
            return LLMResponse({"output_text": "ok"}, 0, 0)

        executor = _executor(FakeProvider(side_effect), max_workers=2)
        inputs = [_make_input(row_id=UUID(int=i)) for i in range(5)]
        worker = threading.Thread(
            target=lambda: list(executor.process_batch_iter(inputs))
        )
        worker.start()
        time.sleep(0.02)
        release.set()
        worker.join(timeout=1)

        assert peak <= 2


_OUTPUT_FORMAT = {
    "type": "json_schema",
    "name": "evaluation_result",
    "schema": {"type": "object", "properties": {}, "additionalProperties": False},
}
_TOOL = {"name": "evaluation_result", "description": "d", "input_schema": {}}


class TestInvalidRequestProfileCombinations:
    """build_request rejects these with ValueError, and the executor fails only that
    row with the message, without calling the provider or crashing the batch."""

    def test_output_format_with_tools_fails_the_row(self):
        provider = FakeProvider()
        row = _make_input(
            tool_config=json.dumps({"output_format": _OUTPUT_FORMAT, "tools": [_TOOL]})
        )

        result = _executor(provider).invoke(row)

        assert result.status == "failed"
        assert "both output_format and tools" in result.error
        assert provider.calls == 0

    def test_thinking_with_a_forced_tool_fails_the_row(self):
        provider = FakeProvider()
        row = _make_input(
            params=json.dumps({"thinking": "adaptive"}),
            tool_config=json.dumps(
                {"tools": [_TOOL], "tool_choice": {"name": "evaluation_result"}}
            ),
        )

        result = _executor(provider).invoke(row)

        assert result.status == "failed"
        assert "forced tool choice" in result.error
        assert provider.calls == 0

    def test_other_rows_in_the_batch_still_complete(self):
        provider = FakeProvider()
        bad = _make_input(
            row_id=UUID("00000000-0000-0000-0000-000000000012"),
            tool_config=json.dumps({"output_format": _OUTPUT_FORMAT, "tools": [_TOOL]}),
        )

        results = list(_executor(provider).process_batch_iter([bad, _make_input()]))

        assert sorted(r.status for r in results) == ["complete", "failed"]


class TestThrottle:
    def test_throttle_starts_a_pause_for_the_model(self):
        def always(*a):
            raise _Throttled("slow down")

        gate = ThrottleGate(30.0, 30.0)
        _executor(FakeProvider(always), throttle_gate=gate).invoke(_make_input())

        # A new throttle now belongs to the running pause.
        assert gate.on_throttle("model-1", call_started_at=gate.now()) is None
        assert gate.on_throttle("model-2", call_started_at=gate.now()) == 30.0

    def test_prefixed_and_bare_ids_share_a_pause(self):
        def always(*a):
            raise _Throttled("slow down")

        gate = ThrottleGate(30.0, 30.0)
        _executor(FakeProvider(always), throttle_gate=gate).invoke(
            _make_input(model_id="mc:model-1")
        )

        # The throttle on "mc:model-1" started the pause for "model-1".
        assert gate.on_throttle("model-1", call_started_at=gate.now()) is None

    def test_success_is_reported_to_the_gate(self, mocker):
        gate = mocker.Mock(spec=ThrottleGate)
        gate.wait.return_value = True
        gate.now.return_value = 5.0
        gate.max_pause = 0.0

        _executor(FakeProvider(), throttle_gate=gate).invoke(_make_input())

        gate.on_success.assert_called_once_with("model-1", 5.0)

    def test_every_retry_attempt_waits_on_the_gate(self, mocker):
        def always(*a):
            raise _Retryable("transient")

        gate = mocker.Mock(spec=ThrottleGate)
        gate.wait.return_value = True
        gate.now.return_value = 0.0
        gate.max_pause = 0.0

        result = _executor(
            FakeProvider(always), retry_max_attempts=3, throttle_gate=gate
        ).invoke(_make_input())

        assert result.status == "failed"
        assert gate.wait.call_count == 3
        # Only throttles are reported to the gate, not other errors.
        gate.on_throttle.assert_not_called()

    def test_abort_wins_over_deferral(self):
        abort_event = threading.Event()

        def abort_then_throttle(*a):
            abort_event.set()  # another row's auth failure aborted the batch
            raise _Throttled("slow down")

        result = _executor(FakeProvider(abort_then_throttle)).invoke(
            _make_input(), abort_event=abort_event
        )

        assert result.status == "failed"
        assert result.error == "Batch aborted"

    def test_stopping_during_a_pause_defers_without_calling(self):
        gate = ThrottleGate(30.0, 30.0)
        gate.on_throttle("model-1", call_started_at=gate.now())
        stop_event = threading.Event()
        stop_event.set()
        provider = FakeProvider()

        started = time.monotonic()
        result = _executor(provider, throttle_gate=gate, stop_event=stop_event).invoke(
            _make_input()
        )

        assert result == Deferred(BATCH_1, ROW_1, "model-1")
        assert provider.calls == 0
        assert time.monotonic() - started < 1

    def test_stopping_cuts_a_retry_backoff_short(self):
        stop_event = threading.Event()

        def fail_then_stop(*a):
            stop_event.set()  # SIGTERM arrives during the first attempt
            raise _Retryable("transient")

        provider = FakeProvider(fail_then_stop)
        started = time.monotonic()
        result = _executor(
            provider, retry_max_attempts=3, retry_max_backoff=30, stop_event=stop_event
        ).invoke(_make_input())

        assert result == Deferred(BATCH_1, ROW_1, "model-1")
        assert provider.calls == 1
        assert time.monotonic() - started < 0.5  # first backoff would be 1s

    def test_abort_wins_over_a_deferral_earlier_in_the_page(self):
        def throttle_then_auth_fail(request):
            if request.prompt == "first":
                raise _Throttled("slow down")
            raise _Auth("bad credentials")

        row_1, row_2 = UUID(int=1), UUID(int=2)
        inputs = [
            _make_input(row_id=row_1, prompt="first"),
            _make_input(row_id=row_2, prompt="second"),
        ]

        # One worker, so row 1 is deferred before row 2 aborts the batch.
        outcomes = _executor(
            FakeProvider(throttle_then_auth_fail), max_workers=1
        ).process_batch(inputs)

        by_row = {o.row_id: o for o in outcomes}
        assert len(outcomes) == 2
        assert not any(isinstance(o, Deferred) for o in outcomes)
        assert by_row[row_1].status == "failed"
        assert by_row[row_1].error == "Batch aborted"
        assert by_row[row_2].status == "failed"
        assert by_row[row_2].error == "bad credentials"

    def test_abort_during_a_pause_wakes_waiting_rows_without_a_call(self):
        gate = ThrottleGate(3.0, 3.0)
        gate.on_throttle("model-1", call_started_at=gate.now())
        called_models = []

        def auth_fail_on_model_2(request):
            called_models.append(request.model_id)
            time.sleep(0.1)  # let the model-1 rows reach the pause first
            raise _Auth("bad credentials")

        inputs = [
            _make_input(row_id=UUID(int=1), model_id="model-1"),
            _make_input(row_id=UUID(int=2), model_id="model-1"),
            _make_input(row_id=UUID(int=3), model_id="model-2"),
        ]

        started = time.monotonic()
        outcomes = _executor(
            FakeProvider(auth_fail_on_model_2), throttle_gate=gate
        ).process_batch(inputs)

        by_row = {o.row_id: o for o in outcomes}
        assert time.monotonic() - started < 1
        assert called_models == ["model-2"]
        assert by_row[UUID(int=1)].error == "Batch aborted"
        assert by_row[UUID(int=2)].error == "Batch aborted"
        assert by_row[UUID(int=3)].error == "bad credentials"

    def test_row_aborted_while_waiting_is_not_called(self):
        gate = ThrottleGate(0.2, 0.2)
        gate.on_throttle("model-1", call_started_at=gate.now())
        abort_event = threading.Event()
        threading.Timer(0.05, abort_event.set).start()
        provider = FakeProvider()

        result = _executor(provider, throttle_gate=gate).invoke(
            _make_input(), abort_event=abort_event
        )

        assert result.error == "Batch aborted"
        assert provider.calls == 0

    def test_throttled_deferral_carries_reason_and_error_code(self):
        def always(*a):
            raise _Throttled("slow down")

        provider = FakeProvider(always)
        provider.error_code = lambda exc: "TooManyRequests"

        result = _executor(provider).invoke(_make_input())

        assert isinstance(result, Deferred)
        assert result.reason is DeferralReason.THROTTLED
        assert result.error_code == "TooManyRequests"

    def test_batch_yields_each_row_once_with_deferrals(self):
        def throttle_odd_rows(request):
            if request.prompt.endswith(("1", "3")):
                raise _Throttled("slow down")
            return LLMResponse({"output_text": "ok"}, 0, 0)

        inputs = [
            _make_input(row_id=UUID(int=i), model_id=f"model-{i}", prompt=f"p{i}")
            for i in range(4)
        ]

        outcomes = _executor(FakeProvider(throttle_odd_rows)).process_batch(inputs)

        by_row = {o.row_id: o for o in outcomes}
        assert len(outcomes) == 4
        assert by_row[UUID(int=1)] == Deferred(BATCH_1, UUID(int=1), "model-1")
        assert by_row[UUID(int=3)] == Deferred(BATCH_1, UUID(int=3), "model-3")
        assert by_row[UUID(int=0)].status == "complete"
        assert by_row[UUID(int=2)].status == "complete"


class TestThrottleRowAge:
    @staticmethod
    def _always_throttled(*a):
        raise _Throttled("slow down")

    def test_row_well_under_the_limit_is_deferred(self):
        executor = _executor(
            FakeProvider(self._always_throttled),
            max_workers=1,
            throttle_gate=ThrottleGate(0.01, 0.01),
        )

        result = executor.invoke(_make_input(age_seconds=500))

        assert result == Deferred(BATCH_1, ROW_1, "model-1")

    def test_row_at_the_limit_fails(self):
        executor = _executor(
            FakeProvider(self._always_throttled),
            max_workers=1,
            throttle_gate=ThrottleGate(0.01, 0.01),
        )

        result = executor.invoke(_make_input(age_seconds=600))

        assert result.status == "failed"
        assert result.error == "throttled and older than 600 seconds"

    def test_age_limit_only_applies_to_throttles(self):
        def always(*a):
            raise _Fatal("bad request")

        executor = _executor(
            FakeProvider(always), max_workers=1, throttle_gate=ThrottleGate(0.01, 0.01)
        )

        result = executor.invoke(_make_input(age_seconds=10_000))

        assert result.error == "bad request"


class _FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestThrottleRowAgeIsCurrent:
    @staticmethod
    def _always_throttled(*a):
        raise _Throttled("slow down")

    def test_row_that_ages_past_the_limit_while_in_flight_fails(self):
        clock = _FakeClock()
        gate = ThrottleGate(1.0, 1.0, clock=clock)

        def advance_then_throttle(request):
            clock.advance(20)
            raise _Throttled("slow down")

        executor = _executor(
            FakeProvider(advance_then_throttle), max_workers=1, throttle_gate=gate
        )

        result = executor.invoke(_make_input(age_seconds=590))

        assert result.status == "failed"
        assert result.error == "throttled and older than 600 seconds"

    def test_row_near_the_limit_fails_instead_of_deferring(self):
        executor = _executor(
            FakeProvider(self._always_throttled),
            max_workers=1,
            throttle_gate=ThrottleGate(60.0, 60.0),
        )

        result = executor.invoke(_make_input(age_seconds=550))

        assert result.status == "failed"
        assert result.error == "throttled and older than 600 seconds"

    def test_old_row_waits_out_a_pause_and_gets_a_call(self):
        gate = ThrottleGate(0.2, 0.2)
        gate.on_throttle("model-1", call_started_at=gate.now())
        provider = FakeProvider()
        executor = _executor(provider, max_workers=1, throttle_gate=gate)

        started = time.monotonic()
        result = executor.invoke(_make_input(age_seconds=700))

        assert result.status == "complete"
        assert provider.calls == 1
        assert time.monotonic() - started >= 0.15

    def test_old_row_throttled_after_its_pause_fails_after_one_call(self):
        gate = ThrottleGate(0.05, 0.05)
        gate.on_throttle("model-1", call_started_at=gate.now())
        provider = FakeProvider(self._always_throttled)
        executor = _executor(provider, max_workers=1, throttle_gate=gate)

        result = executor.invoke(_make_input(age_seconds=700))

        assert result.status == "failed"
        assert result.error == "throttled and older than 600 seconds"
        assert provider.calls == 1

    def test_page_of_old_rows_fails_no_row_without_a_call(self):
        sent = set()
        lock = threading.Lock()

        def throttle_first_call(request):
            with lock:
                first = not sent
                sent.add(request.prompt)
            if first:
                raise _Throttled("slow down")
            return LLMResponse({"output_text": "ok"}, 0, 0)

        inputs = [
            _make_input(row_id=UUID(int=i), prompt=f"p{i}", age_seconds=700)
            for i in range(100)
        ]
        executor = _executor(
            FakeProvider(throttle_first_call),
            max_workers=20,
            throttle_gate=ThrottleGate(0.05, 0.05),
        )

        outcomes = executor.process_batch(inputs)

        failed = [o for o in outcomes if o.status == "failed"]
        prompts = {o.row_id: f"p{o.row_id.int}" for o in outcomes}
        assert len(outcomes) == 100
        assert len(failed) == 1
        assert all(prompts[o.row_id] in sent for o in failed)

    def test_old_row_still_gets_a_call_when_the_model_is_not_paused(self):
        provider = FakeProvider()
        executor = _executor(
            provider, max_workers=1, throttle_gate=ThrottleGate(2.0, 2.0)
        )

        result = executor.invoke(_make_input(age_seconds=10_000))

        assert result.status == "complete"
        assert provider.calls == 1

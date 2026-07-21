import json
import threading
import time
from uuid import UUID

from llm_worker.clickhouse import LLMInput
from llm_worker.executor import BatchExecutor
from llm_worker.providers.base import ErrorDisposition, LLMProvider, LLMResponse

BATCH_1 = UUID("00000000-0000-0000-0000-000000000001")
ROW_1 = UUID("00000000-0000-0000-0000-000000000011")


def _make_input(
    batch_id=BATCH_1,
    row_id=ROW_1,
    model_id="model-1",
    prompt="hello",
    params="{}",
    tool_config="",
):
    return LLMInput(batch_id, row_id, model_id, prompt, params, tool_config)


class _Retryable(Exception):
    pass


class _Fatal(Exception):
    pass


class _Auth(Exception):
    pass


_DISPOSITIONS = {
    _Retryable: ErrorDisposition.RETRY,
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


def _executor(provider, max_workers=3, retry_max_attempts=1, retry_max_backoff=0):
    return BatchExecutor(provider, max_workers, retry_max_attempts, retry_max_backoff)


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
        # build_request -> _normalize_tools -> _normalize_tool does unguarded
        # `spec["name"]` access, which raises an uncaught KeyError that must
        # not escape invoke() -- the row should come back failed instead.
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
        lock = threading.Lock()
        both_started = threading.Event()
        release = threading.Event()
        started = {"n": 0}

        def side_effect(request):
            with lock:
                started["n"] += 1
                if started["n"] == 2:
                    both_started.set()
            both_started.wait(timeout=2)
            release.wait(timeout=2)
            if request.model_id == "abort-me":
                raise _Auth("denied")
            return LLMResponse({"output_text": "ok"}, 0, 0)

        provider = FakeProvider(side_effect)
        inputs = [
            _make_input(row_id=UUID(int=0), model_id="abort-me"),
            _make_input(row_id=UUID(int=1), model_id="ok"),
            _make_input(row_id=UUID(int=2), model_id="ok"),
            _make_input(row_id=UUID(int=3), model_id="ok"),
            _make_input(row_id=UUID(int=4), model_id="ok"),
        ]

        results: list = []
        worker = threading.Thread(
            target=lambda: results.extend(
                _executor(provider, max_workers=2).process_batch_iter(inputs)
            )
        )
        worker.start()
        assert both_started.wait(timeout=2)
        release.set()
        worker.join(timeout=2)

        assert not worker.is_alive()
        assert len(results) == len(inputs)
        row_ids = [r.row_id for r in results]
        assert len(set(row_ids)) == len(inputs)  # every row yielded exactly once
        assert set(row_ids) == {i.row_id for i in inputs}  # none lost

        by_row = {r.row_id: r for r in results}
        assert by_row[UUID(int=0)].status == "failed"
        assert "denied" in by_row[UUID(int=0)].error
        assert by_row[UUID(int=1)].status == "complete"
        for i in range(2, 5):
            assert by_row[UUID(int=i)].status == "failed"
            assert by_row[UUID(int=i)].error == "Batch aborted"

        assert provider.calls == 2  # unsubmitted rows never reached the provider


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

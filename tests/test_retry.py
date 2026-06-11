import threading
from uuid import UUID

import pytest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,  # noqa: F401 (used in parametrize)
    NoCredentialsError,  # noqa: F401 (used in parametrize)
    NoRegionError,  # noqa: F401 (used in parametrize)
    ParamValidationError,  # noqa: F401 (used in parametrize)
    ReadTimeoutError,  # noqa: F401 (used in parametrize)
)

from llm_worker.bedrock import (
    BedrockClient,
    NON_RETRYABLE_ERROR_CODES,
    RETRYABLE_ERROR_CODES,
    RETRYABLE_TRANSPORT_ERRORS,
    _is_retryable,
)
from llm_worker.clickhouse import LLMInput
from llm_worker.config import BedrockConfig

BATCH_1 = UUID("00000000-0000-0000-0000-000000000001")
ROW_1 = UUID("00000000-0000-0000-0000-000000000011")


def _make_input(row_id=ROW_1):
    return LLMInput(BATCH_1, row_id, "model-1", "test", "{}", "")


def _client_error(code, message="error"):
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "Converse",
    )


def _text_response(text="ok"):
    return {"output": {"message": {"content": [{"text": text}]}}}


def _make_client(mock_boto, max_attempts=3, max_workers=1):
    config = BedrockConfig(
        region="us-east-1",
        max_workers=max_workers,
        retry_max_attempts=max_attempts,
        retry_max_backoff=0,
    )
    return BedrockClient(config, boto_client=mock_boto)


class TestIsRetryable:
    def test_throttling_is_retryable(self):
        assert _is_retryable(_client_error("ThrottlingException"))

    def test_service_error_is_retryable(self):
        assert _is_retryable(_client_error("InternalServerException"))

    def test_model_not_ready_is_retryable(self):
        assert _is_retryable(_client_error("ModelNotReadyException"))

    @pytest.mark.parametrize("exc_class", RETRYABLE_TRANSPORT_ERRORS)
    def test_transport_exceptions_are_retryable(self, exc_class):
        exc = exc_class(endpoint_url="https://bedrock.us-east-1.amazonaws.com")
        assert _is_retryable(exc)

    def test_bare_botocore_error_is_not_retryable(self):
        assert not _is_retryable(BotoCoreError())

    @pytest.mark.parametrize(
        "exc_class", [ParamValidationError, NoCredentialsError, NoRegionError]
    )
    def test_permanent_botocore_errors_are_not_retryable(self, exc_class):
        exc = exc_class(report="test")
        assert not _is_retryable(exc)

    def test_generic_exception_is_not_retryable(self):
        assert not _is_retryable(Exception("timeout"))

    @pytest.mark.parametrize("code", list(NON_RETRYABLE_ERROR_CODES))
    def test_non_retryable_codes(self, code):
        assert not _is_retryable(_client_error(code))

    def test_unknown_client_error_is_not_retryable(self):
        assert not _is_retryable(_client_error("UnknownException"))


class TestRetryBehavior:
    def test_retries_on_throttle_then_succeeds(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = [
            _client_error("ThrottlingException"),
            _client_error("ThrottlingException"),
            _text_response("done"),
        ]
        client = _make_client(mock_boto, max_attempts=5)

        result = client.invoke(_make_input())

        assert result.status == "complete"
        assert mock_boto.converse.call_count == 3

    def test_fails_after_max_retries(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error("ThrottlingException")
        client = _make_client(mock_boto, max_attempts=3)

        result = client.invoke(_make_input())

        assert result.status == "failed"
        assert mock_boto.converse.call_count == 3

    def test_no_retry_on_validation_error(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error("ValidationException")
        client = _make_client(mock_boto, max_attempts=3)

        result = client.invoke(_make_input())

        assert result.status == "failed"
        assert mock_boto.converse.call_count == 1

    def test_no_retry_on_access_denied(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error("AccessDeniedException")
        client = _make_client(mock_boto, max_attempts=3)

        result = client.invoke(_make_input())

        assert result.status == "failed"
        assert mock_boto.converse.call_count == 1

    @pytest.mark.parametrize("code", sorted(RETRYABLE_ERROR_CODES))
    def test_retries_known_retryable_client_errors(self, mocker, code):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = [_client_error(code), _text_response("done")]
        client = _make_client(mock_boto, max_attempts=2)

        result = client.invoke(_make_input())

        assert result.status == "complete"
        assert mock_boto.converse.call_count == 2


class TestBatchAbort:
    def test_access_denied_sets_abort_event(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error("AccessDeniedException")
        client = _make_client(mock_boto)
        abort_event = threading.Event()

        client.invoke(_make_input(), abort_event=abort_event)

        assert abort_event.is_set()

    def test_aborted_rows_return_failed(self, mocker):
        mock_boto = mocker.Mock()
        client = _make_client(mock_boto)
        abort_event = threading.Event()
        abort_event.set()

        result = client.invoke(_make_input(), abort_event=abort_event)

        assert result.status == "failed"
        assert result.error == "Batch aborted"
        mock_boto.converse.assert_not_called()

    def test_batch_abort_on_auth_error(self, mocker):
        mock_boto = mocker.Mock()
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _client_error("AccessDeniedException")
            return _text_response("ok")

        mock_boto.converse.side_effect = side_effect
        client = _make_client(mock_boto, max_workers=1)
        inputs = [_make_input(row_id=UUID(int=i)) for i in range(5)]

        results = client.process_batch(inputs)

        assert len(results) == 5
        failed = [r for r in results if r.status == "failed"]
        assert len(failed) >= 2
        assert mock_boto.converse.call_count < len(inputs)

    def test_throttle_does_not_abort_batch(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error("ThrottlingException")
        client = _make_client(mock_boto, max_attempts=1)
        abort_event = threading.Event()

        client.invoke(_make_input(), abort_event=abort_event)

        assert not abort_event.is_set()

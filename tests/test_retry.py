import pytest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    NoRegionError,
    ParamValidationError,
)

from llm_worker.providers.bedrock import (
    NON_RETRYABLE_ERROR_CODES,
    RETRYABLE_TRANSPORT_ERRORS,
    _is_retryable,
)


def _client_error(code, message="error"):
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "Converse",
    )


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

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from llm_worker.config import BedrockConfig
from llm_worker.contract import ContractRequest, Tool
from llm_worker.providers.base import ErrorDisposition
from llm_worker.providers.bedrock import (
    BedrockProvider,
    _build_converse_kwargs,
    _extract_output,
)

BEDROCK_CONFIG = BedrockConfig(region="us-east-1")


def _provider(mock_boto):
    return BedrockProvider(BEDROCK_CONFIG, boto_client=mock_boto)


def _req(
    model_id="model-1",
    prompt="hello",
    max_output_tokens=512,
    temperature=0.0,
    tools=None,
    forced_tool=None,
):
    return ContractRequest(
        model_id=model_id,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        tools=tools or [],
        forced_tool=forced_tool,
    )


def _text_response(text="hello"):
    return {"output": {"message": {"content": [{"text": text}]}}}


def _tool_response(tool_input=None):
    if tool_input is None:
        tool_input = {"category": "error", "severity": "high"}
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool-1",
                            "name": "classify",
                            "input": tool_input,
                        }
                    }
                ]
            }
        }
    }


def _client_error(code, message="error"):
    return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")


class TestBuildConverseKwargs:
    def test_basic_request(self):
        kw = _build_converse_kwargs(_req())
        assert kw["modelId"] == "model-1"
        assert kw["messages"][0]["role"] == "user"
        assert kw["messages"][0]["content"] == [{"text": "hello"}]
        assert kw["inferenceConfig"]["maxTokens"] == 512
        assert kw["inferenceConfig"]["temperature"] == 0.0
        assert "toolConfig" not in kw

    def test_uses_request_inference_params(self):
        kw = _build_converse_kwargs(_req(max_output_tokens=1024, temperature=0.5))
        assert kw["inferenceConfig"]["maxTokens"] == 1024
        assert kw["inferenceConfig"]["temperature"] == 0.5

    def test_resolves_model_ref(self):
        kw = _build_converse_kwargs(_req(model_id="provider:custom-model"))
        assert kw["modelId"] == "custom-model"

    def test_unflattens_tool_config(self):
        req = _req(
            tools=[Tool("classify", "Classify it", {"type": "object"})],
            forced_tool="classify",
        )
        kw = _build_converse_kwargs(req)
        spec = kw["toolConfig"]["tools"][0]["toolSpec"]
        assert spec["name"] == "classify"
        assert spec["description"] == "Classify it"
        assert spec["inputSchema"] == {"json": {"type": "object"}}
        assert kw["toolConfig"]["toolChoice"] == {"tool": {"name": "classify"}}

    def test_omits_empty_description(self):
        kw = _build_converse_kwargs(_req(tools=[Tool("t", "", {})]))
        assert "description" not in kw["toolConfig"]["tools"][0]["toolSpec"]

    def test_no_tool_choice_when_not_forced(self):
        kw = _build_converse_kwargs(_req(tools=[Tool("t", "", {})], forced_tool=None))
        assert "toolChoice" not in kw["toolConfig"]

    def test_omits_temperature_when_disabled(self):
        kw = _build_converse_kwargs(_req(), include_temperature=False)
        assert "temperature" not in kw["inferenceConfig"]
        assert kw["inferenceConfig"]["maxTokens"] == 512


class TestExtractOutput:
    def test_text_response(self):
        assert _extract_output(_text_response("world")) == {"output_text": "world"}

    def test_tool_response(self):
        tool_input = {"category": "error", "severity": "high"}
        assert _extract_output(_tool_response(tool_input)) == {
            "output_text": "",
            "tool_uses": [tool_input],
        }

    def test_multi_text_blocks(self):
        response = {
            "output": {"message": {"content": [{"text": "hello "}, {"text": "world"}]}}
        }
        assert _extract_output(response) == {"output_text": "hello world"}

    def test_empty_content(self):
        response = {"output": {"message": {"content": []}}}
        assert _extract_output(response) == {"output_text": ""}

    def test_text_and_tool_mixed(self):
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "reasoning here"},
                        {"toolUse": {"name": "classify", "input": {"cat": "error"}}},
                    ]
                }
            }
        }
        assert _extract_output(response) == {
            "output_text": "reasoning here",
            "tool_uses": [{"cat": "error"}],
        }

    def test_missing_output(self):
        assert _extract_output({}) == {"output_text": ""}


class TestComplete:
    def test_returns_canonical_text_response(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("analysis complete")

        response = _provider(mock_boto).complete(_req())

        assert response.output == {"output_text": "analysis complete"}

    def test_returns_tool_uses(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _tool_response({"category": "timeout"})

        response = _provider(mock_boto).complete(
            _req(tools=[Tool("classify", "", {})], forced_tool="classify")
        )

        assert response.output == {
            "output_text": "",
            "tool_uses": [{"category": "timeout"}],
        }

    def test_extracts_usage_tokens(self, mocker):
        mock_boto = mocker.Mock()
        resp = _text_response("ok")
        resp["usage"] = {"inputTokens": 11, "outputTokens": 22}
        mock_boto.converse.return_value = resp

        response = _provider(mock_boto).complete(_req())

        assert response.input_tokens == 11
        assert response.output_tokens == 22

    def test_passes_inference_config_to_backend(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("ok")

        _provider(mock_boto).complete(_req(max_output_tokens=2048, temperature=0.0))

        kwargs = mock_boto.converse.call_args[1]
        assert kwargs["inferenceConfig"]["maxTokens"] == 2048
        assert kwargs["inferenceConfig"]["temperature"] == 0.0

    def test_raises_on_backend_error(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error("ThrottlingException")

        with pytest.raises(ClientError):
            _provider(mock_boto).complete(_req())


class TestTemperatureFallback:
    """Adaptive-thinking models (e.g. Claude Opus 4.8, Sonnet 5) reject a custom
    temperature on Bedrock with ValidationException 'temperature is deprecated for
    this model'. The adapter learns this per-model and retries once without it."""

    @staticmethod
    def _rejection():
        return _client_error(
            "ValidationException",
            "The model returned the following errors: `temperature` is "
            "deprecated for this model.",
        )

    def test_retries_without_temperature_on_rejection(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = [self._rejection(), _text_response("ok")]

        response = _provider(mock_boto).complete(_req())

        assert response.output == {"output_text": "ok"}
        assert mock_boto.converse.call_count == 2
        first, second = mock_boto.converse.call_args_list
        assert "temperature" in first.kwargs["inferenceConfig"]
        assert "temperature" not in second.kwargs["inferenceConfig"]

    def test_learned_model_skips_temperature_on_next_call(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = [
            self._rejection(),
            _text_response("ok"),
            _text_response("ok2"),
        ]
        provider = _provider(mock_boto)

        provider.complete(_req())  # rejects, then retries without temperature
        provider.complete(_req())  # same model: omit temperature up front

        assert mock_boto.converse.call_count == 3
        third = mock_boto.converse.call_args_list[2]
        assert "temperature" not in third.kwargs["inferenceConfig"]

    def test_rejection_is_per_model(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = [
            self._rejection(),
            _text_response("ok"),
            _text_response("ok2"),
        ]
        provider = _provider(mock_boto)

        provider.complete(_req(model_id="us.anthropic.claude-opus-4-8"))  # learns
        provider.complete(_req(model_id="us.anthropic.claude-haiku-4-5"))  # unaffected

        haiku_call = mock_boto.converse.call_args_list[2]
        assert "temperature" in haiku_call.kwargs["inferenceConfig"]

    def test_non_temperature_validation_reraises(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error(
            "ValidationException", "maxTokens exceeds the model limit"
        )

        with pytest.raises(ClientError):
            _provider(mock_boto).complete(_req())

        assert mock_boto.converse.call_count == 1


class TestClassifyError:
    def test_access_denied_aborts_batch(self, mocker):
        provider = _provider(mocker.Mock())
        assert (
            provider.classify_error(_client_error("AccessDeniedException"))
            == ErrorDisposition.ABORT_BATCH
        )

    @pytest.mark.parametrize(
        "code",
        [
            "ThrottlingException",
            "InternalServerException",
            "ModelNotReadyException",
            "ServiceUnavailableException",
        ],
    )
    def test_retryable_codes_retry(self, mocker, code):
        provider = _provider(mocker.Mock())
        assert provider.classify_error(_client_error(code)) == ErrorDisposition.RETRY

    @pytest.mark.parametrize(
        "code", ["ValidationException", "ResourceNotFoundException"]
    )
    def test_non_retryable_codes_fail_row(self, mocker, code):
        provider = _provider(mocker.Mock())
        assert provider.classify_error(_client_error(code)) == ErrorDisposition.FAIL_ROW

    def test_transport_error_retries(self, mocker):
        provider = _provider(mocker.Mock())
        exc = EndpointConnectionError(endpoint_url="https://bedrock.example.com")
        assert provider.classify_error(exc) == ErrorDisposition.RETRY

    def test_generic_exception_fails_row(self, mocker):
        provider = _provider(mocker.Mock())
        assert provider.classify_error(Exception("boom")) == ErrorDisposition.FAIL_ROW

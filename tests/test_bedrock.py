import json
import time

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from llm_worker.config import BedrockConfig
from llm_worker.contract import ContractRequest, OutputFormat, Tool
from llm_worker.providers.base import ErrorDisposition
from llm_worker.providers.bedrock import (
    NON_RETRYABLE_ERROR_CODES,
    RETRYABLE_ERROR_CODES,
    BedrockProvider,
    _build_converse_kwargs,
    _extract_output,
    _PROFILE_ARN_COOLDOWN_SECONDS,
)


def _provider(mock_boto, inference_profiles=None):
    config = BedrockConfig(
        region="us-east-1", inference_profiles=inference_profiles or {}
    )
    return BedrockProvider(config, boto_client=mock_boto)


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


class TestInferenceProfileResolution:
    ARN = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
    ARN_2 = (
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/def456"
    )

    def test_resolves_to_profile_arn_when_configured(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("ok")

        _provider(mock_boto, inference_profiles={"model-1": self.ARN}).complete(
            _req(model_id="model-1")
        )

        assert mock_boto.converse.call_args[1]["modelId"] == self.ARN

    def test_no_profile_configured_uses_bare_model_id(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("ok")

        _provider(mock_boto).complete(_req(model_id="model-1"))

        assert mock_boto.converse.call_args[1]["modelId"] == "model-1"

    def test_profile_lookup_uses_resolved_model_id(self, mocker):
        """The inference-profile map is keyed on the bare model id produced by
        resolve_model_ref, not the raw request.model_id -- a namespaced id
        must still resolve to the configured ARN."""
        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("ok")

        _provider(mock_boto, inference_profiles={"model-1": self.ARN}).complete(
            _req(model_id="provider:model-1")
        )

        assert mock_boto.converse.call_args[1]["modelId"] == self.ARN

    def test_invalid_profile_arn_retries_with_bare_model_id(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = [
            _client_error("ResourceNotFoundException"),
            _text_response("ok"),
        ]

        response = _provider(
            mock_boto, inference_profiles={"model-1": self.ARN}
        ).complete(_req(model_id="model-1"))

        assert mock_boto.converse.call_count == 2
        assert mock_boto.converse.call_args_list[0].kwargs["modelId"] == self.ARN
        assert mock_boto.converse.call_args_list[1].kwargs["modelId"] == "model-1"
        assert response.output == {"output_text": "ok"}

    def test_non_profile_error_with_resolved_arn_still_raises(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error("ThrottlingException")

        with pytest.raises(ClientError):
            _provider(mock_boto, inference_profiles={"model-1": self.ARN}).complete(
                _req(model_id="model-1")
            )

        assert mock_boto.converse.call_count == 1

    def test_no_profile_configured_resource_not_found_raises_without_retry(
        self, mocker
    ):
        """With no inference profile configured, resolved_model == model, so
        a ResourceNotFoundException here means the bare model id itself is
        invalid -- not a bad profile ARN. It must raise immediately, without
        a doomed retry against the same id or a bogus cooldown entry."""
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error("ResourceNotFoundException")
        provider = _provider(mock_boto)

        with pytest.raises(ClientError):
            provider.complete(_req(model_id="model-1"))

        assert mock_boto.converse.call_count == 1
        assert "model-1" not in provider._profile_invalid_since

    def test_unrelated_validation_error_with_resolved_arn_still_raises(self, mocker):
        """A ValidationException unrelated to the ARN (e.g. a token-limit
        error) must not be misdiagnosed as a broken profile -- it should
        propagate immediately, not burn a second call retrying bare."""
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error(
            "ValidationException", "maxTokens exceeds the model limit"
        )

        with pytest.raises(ClientError):
            _provider(mock_boto, inference_profiles={"model-1": self.ARN}).complete(
                _req(model_id="model-1")
            )

        assert mock_boto.converse.call_count == 1

    def test_access_denied_with_resolved_arn_still_raises_immediately(self, mocker):
        """AccessDeniedException signals a broad credentials problem
        (classify_error maps it to ABORT_BATCH) -- it must not be retried
        as if it were an ARN-specific issue, which would delay that."""
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error("AccessDeniedException")

        with pytest.raises(ClientError):
            _provider(mock_boto, inference_profiles={"model-1": self.ARN}).complete(
                _req(model_id="model-1")
            )

        assert mock_boto.converse.call_count == 1

    def test_invalid_arn_is_not_retried_within_cooldown(self, mocker):
        """Once a model's ARN is known-bad, subsequent rows for that model
        should skip straight to the bare model id instead of repeating the
        guaranteed-failing call every time."""
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = [
            _client_error("ResourceNotFoundException"),
            _text_response("ok"),
            _text_response("ok2"),
        ]
        provider = _provider(mock_boto, inference_profiles={"model-1": self.ARN})

        provider.complete(_req(model_id="model-1"))  # fails on ARN, retries bare
        provider.complete(_req(model_id="model-1"))  # should go straight to bare

        assert mock_boto.converse.call_count == 3
        assert mock_boto.converse.call_args_list[2].kwargs["modelId"] == "model-1"

    def test_invalid_arn_is_retried_again_after_cooldown(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = [
            _client_error("ResourceNotFoundException"),
            _text_response("ok"),
            _text_response("ok2"),
        ]
        provider = _provider(mock_boto, inference_profiles={"model-1": self.ARN})

        provider.complete(_req(model_id="model-1"))  # fails on ARN, retries bare
        provider._profile_invalid_since["model-1"] -= _PROFILE_ARN_COOLDOWN_SECONDS + 1
        provider.complete(_req(model_id="model-1"))  # cooldown elapsed, tries ARN again

        assert mock_boto.converse.call_count == 3
        assert mock_boto.converse.call_args_list[2].kwargs["modelId"] == self.ARN

    def test_cooldown_is_per_model(self, mocker):
        """A stale ARN cooldown for one model must not suppress ARN
        resolution for a different model."""
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = [
            _client_error("ResourceNotFoundException"),
            _text_response("ok"),
            _text_response("ok2"),
        ]
        provider = _provider(
            mock_boto,
            inference_profiles={"model-1": self.ARN, "model-2": self.ARN_2},
        )

        provider.complete(_req(model_id="model-1"))  # fails on ARN, enters cooldown
        provider.complete(_req(model_id="model-2"))  # different model, unaffected

        assert mock_boto.converse.call_count == 3
        assert mock_boto.converse.call_args_list[2].kwargs["modelId"] == self.ARN_2


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

    def test_out_of_range_temperature_is_not_treated_as_rejection(self, mocker):
        """A ValidationException that merely mentions 'temperature' because the
        value is out of range is a real input error, not a model-level rejection
        of the parameter — it must not be swallowed-and-retried, and must not
        poison _omit_temperature for later calls to the same model."""
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = _client_error(
            "ValidationException",
            "temperature must be between 0.0 and 1.0",
        )
        provider = _provider(mock_boto)

        with pytest.raises(ClientError):
            provider.complete(_req(model_id="model-1"))

        assert mock_boto.converse.call_count == 1
        assert "model-1" not in provider._omit_temperature

    def test_learned_rejection_survives_profile_resolution_flip(self, mocker):
        """The learned _omit_temperature state must key on the stable bare
        model id, not whichever ARN/bare-id form resolution returns --
        otherwise the ARN-invalid cooldown flipping a model from its profile
        ARN to the bare id would make the provider re-learn a rejection it
        already knows about."""
        arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = [
            self._rejection(),
            _text_response("ok"),
            _text_response("ok2"),
        ]
        provider = _provider(mock_boto, inference_profiles={"model-1": arn})

        provider.complete(
            _req(model_id="model-1")
        )  # resolves to the ARN, rejects, learns

        # Force the ARN into cooldown the same way an invalid-ARN response
        # would, without reassigning _inference_profiles (production never
        # mutates that after construction).
        provider._profile_invalid_since["model-1"] = time.monotonic()
        provider.complete(_req(model_id="model-1"))  # resolves to the bare id this time

        assert mock_boto.converse.call_count == 3
        third = mock_boto.converse.call_args_list[2]
        assert "temperature" not in third.kwargs["inferenceConfig"]


class TestClassifyError:
    def test_access_denied_aborts_batch(self, mocker):
        provider = _provider(mocker.Mock())
        assert (
            provider.classify_error(_client_error("AccessDeniedException"))
            == ErrorDisposition.ABORT_BATCH
        )

    # The parametrized tests below cover *add*-drift (a new code must be classified);
    # these set-equality checks cover *remove*-drift (a code dropped from the set in a
    # bad merge would otherwise pass silently). Together they pin both sets exactly.
    def test_retryable_codes_are_exactly(self):
        assert RETRYABLE_ERROR_CODES == {
            "InternalServerException",
            "ModelNotReadyException",
            "ServiceUnavailableException",
            "ThrottlingException",
        }

    def test_non_retryable_codes_are_exactly(self):
        assert NON_RETRYABLE_ERROR_CODES == {
            "AccessDeniedException",
            "ValidationException",
            "ResourceNotFoundException",
        }

    @pytest.mark.parametrize("code", sorted(RETRYABLE_ERROR_CODES))
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


_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}},
    "required": ["score"],
    "additionalProperties": False,
}


def _profiled_req(*, description="Return the result", thinking="adaptive", effort=None):
    return ContractRequest(
        model_id="us.anthropic.claude-sonnet-5-5",
        prompt="judge this",
        max_output_tokens=4012,
        temperature=0.0,
        tools=[],
        forced_tool=None,
        output_format=OutputFormat("evaluation_result", description, _SCHEMA),
        thinking=thinking,
        effort=effort,
    )


def _json_response(text, stop_reason="end_turn", reasoning=True):
    content = []
    if reasoning:
        content.append({"reasoningContent": {"reasoningText": {"text": "thinking"}}})
    content.append({"text": text})
    return {"output": {"message": {"content": content}}, "stopReason": stop_reason}


class TestRequestProfiles:
    def test_maps_json_outputs_to_output_config(self):
        kw = _build_converse_kwargs(_profiled_req())

        assert kw["outputConfig"] == {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "schema": json.dumps(_SCHEMA),
                        "name": "evaluation_result",
                        "description": "Return the result",
                    }
                },
            }
        }
        assert "toolConfig" not in kw

    def test_omits_an_empty_schema_description(self):
        kw = _build_converse_kwargs(_profiled_req(description=""))
        json_schema = kw["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
        assert "description" not in json_schema

    def test_sends_thinking_and_effort_as_additional_fields(self):
        kw = _build_converse_kwargs(_profiled_req(effort="medium"))
        assert kw["additionalModelRequestFields"] == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        }

    def test_sends_no_effort_when_unset(self):
        kw = _build_converse_kwargs(_profiled_req())
        assert kw["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"}}

    def test_existing_requests_send_no_new_fields(self):
        kw = _build_converse_kwargs(_req())
        assert "outputConfig" not in kw
        assert "additionalModelRequestFields" not in kw
        assert "additionalModelResponseFieldPaths" not in kw

    def test_json_outputs_ask_for_the_stop_details(self):
        kw = _build_converse_kwargs(_profiled_req())
        assert kw["additionalModelResponseFieldPaths"] == ["/stop_details"]

    def test_sends_no_temperature_with_thinking(self, mocker):
        client = mocker.Mock()
        client.converse.return_value = _json_response('{"score": 2}')

        _provider(client).complete(_profiled_req())

        client.converse.assert_called_once()
        assert "temperature" not in client.converse.call_args.kwargs["inferenceConfig"]

    def test_stores_the_json_reply_as_the_tool_input(self, mocker):
        client = mocker.Mock()
        client.converse.return_value = _json_response('{"score": 2}')

        response = _provider(client).complete(_profiled_req())

        assert response.output == {
            "output_text": '{"score": 2}',
            "tool_uses": [{"score": 2}],
        }

    def test_records_a_token_cap_stop_without_a_result(self):
        output = _extract_output(
            _json_response('{"score": ', stop_reason="max_tokens"), json_output=True
        )
        assert output == {
            "output_text": '{"score": ',
            "result_error": "reply is not a JSON object",
            "stop_reason": "max_tokens",
        }

    @pytest.mark.parametrize("stop_reason", ["max_tokens", "malformed_model_output"])
    def test_a_reply_that_did_not_finish_is_not_a_result(self, stop_reason):
        output = _extract_output(
            _json_response('{"score": 1}', stop_reason=stop_reason), json_output=True
        )
        assert output == {
            "output_text": '{"score": 1}',
            "result_error": "reply did not finish",
            "stop_reason": stop_reason,
        }

    def test_a_stop_sequence_reply_is_a_result(self):
        output = _extract_output(
            _json_response('{"score": 1}', stop_reason="stop_sequence"),
            json_output=True,
        )
        assert output["tool_uses"] == [{"score": 1}]

    def test_records_a_refusal_and_its_category(self):
        response = {
            "output": {"message": {"content": []}},
            "stopReason": "content_filtered",
            "additionalModelResponseFields": {
                "stop_details": {
                    "type": "refusal",
                    "category": "reasoning_extraction",
                    "explanation": "This request was blocked.",
                }
            },
        }
        output = _extract_output(response, json_output=True)
        assert output == {
            "output_text": "",
            "stop_reason": "refusal",
            "refusal_category": "reasoning_extraction",
        }

    def test_records_a_reply_without_text(self):
        response = {
            "output": {"message": {"content": []}},
            "stopReason": "content_filtered",
        }
        output = _extract_output(response, json_output=True)
        assert "tool_uses" not in output
        assert output["stop_reason"] == "content_filtered"


def test_client_waits_120_seconds_and_keeps_botocore_retries(mocker):
    boto_client = mocker.patch("llm_worker.providers.bedrock.boto3.client")

    BedrockProvider(BedrockConfig(region="us-east-1", inference_profiles={}))

    config = boto_client.call_args.kwargs["config"]
    assert config.read_timeout >= 120
    # botocore's retries absorb throttling for every Bedrock model.
    assert config.retries is None


def test_publishes_all_request_profile_capabilities():
    assert BedrockProvider.capabilities == ("output_format", "thinking", "effort")


def test_json_outputs_alone_send_no_temperature(mocker):
    client = mocker.Mock()
    client.converse.return_value = _json_response('{"score": 2}', reasoning=False)

    _provider(client).complete(_profiled_req(thinking=None))

    client.converse.assert_called_once()
    assert "temperature" not in client.converse.call_args.kwargs["inferenceConfig"]

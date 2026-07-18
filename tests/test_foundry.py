from types import SimpleNamespace

import anthropic
import httpx
import pytest

from llm_worker.config import FoundryConfig
from llm_worker.contract import ContractRequest, Tool, resolve_model_ref
from llm_worker.providers.base import ErrorDisposition
from llm_worker.providers.foundry import FoundryProvider

FOUNDRY_CONFIG = FoundryConfig(resource="mc-foundry")


def _provider(mock_client):
    # Passing a client skips _build_client, so Azure credentials are never used.
    return FoundryProvider(FOUNDRY_CONFIG, client=mock_client)


def _req(
    prompt="hello",
    max_output_tokens=512,
    temperature=0.0,
    tools=None,
    forced_tool=None,
    model_id="provider:claude-sonnet-4-5",
):
    return ContractRequest(
        model_id=model_id,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        tools=tools or [],
        forced_tool=forced_tool,
    )


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name, tool_input):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input)


def _response(blocks, input_tokens=0, output_tokens=0):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=blocks, usage=usage)


def _status_error(cls, status, message="err"):
    request = httpx.Request("POST", "https://foundry.example.com")
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body=None)


class TestComplete:
    def test_text_response(self, mocker):
        client = mocker.Mock()
        client.messages.create.return_value = _response(
            [_text_block("done")], input_tokens=3, output_tokens=4
        )

        response = _provider(client).complete(_req())

        assert response.output == {"output_text": "done"}
        assert response.input_tokens == 3
        assert response.output_tokens == 4

    def test_tool_use_response(self, mocker):
        client = mocker.Mock()
        client.messages.create.return_value = _response(
            [_tool_use_block("classify", {"cat": "err"})]
        )

        response = _provider(client).complete(
            _req(tools=[Tool("classify", "d", {})], forced_tool="classify")
        )

        assert response.output == {"output_text": "", "tool_uses": [{"cat": "err"}]}

    def test_omits_tool_uses_when_none(self, mocker):
        client = mocker.Mock()
        client.messages.create.return_value = _response([_text_block("text only")])

        response = _provider(client).complete(_req())

        assert response.output == {"output_text": "text only"}

    def test_sends_resolved_model_id(self, mocker):
        client = mocker.Mock()
        client.messages.create.return_value = _response([_text_block("ok")])

        _provider(client).complete(_req())

        assert client.messages.create.call_args.kwargs["model"] == resolve_model_ref(
            "provider:claude-sonnet-4-5"
        )

    def test_sends_max_tokens_and_temperature(self, mocker):
        client = mocker.Mock()
        client.messages.create.return_value = _response([_text_block("ok")])

        _provider(client).complete(_req(max_output_tokens=256, temperature=0.0))

        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.0

    def test_translates_tools_to_anthropic_shape(self, mocker):
        client = mocker.Mock()
        client.messages.create.return_value = _response([_text_block("ok")])

        _provider(client).complete(
            _req(
                tools=[Tool("classify", "Classify", {"type": "object"})],
                forced_tool="classify",
            )
        )

        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["tools"] == [
            {
                "name": "classify",
                "description": "Classify",
                "input_schema": {"type": "object"},
            }
        ]
        assert kwargs["tool_choice"] == {"type": "tool", "name": "classify"}


class TestBuildClient:
    def test_passes_timeout_to_client(self, mocker):
        # A bounded timeout is essential: without it a hung Foundry request blocks the
        # row (and the batch) forever. The client must receive config.timeout.
        fake_foundry = mocker.patch("llm_worker.providers.foundry.AnthropicFoundry")
        mocker.patch("azure.identity.DefaultAzureCredential")
        mocker.patch(
            "azure.identity.get_bearer_token_provider", return_value=lambda: "token"
        )

        FoundryProvider(FoundryConfig(resource="mc-foundry", timeout=90.0))

        assert fake_foundry.call_args.kwargs["timeout"] == 90.0


class TestTemperatureFallback:
    """Some models (e.g. Claude Opus 4.8 on Foundry) reject a custom temperature
    with 'temperature is deprecated for this model'. The adapter learns this
    per-model and retries once without temperature."""

    @staticmethod
    def _reject():
        return _status_error(
            anthropic.BadRequestError,
            400,
            message="temperature is deprecated for this model",
        )

    def test_retries_without_temperature_on_rejection(self, mocker):
        client = mocker.Mock()
        client.messages.create.side_effect = [
            self._reject(),
            _response([_text_block("ok")]),
        ]

        response = _provider(client).complete(_req())

        assert response.output == {"output_text": "ok"}
        assert client.messages.create.call_count == 2
        first, second = client.messages.create.call_args_list
        assert "temperature" in first.kwargs
        assert "temperature" not in second.kwargs

    def test_learned_model_skips_temperature_on_next_call(self, mocker):
        client = mocker.Mock()
        client.messages.create.side_effect = [
            self._reject(),
            _response([_text_block("ok")]),
            _response([_text_block("ok2")]),
        ]
        provider = _provider(client)

        provider.complete(_req())  # rejects, then retries without temperature
        provider.complete(
            _req()
        )  # same model: omit temperature up front (no re-reject)

        assert client.messages.create.call_count == 3
        assert "temperature" not in client.messages.create.call_args_list[2].kwargs

    def test_rejection_is_per_model(self, mocker):
        client = mocker.Mock()
        client.messages.create.side_effect = [
            self._reject(),
            _response([_text_block("ok")]),
            _response([_text_block("ok2")]),
        ]
        provider = _provider(client)

        provider.complete(_req(model_id="provider:claude-opus-4-8"))  # learns to omit
        provider.complete(_req(model_id="provider:claude-haiku-4-5"))  # unaffected

        haiku_call = client.messages.create.call_args_list[2]
        assert "temperature" in haiku_call.kwargs

    def test_non_temperature_bad_request_reraises(self, mocker):
        client = mocker.Mock()
        client.messages.create.side_effect = _status_error(
            anthropic.BadRequestError, 400, message="max_tokens exceeds the limit"
        )

        with pytest.raises(anthropic.BadRequestError):
            _provider(client).complete(_req())

        assert client.messages.create.call_count == 1


class TestClassifyError:
    def test_rate_limit_retries(self, mocker):
        provider = _provider(mocker.Mock())
        assert (
            provider.classify_error(_status_error(anthropic.RateLimitError, 429))
            == ErrorDisposition.RETRY
        )

    def test_internal_server_retries(self, mocker):
        provider = _provider(mocker.Mock())
        assert (
            provider.classify_error(_status_error(anthropic.InternalServerError, 500))
            == ErrorDisposition.RETRY
        )

    def test_timeout_retries(self, mocker):
        provider = _provider(mocker.Mock())
        request = httpx.Request("POST", "https://foundry.example.com")
        assert (
            provider.classify_error(anthropic.APITimeoutError(request=request))
            == ErrorDisposition.RETRY
        )

    def test_auth_aborts_batch(self, mocker):
        provider = _provider(mocker.Mock())
        assert (
            provider.classify_error(_status_error(anthropic.AuthenticationError, 401))
            == ErrorDisposition.ABORT_BATCH
        )

    def test_permission_denied_aborts_batch(self, mocker):
        provider = _provider(mocker.Mock())
        assert (
            provider.classify_error(_status_error(anthropic.PermissionDeniedError, 403))
            == ErrorDisposition.ABORT_BATCH
        )

    def test_bad_request_fails_row(self, mocker):
        provider = _provider(mocker.Mock())
        assert (
            provider.classify_error(_status_error(anthropic.BadRequestError, 400))
            == ErrorDisposition.FAIL_ROW
        )

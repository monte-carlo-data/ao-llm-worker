from types import SimpleNamespace

import anthropic
import httpx

from llm_worker.config import VertexConfig
from llm_worker.contract import ContractRequest, Tool
from llm_worker.providers.base import ErrorDisposition
from llm_worker.providers.vertex import VertexProvider

VERTEX_CONFIG = VertexConfig(
    project="mc-proj",
    region="us-east5",
)


def _provider(mock_client):
    return VertexProvider(VERTEX_CONFIG, client=mock_client)


def _req(
    prompt="hello", max_output_tokens=512, temperature=0.0, tools=None, forced_tool=None
):
    return ContractRequest(
        model_id="provider:claude-sonnet-4-5@20250929",
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
    request = httpx.Request("POST", "https://vertex.example.com")
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

        assert (
            client.messages.create.call_args.kwargs["model"]
            == "claude-sonnet-4-5@20250929"
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
        # A bounded timeout keeps a hung Vertex request from wedging the batch; the
        # client must receive config.timeout.
        fake_vertex = mocker.patch("llm_worker.providers.vertex.AnthropicVertex")

        VertexProvider(VertexConfig(project="mc-proj", region="global", timeout=90.0))

        assert fake_vertex.call_args.kwargs["timeout"] == 90.0

    def test_passes_project_and_region_to_client(self, mocker):
        # Vertex routes requests using project_id/region; the client must receive
        # both from VertexConfig.
        fake_vertex = mocker.patch("llm_worker.providers.vertex.AnthropicVertex")

        VertexProvider(VertexConfig(project="mc-proj", region="us-east5"))

        assert fake_vertex.call_args.kwargs["project_id"] == "mc-proj"
        assert fake_vertex.call_args.kwargs["region"] == "us-east5"


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
        request = httpx.Request("POST", "https://vertex.example.com")
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

    def test_connection_error_retries(self, mocker):
        provider = _provider(mocker.Mock())
        request = httpx.Request("POST", "https://vertex.example.com")
        assert (
            provider.classify_error(anthropic.APIConnectionError(request=request))
            == ErrorDisposition.RETRY
        )

    def test_generic_exception_fails_row(self, mocker):
        provider = _provider(mocker.Mock())
        assert provider.classify_error(ValueError("boom")) == ErrorDisposition.FAIL_ROW

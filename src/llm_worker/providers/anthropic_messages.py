"""Shared Anthropic Messages API adapter.

Claude on Vertex AI (``AnthropicVertex``) and Microsoft Foundry
(``AnthropicFoundry``) both speak the Anthropic Messages API, so the
request/response translation and error classification are identical — only
client construction differs. That shared logic lives here; the concrete
:mod:`~llm_worker.providers.vertex` and :mod:`~llm_worker.providers.foundry`
adapters are thin subclasses that only build their client.

Like the Bedrock adapter, this honors the row's ``model_id`` (resolved via
:func:`~llm_worker.contract.resolve_model_ref`) rather than a configured model.

Anthropic's tool-use API takes the flat ``{name, description, input_schema}``
shape and ``{"type": "tool", "name": ...}`` choice — which is the MC contract
v1 shape verbatim — so tool translation here is near-passthrough. Most Claude
models accept ``temperature=0``, but some (e.g. Opus 4.8 on Foundry) reject a
custom temperature outright — so this adapter learns, per model, to drop it and
retry (see :meth:`AnthropicMessagesProvider.complete`).

A JSON-outputs request maps to ``output_config.format``, thinking to ``thinking``,
and effort to ``output_config.effort``.
"""

from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from llm_worker.contract import ContractRequest, resolve_model_ref
from llm_worker.providers.base import (
    ErrorDisposition,
    LLMProvider,
    LLMResponse,
    build_output,
    is_temperature_rejection_message,
    omits_temperature,
)


class AnthropicMessagesProvider(LLMProvider):
    """Base adapter for backends that speak the Anthropic Messages API."""

    # No "effort" until it's been tried on Vertex and Foundry.
    capabilities = ("output_format", "thinking")

    def __init__(self, client):
        super().__init__()
        self._client = client

    def complete(self, request: ContractRequest) -> LLMResponse:
        model = resolve_model_ref(request.model_id)
        response = self._complete_with_temperature_fallback(
            model,
            call=lambda include_temperature: self._client.messages.create(
                **_build_kwargs(request, model, include_temperature=include_temperature)
            ),
            exception_type=BadRequestError,
            is_rejection=_is_temperature_rejection,
            log_event="anthropic_temperature_unsupported",
            omit_temperature=omits_temperature(request),
        )
        return _to_llm_response(response, request)

    def classify_error(self, exc: BaseException) -> ErrorDisposition:
        if isinstance(
            exc,
            (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError),
        ):
            return ErrorDisposition.RETRY
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return ErrorDisposition.ABORT_BATCH
        return ErrorDisposition.FAIL_ROW  # BadRequestError, NotFoundError, other


def _is_temperature_rejection(exc: BadRequestError) -> bool:
    return is_temperature_rejection_message(str(exc))


def _build_kwargs(
    request: ContractRequest, model: str, *, include_temperature: bool
) -> dict:
    kwargs: dict = {
        "model": model,
        "max_tokens": request.max_output_tokens,
        "messages": [{"role": "user", "content": request.prompt}],
    }
    if include_temperature:
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in request.tools
        ]
        if request.forced_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": request.forced_tool}
    output_config: dict = {}
    if request.output_format is not None:
        # The Messages API takes only the schema; name and description are for Bedrock.
        output_config["format"] = {
            "type": "json_schema",
            "schema": request.output_format.schema,
        }
    if request.effort is not None:
        output_config["effort"] = request.effort
    if output_config:
        kwargs["output_config"] = output_config
    if request.thinking is not None:
        kwargs["thinking"] = {"type": request.thinking}
    return kwargs


def _to_llm_response(response, request: ContractRequest) -> LLMResponse:
    texts: list[str] = []
    tool_uses = []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            tool_uses.append(block.input)
    output = build_output(
        texts,
        tool_uses,
        json_output=request.output_format is not None,
        stop_reason=getattr(response, "stop_reason", None),
        refusal_category=_refusal_category(response),
    )

    usage = response.usage
    return LLMResponse(
        output=output,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
    )


def _refusal_category(response) -> str | None:
    stop_details = getattr(response, "stop_details", None)
    return getattr(stop_details, "category", None) if stop_details else None

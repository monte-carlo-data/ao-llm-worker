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
"""

import logging

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
from llm_worker.providers.base import ErrorDisposition, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicMessagesProvider(LLMProvider):
    """Base adapter for backends that speak the Anthropic Messages API."""

    def __init__(self, client):
        self._client = client
        # Resolved model refs learned to reject a custom temperature. Per-model,
        # not process-wide: this adapter serves many models (it honors model_id),
        # and only some deprecate temperature — a shared flag would needlessly
        # strip determinism from the models that still accept it.
        self._omit_temperature: set[str] = set()

    def complete(self, request: ContractRequest) -> LLMResponse:
        model = resolve_model_ref(request.model_id)
        include_temperature = model not in self._omit_temperature
        try:
            response = self._client.messages.create(
                **_build_kwargs(request, model, include_temperature=include_temperature)
            )
        except BadRequestError as exc:
            if not include_temperature or not _is_temperature_rejection(exc):
                raise
            # Learn that this model rejects a custom temperature and retry once
            # without it. Determinism degrades to the model default here.
            logger.warning("anthropic_temperature_unsupported", extra={"model": model})
            self._omit_temperature.add(model)
            response = self._client.messages.create(
                **_build_kwargs(request, model, include_temperature=False)
            )
        return _to_llm_response(response)

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
    return "temperature" in str(exc).lower()


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
    return kwargs


def _to_llm_response(response) -> LLMResponse:
    output_text = ""
    tool_uses = []
    for block in response.content:
        if block.type == "text":
            output_text += block.text
        elif block.type == "tool_use":
            tool_uses.append(block.input)

    output: dict = {"output_text": output_text}
    if tool_uses:
        output["tool_uses"] = tool_uses

    usage = response.usage
    return LLMResponse(
        output=output,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
    )

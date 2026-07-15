"""Vertex AI adapter for the LLM worker.

Translates a v1 :class:`~llm_worker.contract.ContractRequest` to Claude on
Vertex AI (via the ``AnthropicVertex`` SDK) and back into the canonical
envelope, and classifies ``anthropic`` exceptions. Auth is ambient Application
Default Credentials via GKE Workload Identity — no API key. Retry and
orchestration live in the executor.

Anthropic's tool-use API takes the flat ``{name, description, input_schema}``
shape and ``{"type": "tool", "name": ...}`` choice — which is the MC contract
v1 shape verbatim — so tool translation here is near-passthrough. Claude
accepts ``temperature=0``, so there is no temperature-rejection fallback.
"""

import logging

from anthropic import (
    AnthropicVertex,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from llm_worker.config import VertexConfig
from llm_worker.contract import ContractRequest
from llm_worker.providers.base import ErrorDisposition, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class VertexProvider(LLMProvider):
    def __init__(self, config: VertexConfig, client: AnthropicVertex | None = None):
        self._model = config.model
        # AnthropicVertex resolves credentials via google.auth ADC (Workload
        # Identity on GKE); no key is passed. Tests inject a mock client.
        self._client = client or AnthropicVertex(
            project_id=config.project, region=config.region
        )

    def complete(self, request: ContractRequest) -> LLMResponse:
        response = self._client.messages.create(**self._build_kwargs(request))
        return _to_llm_response(response)

    def _build_kwargs(self, request: ContractRequest) -> dict:
        # The incoming model_id is a provider-blind ref; a single-model adapter
        # always targets its configured Vertex model (Decision #3).
        kwargs: dict = {
            "model": self._model,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
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

    def classify_error(self, exc: BaseException) -> ErrorDisposition:
        if isinstance(
            exc,
            (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError),
        ):
            return ErrorDisposition.RETRY
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return ErrorDisposition.ABORT_BATCH
        return ErrorDisposition.FAIL_ROW  # BadRequestError, NotFoundError, other


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

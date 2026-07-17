"""AWS Bedrock adapter for the LLM worker.

Translates a v1 :class:`~llm_worker.contract.ContractRequest` to the Bedrock
Converse API and back — re-nesting the flat contract tool spec into Bedrock's
`toolSpec`/`inputSchema`/`toolChoice` shape — and classifies botocore/Bedrock
exceptions. Retry and orchestration live in the executor.
"""

import logging

import boto3
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from llm_worker.config import BedrockConfig
from llm_worker.contract import ContractRequest, resolve_model_ref
from llm_worker.providers.base import ErrorDisposition, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

NON_RETRYABLE_ERROR_CODES = {
    "AccessDeniedException",
    "ValidationException",
    "ResourceNotFoundException",
}

RETRYABLE_ERROR_CODES = {
    "InternalServerException",
    "ModelNotReadyException",
    "ServiceUnavailableException",
    "ThrottlingException",
}

RETRYABLE_TRANSPORT_ERRORS = (
    EndpointConnectionError,
    ConnectTimeoutError,
    ReadTimeoutError,
    ConnectionClosedError,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ClientError):
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in NON_RETRYABLE_ERROR_CODES:
            return False
        return error_code in RETRYABLE_ERROR_CODES
    return isinstance(exc, RETRYABLE_TRANSPORT_ERRORS)


class BedrockProvider(LLMProvider):
    def __init__(self, config: BedrockConfig, boto_client=None):
        self._client = boto_client or boto3.client(
            "bedrock-runtime", region_name=config.region
        )
        # Resolved model refs learned to reject a custom temperature — the
        # adaptive-thinking models (Opus 4.7+, Sonnet 5) return a ValidationException
        # "temperature is deprecated for this model". Per-model, not process-wide: the
        # adapter honors model_id and only some models deprecate it (Haiku 4.5 still
        # accepts it), so a shared flag would needlessly drop determinism for the rest.
        self._omit_temperature: set[str] = set()

    def complete(self, request: ContractRequest) -> LLMResponse:
        model = resolve_model_ref(request.model_id)
        include_temperature = model not in self._omit_temperature
        try:
            response = self._client.converse(
                **_build_converse_kwargs(
                    request, model, include_temperature=include_temperature
                )
            )
        except ClientError as exc:
            if not include_temperature or not _is_temperature_rejection(exc):
                raise
            # Learn that this model rejects a custom temperature and retry once
            # without it. Determinism degrades to the model default here.
            logger.warning("bedrock_temperature_unsupported", extra={"model": model})
            self._omit_temperature.add(model)
            response = self._client.converse(
                **_build_converse_kwargs(request, model, include_temperature=False)
            )
        usage = response.get("usage", {})
        return LLMResponse(
            output=_extract_output(response),
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )

    def classify_error(self, exc: BaseException) -> ErrorDisposition:
        if isinstance(exc, ClientError):
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code == "AccessDeniedException":
                return ErrorDisposition.ABORT_BATCH
        if _is_retryable(exc):
            return ErrorDisposition.RETRY
        return ErrorDisposition.FAIL_ROW


def _is_temperature_rejection(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    return (
        error.get("Code") == "ValidationException"
        and "temperature" in str(error.get("Message", "")).lower()
    )


def _build_converse_kwargs(
    request: ContractRequest,
    model: str | None = None,
    *,
    include_temperature: bool = True,
) -> dict:
    inference_config: dict = {"maxTokens": request.max_output_tokens}
    if include_temperature:
        inference_config["temperature"] = request.temperature
    kwargs = {
        "modelId": model if model is not None else resolve_model_ref(request.model_id),
        "messages": [{"role": "user", "content": [{"text": request.prompt}]}],
        "inferenceConfig": inference_config,
    }
    if request.tools:
        kwargs["toolConfig"] = _build_tool_config(request)
    return kwargs


def _build_tool_config(request: ContractRequest) -> dict:
    tools = []
    for tool in request.tools:
        spec: dict = {"name": tool.name, "inputSchema": {"json": tool.input_schema}}
        if tool.description:
            spec["description"] = tool.description
        tools.append({"toolSpec": spec})

    tool_config: dict = {"tools": tools}
    if request.forced_tool:
        tool_config["toolChoice"] = {"tool": {"name": request.forced_tool}}
    return tool_config


def _extract_output(response: dict) -> dict:
    output_message = response.get("output", {}).get("message", {})
    content = output_message.get("content", [])

    output_text = ""
    tool_uses = []
    for block in content:
        if block.get("text"):
            output_text += block["text"]
        elif block.get("toolUse"):
            tool_use = block["toolUse"]
            if tool_use.get("input"):
                tool_uses.append(tool_use["input"])

    if tool_uses:
        return {"output_text": output_text, "tool_uses": tool_uses}
    return {"output_text": output_text}

"""AWS Bedrock adapter for the LLM worker.

Translates a v1 :class:`~llm_worker.contract.ContractRequest` to the Bedrock
Converse API and back — re-nesting the flat contract tool spec into Bedrock's
`toolSpec`/`inputSchema`/`toolChoice` shape — and classifies botocore/Bedrock
exceptions. Retry and orchestration live in the executor.
"""

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
        super().__init__()
        self._client = boto_client or boto3.client(
            "bedrock-runtime", region_name=config.region
        )

    def complete(self, request: ContractRequest) -> LLMResponse:
        model = resolve_model_ref(request.model_id)
        response = self._complete_with_temperature_fallback(
            model,
            call=lambda include_temperature: self._client.converse(
                **_build_converse_kwargs(
                    request, model, include_temperature=include_temperature
                )
            ),
            exception_type=ClientError,
            is_rejection=_is_temperature_rejection,
            log_event="bedrock_temperature_unsupported",
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


_TEMPERATURE_REJECTION_KEYWORDS = ("not support", "unsupported", "deprecated")


def _is_temperature_rejection(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    if error.get("Code") != "ValidationException":
        return False
    message = str(error.get("Message", "")).lower()
    # Genuine rejections read like "`temperature` is deprecated for this
    # model" or "model does not support temperature" — not range/format
    # validation errors like "temperature must be between 0.0 and 1.0", which
    # must propagate rather than being swallowed and retried.
    return "temperature" in message and any(
        keyword in message for keyword in _TEMPERATURE_REJECTION_KEYWORDS
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
            if "input" in tool_use:
                tool_uses.append(tool_use["input"])

    if tool_uses:
        return {"output_text": output_text, "tool_uses": tool_uses}
    return {"output_text": output_text}

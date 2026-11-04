"""AWS Bedrock adapter for the LLM worker.

Translates a v1 :class:`~llm_worker.contract.ContractRequest` to the Bedrock
Converse API and back — re-nesting the flat contract tool spec into Bedrock's
`toolSpec`/`inputSchema`/`toolChoice` shape — and classifies botocore/Bedrock
exceptions. Retry and orchestration live in the executor, with one exception:
this adapter retries once internally when the configured cost-attribution
inference-profile ARN turns out to be stale or invalid, falling back to the
bare model id before deferring to the executor's retry loop for everything
else.

A JSON-outputs request maps to Converse's ``outputConfig.textFormat``. Converse has
no fields for Claude's thinking and effort, so they go through
``additionalModelRequestFields`` in the Messages API's own shape.
"""

import json
import logging
import time

import boto3
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from llm_worker.config import BedrockConfig
from llm_worker.contract import ContractRequest, resolve_model_ref
from llm_worker.providers.base import (
    ErrorDisposition,
    LLMProvider,
    LLMResponse,
    is_temperature_rejection_message,
    json_output_result,
)

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

# Narrower than NON_RETRYABLE_ERROR_CODES: only a missing-ARN signal.
_PROFILE_ARN_ERROR_CODES = {"ResourceNotFoundException"}

# Once a model's configured ARN is seen to be invalid, stop retrying it for
# this long -- self-heals a transient rejection without needing a redeploy,
# while not repeating the doubled-call cost on every single row in between.
_PROFILE_ARN_COOLDOWN_SECONDS = 300

# botocore's default read timeout is 60s. Bedrock compiles a new JSON-outputs schema
# on its first request, which can take about 90s.
BEDROCK_READ_TIMEOUT_SECONDS = 120


def _error_code(exc: ClientError) -> str:
    return exc.response.get("Error", {}).get("Code", "")


def _is_profile_arn_error(exc: ClientError) -> bool:
    return _error_code(exc) in _PROFILE_ARN_ERROR_CODES


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ClientError):
        error_code = _error_code(exc)
        if error_code in NON_RETRYABLE_ERROR_CODES:
            return False
        return error_code in RETRYABLE_ERROR_CODES
    return isinstance(exc, RETRYABLE_TRANSPORT_ERRORS)


class BedrockProvider(LLMProvider):
    def __init__(self, config: BedrockConfig, boto_client=None):
        super().__init__()
        self._client = boto_client or boto3.client(
            "bedrock-runtime",
            region_name=config.region,
            config=Config(read_timeout=BEDROCK_READ_TIMEOUT_SECONDS),
        )
        self._inference_profiles = config.inference_profiles
        self._profile_invalid_since: dict[str, float] = {}
        self._unmapped_models_warned: set[str] = set()

    def complete(self, request: ContractRequest) -> LLMResponse:
        model = resolve_model_ref(request.model_id)
        resolved_model = self._resolve_profile(model)
        try:
            response = self._converse(request, model, resolved_model)
        except ClientError as exc:
            if resolved_model == model or not _is_profile_arn_error(exc):
                raise
            self._profile_invalid_since[model] = time.monotonic()
            logger.warning(
                "bedrock_inference_profile_invalid",
                extra={
                    "model_id": model,
                    "resolved_model": resolved_model,
                    "error": str(exc),
                },
            )
            response = self._converse(request, model, model)
        usage = response.get("usage", {})
        return LLMResponse(
            output=_extract_output(
                response, json_output=request.output_format is not None
            ),
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )

    def _resolve_profile(self, model: str) -> str:
        invalid_since = self._profile_invalid_since.get(model)
        if (
            invalid_since is not None
            and time.monotonic() - invalid_since < _PROFILE_ARN_COOLDOWN_SECONDS
        ):
            return model
        if model not in self._inference_profiles:
            if model not in self._unmapped_models_warned:
                self._unmapped_models_warned.add(model)
                logger.warning(
                    "bedrock_inference_profile_unmapped",
                    extra={"model_id": model},
                )
            return model
        return self._inference_profiles[model]

    def _converse(self, request: ContractRequest, model: str, resolved_model: str):
        # `model` (stable, never an ARN) keys the temperature-fallback learned
        # state so it isn't lost when resolution flips between ARN/bare id.
        return self._complete_with_temperature_fallback(
            model,
            call=lambda include_temperature: self._client.converse(
                **_build_converse_kwargs(
                    request, resolved_model, include_temperature=include_temperature
                )
            ),
            exception_type=ClientError,
            is_rejection=_is_temperature_rejection,
            log_event="bedrock_temperature_unsupported",
            omit_temperature=request.thinking is not None,
        )

    def classify_error(self, exc: BaseException) -> ErrorDisposition:
        if isinstance(exc, ClientError):
            if _error_code(exc) == "AccessDeniedException":
                return ErrorDisposition.ABORT_BATCH
        if _is_retryable(exc):
            return ErrorDisposition.RETRY
        return ErrorDisposition.FAIL_ROW


def _is_temperature_rejection(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    if error.get("Code") != "ValidationException":
        return False
    return is_temperature_rejection_message(str(error.get("Message", "")))


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
    if request.output_format is not None:
        kwargs["outputConfig"] = _build_output_config(request)
    additional_fields: dict = {}
    if request.thinking is not None:
        additional_fields["thinking"] = {"type": request.thinking}
    if request.effort is not None:
        additional_fields["output_config"] = {"effort": request.effort}
    if additional_fields:
        kwargs["additionalModelRequestFields"] = additional_fields
    return kwargs


def _build_output_config(request: ContractRequest) -> dict:
    output_format = request.output_format
    assert output_format is not None
    # Converse takes the schema as a JSON string.
    json_schema: dict = {
        "schema": json.dumps(output_format.schema),
        "name": output_format.name,
    }
    if output_format.description:
        json_schema["description"] = output_format.description
    return {
        "textFormat": {
            "type": "json_schema",
            "structure": {"jsonSchema": json_schema},
        }
    }


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


def _extract_output(response: dict, *, json_output: bool = False) -> dict:
    output_message = response.get("output", {}).get("message", {})
    content = output_message.get("content", [])

    # Thinking arrives as ``reasoningContent`` blocks, which have no ``text`` key.
    texts: list[str] = []
    tool_uses = []
    for block in content:
        if block.get("text"):
            texts.append(block["text"])
        elif block.get("toolUse"):
            tool_use = block["toolUse"]
            if "input" in tool_use:
                tool_uses.append(tool_use["input"])
    output_text = "".join(texts)

    if json_output:
        # Converse has no ``refusal`` stop reason and no refusal category, so a
        # refusal is recorded under whatever stop reason Converse maps it to.
        return json_output_result(
            output_text=output_text,
            final_text=texts[-1] if texts else "",
            stop_reason=response.get("stopReason"),
        )

    if tool_uses:
        return {"output_text": output_text, "tool_uses": tool_uses}
    return {"output_text": output_text}

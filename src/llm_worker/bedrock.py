"""Bedrock client for making LLM calls."""

import json
import logging
import threading
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import boto3
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from llm_worker.clickhouse import LLMInput, LLMResult
from llm_worker.config import BedrockConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PARAMS = {
    "maxTokens": 512,
    "temperature": 0,
}

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


class BedrockClient:
    def __init__(self, config: BedrockConfig, boto_client=None):
        self._client = boto_client or boto3.client(
            "bedrock-runtime", region_name=config.region
        )
        self._max_workers = config.max_workers
        self._retry_max_attempts = config.retry_max_attempts
        self._retry_max_backoff = config.retry_max_backoff

    def _converse(self, **kwargs) -> dict:
        retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._retry_max_attempts),
            wait=wait_exponential(multiplier=1, max=self._retry_max_backoff),
            reraise=True,
            before_sleep=lambda state: logger.warning(
                "bedrock_retry",
                extra={
                    "attempt": state.attempt_number,
                    "model": kwargs.get("modelId", "unknown"),
                    "error": str(state.outcome.exception())
                    if state.outcome
                    else "unknown",
                },
            ),
        )
        for attempt in retrying:
            with attempt:
                return self._client.converse(**kwargs)
        raise RuntimeError("Retrying exhausted without result")

    @staticmethod
    def _aborted_result(input_row: LLMInput) -> LLMResult:
        return LLMResult(
            batch_id=input_row.batch_id,
            row_id=input_row.row_id,
            response="",
            status="failed",
            error="Batch aborted",
        )

    def invoke(
        self,
        input_row: LLMInput,
        abort_event: threading.Event | None = None,
    ) -> LLMResult:
        if abort_event and abort_event.is_set():
            logger.info(
                "row_skipped_abort",
                extra={
                    "row_id": str(input_row.row_id),
                    "batch_id": str(input_row.batch_id),
                },
            )
            return self._aborted_result(input_row)

        try:
            params = json.loads(input_row.params) if input_row.params else {}
            tool_config = (
                json.loads(input_row.tool_config) if input_row.tool_config else {}
            )
        except json.JSONDecodeError as e:
            logger.warning(
                "row_invalid_json",
                extra={"row_id": str(input_row.row_id), "error": str(e)},
            )
            return LLMResult(
                batch_id=input_row.batch_id,
                row_id=input_row.row_id,
                response="",
                status="failed",
                error=f"Invalid JSON in params or tool_config: {e}",
            )

        try:
            request_params = _build_request(
                input_row.model_id, input_row.prompt, params, tool_config
            )
            response = self._converse(**request_params)
            output = _extract_output(response)
            usage = response.get("usage", {})
            logger.info(
                "row_complete",
                extra={
                    "row_id": str(input_row.row_id),
                    "batch_id": str(input_row.batch_id),
                    "model_id": input_row.model_id,
                    "input_tokens": usage.get("inputTokens", 0),
                    "output_tokens": usage.get("outputTokens", 0),
                },
            )
            return LLMResult(
                batch_id=input_row.batch_id,
                row_id=input_row.row_id,
                response=json.dumps(output),
                status="complete",
                error="",
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "AccessDeniedException" and abort_event:
                logger.error(
                    "row_access_denied",
                    extra={
                        "row_id": str(input_row.row_id),
                        "batch_id": str(input_row.batch_id),
                        "model_id": input_row.model_id,
                        "action": "abort_batch",
                    },
                )
                abort_event.set()
            else:
                logger.error(
                    "row_client_error",
                    extra={
                        "row_id": str(input_row.row_id),
                        "code": error_code,
                        "error": str(e),
                    },
                )
            return LLMResult(
                batch_id=input_row.batch_id,
                row_id=input_row.row_id,
                response="",
                status="failed",
                error=str(e),
            )
        except Exception as e:
            logger.error(
                "row_unexpected_error",
                extra={"row_id": str(input_row.row_id), "error": str(e)},
            )
            return LLMResult(
                batch_id=input_row.batch_id,
                row_id=input_row.row_id,
                response="",
                status="failed",
                error=str(e),
            )

    def process_batch_iter(self, inputs: list[LLMInput]) -> Iterator[LLMResult]:
        abort_event = threading.Event()
        pending_inputs = iter(inputs)
        aborted_rows: list[LLMInput] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            in_flight: set[Future[LLMResult]] = set()

            while len(in_flight) < self._max_workers:
                try:
                    row = next(pending_inputs)
                except StopIteration:
                    break
                in_flight.add(executor.submit(self.invoke, row, abort_event))

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()

                if abort_event.is_set():
                    aborted_rows.extend(list(pending_inputs))
                    break

                while len(in_flight) < self._max_workers:
                    try:
                        row = next(pending_inputs)
                    except StopIteration:
                        break
                    in_flight.add(executor.submit(self.invoke, row, abort_event))

            for future in in_flight:
                future.cancel()

            for future in in_flight:
                if future.cancelled():
                    continue
                yield future.result()

        for row in aborted_rows:
            logger.info(
                "row_skipped_unsubmitted",
                extra={"row_id": str(row.row_id), "batch_id": str(row.batch_id)},
            )
            yield self._aborted_result(row)

    def process_batch(self, inputs: list[LLMInput]) -> list[LLMResult]:
        return list(self.process_batch_iter(inputs))


def _build_request(model_id: str, prompt: str, params: dict, tool_config: dict) -> dict:
    request = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        "inferenceConfig": {
            "maxTokens": params.get("maxTokens", DEFAULT_MODEL_PARAMS["maxTokens"]),
            "temperature": params.get(
                "temperature", DEFAULT_MODEL_PARAMS["temperature"]
            ),
        },
    }
    if tool_config:
        request["toolConfig"] = tool_config
    return request


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

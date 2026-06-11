import json
import threading
import time
from uuid import UUID

from llm_worker.bedrock import BedrockClient, _build_request, _extract_output
from llm_worker.clickhouse import LLMInput
from llm_worker.config import BedrockConfig

BEDROCK_CONFIG = BedrockConfig(
    region="us-east-1",
    max_workers=3,
    retry_max_attempts=1,
    retry_max_backoff=1,
)

BATCH_1 = UUID("00000000-0000-0000-0000-000000000001")
ROW_1 = UUID("00000000-0000-0000-0000-000000000011")


def _make_input(
    batch_id=BATCH_1,
    row_id=ROW_1,
    model_id="anthropic.claude-3-sonnet",
    prompt="test prompt",
    params="{}",
    tool_config="",
):
    return LLMInput(batch_id, row_id, model_id, prompt, params, tool_config)


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


class TestBuildRequest:
    def test_basic_request(self):
        req = _build_request("model-1", "hello", {}, {})
        assert req["modelId"] == "model-1"
        assert req["messages"][0]["role"] == "user"
        assert req["messages"][0]["content"] == [{"text": "hello"}]
        assert "toolConfig" not in req

    def test_default_params(self):
        req = _build_request("model-1", "hello", {}, {})
        assert req["inferenceConfig"]["maxTokens"] == 512
        assert req["inferenceConfig"]["temperature"] == 0
        assert "topP" not in req["inferenceConfig"]

    def test_custom_params(self):
        req = _build_request(
            "model-1", "hello", {"maxTokens": 1024, "temperature": 0.5}, {}
        )
        assert req["inferenceConfig"]["maxTokens"] == 1024
        assert req["inferenceConfig"]["temperature"] == 0.5
        assert "topP" not in req["inferenceConfig"]

    def test_top_p_in_params_is_ignored(self):
        req = _build_request("model-1", "hello", {"topP": 0.95}, {})
        assert "topP" not in req["inferenceConfig"]

    def test_tool_config_included(self):
        tools = {"tools": [{"name": "classify"}], "toolChoice": {"auto": {}}}
        req = _build_request("model-1", "hello", {}, tools)
        assert req["toolConfig"] == tools

    def test_empty_tool_config_excluded(self):
        req = _build_request("model-1", "hello", {}, {})
        assert "toolConfig" not in req


class TestExtractOutput:
    def test_text_response(self):
        result = _extract_output(_text_response("world"))
        assert result == {"output_text": "world"}

    def test_tool_response(self):
        tool_input = {"category": "error", "severity": "high"}
        result = _extract_output(_tool_response(tool_input))
        assert result == {"output_text": "", "tool_uses": [tool_input]}

    def test_multi_text_blocks(self):
        response = {
            "output": {"message": {"content": [{"text": "hello "}, {"text": "world"}]}}
        }
        result = _extract_output(response)
        assert result == {"output_text": "hello world"}

    def test_empty_content(self):
        response = {"output": {"message": {"content": []}}}
        result = _extract_output(response)
        assert result == {"output_text": ""}

    def test_text_and_tool_mixed(self):
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "reasoning here"},
                        {
                            "toolUse": {
                                "toolUseId": "1",
                                "name": "classify",
                                "input": {"cat": "error"},
                            }
                        },
                    ]
                }
            }
        }
        result = _extract_output(response)
        assert result == {
            "output_text": "reasoning here",
            "tool_uses": [{"cat": "error"}],
        }

    def test_multiple_tool_uses(self):
        response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "1",
                                "name": "classify",
                                "input": {"cat": "error"},
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "2",
                                "name": "extract",
                                "input": {"sev": "high"},
                            }
                        },
                    ]
                }
            }
        }
        result = _extract_output(response)
        assert result == {
            "output_text": "",
            "tool_uses": [{"cat": "error"}, {"sev": "high"}],
        }

    def test_missing_output(self):
        result = _extract_output({})
        assert result == {"output_text": ""}


class TestInvoke:
    def test_successful_text_call(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("analysis complete")
        client = BedrockClient(BEDROCK_CONFIG, boto_client=mock_boto)

        result = client.invoke(_make_input())

        assert result.status == "complete"
        assert result.error == ""
        assert json.loads(result.response) == {"output_text": "analysis complete"}

    def test_successful_tool_call(self, mocker):
        mock_boto = mocker.Mock()
        tool_input = {"category": "timeout"}
        mock_boto.converse.return_value = _tool_response(tool_input)
        client = BedrockClient(BEDROCK_CONFIG, boto_client=mock_boto)

        result = client.invoke(
            _make_input(tool_config='{"tools": [{"name": "classify"}]}')
        )

        assert result.status == "complete"
        assert json.loads(result.response) == {
            "output_text": "",
            "tool_uses": [tool_input],
        }

    def test_bedrock_error(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.side_effect = Exception("Bedrock unavailable")
        client = BedrockClient(BEDROCK_CONFIG, boto_client=mock_boto)

        result = client.invoke(_make_input())

        assert result.status == "failed"
        assert "Bedrock unavailable" in result.error

    def test_invalid_params_json(self, mocker):
        mock_boto = mocker.Mock()
        client = BedrockClient(BEDROCK_CONFIG, boto_client=mock_boto)

        result = client.invoke(_make_input(params="not json"))

        assert result.status == "failed"
        assert "Invalid JSON" in result.error
        mock_boto.converse.assert_not_called()

    def test_invalid_tool_config_json(self, mocker):
        mock_boto = mocker.Mock()
        client = BedrockClient(BEDROCK_CONFIG, boto_client=mock_boto)

        result = client.invoke(_make_input(tool_config="not json"))

        assert result.status == "failed"
        assert "Invalid JSON" in result.error
        mock_boto.converse.assert_not_called()

    def test_preserves_batch_and_row_id(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("ok")
        client = BedrockClient(BEDROCK_CONFIG, boto_client=mock_boto)

        b = UUID("00000000-0000-0000-0000-000000000099")
        r = UUID("00000000-0000-0000-0000-000000000042")
        result = client.invoke(_make_input(batch_id=b, row_id=r))

        assert result.batch_id == b
        assert result.row_id == r

    def test_custom_params_passed_to_bedrock(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("ok")
        client = BedrockClient(BEDROCK_CONFIG, boto_client=mock_boto)

        client.invoke(_make_input(params='{"maxTokens": 2048, "temperature": 0.0}'))

        call_kwargs = mock_boto.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["maxTokens"] == 2048
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.0


class TestProcessBatch:
    def test_processes_all_rows(self, mocker):
        mock_boto = mocker.Mock()
        mock_boto.converse.return_value = _text_response("done")
        client = BedrockClient(BEDROCK_CONFIG, boto_client=mock_boto)
        inputs = [_make_input(row_id=UUID(int=i)) for i in range(5)]

        results = client.process_batch(inputs)

        assert len(results) == 5
        assert all(r.status == "complete" for r in results)

    def test_mixed_success_and_failure(self, mocker):
        mock_boto = mocker.Mock()
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception("fail")
            return _text_response("ok")

        mock_boto.converse.side_effect = side_effect
        config = BedrockConfig(
            region="us-east-1", max_workers=1, retry_max_attempts=1, retry_max_backoff=1
        )
        client = BedrockClient(config, boto_client=mock_boto)
        inputs = [_make_input(row_id=UUID(int=i)) for i in range(4)]

        results = client.process_batch(inputs)

        assert len(results) == 4
        completed = [r for r in results if r.status == "complete"]
        failed = [r for r in results if r.status == "failed"]
        assert len(completed) == 2
        assert len(failed) == 2

    def test_empty_input(self, mocker):
        mock_boto = mocker.Mock()
        client = BedrockClient(BEDROCK_CONFIG, boto_client=mock_boto)

        results = client.process_batch([])

        assert results == []
        mock_boto.converse.assert_not_called()

    def test_limits_in_flight_work_to_max_workers(self, mocker):
        mock_boto = mocker.Mock()
        config = BedrockConfig(
            region="us-east-1", max_workers=2, retry_max_attempts=1, retry_max_backoff=1
        )
        client = BedrockClient(config, boto_client=mock_boto)
        inputs = [_make_input(row_id=UUID(int=i)) for i in range(5)]
        started = 0
        peak_started = 0
        lock = threading.Lock()
        release = threading.Event()

        def side_effect(**kwargs):
            nonlocal started, peak_started
            with lock:
                started += 1
                peak_started = max(peak_started, started)
            release.wait(timeout=0.1)
            with lock:
                started -= 1
            return _text_response("done")

        mock_boto.converse.side_effect = side_effect
        worker = threading.Thread(
            target=lambda: list(client.process_batch_iter(inputs))
        )
        worker.start()
        time.sleep(0.02)
        release.set()
        worker.join(timeout=1)

        assert peak_started <= 2

"""Cross-adapter golden tests.

Both adapters translate the *same* canonical v1 input, and — given equivalent
backend responses — produce the *same* canonical output envelope. This pins the
monolith↔worker contract with a shared artifact so the adapters can't drift.
"""

from types import SimpleNamespace

from llm_worker.config import BedrockConfig, FoundryConfig, VertexConfig
from llm_worker.contract import ContractRequest, Tool
from llm_worker.providers.bedrock import BedrockProvider
from llm_worker.providers.foundry import FoundryProvider
from llm_worker.providers.vertex import VertexProvider

# --- shared canonical v1 input fixtures ---
GOLDEN_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}},
    "required": ["score"],
}
GOLDEN_TOOL_REQUEST = ContractRequest(
    model_id="mc:eval-default",
    prompt="Rate this.",
    max_output_tokens=512,
    temperature=0.0,
    tools=[Tool("evaluation_result", "Return the evaluation result", GOLDEN_SCHEMA)],
    forced_tool="evaluation_result",
)
GOLDEN_TEXT_REQUEST = ContractRequest(
    model_id="mc:eval-default",
    prompt="Summarize.",
    max_output_tokens=512,
    temperature=0.0,
    tools=[],
    forced_tool=None,
)
GOLDEN_TOOL_OUTPUT = {"score": 0.9}
EMPTY_TOOL_OUTPUT: dict = {}


def _bedrock(mock_boto):
    return BedrockProvider(BedrockConfig(region="us-east-1"), boto_client=mock_boto)


def _vertex(mock_client):
    return VertexProvider(VertexConfig("mc-proj", "global"), client=mock_client)


def _foundry(mock_client):
    return FoundryProvider(FoundryConfig("mc-foundry"), client=mock_client)


def _anthropic_response(content):
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=0, output_tokens=0),
    )


class TestInputTranslationFromSharedFixture:
    def test_bedrock_translates_tool_request(self, mocker):
        boto = mocker.Mock()
        boto.converse.return_value = {
            "output": {"message": {"content": [{"text": ""}]}}
        }

        _bedrock(boto).complete(GOLDEN_TOOL_REQUEST)

        spec = boto.converse.call_args.kwargs["toolConfig"]["tools"][0]["toolSpec"]
        assert spec["name"] == "evaluation_result"
        assert spec["inputSchema"]["json"] == GOLDEN_SCHEMA
        assert boto.converse.call_args.kwargs["toolConfig"]["toolChoice"] == {
            "tool": {"name": "evaluation_result"}
        }

    def test_vertex_translates_tool_request(self, mocker):
        client = mocker.Mock()
        client.messages.create.return_value = _anthropic_response(
            [SimpleNamespace(type="text", text="")]
        )

        _vertex(client).complete(GOLDEN_TOOL_REQUEST)

        kw = client.messages.create.call_args.kwargs
        assert kw["tools"][0]["name"] == "evaluation_result"
        assert kw["tools"][0]["input_schema"] == GOLDEN_SCHEMA
        assert kw["tool_choice"] == {"type": "tool", "name": "evaluation_result"}

    def test_foundry_translates_tool_request(self, mocker):
        client = mocker.Mock()
        client.messages.create.return_value = _anthropic_response(
            [SimpleNamespace(type="text", text="")]
        )

        _foundry(client).complete(GOLDEN_TOOL_REQUEST)

        kw = client.messages.create.call_args.kwargs
        assert kw["tools"][0]["name"] == "evaluation_result"
        assert kw["tools"][0]["input_schema"] == GOLDEN_SCHEMA
        assert kw["tool_choice"] == {"type": "tool", "name": "evaluation_result"}


class TestOutputEnvelopeGolden:
    def test_tool_branch_identical_envelope(self, mocker):
        boto = mocker.Mock()
        boto.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "evaluation_result",
                                "input": GOLDEN_TOOL_OUTPUT,
                            }
                        }
                    ]
                }
            }
        }
        bedrock_env = _bedrock(boto).complete(GOLDEN_TOOL_REQUEST).output

        vclient = mocker.Mock()
        vclient.messages.create.return_value = _anthropic_response(
            [
                SimpleNamespace(
                    type="tool_use",
                    name="evaluation_result",
                    input=GOLDEN_TOOL_OUTPUT,
                )
            ]
        )
        vertex_env = _vertex(vclient).complete(GOLDEN_TOOL_REQUEST).output

        fclient = mocker.Mock()
        fclient.messages.create.return_value = _anthropic_response(
            [
                SimpleNamespace(
                    type="tool_use",
                    name="evaluation_result",
                    input=GOLDEN_TOOL_OUTPUT,
                )
            ]
        )
        foundry_env = _foundry(fclient).complete(GOLDEN_TOOL_REQUEST).output

        assert (
            bedrock_env
            == vertex_env
            == foundry_env
            == {"output_text": "", "tool_uses": [GOLDEN_TOOL_OUTPUT]}
        )

    def test_multiple_tool_uses_accumulate_in_order(self, mocker):
        # Exercises the accumulation loops in bedrock._extract_output and
        # anthropic._to_llm_response across all three adapters — multi-tool
        # extraction is otherwise unverified.
        first, second = {"score": 0.9}, {"score": 0.1}

        boto = mocker.Mock()
        boto.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"name": "evaluation_result", "input": first}},
                        {"toolUse": {"name": "evaluation_result", "input": second}},
                    ]
                }
            }
        }
        bedrock_env = _bedrock(boto).complete(GOLDEN_TOOL_REQUEST).output

        vclient = mocker.Mock()
        vclient.messages.create.return_value = _anthropic_response(
            [
                SimpleNamespace(type="tool_use", name="evaluation_result", input=first),
                SimpleNamespace(type="tool_use", name="evaluation_result", input=second),
            ]
        )
        vertex_env = _vertex(vclient).complete(GOLDEN_TOOL_REQUEST).output

        fclient = mocker.Mock()
        fclient.messages.create.return_value = _anthropic_response(
            [
                SimpleNamespace(type="tool_use", name="evaluation_result", input=first),
                SimpleNamespace(type="tool_use", name="evaluation_result", input=second),
            ]
        )
        foundry_env = _foundry(fclient).complete(GOLDEN_TOOL_REQUEST).output

        assert (
            bedrock_env
            == vertex_env
            == foundry_env
            == {"output_text": "", "tool_uses": [first, second]}
        )

    def test_empty_tool_output_envelope_parity(self, mocker):
        boto = mocker.Mock()
        boto.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "evaluation_result",
                                "input": EMPTY_TOOL_OUTPUT,
                            }
                        }
                    ]
                }
            }
        }
        bedrock_env = _bedrock(boto).complete(GOLDEN_TOOL_REQUEST).output

        vclient = mocker.Mock()
        vclient.messages.create.return_value = _anthropic_response(
            [
                SimpleNamespace(
                    type="tool_use",
                    name="evaluation_result",
                    input=EMPTY_TOOL_OUTPUT,
                )
            ]
        )
        vertex_env = _vertex(vclient).complete(GOLDEN_TOOL_REQUEST).output

        fclient = mocker.Mock()
        fclient.messages.create.return_value = _anthropic_response(
            [
                SimpleNamespace(
                    type="tool_use",
                    name="evaluation_result",
                    input=EMPTY_TOOL_OUTPUT,
                )
            ]
        )
        foundry_env = _foundry(fclient).complete(GOLDEN_TOOL_REQUEST).output

        assert (
            bedrock_env
            == vertex_env
            == foundry_env
            == {"output_text": "", "tool_uses": [EMPTY_TOOL_OUTPUT]}
        )

    def test_text_branch_identical_envelope(self, mocker):
        boto = mocker.Mock()
        boto.converse.return_value = {
            "output": {"message": {"content": [{"text": "a summary"}]}}
        }
        bedrock_env = _bedrock(boto).complete(GOLDEN_TEXT_REQUEST).output

        vclient = mocker.Mock()
        vclient.messages.create.return_value = _anthropic_response(
            [SimpleNamespace(type="text", text="a summary")]
        )
        vertex_env = _vertex(vclient).complete(GOLDEN_TEXT_REQUEST).output

        fclient = mocker.Mock()
        fclient.messages.create.return_value = _anthropic_response(
            [SimpleNamespace(type="text", text="a summary")]
        )
        foundry_env = _foundry(fclient).complete(GOLDEN_TEXT_REQUEST).output

        assert bedrock_env == vertex_env == foundry_env == {"output_text": "a summary"}
        assert "tool_uses" not in bedrock_env

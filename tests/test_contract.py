from llm_worker.contract import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    ContractRequest,
    Tool,
    build_request,
    resolve_model_ref,
)

# v0 (Bedrock Converse) tool_config, as monolith emits today.
V0_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "evaluation_result",
                "description": "Return the evaluation result",
                "inputSchema": {
                    "json": {"type": "object", "properties": {"score": {}}}
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": "evaluation_result"}},
}

# v1 (flat) tool_config — the same intent, post-migration.
V1_TOOL_CONFIG = {
    "tools": [
        {
            "name": "evaluation_result",
            "description": "Return the evaluation result",
            "input_schema": {"type": "object", "properties": {"score": {}}},
        }
    ],
    "tool_choice": {"name": "evaluation_result"},
}


class TestResolveModelRef:
    def test_legacy_passthrough(self):
        assert resolve_model_ref("us.anthropic.claude") == "us.anthropic.claude"

    def test_legacy_with_colon_roundtrips(self):
        # A Bedrock inference-profile version suffix contains a colon.
        assert (
            resolve_model_ref("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
            == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )

    def test_provider_prefix_stripped(self):
        assert resolve_model_ref("provider:gpt-5.4") == "gpt-5.4"

    def test_provider_prefix_strips_first_only(self):
        assert resolve_model_ref("provider:ft:gpt-4o:acme") == "ft:gpt-4o:acme"

    def test_mc_prefix_stripped(self):
        assert resolve_model_ref("mc:eval-default") == "eval-default"


class TestBuildRequest:
    def test_v0_upconverts_to_v1(self):
        req = build_request("model-1", "hi", {"maxTokens": 256}, V0_TOOL_CONFIG)
        assert req.max_output_tokens == 256
        assert req.forced_tool == "evaluation_result"
        assert req.tools == [
            Tool(
                name="evaluation_result",
                description="Return the evaluation result",
                input_schema={"type": "object", "properties": {"score": {}}},
            )
        ]

    def test_v1_passthrough(self):
        req = build_request(
            "model-1",
            "hi",
            {"max_output_tokens": 256, "temperature": 0.0},
            V1_TOOL_CONFIG,
        )
        assert req.max_output_tokens == 256
        assert req.forced_tool == "evaluation_result"
        assert req.tools[0].input_schema == {
            "type": "object",
            "properties": {"score": {}},
        }

    def test_v0_and_v1_produce_equal_requests(self):
        v0 = build_request("m", "hi", {"maxTokens": 256}, V0_TOOL_CONFIG)
        v1 = build_request("m", "hi", {"max_output_tokens": 256}, V1_TOOL_CONFIG)
        assert v0 == v1

    def test_defaults_when_params_empty(self):
        req = build_request("m", "hi", {}, {})
        assert req.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
        assert req.temperature == 0.0
        assert req.tools == []
        assert req.forced_tool is None

    def test_producer_value_wins_over_default(self):
        req = build_request(
            "m", "hi", {"max_output_tokens": 1024, "temperature": 0.7}, {}
        )
        assert req.max_output_tokens == 1024
        assert req.temperature == 0.7

    def test_auto_tool_choice_is_not_forced(self):
        cfg = {"tools": [{"name": "t", "input_schema": {}}], "tool_choice": "auto"}
        req = build_request("m", "hi", {}, cfg)
        assert req.forced_tool is None
        assert len(req.tools) == 1

    def test_returns_contract_request(self):
        assert isinstance(build_request("m", "hi", {}, {}), ContractRequest)

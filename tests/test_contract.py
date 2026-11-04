import pytest

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

    @pytest.mark.parametrize("v0_tool_choice", [{"auto": {}}, {"any": {}}])
    def test_v0_dict_tool_choice_is_not_forced(self, v0_tool_choice):
        # v0 (Bedrock) {"auto": {}} / {"any": {}} both mean "optional tool use",
        # same as v1's plain "auto" string -- only {"tool": {"name": ...}} forces.
        cfg = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "t",
                        "description": "",
                        "inputSchema": {"json": {}},
                    }
                }
            ],
            "toolChoice": v0_tool_choice,
        }
        req = build_request("m", "hi", {}, cfg)
        assert req.forced_tool is None
        assert len(req.tools) == 1

    def test_returns_contract_request(self):
        assert isinstance(build_request("m", "hi", {}, {}), ContractRequest)


# What the monolith sends for a model that rejects a forced tool call.
OUTPUT_FORMAT_TOOL_CONFIG = {
    "output_format": {
        "type": "json_schema",
        "name": "evaluation_result",
        "description": "Return the evaluation result",
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
            "additionalProperties": False,
        },
    }
}


class TestRequestProfileFields:
    def test_parses_output_format_thinking_and_effort(self):
        req = build_request(
            "m",
            "p",
            {"max_output_tokens": 4012, "thinking": "adaptive", "effort": "high"},
            OUTPUT_FORMAT_TOOL_CONFIG,
        )

        assert req.tools == []
        assert req.forced_tool is None
        assert req.output_format is not None
        assert req.output_format.name == "evaluation_result"
        assert req.output_format.description == "Return the evaluation result"
        assert (
            req.output_format.schema
            == OUTPUT_FORMAT_TOOL_CONFIG["output_format"]["schema"]
        )
        assert req.thinking == "adaptive"
        assert req.effort == "high"
        assert req.max_output_tokens == 4012

    def test_effort_is_optional(self):
        req = build_request(
            "m", "p", {"thinking": "adaptive"}, OUTPUT_FORMAT_TOOL_CONFIG
        )
        assert req.effort is None

    @pytest.mark.parametrize("tool_config", [V0_TOOL_CONFIG, V1_TOOL_CONFIG, {}])
    def test_existing_rows_get_none_for_the_new_fields(self, tool_config):
        req = build_request("m", "p", {"max_output_tokens": 512}, tool_config)
        assert req.output_format is None
        assert req.thinking is None
        assert req.effort is None

    @pytest.mark.parametrize(
        "output_format, message",
        [
            ({"type": "text", "name": "n", "schema": {}}, "json_schema"),
            ({"type": "json_schema", "name": "n"}, "schema"),
            ({"type": "json_schema", "schema": {}}, "name"),
        ],
    )
    def test_rejects_a_malformed_output_format(self, output_format, message):
        with pytest.raises(ValueError, match=message):
            build_request("m", "p", {}, {"output_format": output_format})

    @pytest.mark.parametrize("params", [{"thinking": ""}, {"effort": 3}])
    def test_rejects_a_non_string_thinking_or_effort(self, params):
        with pytest.raises(ValueError, match="non-empty string"):
            build_request("m", "p", params, OUTPUT_FORMAT_TOOL_CONFIG)


class TestInvalidCombinations:
    def test_rejects_output_format_with_tools(self):
        tool_config = {**OUTPUT_FORMAT_TOOL_CONFIG, "tools": V1_TOOL_CONFIG["tools"]}
        with pytest.raises(ValueError, match="both output_format and tools"):
            build_request("m", "p", {}, tool_config)

    def test_rejects_thinking_with_a_forced_tool(self):
        with pytest.raises(ValueError, match="forced tool choice"):
            build_request("m", "p", {"thinking": "adaptive"}, V1_TOOL_CONFIG)

    def test_allows_thinking_with_tools_on_auto(self):
        tool_config = {"tools": V1_TOOL_CONFIG["tools"], "tool_choice": "auto"}
        req = build_request("m", "p", {"thinking": "adaptive"}, tool_config)
        assert req.thinking == "adaptive"
        assert req.forced_tool is None

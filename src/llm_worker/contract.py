"""The MC LLM eval contract (v1) and the v0→v1 upconverter.

Monolith emits Bedrock-flavored rows today (**v0**): `params` uses `maxTokens`,
`tool_config` is a Bedrock Converse `toolConfig` (nested `toolSpec` /
`inputSchema.json` / `toolChoice.tool`). The target is a flat, neutral **v1**
shape (`max_output_tokens`, `tools:[{name,description,input_schema}]`,
`tool_choice`).

The executor normalizes every input row into a :class:`ContractRequest` before
handing it to a provider adapter, so adapters only ever see v1. Dispatch is by
**payload-shape sniff** (a v0 row is recognized by its `toolSpec`/`maxTokens`
spellings), not a `contract_version` column — so the worker depends on no schema
change and reads v0 and v1 rows interchangeably during the migration.

Some models reject a forced tool call. For those, the monolith sends
``tool_config = {"output_format": {...}}`` instead of tools: the model replies with
JSON that matches the schema. ``params`` can also carry ``thinking`` and ``effort``.
All three are optional, so rows without them parse exactly as before.
"""

from dataclasses import dataclass

DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
_MODEL_REF_PREFIXES = ("mc:", "provider:")


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict


@dataclass
class OutputFormat:
    """A JSON schema the model's reply must match (JSON outputs)."""

    name: str
    description: str
    schema: dict


@dataclass
class ContractRequest:
    """A normalized v1 request. Adapters translate this to their backend.

    Attributes:
        output_format: Ask for a JSON reply that matches this schema. The adapter
            stores the parsed reply as ``tool_uses[0]``, where a forced tool call's
            input goes, so readers of the result don't change.
        thinking: The thinking type to send, such as ``adaptive``. When set, no
            temperature is sent, because models that think reject one.
        effort: The effort level to send. ``None`` leaves the model's default.
    """

    model_id: str
    prompt: str
    max_output_tokens: int
    temperature: float
    tools: list[Tool]
    forced_tool: str | None  # tool name to force, or None for auto / no tools
    output_format: OutputFormat | None = None
    thinking: str | None = None
    effort: str | None = None


def resolve_model_ref(model_id: str) -> str:
    """Strip an ``mc:``/``provider:`` namespace prefix (the first prefix only, so
    values that themselves contain colons — ARNs, fine-tune ids — round-trip).
    Unprefixed values are legacy and returned as-is.
    """
    for prefix in _MODEL_REF_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return model_id


def build_request(
    model_id: str, prompt: str, params: dict, tool_config: dict
) -> ContractRequest:
    """Normalize a raw input row (v0 or v1) into a canonical v1 request, filling
    defaults where absent."""
    max_output_tokens = params.get("max_output_tokens")
    if max_output_tokens is None:
        max_output_tokens = params.get("maxTokens", DEFAULT_MAX_OUTPUT_TOKENS)
    temperature = params.get("temperature", DEFAULT_TEMPERATURE)

    tools, forced_tool = _normalize_tools(tool_config)
    output_format = _normalize_output_format(tool_config)
    thinking = _optional_str(params, "thinking")
    # The executor fails the row with these messages. Both combinations would
    # otherwise reach the API as a request it rejects or answers unpredictably.
    if output_format is not None and tools:
        raise ValueError("tool_config can't have both output_format and tools")
    if thinking is not None and forced_tool is not None:
        raise ValueError("params.thinking can't be combined with a forced tool choice")

    return ContractRequest(
        model_id=model_id,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        tools=tools,
        forced_tool=forced_tool,
        output_format=output_format,
        thinking=thinking,
        effort=_optional_str(params, "effort"),
    )


def _normalize_tools(tool_config: dict) -> tuple[list[Tool], str | None]:
    if not tool_config:
        return [], None
    tools = [_normalize_tool(t) for t in tool_config.get("tools", [])]
    forced_tool = _normalize_tool_choice(
        tool_config.get("tool_choice", tool_config.get("toolChoice"))
    )
    return tools, forced_tool


def _normalize_output_format(tool_config: dict) -> OutputFormat | None:
    output_format = (tool_config or {}).get("output_format")
    if output_format is None:
        return None
    if output_format.get("type") != "json_schema":
        raise ValueError(
            f"output_format type must be 'json_schema', got {output_format.get('type')!r}"
        )
    schema = output_format.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("output_format missing required 'schema' object")
    name = output_format.get("name")
    if not name:
        raise ValueError("output_format missing required 'name'")
    return OutputFormat(
        name=name,
        description=output_format.get("description", ""),
        schema=schema,
    )


def _optional_str(params: dict, key: str) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"params.{key} must be a non-empty string, got {value!r}")
    return value


def _normalize_tool(tool: dict) -> Tool:
    # v0 (Bedrock): {"toolSpec": {"name", "description", "inputSchema": {"json": schema}}}
    if "toolSpec" in tool:
        spec = tool["toolSpec"]
        name = spec.get("name")
        if not name:
            raise ValueError("tool spec missing required 'name'")
        return Tool(
            name=name,
            description=spec.get("description", ""),
            input_schema=spec.get("inputSchema", {}).get("json", {}),
        )
    # v1 (flat): {"name", "description", "input_schema": schema}
    name = tool.get("name")
    if not name:
        raise ValueError("tool spec missing required 'name'")
    return Tool(
        name=name,
        description=tool.get("description", ""),
        input_schema=tool.get("input_schema", {}),
    )


def _normalize_tool_choice(tool_choice) -> str | None:
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):  # v1: "auto"
        return None
    if "tool" in tool_choice:  # v0: {"tool": {"name": ...}}
        return tool_choice["tool"].get("name")
    if "name" in tool_choice:  # v1: {"name": ...}
        return tool_choice["name"]
    return None  # v0 {"auto": {}} / {"any": {}} or anything unrecognized

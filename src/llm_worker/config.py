import os
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    ca_cert: str


# Each provider config carries its own identity — the LLM_PROVIDER name and the
# cloud platform it runs on (kept 1:1 with the per-cloud image variants; see
# Dockerfile ARG CLOUD). `provider`/`cloud` are ClassVars (not init fields), so
# the config's *type* is the single source of truth for which provider it is —
# there is no separate discriminator string to keep in sync.
@dataclass(frozen=True)
class BedrockConfig:
    provider: ClassVar[str] = "bedrock"
    cloud: ClassVar[str] = "aws"
    region: str


@dataclass(frozen=True)
class VertexConfig:
    provider: ClassVar[str] = "vertex"
    cloud: ClassVar[str] = "gcp"
    project: str
    region: str
    # Per-request timeout (seconds) for the Anthropic client. Bounds a hung/slow
    # upstream request so it fails fast (→ retry → failed row) instead of blocking
    # the whole batch forever.
    timeout: float = 120.0


@dataclass(frozen=True)
class FoundryConfig:
    provider: ClassVar[str] = "foundry"
    cloud: ClassVar[str] = "azure"
    resource: str
    # See VertexConfig.timeout.
    timeout: float = 120.0


# The one provider config for this deployment — exactly one variant, so illegal
# states (no provider, or two providers at once) are unrepresentable.
ProviderConfig = BedrockConfig | VertexConfig | FoundryConfig


@dataclass(frozen=True)
class ServiceConfig:
    clickhouse: ClickHouseConfig
    provider_config: ProviderConfig
    max_workers: int
    retry_max_attempts: int
    retry_max_backoff: int
    poll_interval: float
    pending_batch_limit: int
    batch_page_size: int

    @property
    def provider(self) -> str:
        """LLM_PROVIDER name for this deployment (from the provider config's type)."""
        return self.provider_config.provider

    @property
    def cloud(self) -> str:
        """Cloud platform for this deployment (from the provider config's type)."""
        return self.provider_config.cloud


def _parse_env_int(name: str, default: str, *, min_val: int = 1) -> int:
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid integer") from None
    if value < min_val:
        raise ValueError(f"{name}={value} must be >= {min_val}")
    return value


def _parse_env_float(name: str, default: str, *, min_val: float = 0.1) -> float:
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid number") from None
    if value < min_val:
        raise ValueError(f"{name}={value} must be >= {min_val}")
    return value


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required for the selected LLM_PROVIDER")
    return value


def _load_bedrock_config() -> BedrockConfig:
    # Bedrock's timeouts are handled by botocore (bounded by default), so the
    # Anthropic-client request timeout doesn't apply here.
    return BedrockConfig(region=os.environ.get("AWS_REGION", "us-east-1"))


def _load_vertex_config() -> VertexConfig:
    return VertexConfig(
        project=_require_env("ANTHROPIC_VERTEX_PROJECT_ID"),
        region=os.environ.get("CLOUD_ML_REGION", "global"),
        # Anthropic-client providers (Vertex/Foundry) share one request timeout.
        timeout=_parse_env_float("LLM_REQUEST_TIMEOUT_SECONDS", "120"),
    )


def _load_foundry_config() -> FoundryConfig:
    return FoundryConfig(
        resource=_require_env("ANTHROPIC_FOUNDRY_RESOURCE"),
        timeout=_parse_env_float("LLM_REQUEST_TIMEOUT_SECONDS", "120"),
    )


# provider name -> env-config loader. The single source of truth for the set of
# supported providers on the config side; adding a provider is one row here plus
# one row in providers/__init__._PROVIDER_ADAPTERS.
_PROVIDER_CONFIG_LOADERS = {
    "bedrock": _load_bedrock_config,
    "vertex": _load_vertex_config,
    "foundry": _load_foundry_config,
}


def _load_provider_config(provider: str) -> ProviderConfig:
    loader = _PROVIDER_CONFIG_LOADERS.get(provider)
    if loader is None:
        supported = ", ".join(repr(p) for p in _PROVIDER_CONFIG_LOADERS)
        raise ValueError(
            f"LLM_PROVIDER={provider!r} is not supported (expected {supported})"
        )
    return loader()


def load_config() -> ServiceConfig:
    provider = os.environ.get("LLM_PROVIDER", "bedrock")

    return ServiceConfig(
        clickhouse=ClickHouseConfig(
            host=os.environ.get("CH_HOST", "localhost"),
            port=_parse_env_int("CH_PORT", "8123"),
            user=os.environ.get("CH_USER", "default"),
            password=os.environ.get("CH_PASSWORD", ""),
            database=os.environ.get("CH_DATABASE", "default"),
            ca_cert=os.environ.get("CH_CA_CERT", ""),
        ),
        provider_config=_load_provider_config(provider),
        max_workers=_parse_env_int("MAX_WORKERS", "20"),
        retry_max_attempts=_parse_env_int("RETRY_MAX_ATTEMPTS", "5"),
        retry_max_backoff=_parse_env_int("RETRY_MAX_BACKOFF", "30"),
        poll_interval=_parse_env_float("POLL_INTERVAL", "10"),
        pending_batch_limit=_parse_env_int("PENDING_BATCH_LIMIT", "100"),
        batch_page_size=_parse_env_int("BATCH_PAGE_SIZE", "100"),
    )

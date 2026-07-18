import os
from dataclasses import dataclass

# Cloud platform each LLM provider runs on. Kept 1:1 with the per-cloud image
# variants (see Dockerfile ARG CLOUD). The worker publishes its cloud so the
# monolith can resolve a deployment's cloud-native model pool.
_PROVIDER_TO_CLOUD = {"bedrock": "aws", "vertex": "gcp", "foundry": "azure"}


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    ca_cert: str


@dataclass(frozen=True)
class BedrockConfig:
    region: str


@dataclass(frozen=True)
class VertexConfig:
    project: str
    region: str
    # Per-request timeout (seconds) for the Anthropic client. Bounds a hung/slow
    # upstream request so it fails fast (→ retry → failed row) instead of blocking
    # the whole batch forever.
    timeout: float = 120.0


@dataclass(frozen=True)
class FoundryConfig:
    resource: str
    # See VertexConfig.timeout.
    timeout: float = 120.0


@dataclass(frozen=True)
class ServiceConfig:
    clickhouse: ClickHouseConfig
    provider: str
    max_workers: int
    retry_max_attempts: int
    retry_max_backoff: int
    bedrock: BedrockConfig | None
    vertex: VertexConfig | None
    foundry: FoundryConfig | None
    poll_interval: float
    pending_batch_limit: int
    batch_page_size: int

    @property
    def cloud(self) -> str:
        """Cloud platform for this deployment, derived from the LLM provider."""
        return _PROVIDER_TO_CLOUD.get(self.provider, self.provider)


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


def _load_provider_config(
    provider: str,
) -> tuple[BedrockConfig | None, VertexConfig | None, FoundryConfig | None]:
    if provider == "bedrock":
        # Bedrock's timeouts are handled by botocore (bounded by default), so the
        # Anthropic-client request timeout doesn't apply here.
        region = os.environ.get("AWS_REGION", "us-east-1")
        return BedrockConfig(region=region), None, None
    # Anthropic-client providers (Vertex/Foundry) share one request timeout.
    request_timeout = _parse_env_float("LLM_REQUEST_TIMEOUT_SECONDS", "120")
    if provider == "vertex":
        return (
            None,
            VertexConfig(
                project=_require_env("ANTHROPIC_VERTEX_PROJECT_ID"),
                region=os.environ.get("CLOUD_ML_REGION", "global"),
                timeout=request_timeout,
            ),
            None,
        )
    if provider == "foundry":
        return (
            None,
            None,
            FoundryConfig(
                resource=_require_env("ANTHROPIC_FOUNDRY_RESOURCE"),
                timeout=request_timeout,
            ),
        )
    raise ValueError(
        f"LLM_PROVIDER={provider!r} is not supported "
        f"(expected 'bedrock', 'vertex', or 'foundry')"
    )


def load_config() -> ServiceConfig:
    provider = os.environ.get("LLM_PROVIDER", "bedrock")
    bedrock, vertex, foundry = _load_provider_config(provider)

    return ServiceConfig(
        clickhouse=ClickHouseConfig(
            host=os.environ.get("CH_HOST", "localhost"),
            port=_parse_env_int("CH_PORT", "8123"),
            user=os.environ.get("CH_USER", "default"),
            password=os.environ.get("CH_PASSWORD", ""),
            database=os.environ.get("CH_DATABASE", "default"),
            ca_cert=os.environ.get("CH_CA_CERT", ""),
        ),
        provider=provider,
        max_workers=_parse_env_int("MAX_WORKERS", "20"),
        retry_max_attempts=_parse_env_int("RETRY_MAX_ATTEMPTS", "5"),
        retry_max_backoff=_parse_env_int("RETRY_MAX_BACKOFF", "30"),
        bedrock=bedrock,
        vertex=vertex,
        foundry=foundry,
        poll_interval=_parse_env_float("POLL_INTERVAL", "10"),
        pending_batch_limit=_parse_env_int("PENDING_BATCH_LIMIT", "100"),
        batch_page_size=_parse_env_int("BATCH_PAGE_SIZE", "100"),
    )

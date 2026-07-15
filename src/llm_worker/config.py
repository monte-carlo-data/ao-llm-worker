import os
from dataclasses import dataclass


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
    model: str


@dataclass(frozen=True)
class ServiceConfig:
    clickhouse: ClickHouseConfig
    provider: str
    max_workers: int
    retry_max_attempts: int
    retry_max_backoff: int
    bedrock: BedrockConfig | None
    vertex: VertexConfig | None
    poll_interval: float
    pending_batch_limit: int
    batch_page_size: int


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
) -> tuple[BedrockConfig | None, VertexConfig | None]:
    if provider == "bedrock":
        region = os.environ.get("AWS_REGION", "us-east-1")
        return BedrockConfig(region=region), None
    if provider == "vertex":
        return (
            None,
            VertexConfig(
                project=_require_env("ANTHROPIC_VERTEX_PROJECT_ID"),
                region=os.environ.get("CLOUD_ML_REGION", "global"),
                model=_require_env("VERTEX_MODEL"),
            ),
        )
    raise ValueError(
        f"LLM_PROVIDER={provider!r} is not supported (expected 'bedrock' or 'vertex')"
    )


def load_config() -> ServiceConfig:
    provider = os.environ.get("LLM_PROVIDER", "bedrock")
    bedrock, vertex = _load_provider_config(provider)

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
        poll_interval=_parse_env_float("POLL_INTERVAL", "10"),
        pending_batch_limit=_parse_env_int("PENDING_BATCH_LIMIT", "100"),
        batch_page_size=_parse_env_int("BATCH_PAGE_SIZE", "100"),
    )

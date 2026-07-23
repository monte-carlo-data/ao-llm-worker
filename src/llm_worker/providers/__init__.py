"""Pluggable LLM backends.

`base` defines the provider interface; concrete adapters (e.g. `bedrock`,
`vertex`) live alongside it. A single registry (`_PROVIDERS`) maps each
`LLM_PROVIDER` name to its env-config loader and adapter factory; `load_config`
consults it to build the provider config and `create_provider` to build the
adapter. One registry means the config side and the adapter side can't drift out
of sync, and factories are plain functions (statically checkable) rather than
`importlib`/`getattr` on strings.
"""

from collections.abc import Callable
from dataclasses import dataclass

from llm_worker.config import (
    BedrockConfig,
    FoundryConfig,
    ProviderConfig,
    ServiceConfig,
    VertexConfig,
    load_bedrock_config,
    load_foundry_config,
    load_vertex_config,
)
from llm_worker.providers.base import LLMProvider


@dataclass(frozen=True)
class _Provider:
    """One provider's wiring: how to load its env-config, how to build its
    adapter, and which image variant ships its SDK."""

    config_loader: Callable[[], ProviderConfig]
    factory: Callable[[ProviderConfig], LLMProvider]
    image_variant: str


# Adapter factories import their concrete module lazily, so an image that ships
# only one provider's SDK never imports the others; a missing adapter/SDK raises
# ImportError, which create_provider turns into a clear startup error.
def _build_bedrock(config: ProviderConfig) -> LLMProvider:
    from llm_worker.providers.bedrock import BedrockProvider

    assert isinstance(config, BedrockConfig)
    return BedrockProvider(config)


def _build_vertex(config: ProviderConfig) -> LLMProvider:
    from llm_worker.providers.vertex import VertexProvider

    assert isinstance(config, VertexConfig)
    return VertexProvider(config)


def _build_foundry(config: ProviderConfig) -> LLMProvider:
    from llm_worker.providers.foundry import FoundryProvider

    assert isinstance(config, FoundryConfig)
    return FoundryProvider(config)


# Single source of truth for provider dispatch: LLM_PROVIDER name -> wiring.
# Adding a provider is one row here plus its config loader and adapter module.
_PROVIDERS: dict[str, _Provider] = {
    "bedrock": _Provider(load_bedrock_config, _build_bedrock, "aws"),
    "vertex": _Provider(load_vertex_config, _build_vertex, "gcp"),
    "foundry": _Provider(load_foundry_config, _build_foundry, "azure"),
}


def _get(name: str) -> _Provider:
    entry = _PROVIDERS.get(name)
    if entry is None:
        supported = ", ".join(repr(p) for p in _PROVIDERS)
        raise ValueError(
            f"LLM_PROVIDER={name!r} is not supported (expected {supported})"
        )
    return entry


def load_provider_config(provider: str) -> ProviderConfig:
    """Load the env-config for ``provider``, or raise on an unsupported name."""
    return _get(provider).config_loader()


def create_provider(config: ServiceConfig) -> LLMProvider:
    """Build the provider adapter for ``config.provider_config``.

    The adapter module is imported lazily inside its factory, so an image that
    ships only one provider's SDK never imports the others. A missing adapter/SDK
    surfaces as a clear startup error (e.g. running ``LLM_PROVIDER=vertex`` on the
    aws image variant) rather than an opaque ``ImportError``.
    """
    provider_config = config.provider_config
    entry = _get(provider_config.provider)
    try:
        return entry.factory(provider_config)
    except ImportError as e:
        raise RuntimeError(
            f"LLM_PROVIDER={provider_config.provider} selected but its adapter/SDK "
            f"is unavailable ({e}). Is this the {entry.image_variant} image variant?"
        ) from e

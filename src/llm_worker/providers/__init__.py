"""Pluggable LLM backends.

`base` defines the provider interface; concrete adapters (e.g. `bedrock`,
`vertex`) live alongside it. `create_provider` selects one from config.
"""

import importlib

from llm_worker.config import ServiceConfig
from llm_worker.providers.base import LLMProvider

# provider name -> (adapter module suffix, adapter class, image-variant hint).
# The adapter lives at ``llm_worker.providers.<module>``. Single source of truth
# for provider dispatch; adding a provider is one row here plus its config loader
# in config._PROVIDER_CONFIG_LOADERS — no if-chain to keep in sync.
_PROVIDER_ADAPTERS: dict[str, tuple[str, str, str]] = {
    "bedrock": ("bedrock", "BedrockProvider", "aws"),
    "vertex": ("vertex", "VertexProvider", "gcp"),
    "foundry": ("foundry", "FoundryProvider", "azure"),
}


def create_provider(config: ServiceConfig) -> LLMProvider:
    """Build the provider for ``config.provider_config``.

    Adapter modules are imported lazily, so an image that ships only one
    provider's SDK never imports the others. A missing adapter or SDK surfaces
    as a clear startup error rather than an opaque ``ImportError`` (e.g. running
    ``LLM_PROVIDER=vertex`` on the aws image variant).
    """
    provider_config = config.provider_config
    name = provider_config.provider
    entry = _PROVIDER_ADAPTERS.get(name)
    if entry is None:
        supported = ", ".join(repr(p) for p in _PROVIDER_ADAPTERS)
        raise ValueError(
            f"LLM_PROVIDER={name!r} is not supported (expected {supported})"
        )

    module_suffix, class_name, image_variant = entry
    try:
        module = importlib.import_module(f"llm_worker.providers.{module_suffix}")
    except ImportError as e:
        raise RuntimeError(
            f"LLM_PROVIDER={name} selected but its adapter/SDK is unavailable "
            f"({e}). Is this the {image_variant} image variant?"
        ) from e

    provider_cls = getattr(module, class_name)
    return provider_cls(provider_config)

"""Pluggable LLM backends.

`base` defines the provider interface; concrete adapters (e.g. `bedrock`,
`vertex`) live alongside it. `create_provider` selects one from config.
"""

from llm_worker.config import ServiceConfig
from llm_worker.providers.base import LLMProvider


def create_provider(config: ServiceConfig) -> LLMProvider:
    """Build the provider selected by ``config.provider``.

    Adapter modules are imported lazily, so an image that ships only one
    provider's SDK never imports the others. A missing adapter or SDK surfaces
    as a clear startup error rather than an opaque ``ImportError`` (e.g. running
    ``LLM_PROVIDER=vertex`` on the aws image variant).
    """
    if config.provider == "bedrock":
        from llm_worker.providers.bedrock import BedrockProvider

        assert config.bedrock is not None  # guaranteed by load_config
        return BedrockProvider(config.bedrock)

    if config.provider == "vertex":
        try:
            from llm_worker.providers.vertex import VertexProvider
        except ImportError as e:
            raise RuntimeError(
                f"LLM_PROVIDER=vertex selected but its adapter/SDK is "
                f"unavailable ({e}). Is this the gcp image variant?"
            ) from e

        assert config.vertex is not None  # guaranteed by load_config
        return VertexProvider(config.vertex)

    raise ValueError(
        f"LLM_PROVIDER={config.provider!r} is not supported "
        f"(expected 'bedrock' or 'vertex')"
    )

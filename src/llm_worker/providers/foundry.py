"""Microsoft Foundry adapter for the LLM worker.

A thin :class:`~llm_worker.providers.anthropic_messages.AnthropicMessagesProvider`
subclass: all request/response translation and error classification are shared
(Vertex and Foundry both speak the Anthropic Messages API); only client
construction differs. Auth is Entra ID via ``DefaultAzureCredential`` (AKS
Workload Identity), scope ``https://ai.azure.com/.default`` — no API key. The
SDK builds ``https://<resource>.services.ai.azure.com/anthropic/`` from
``resource``.
"""

from anthropic import AnthropicFoundry

from llm_worker.config import FoundryConfig
from llm_worker.providers.anthropic_messages import AnthropicMessagesProvider


class FoundryProvider(AnthropicMessagesProvider):
    def __init__(self, config: FoundryConfig, client: AnthropicFoundry | None = None):
        super().__init__(client or self._build_client(config))

    @staticmethod
    def _build_client(config: FoundryConfig) -> AnthropicFoundry:
        from azure.identity import (
            DefaultAzureCredential,
            get_bearer_token_provider,
        )

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )
        return AnthropicFoundry(
            resource=config.resource,
            azure_ad_token_provider=token_provider,
            timeout=config.timeout,
        )

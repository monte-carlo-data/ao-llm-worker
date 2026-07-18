"""Vertex AI adapter for the LLM worker.

A thin :class:`~llm_worker.providers.anthropic_messages.AnthropicMessagesProvider`
subclass: all request/response translation and error classification are shared
(Vertex and Foundry both speak the Anthropic Messages API); only client
construction differs. Auth is ambient Application Default Credentials via GKE
Workload Identity — no API key.
"""

from anthropic import AnthropicVertex

from llm_worker.config import VertexConfig
from llm_worker.providers.anthropic_messages import AnthropicMessagesProvider


class VertexProvider(AnthropicMessagesProvider):
    def __init__(self, config: VertexConfig, client: AnthropicVertex | None = None):
        # AnthropicVertex resolves credentials via google.auth ADC (Workload
        # Identity on GKE); no key is passed. Tests inject a mock client.
        super().__init__(
            client
            or AnthropicVertex(
                project_id=config.project,
                region=config.region,
                timeout=config.timeout,
            )
        )

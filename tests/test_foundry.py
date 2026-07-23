"""Foundry-specific client construction.

The shared request/response, error-classification, and temperature-fallback
behavior lives in ``test_anthropic_messages_providers.py``; this file covers only
what's unique to Foundry — how the ``AnthropicFoundry`` client is built and
authenticated.
"""

from llm_worker.config import FoundryConfig
from llm_worker.providers.foundry import FoundryProvider


class TestBuildClient:
    def test_passes_timeout_to_client(self, mocker):
        # A bounded timeout is essential: without it a hung Foundry request blocks the
        # row (and the batch) forever. The client must receive config.timeout.
        fake_foundry = mocker.patch("llm_worker.providers.foundry.AnthropicFoundry")
        mocker.patch("azure.identity.DefaultAzureCredential")
        mocker.patch(
            "azure.identity.get_bearer_token_provider", return_value=lambda: "token"
        )

        FoundryProvider(FoundryConfig(resource="mc-foundry", timeout=90.0))

        assert fake_foundry.call_args.kwargs["timeout"] == 90.0

    def test_passes_resource_and_token_provider_to_client(self, mocker):
        # Foundry builds its endpoint from `resource` and authenticates via a
        # bearer-token provider (Entra ID); the client must receive both.
        fake_foundry = mocker.patch("llm_worker.providers.foundry.AnthropicFoundry")
        mocker.patch("azure.identity.DefaultAzureCredential")
        mock_get_bearer = mocker.patch(
            "azure.identity.get_bearer_token_provider", return_value=lambda: "token"
        )

        FoundryProvider(FoundryConfig(resource="mc-foundry"))

        assert fake_foundry.call_args.kwargs["resource"] == "mc-foundry"
        assert (
            fake_foundry.call_args.kwargs["azure_ad_token_provider"]
            == mock_get_bearer.return_value
        )

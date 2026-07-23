"""Vertex-specific client construction.

The shared request/response, error-classification, and temperature-fallback
behavior lives in ``test_anthropic_messages_providers.py``; this file covers only
what's unique to Vertex — how the ``AnthropicVertex`` client is built.
"""

from llm_worker.config import VertexConfig
from llm_worker.providers.vertex import VertexProvider


class TestBuildClient:
    def test_passes_timeout_to_client(self, mocker):
        # A bounded timeout keeps a hung Vertex request from wedging the batch; the
        # client must receive config.timeout.
        fake_vertex = mocker.patch("llm_worker.providers.vertex.AnthropicVertex")

        VertexProvider(VertexConfig(project="mc-proj", region="global", timeout=90.0))

        assert fake_vertex.call_args.kwargs["timeout"] == 90.0

    def test_passes_project_and_region_to_client(self, mocker):
        # Vertex routes requests using project_id/region; the client must receive
        # both from VertexConfig.
        fake_vertex = mocker.patch("llm_worker.providers.vertex.AnthropicVertex")

        VertexProvider(VertexConfig(project="mc-proj", region="us-east5"))

        assert fake_vertex.call_args.kwargs["project_id"] == "mc-proj"
        assert fake_vertex.call_args.kwargs["region"] == "us-east5"

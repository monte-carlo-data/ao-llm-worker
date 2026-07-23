import sys

import pytest

from llm_worker.config import BedrockConfig, FoundryConfig, VertexConfig
from llm_worker.providers import create_provider, load_provider_config
from llm_worker.providers.bedrock import BedrockProvider


# --- load_provider_config: env-config resolution + dispatch ---
def test_load_provider_config_default_bedrock(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    cfg = load_provider_config("bedrock")
    assert isinstance(cfg, BedrockConfig)
    assert cfg.provider == "bedrock"
    assert cfg.cloud == "aws"
    assert cfg.region == "us-east-1"


def test_bedrock_region_from_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert load_provider_config("bedrock").region == "us-west-2"


def test_load_vertex_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "mc-proj")
    monkeypatch.setenv("CLOUD_ML_REGION", "us-east5")
    cfg = load_provider_config("vertex")
    assert isinstance(cfg, VertexConfig)
    assert cfg.cloud == "gcp"
    assert cfg.project == "mc-proj"
    assert cfg.region == "us-east5"
    assert cfg.timeout == 120.0


def test_vertex_region_defaults_to_global(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "mc-proj")
    monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
    assert load_provider_config("vertex").region == "global"


def test_load_foundry_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "mc-foundry")
    cfg = load_provider_config("foundry")
    assert isinstance(cfg, FoundryConfig)
    assert cfg.cloud == "azure"
    assert cfg.resource == "mc-foundry"
    assert cfg.timeout == 120.0


def test_request_timeout_override(monkeypatch):
    # A bounded per-request timeout on the Anthropic (Foundry/Vertex) client keeps a
    # hung upstream request from wedging the whole batch (it raises APITimeoutError →
    # retried → failed row instead of blocking forever). Configurable per deployment.
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "mc-foundry")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "45")
    assert load_provider_config("foundry").timeout == 45.0


def test_vertex_missing_env_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_VERTEX_PROJECT_ID is required"):
        load_provider_config("vertex")


def test_foundry_missing_env_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_FOUNDRY_RESOURCE is required"):
        load_provider_config("foundry")


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="not supported"):
        load_provider_config("nonexistent_provider")


# --- create_provider: adapter construction from a resolved config ---
def test_creates_bedrock_provider():
    provider = create_provider(load_provider_config("bedrock"))
    assert isinstance(provider, BedrockProvider)


def test_bedrock_path_does_not_import_vertex_module(monkeypatch):
    # Use monkeypatch.delitem (not a bare sys.modules.pop) so the original module is
    # restored on teardown — otherwise the module stays absent and a later
    # mocker.patch("...vertex.<sym>") re-imports a fresh copy, leaving that patch out
    # of sync with symbols imported from the original module at collection time.
    monkeypatch.delitem(sys.modules, "llm_worker.providers.vertex", raising=False)
    create_provider(load_provider_config("bedrock"))
    assert "llm_worker.providers.vertex" not in sys.modules


def test_vertex_selection_fails_fast_when_sdk_unavailable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "mc-proj")
    provider_config = load_provider_config("vertex")
    # Simulate the adapter/SDK being absent (e.g. running on the aws image).
    monkeypatch.setitem(sys.modules, "llm_worker.providers.vertex", None)

    with pytest.raises(RuntimeError, match="gcp image variant"):
        create_provider(provider_config)


def test_foundry_selection_fails_fast_when_sdk_unavailable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "mc-foundry")
    provider_config = load_provider_config("foundry")
    # Simulate the adapter/SDK being absent (e.g. running on the aws image).
    monkeypatch.setitem(sys.modules, "llm_worker.providers.foundry", None)

    with pytest.raises(RuntimeError, match="azure image variant"):
        create_provider(provider_config)


def test_creates_foundry_provider(monkeypatch, mocker):
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "mc-foundry")
    # Patch client construction so no real Azure credentials are required.
    mocker.patch(
        "llm_worker.providers.foundry.FoundryProvider._build_client",
        return_value=mocker.Mock(),
    )
    from llm_worker.providers.foundry import FoundryProvider

    provider = create_provider(load_provider_config("foundry"))
    assert isinstance(provider, FoundryProvider)


def test_creates_vertex_provider(monkeypatch, mocker):
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "mc-proj")
    # Patch the SDK client so no real GCP credentials are required (VertexProvider
    # constructs AnthropicVertex directly, so patch at the adapter module).
    mocker.patch(
        "llm_worker.providers.vertex.AnthropicVertex", return_value=mocker.Mock()
    )
    from llm_worker.providers.vertex import VertexProvider

    provider = create_provider(load_provider_config("vertex"))
    assert isinstance(provider, VertexProvider)

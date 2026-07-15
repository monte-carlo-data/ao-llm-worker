import sys
from dataclasses import replace

import pytest

from llm_worker.config import load_config
from llm_worker.providers import create_provider
from llm_worker.providers.bedrock import BedrockProvider


def test_creates_bedrock_provider():
    provider = create_provider(load_config())
    assert isinstance(provider, BedrockProvider)


def test_bedrock_path_does_not_import_vertex_module():
    sys.modules.pop("llm_worker.providers.vertex", None)
    create_provider(load_config())
    assert "llm_worker.providers.vertex" not in sys.modules


def test_vertex_selection_fails_fast_when_sdk_unavailable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "mc-proj")
    config = load_config()
    # Simulate the adapter/SDK being absent (e.g. running on the aws image).
    monkeypatch.setitem(sys.modules, "llm_worker.providers.vertex", None)

    with pytest.raises(RuntimeError, match="gcp image variant"):
        create_provider(config)


def test_foundry_selection_fails_fast_when_sdk_unavailable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "foundry")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "mc-foundry")
    config = load_config()
    # Simulate the adapter/SDK being absent (e.g. running on the aws image).
    monkeypatch.setitem(sys.modules, "llm_worker.providers.foundry", None)

    with pytest.raises(RuntimeError, match="azure image variant"):
        create_provider(config)


def test_creates_foundry_provider(monkeypatch, mocker):
    monkeypatch.setenv("LLM_PROVIDER", "foundry")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "mc-foundry")
    # Patch client construction so no real Azure credentials are required.
    mocker.patch(
        "llm_worker.providers.foundry.FoundryProvider._build_client",
        return_value=mocker.Mock(),
    )
    from llm_worker.providers.foundry import FoundryProvider

    provider = create_provider(load_config())
    assert isinstance(provider, FoundryProvider)


def test_unknown_provider_raises():
    config = replace(load_config(), provider="nonexistent_provider")
    with pytest.raises(ValueError, match="not supported"):
        create_provider(config)

from __future__ import annotations

import pytest

from gpu_orchestrator.config import OrchestratorConfig


def test_route_contract_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "REIGH_BACKEND",
        "WORKER_BACKEND",
        "REIGH_WORKER_PROFILE",
        "WGP_PROFILE",
        "REIGH_WORKER_POOL",
        "WORKER_POOL",
        "REIGH_SELECTOR_NAMESPACE",
        "ROUTE_SELECTOR_NAMESPACE",
        "REIGH_SELECTOR_VERSION",
        "ROUTE_SELECTOR_VERSION",
        "REIGH_WORKER_CONTRACT_VERSION",
        "REIGH_WORKER_RUN_ID",
        "WORKER_RUN_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    config = OrchestratorConfig.from_env()

    assert config.worker_backend == "wgp"
    assert config.capacity_reconciler_mode == "off"
    assert config.worker_profile == "1"
    assert config.selector_namespace == "production"
    assert config.selector_version is None
    assert config.worker_contract_version == 1
    assert config.worker_run_id is None
    assert config.worker_pool == "gpu-wgp-production"


def test_route_contract_config_parses_pool_backend_profile_and_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_WORKER_POOL", "canary-vibecomfy")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "staging")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "42")
    monkeypatch.setenv("REIGH_WORKER_CONTRACT_VERSION", "7")
    monkeypatch.setenv("REIGH_WORKER_RUN_ID", "run-123")

    config = OrchestratorConfig.from_env()

    assert config.worker_backend == "vibecomfy"
    assert config.worker_profile == "3"
    assert config.worker_pool == "canary-vibecomfy"
    assert config.selector_namespace == "staging"
    assert config.selector_version == "42"
    assert config.worker_contract_version == 7
    assert config.worker_run_id == "run-123"


def test_route_contract_config_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REIGH_BACKEND", "comfy")

    with pytest.raises(ValueError, match="Unsupported worker backend"):
        OrchestratorConfig.from_env()


def test_capacity_reconciler_mode_parses_shadow_and_rejects_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCHESTRATOR_CAPACITY_RECONCILER_MODE", "shadow")

    config = OrchestratorConfig.from_env()

    assert config.capacity_reconciler_mode == "shadow"

    monkeypatch.setenv("ORCHESTRATOR_CAPACITY_RECONCILER_MODE", "enabled")
    with pytest.raises(ValueError, match="Unsupported ORCHESTRATOR_CAPACITY_RECONCILER_MODE"):
        OrchestratorConfig.from_env()

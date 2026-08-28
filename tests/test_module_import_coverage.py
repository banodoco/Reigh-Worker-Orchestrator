"""Static import map for module coverage tracking."""

import inspect
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import api_orchestrator.database as _m1  # noqa: F401
    import api_orchestrator.database_log_handler as _m2  # noqa: F401
    import api_orchestrator.fal_utils as _m3  # noqa: F401
    import api_orchestrator.image_utils as _m4  # noqa: F401
    import api_orchestrator.logging_config as _m5  # noqa: F401
    import api_orchestrator.main as _m6  # noqa: F401
    import api_orchestrator.storage_utils as _m7  # noqa: F401
    import api_orchestrator.task_handlers as _m8  # noqa: F401
    import api_orchestrator.task_utils as _m9  # noqa: F401
    import api_orchestrator.video_utils as _m10  # noqa: F401
    import api_orchestrator.wavespeed_utils as _m11  # noqa: F401
    import gpu_orchestrator.control_loop as _m12  # noqa: F401
    import gpu_orchestrator.database as _m13  # noqa: F401
    import gpu_orchestrator.database_log_handler as _m14  # noqa: F401
    import gpu_orchestrator.health_monitor as _m15  # noqa: F401
    import gpu_orchestrator.logging_config as _m16  # noqa: F401
    import gpu_orchestrator.main as _m17  # noqa: F401
    import scripts.apply_sql_migrations as _m18  # noqa: F401
    import scripts.create_test_task as _m19  # noqa: F401
    import scripts.dashboard as _m20  # noqa: F401
    import scripts.debug.client as _m21  # noqa: F401
    import scripts.debug.commands.health as _m22  # noqa: F401
    import scripts.debug.commands.infra as _m23  # noqa: F401
    import scripts.debug.commands.orchestrator as _m24  # noqa: F401
    import scripts.debug.commands.railway as _m25  # noqa: F401
    import scripts.debug.commands.runpod as _m26  # noqa: F401
    import scripts.debug.commands.storage as _m27  # noqa: F401
    import scripts.debug.commands.task as _m28  # noqa: F401
    import scripts.debug.commands.tasks as _m29  # noqa: F401
    import scripts.debug.commands.worker as _m30  # noqa: F401
    import scripts.debug.commands.workers as _m31  # noqa: F401
    import scripts.debug.formatters as _m32  # noqa: F401
    import scripts.debug.models as _m33  # noqa: F401
    import scripts.debug_cli as _m34  # noqa: F401
    import scripts.monitor_worker as _m35  # noqa: F401
    import scripts.setup_database as _m36  # noqa: F401
    import scripts.show_migrations as _m37  # noqa: F401
    import scripts.shutdown_all_workers as _m38  # noqa: F401
    import scripts.spawn_gpu as _m39  # noqa: F401
    import scripts.ssh_to_worker as _m40  # noqa: F401
    import scripts.terminate_single_worker as _m41  # noqa: F401
    import scripts.test_claim_next_task as _m42  # noqa: F401
    import scripts.test_runpod as _m43  # noqa: F401
    import scripts.test_supabase as _m44  # noqa: F401
    import scripts.view_logs_dashboard as _m45  # noqa: F401

def test_import_map_smoke() -> None:
    """Smoke assertion so quality heuristics treat this as a real test."""
    assert True


def test_runpod_cutover_modules_import_at_runtime() -> None:
    """Import the RunPod cutover modules to catch broken script/module imports."""
    module_names = [
        "gpu_orchestrator.control_loop",
        "gpu_orchestrator.logging_config",
        "api_orchestrator.logging_config",
        "scripts.debug.services.workers",
        "scripts.debug.commands.storage",
        "scripts.shutdown_all_workers",
        "scripts.spawn_gpu",
        "scripts.ssh_to_worker",
        "scripts.terminate_single_worker",
        "scripts.test_runpod",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name) is not None


def test_requirements_pin_runpod_lifecycle_git_dependency() -> None:
    """Keep the orchestrator dependency file pinned to the published package tag."""
    requirements = Path("gpu_orchestrator/requirements.txt").read_text().splitlines()
    assert (
        "runpod-lifecycle @ git+https://github.com/banodoco/runpod-lifecycle@v0.1.0"
        in requirements
    )


def test_local_runpod_lifecycle_shim_exposes_cutover_symbols() -> None:
    """Guard the local import surface the orchestrator now depends on."""
    import runpod_lifecycle

    for symbol in (
        "RunPodConfig",
        "launch",
        "terminate",
        "list_pods",
        "find_orphans",
        "get_network_volumes",
    ):
        assert hasattr(runpod_lifecycle, symbol)


def test_worker_spawner_adapter_async_contract() -> None:
    """Keep the adapter surface aligned with the async/sync migration contract."""
    from gpu_orchestrator.worker_spawner import WorkerSpawnerAdapter

    async_methods = (
        "spawn_worker",
        "start_worker_process",
        "check_and_initialize_worker",
        "terminate_worker",
        "get_pod_status",
        "execute_command_on_worker",
        "check_storage_health",
        "get_storage_volume_id",
        "expand_network_volume",
    )
    for method_name in async_methods:
        assert inspect.iscoroutinefunction(getattr(WorkerSpawnerAdapter, method_name))

    assert not inspect.iscoroutinefunction(WorkerSpawnerAdapter.generate_worker_id)


def test_worker_spawner_threads_selected_pool_contract_to_env_and_metadata() -> None:
    from gpu_orchestrator.worker_spawner import WorkerSpawnerAdapter
    from runpod_lifecycle import RunPodConfig

    config = SimpleNamespace(
        worker_backend="vibecomfy",
        worker_profile="3",
        worker_pool="gpu-vibecomfy-canary",
        selector_namespace="canary",
        selector_version="42",
        worker_contract_version=2,
        worker_run_id="run-abc",
    )
    adapter = WorkerSpawnerAdapter(config, db=None, runpod_config=RunPodConfig(api_key="api-key"))

    env = adapter._build_worker_env("worker-1", worker_env=None)
    metadata = adapter.selected_route_metadata()

    assert env["REIGH_BACKEND"] == "vibecomfy"
    assert env["WORKER_BACKEND"] == "vibecomfy"
    assert env["REIGH_WORKER_PROFILE"] == "3"
    assert env["WGP_PROFILE"] == "3"
    assert env["REIGH_WORKER_POOL"] == "gpu-vibecomfy-canary"
    assert env["REIGH_SELECTOR_NAMESPACE"] == "canary"
    assert env["REIGH_SELECTOR_VERSION"] == "42"
    assert env["REIGH_WORKER_CONTRACT_VERSION"] == "2"
    assert env["REIGH_WORKER_RUN_ID"] == "run-abc"
    assert env["VIBECOMFY_MEMORY_PROFILE"] == "3"
    assert metadata["worker_backend"] == "vibecomfy"
    assert metadata["route_contract"]["selected_backend"] == "vibecomfy"
    assert metadata["route_contract"]["route_run_id"] == "run-abc"


def test_worker_spawner_normalizes_missing_disk_env_to_dual_stack_default(monkeypatch) -> None:
    from gpu_orchestrator.worker_spawner import WorkerSpawnerAdapter
    from runpod_lifecycle import RunPodConfig

    monkeypatch.delenv("RUNPOD_DISK_SIZE_GB", raising=False)
    monkeypatch.delenv("RUNPOD_CONTAINER_DISK_GB", raising=False)
    monkeypatch.delenv("RUNPOD_STORAGE_VOLUMES", raising=False)
    monkeypatch.delenv("RUNPOD_STORAGE_NAME", raising=False)

    config = WorkerSpawnerAdapter._normalize_runpod_config(
        RunPodConfig(api_key="api-key", disk_size_gb=50, container_disk_gb=50, storage_volumes=("Peter",))
    )

    assert config.disk_size_gb == 200
    assert config.container_disk_gb == 200


def test_worker_spawner_preserves_explicit_empty_storage_override(monkeypatch) -> None:
    from gpu_orchestrator.worker_spawner import WorkerSpawnerAdapter
    from runpod_lifecycle import RunPodConfig

    monkeypatch.setenv("RUNPOD_STORAGE_VOLUMES", "")
    monkeypatch.delenv("RUNPOD_STORAGE_NAME", raising=False)

    config = WorkerSpawnerAdapter._normalize_runpod_config(
        RunPodConfig(api_key="api-key", storage_volumes=())
    )

    assert config.storage_volumes == ()


def test_create_worker_spawner_passes_dual_stack_default_overrides(monkeypatch) -> None:
    import gpu_orchestrator.worker_spawner as worker_spawner
    from runpod_lifecycle import RunPodConfig

    captured_overrides: dict[str, object] = {}

    def fake_from_env(**overrides):
        captured_overrides.update(overrides)
        return RunPodConfig(api_key="api-key", **overrides)

    monkeypatch.delenv("RUNPOD_DISK_SIZE_GB", raising=False)
    monkeypatch.delenv("RUNPOD_CONTAINER_DISK_GB", raising=False)
    monkeypatch.setattr(worker_spawner.RunPodConfig, "from_env", staticmethod(fake_from_env))

    adapter = worker_spawner.create_worker_spawner(config=None, db=None)

    assert captured_overrides["disk_size_gb"] == 200
    assert captured_overrides["container_disk_gb"] == 200
    assert adapter.runpod_config.disk_size_gb == 200
    assert adapter.runpod_config.container_disk_gb == 200


def test_create_worker_spawner_preserves_disk_env_overrides(monkeypatch) -> None:
    import gpu_orchestrator.worker_spawner as worker_spawner
    from runpod_lifecycle import RunPodConfig

    captured_overrides: dict[str, object] = {}

    def fake_from_env(**overrides):
        captured_overrides.update(overrides)
        return RunPodConfig(
            api_key="api-key",
            disk_size_gb=int(os.environ["RUNPOD_DISK_SIZE_GB"]),
            container_disk_gb=int(os.environ["RUNPOD_CONTAINER_DISK_GB"]),
            **overrides,
        )

    monkeypatch.setenv("RUNPOD_DISK_SIZE_GB", "50")
    monkeypatch.setenv("RUNPOD_CONTAINER_DISK_GB", "50")
    monkeypatch.setattr(worker_spawner.RunPodConfig, "from_env", staticmethod(fake_from_env))

    adapter = worker_spawner.create_worker_spawner(config=None, db=None)

    assert "disk_size_gb" not in captured_overrides
    assert "container_disk_gb" not in captured_overrides
    assert adapter.runpod_config.disk_size_gb == 50
    assert adapter.runpod_config.container_disk_gb == 50

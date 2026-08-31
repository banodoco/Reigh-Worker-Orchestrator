"""Control-loop ordering tests for capacity reconciler modes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from gpu_orchestrator.control_loop import OrchestratorControlLoop
from gpu_orchestrator.worker_state import CycleSummary, TaskCounts


class FakeHealthMonitor:
    def check_scaling_anomaly(self, **_: Any) -> None:
        return None

    def log_health_summary(self, _: int) -> None:
        return None

    def check_logging_health(self) -> bool:
        return True


def _run(value: Any) -> Any:
    return asyncio.run(value)


def _loop(mode: str, calls: list[str]) -> OrchestratorControlLoop:
    loop = OrchestratorControlLoop.__new__(OrchestratorControlLoop)
    loop.config = SimpleNamespace(capacity_reconciler_mode=mode)
    loop.cycle_count = 0
    loop.health_monitor = FakeHealthMonitor()

    async def fetch_current_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]], TaskCounts, dict[str, Any]]:
        calls.append("fetch")
        return [], [], TaskCounts(queued=1, active_cloud=0, total=1), {}

    async def derive_all_worker_states(workers: list[dict[str, Any]], queued_count: int) -> list[Any]:
        calls.append("derive")
        return []

    async def handle_early_termination(worker_states: list[Any], task_counts: TaskCounts, summary: CycleSummary) -> list[Any]:
        calls.append("early")
        return worker_states

    async def handle_spawning_workers(worker_states: list[Any], task_counts: TaskCounts, summary: CycleSummary) -> None:
        calls.append("spawning")

    async def health_check_active_workers(worker_states: list[Any], task_counts: TaskCounts, summary: CycleSummary) -> None:
        calls.append("health")

    async def cleanup_error_workers(worker_states: list[Any], summary: CycleSummary) -> None:
        calls.append("cleanup")

    async def reconcile_orphaned_tasks(worker_states: list[Any], summary: CycleSummary) -> None:
        calls.append("orphan")

    async def handle_stale_route_workers(worker_states: list[Any], summary: CycleSummary) -> None:
        calls.append("stale_route")

    async def run_capacity_reconciler(reconciler_mode: str, summary: CycleSummary) -> None:
        calls.append(f"reconciler:{reconciler_mode}")

    async def calculate_scaling_decision(worker_states: list[Any], task_counts: TaskCounts) -> object:
        calls.append("calculate")
        return object()

    async def execute_scaling(
        scaling: object,
        worker_states: list[Any],
        task_counts: TaskCounts,
        summary: CycleSummary,
        detailed_counts: dict[str, Any],
    ) -> None:
        calls.append("execute")

    async def run_periodic_checks(
        worker_states: list[Any],
        excluded_workers: list[dict[str, Any]],
        queued_count: int,
        summary: CycleSummary,
    ) -> None:
        calls.append("periodic")

    loop._fetch_current_state = fetch_current_state
    loop._derive_all_worker_states = derive_all_worker_states
    loop._handle_early_termination = handle_early_termination
    loop._handle_spawning_workers = handle_spawning_workers
    loop._health_check_active_workers = health_check_active_workers
    loop._cleanup_error_workers = cleanup_error_workers
    loop._reconcile_orphaned_tasks = reconcile_orphaned_tasks
    loop._handle_stale_route_workers = handle_stale_route_workers
    loop._run_capacity_reconciler = run_capacity_reconciler
    loop._calculate_scaling_decision = calculate_scaling_decision
    loop._execute_scaling = execute_scaling
    loop._run_periodic_checks = run_periodic_checks
    return loop


def test_shadow_mode_observes_after_lifecycle_before_legacy_capacity_actions() -> None:
    calls: list[str] = []

    _run(_loop("shadow", calls).run_single_cycle())

    assert calls[:7] == [
        "fetch",
        "derive",
        "spawning",
        "health",
        "cleanup",
        "orphan",
        "stale_route",
    ]
    assert calls.index("reconciler:shadow") < calls.index("early")
    assert calls.index("reconciler:shadow") < calls.index("calculate")
    assert calls.index("calculate") < calls.index("execute")
    assert calls[-1] == "periodic"


def test_authoritative_mode_skips_only_legacy_capacity_actions() -> None:
    calls: list[str] = []

    _run(_loop("authoritative", calls).run_single_cycle())

    assert calls == [
        "fetch",
        "derive",
        "spawning",
        "health",
        "cleanup",
        "orphan",
        "stale_route",
        "reconciler:authoritative",
        "periodic",
    ]
    assert "early" not in calls
    assert "calculate" not in calls
    assert "execute" not in calls


def test_off_mode_preserves_legacy_capacity_order() -> None:
    calls: list[str] = []

    _run(_loop("off", calls).run_single_cycle())

    assert calls == [
        "fetch",
        "derive",
        "early",
        "spawning",
        "health",
        "cleanup",
        "orphan",
        "calculate",
        "execute",
        "periodic",
    ]

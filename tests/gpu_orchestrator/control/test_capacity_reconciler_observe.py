"""Tests for capacity reconciler observation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from gpu_orchestrator.control.capacity_reconciler import CapacityReconciler, UNATTRIBUTED_ROUTE_KEY
from gpu_orchestrator.control.capacity_types import POOL_FLOOR_ROUTE_KEY


NOW = datetime(2026, 5, 13, 14, 11, tzinfo=timezone.utc)


@dataclass(frozen=True)
class FrozenClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


class FakeDatabase:
    def __init__(
        self,
        *,
        workers: list[dict[str, Any]] | None = None,
        detailed_counts: dict[str, Any] | None = None,
        queued_count: int = 0,
        total_count: int = 0,
        route_backoffs: dict[str, dict[str, Any]] | None = None,
        previous_intent: dict[str, Any] | None = None,
    ) -> None:
        self.workers = workers or []
        self.detailed_counts = detailed_counts
        self.queued_count = queued_count
        self.total_count = total_count
        self.route_backoffs = route_backoffs or {}
        self.previous_intent = previous_intent
        self.derivation_worker_ids: list[str] = []

    async def get_workers(self, status: list[str]) -> list[dict[str, Any]]:
        return [worker for worker in self.workers if worker.get("status") in status]

    async def get_detailed_task_counts_via_edge_function(self) -> dict[str, Any] | None:
        return self.detailed_counts

    async def count_available_tasks_via_edge_function(self, *, include_active: bool) -> int:
        return self.total_count if include_active else self.queued_count

    async def batch_check_ever_claimed(self, worker_ids: list[str]) -> dict[str, bool]:
        self.derivation_worker_ids = list(worker_ids)
        return {worker_id: False for worker_id in worker_ids}

    async def batch_check_active_tasks(self, worker_ids: list[str]) -> dict[str, bool]:
        return {worker_id: self._worker(worker_id).get("has_active_task", False) for worker_id in worker_ids}

    async def get_route_backoff(self, pool: str, route_key: str | None = None) -> dict[str, Any]:
        return self.route_backoffs.get(route_key or "", {})

    async def get_latest_authoritative_capacity_intent(self, pool: str) -> dict[str, Any] | None:
        return self.previous_intent

    def _worker(self, worker_id: str) -> dict[str, Any]:
        for worker in self.workers:
            if worker["id"] == worker_id:
                return worker
        raise AssertionError(f"unknown worker {worker_id}")


def _config() -> SimpleNamespace:
    return SimpleNamespace(worker_pool="gpu-vibecomfy-canary")


def _worker(
    worker_id: str,
    *,
    status: str = "active",
    metadata: dict[str, Any] | None = None,
    has_active_task: bool = False,
    route_stale: bool = False,
    should_terminate: bool = False,
) -> dict[str, Any]:
    return {
        "id": worker_id,
        "status": status,
        "created_at": NOW.isoformat(),
        "metadata": metadata or {},
        "has_active_task": has_active_task,
        "route_stale": route_stale,
        "should_terminate": should_terminate,
    }


def _state_deriver(*, worker: dict[str, Any], now: datetime, has_active_task: bool, **_: Any) -> Any:
    return SimpleNamespace(
        worker_id=worker["id"],
        is_active=worker["status"] == "active",
        is_spawning=worker["status"] == "spawning",
        has_active_task=has_active_task,
        is_route_stale=worker.get("route_stale", False),
        should_terminate=worker.get("should_terminate", False),
        excluded_from_capacity_control=False,
        created_at=now,
    )


def _observe(db: FakeDatabase):
    reconciler = CapacityReconciler(db=db, config=_config(), clock=FrozenClock(), state_deriver=_state_deriver)
    return asyncio.run(reconciler.observe(cycle_id=17))


def _selected_route(
    route_key: str,
    *,
    queued_only: int = 0,
    active_only: int = 0,
    blocked_by_capacity: int = 0,
    potentially_claimable: int | None = None,
) -> dict[str, Any]:
    return {
        "route_key": route_key,
        "queued_only": queued_only,
        "active_only": active_only,
        "blocked_by_capacity": blocked_by_capacity,
        "potentially_claimable": potentially_claimable,
        "selected_backend": "vibecomfy",
        "selected_profile": "3",
        "selector_namespace": "canary",
        "selector_version": "42",
        "worker_contract_version": 1,
    }


def test_observe_excludes_live_test_workers_but_keeps_them_visible(monkeypatch) -> None:
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "canary")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "42")
    monkeypatch.setenv("REIGH_WORKER_CONTRACT_VERSION", "1")
    monkeypatch.setenv("REIGH_WORKER_POOL", "gpu-vibecomfy-canary")
    db = FakeDatabase(
        workers=[
            _worker("prod-active", metadata={"spawn_reason_route_key": "route-a"}),
            _worker("live-test", metadata={"live_test_variant": "smoke", "spawn_reason_route_key": "route-a"}),
        ],
        detailed_counts={"route_totals": [_selected_route("route-a", queued_only=1, active_only=1)]},
    )

    observed = _observe(db)

    assert [worker.worker_id for worker in observed.workers] == ["prod-active"]
    assert [worker["id"] for worker in observed.excluded_workers] == ["live-test"]
    assert db.derivation_worker_ids == ["prod-active"]
    assert observed.effective_capacity == 1


def test_observe_uses_exact_selected_route_totals_and_worker_route_metadata(monkeypatch) -> None:
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "canary")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "42")
    monkeypatch.setenv("REIGH_WORKER_CONTRACT_VERSION", "1")
    db = FakeDatabase(
        workers=[_worker("spawning", status="spawning", metadata={"route_contract": {"route_key": "route-a"}})],
        detailed_counts={
            "route_totals": [
                _selected_route("route-a", queued_only=2, active_only=1, blocked_by_capacity=1),
                {**_selected_route("route-b", queued_only=9), "selector_namespace": "production"},
            ],
        },
    )

    observed = _observe(db)

    assert observed.route_demand_source == "route_totals"
    assert observed.route_demand_fallback is False
    assert [(demand.route_key, demand.queued, demand.in_progress) for demand in observed.route_demands] == [
        ("route-a", 3, 1)
    ]
    assert observed.workers[0].route_key == "route-a"
    assert observed.effective_capacity == 1


def test_observe_infers_single_route_from_selected_pool_totals(monkeypatch) -> None:
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "canary")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "42")
    db = FakeDatabase(
        detailed_counts={
            "selected_pool_totals": {
                "route_keys": ["route-a"],
                "queued_only": 1,
                "active_only": 2,
                "blocked_by_capacity": 1,
            }
        }
    )

    observed = _observe(db)

    assert observed.route_demand_source == "single_route_key_inferred"
    assert observed.route_demand_fallback is False
    assert [(demand.route_key, demand.queued, demand.in_progress) for demand in observed.route_demands] == [
        ("route-a", 2, 2)
    ]


def test_observe_marks_ambiguous_selected_totals_with_unattributed_fallback(monkeypatch) -> None:
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "canary")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "42")
    db = FakeDatabase(
        detailed_counts={
            "selected_pool_totals": {
                "route_keys": ["route-a", "route-b"],
                "potentially_claimable": 4,
                "active_only": 1,
            }
        }
    )

    observed = _observe(db)

    assert observed.route_demand_source == "unattributed_selected_totals"
    assert observed.route_demand_fallback is True
    assert observed.route_demands[0].route_key == UNATTRIBUTED_ROUTE_KEY
    assert observed.route_demands[0].route_key != POOL_FLOOR_ROUTE_KEY
    assert observed.queued_total == 4
    assert observed.in_progress_total == 1


def test_observe_applies_route_backoff_and_excludes_route_stale_from_capacity(monkeypatch) -> None:
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "canary")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "42")
    db = FakeDatabase(
        workers=[
            _worker("stale-active", route_stale=True),
            _worker("fresh-spawning", status="spawning"),
        ],
        detailed_counts={"route_totals": [_selected_route("route-a", queued_only=3)]},
        route_backoffs={"route-a": {"next_spawn_allowed_at": (NOW + timedelta(minutes=5)).isoformat()}},
    )

    observed = _observe(db)

    assert observed.active_count == 0
    assert observed.spawning_count == 1
    assert observed.effective_capacity == 1
    assert observed.route_demands[0].spawn_allowed is False
    assert observed.eligible_queued_total == 0

"""Tests for capacity reconciler orchestration boundaries."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from gpu_orchestrator.control.capacity_reconciler import CapacityReconciler
from gpu_orchestrator.control.capacity_types import (
    ObservedPoolState,
    ReconciliationResult,
    RouteDemand,
    WorkerCapacitySnapshot,
)


NOW = datetime(2026, 5, 13, 14, 11, tzinfo=timezone.utc)


@dataclass(frozen=True)
class FrozenClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


class FakeDatabase:
    def __init__(self, *, lease_acquired: bool = True, existing_worker: dict[str, Any] | None = None) -> None:
        self.lease_acquired = lease_acquired
        self.existing_worker = existing_worker
        self.calls: list[str] = []
        self.intent_rows: list[dict[str, Any]] = []
        self.outcome_updates: list[dict[str, Any]] = []
        self.system_logs: list[dict[str, Any]] = []
        self.reset_backoffs: list[tuple[str, str]] = []
        self.failure_backoffs: list[tuple[str, str, str]] = []

    async def insert_worker_capacity_intent(self, **payload: Any) -> dict[str, Any]:
        self.calls.append("insert_intent")
        row = {"id": f"intent-{len(self.intent_rows) + 1}", **payload}
        self.intent_rows.append(row)
        return row

    async def update_worker_capacity_intent_outcome(self, intent_id: str, *, outcome: dict[str, Any]) -> bool:
        self.calls.append("update_outcome")
        self.outcome_updates.append({"intent_id": intent_id, "outcome": outcome})
        return True

    async def insert_system_log_metadata(self, **payload: Any) -> bool:
        self.calls.append("system_log")
        self.system_logs.append(payload)
        return True

    async def acquire_pool_lease(self, **payload: Any) -> bool:
        self.calls.append("acquire_lease")
        return self.lease_acquired

    async def release_pool_lease(self, **payload: Any) -> bool:
        self.calls.append("release_lease")
        return True

    async def get_worker_by_capacity_action(self, intent_id: str, ordinal: int) -> dict[str, Any] | None:
        self.calls.append(f"lookup_worker:{ordinal}")
        return self.existing_worker

    async def reset_route_backoff(self, pool: str, route_key: str, **_: Any) -> bool:
        self.calls.append("reset_backoff")
        self.reset_backoffs.append((pool, route_key))
        return True

    async def record_route_spawn_failure(self, pool: str, route_key: str, *, error: str, **_: Any) -> dict[str, Any]:
        self.calls.append("record_backoff_failure")
        self.failure_backoffs.append((pool, route_key, error))
        return {}


class FakeExecutor:
    def __init__(self, calls: list[str], *, spawn_success: bool = True) -> None:
        self.calls = calls
        self.spawn_success = spawn_success

    async def spawn(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(f"spawn:{payload['action'].ordinal}")
        return {"success": self.spawn_success, "error": None if self.spawn_success else "runpod_spawn_failed"}

    async def cancel_pending_spawn(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(f"cancel:{payload['action'].worker_id}")
        return {"success": True}

    async def terminate_idle(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(f"terminate:{payload['action'].worker_id}")
        return {"success": True}


class StaticObservationReconciler(CapacityReconciler):
    def __init__(self, observations: list[ObservedPoolState], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._observations = observations

    async def observe(self, cycle_id: int | None = None) -> ObservedPoolState:
        observation = self._observations.pop(0) if self._observations else self._last_observation
        self._last_observation = observation
        return observation


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        worker_pool="gpu-main",
        min_active_gpus=0,
        max_active_gpus=5,
        machines_to_keep_idle=0,
        orchestrator_poll_sec=30,
    )


def _queued_observation(*, route_key: str = "route-a", queued: int = 1) -> ObservedPoolState:
    return ObservedPoolState(
        pool="gpu-main",
        route_demands=(RouteDemand(route_key, queued=queued),),
        observed_at=NOW,
        cycle_id=7,
    )


def _run(result: Any) -> ReconciliationResult:
    return asyncio.run(result)


def test_shadow_reconcile_records_intent_and_logs_without_side_effects() -> None:
    db = FakeDatabase()
    executor = FakeExecutor(db.calls)
    reconciler = StaticObservationReconciler(
        [_queued_observation()],
        db=db,
        config=_config(),
        mode="shadow",
        observer_id="observer-shadow",
        action_executor=executor,
        clock=FrozenClock(),
    )

    result = _run(reconciler.reconcile(7))

    assert result.mode == "shadow"
    assert db.intent_rows[0]["shadow"] is True
    assert db.intent_rows[0]["observer_id"] == "observer-shadow"
    assert db.system_logs
    assert db.system_logs[0]["metadata"]["shadow_intended_action"] == "true"
    assert "acquire_lease" not in db.calls
    assert not any(call.startswith("spawn:") for call in db.calls)
    assert db.reset_backoffs == []
    assert db.failure_backoffs == []


def test_authoritative_reconcile_acquires_lease_records_before_spawn_and_resets_backoff() -> None:
    db = FakeDatabase()
    executor = FakeExecutor(db.calls)
    reconciler = StaticObservationReconciler(
        [_queued_observation(), _queued_observation(queued=2)],
        db=db,
        config=_config(),
        mode="authoritative",
        observer_id="observer-auth",
        holder_id="holder-auth",
        action_executor=executor,
        clock=FrozenClock(),
    )

    result = _run(reconciler.reconcile(7))

    assert result.lease_acquired is True
    assert db.intent_rows[0]["shadow"] is False
    assert db.calls.index("acquire_lease") < db.calls.index("insert_intent") < db.calls.index("spawn:0")
    assert db.calls[-1] == "release_lease"
    assert db.reset_backoffs == [("gpu-main", "route-a"), ("gpu-main", "route-a")]
    assert db.failure_backoffs == []
    assert result.counters.spawned == 2


def test_authoritative_spawn_action_is_idempotent_when_worker_metadata_exists() -> None:
    db = FakeDatabase(existing_worker={"id": "worker-existing"})
    executor = FakeExecutor(db.calls)
    reconciler = StaticObservationReconciler(
        [_queued_observation(), _queued_observation()],
        db=db,
        config=_config(),
        mode="authoritative",
        action_executor=executor,
        clock=FrozenClock(),
    )

    result = _run(reconciler.reconcile(7))

    assert "lookup_worker:0" in db.calls
    assert not any(call.startswith("spawn:") for call in db.calls)
    assert db.reset_backoffs == []
    assert result.outcome["skipped_duplicate_spawn_actions"][0]["worker_id"] == "worker-existing"


def test_authoritative_spawn_failure_updates_route_backoff() -> None:
    db = FakeDatabase()
    executor = FakeExecutor(db.calls, spawn_success=False)
    reconciler = StaticObservationReconciler(
        [_queued_observation(), _queued_observation()],
        db=db,
        config=_config(),
        mode="authoritative",
        action_executor=executor,
        clock=FrozenClock(),
    )

    result = _run(reconciler.reconcile(7))

    assert result.counters.errors == 1
    assert db.reset_backoffs == []
    assert db.failure_backoffs == [("gpu-main", "route-a", "runpod_spawn_failed")]


def test_authoritative_lease_rejection_suppresses_record_and_actions() -> None:
    db = FakeDatabase(lease_acquired=False)
    executor = FakeExecutor(db.calls)
    reconciler = StaticObservationReconciler(
        [_queued_observation(), _queued_observation()],
        db=db,
        config=_config(),
        mode="authoritative",
        action_executor=executor,
        clock=FrozenClock(),
    )

    result = _run(reconciler.reconcile(7))

    assert result.lease_acquired is False
    assert result.outcome["lease_suppressed"] is True
    assert db.intent_rows == []
    assert not any(call.startswith("spawn:") for call in db.calls)
    assert "release_lease" not in db.calls


def test_first_authoritative_observe_records_legacy_spawning_adoption() -> None:
    db = FakeDatabase()
    executor = FakeExecutor(db.calls)
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(WorkerCapacitySnapshot(worker_id="legacy-spawn", is_spawning=True, metadata={}),),
        route_demands=(),
        observed_at=NOW,
    )
    reconciler = StaticObservationReconciler(
        [observed, observed],
        db=db,
        config=_config(),
        mode="authoritative",
        action_executor=executor,
        clock=FrozenClock(),
    )

    result = _run(reconciler.reconcile(7))

    assert result.adopted_worker_ids == ("legacy-spawn",)
    assert db.intent_rows[0]["reason"] == "adoption"
    assert db.intent_rows[0]["outcome"]["adopted_legacy_spawning_worker_ids"] == ["legacy-spawn"]
    assert not any(action["type"] == "cancel_pending_spawn" for action in db.intent_rows[0]["actions"])

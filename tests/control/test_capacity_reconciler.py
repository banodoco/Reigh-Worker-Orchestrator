"""Focused runtime-safety tests for capacity reconciler rollout."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from gpu_orchestrator.control.capacity_actions import (
    CapacityActionExecutor,
    apply_reconciliation_counters_to_summary,
)
from gpu_orchestrator.control.capacity_plan import build_capacity_plan
from gpu_orchestrator.control.capacity_reconciler import CapacityReconciler
from gpu_orchestrator.control.capacity_types import (
    CapacityAction,
    CapacityActionType,
    IntentRecord,
    ObservedPoolState,
    POOL_FLOOR_ROUTE_KEY,
    ResultCounters,
    RouteDemand,
    WorkerCapacitySnapshot,
)
from gpu_orchestrator.database import CAPACITY_POOL_FLOOR_ROUTE_KEY
from gpu_orchestrator.worker_state import CycleSummary
from tests.gpu_orchestrator.test_database_capacity_helpers import _FakeSupabase, _client


NOW = datetime(2026, 5, 13, 14, 11, tzinfo=timezone.utc)


@dataclass(frozen=True)
class FrozenClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


def _run(value: Any) -> Any:
    return asyncio.run(value)


def _worker(
    worker_id: str,
    *,
    active: bool = False,
    spawning: bool = False,
    idle: bool = False,
    route_stale: bool = False,
    route_key: str | None = "route-a",
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> WorkerCapacitySnapshot:
    return WorkerCapacitySnapshot(
        worker_id=worker_id,
        route_key=route_key,
        is_active=active,
        is_spawning=spawning,
        is_idle=idle,
        is_route_stale=route_stale,
        created_at=created_at or NOW,
        metadata=metadata or {},
    )


def _plan(
    observed: ObservedPoolState,
    *,
    min_active_gpus: int = 0,
    max_active_gpus: int = 8,
    machines_to_keep_idle: int = 0,
    now: datetime = NOW,
):
    return build_capacity_plan(
        observed,
        min_active_gpus=min_active_gpus,
        max_active_gpus=max_active_gpus,
        machines_to_keep_idle=machines_to_keep_idle,
        orchestrator_poll_sec=30,
        clock=FrozenClock(now),
    )


class FakeDatabase:
    def __init__(self, *, lease_acquired: bool = True, existing_worker: dict[str, Any] | None = None) -> None:
        self.lease_acquired = lease_acquired
        self.existing_worker = existing_worker
        self.calls: list[str] = []
        self.intent_rows: list[dict[str, Any]] = []
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
        self.intent_rows[-1]["recorded_outcome"] = outcome
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
        return {"success": self.spawn_success, "error": None if self.spawn_success else "spawn_failed"}

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
        self._last_observation = observations[-1]

    async def observe(self, cycle_id: int | None = None) -> ObservedPoolState:
        observation = self._observations.pop(0) if self._observations else self._last_observation
        self._last_observation = observation
        return observation


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        worker_pool="gpu-main",
        min_active_gpus=0,
        max_active_gpus=8,
        machines_to_keep_idle=0,
        orchestrator_poll_sec=30,
    )


def test_queued_zero_incident_replay_does_not_cancel_spawning_worker() -> None:
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(
            _worker("productive", active=True),
            _worker("authorized-spawn", spawning=True, created_at=NOW - timedelta(seconds=250)),
        ),
        route_demands=(RouteDemand("route-a", queued=0, in_progress=5),),
        observed_at=NOW,
    )

    plan = _plan(observed)

    assert plan.effective_capacity == 2
    assert plan.desired_capacity == 2
    assert all(action.action_type != CapacityActionType.CANCEL_PENDING_SPAWN for action in plan.actions)


def test_route_scoped_backoff_fake_clock_delays_and_resets_on_successful_spawn() -> None:
    fake = _FakeSupabase()
    db = _client(fake)
    expected_delays = [30, 60, 120, 240]

    for index, delay in enumerate(expected_delays, start=1):
        row = _run(db.record_route_spawn_failure("gpu-main", "route-a", now=NOW, error="spawn_failed"))
        assert row["route_key"] == "route-a"
        assert row["consecutive_spawn_failures"] == index
        assert row["next_spawn_allowed_at"] == (NOW + timedelta(seconds=delay)).isoformat()

    assert _run(db.reset_route_backoff("gpu-main", "route-a", now=NOW + timedelta(seconds=241)))
    reset = _run(db.get_route_backoff("gpu-main", "route-a"))
    assert reset["consecutive_spawn_failures"] == 0
    assert reset["next_spawn_allowed_at"] is None
    assert reset["last_spawn_succeeded_at"] == (NOW + timedelta(seconds=241)).isoformat()


def test_pool_floor_backoff_sentinel_is_unique_for_nullable_route_keys() -> None:
    fake = _FakeSupabase()
    db = _client(fake)

    _run(db.record_route_spawn_failure("gpu-main", None, now=NOW, error="a"))
    _run(db.record_route_spawn_failure("gpu-main", "", now=NOW, error="b"))

    rows = fake.tables["worker_capacity_route_backoffs"]
    assert len(rows) == 1
    assert rows[0]["route_key"] == CAPACITY_POOL_FLOOR_ROUTE_KEY
    assert rows[0]["consecutive_spawn_failures"] == 2


def test_pool_bounds_apply_once_across_multiple_routes() -> None:
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(_worker("busy", active=True),),
        route_demands=(
            RouteDemand("route-a", queued=4),
            RouteDemand("route-b", queued=4),
        ),
        observed_at=NOW,
    )

    plan = _plan(observed, min_active_gpus=2, max_active_gpus=5, machines_to_keep_idle=2)

    assert plan.desired_capacity == 5
    assert len([action for action in plan.actions if action.action_type == CapacityActionType.SPAWN]) == 4
    assert plan.formula_components["min_active_gpus"] == 2
    assert plan.formula_components["max_active_gpus"] == 5


def test_cancel_hysteresis_requires_wall_clock_or_expired_valid_until() -> None:
    workers = (
        _worker("spawning-old", spawning=True, created_at=NOW - timedelta(seconds=250)),
        _worker("spawning-new", spawning=True, created_at=NOW - timedelta(seconds=240)),
    )
    recent = IntentRecord(desired_capacity=1, effective_capacity=2, stable_since=NOW - timedelta(seconds=30), valid_until=NOW + timedelta(minutes=5), shadow=False)
    elapsed = IntentRecord(desired_capacity=1, effective_capacity=2, stable_since=NOW - timedelta(seconds=61), valid_until=NOW + timedelta(minutes=5), shadow=False)
    expired = IntentRecord(desired_capacity=2, effective_capacity=1, stable_since=NOW, valid_until=NOW - timedelta(seconds=1), shadow=False)

    recent_plan = _plan(ObservedPoolState(pool="gpu-main", workers=workers, previous_intent=recent, observed_at=NOW))
    elapsed_plan = _plan(ObservedPoolState(pool="gpu-main", workers=workers, previous_intent=elapsed, observed_at=NOW))
    expired_plan = _plan(ObservedPoolState(pool="gpu-main", workers=workers, previous_intent=expired, observed_at=NOW))

    assert all(action.action_type != CapacityActionType.CANCEL_PENDING_SPAWN for action in recent_plan.actions)
    assert elapsed_plan.actions[0].action_type == CapacityActionType.CANCEL_PENDING_SPAWN
    assert expired_plan.actions[0].action_type == CapacityActionType.CANCEL_PENDING_SPAWN


def test_route_stale_and_excluded_workers_do_not_count_as_effective_capacity() -> None:
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(
            _worker("stale", active=True, route_stale=True),
            WorkerCapacitySnapshot(worker_id="excluded", is_active=True, excluded_from_capacity_control=True),
            _worker("fresh-spawning", spawning=True),
        ),
        route_demands=(RouteDemand("route-a", queued=1),),
        observed_at=NOW,
    )

    plan = _plan(observed, min_active_gpus=1)

    assert plan.active_count == 0
    assert plan.spawning_count == 1
    assert plan.effective_capacity == 1
    assert plan.actions[0].action_type == CapacityActionType.NOOP


def test_authoritative_record_before_act_single_writer_adoption_and_exact_one_row() -> None:
    db = FakeDatabase()
    executor = FakeExecutor(db.calls)
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(_worker("legacy-spawn", spawning=True, metadata={}),),
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

    result = _run(reconciler.reconcile(44))

    assert result.lease_acquired is True
    assert result.adopted_worker_ids == ("legacy-spawn",)
    assert db.calls.index("acquire_lease") < db.calls.index("insert_intent")
    assert len(db.intent_rows) == 1
    assert db.intent_rows[0]["shadow"] is False
    assert db.intent_rows[0]["cycle_id"] == 44
    assert not any(action["type"] == "cancel_pending_spawn" for action in db.intent_rows[0]["actions"])


def test_authoritative_lease_rejection_suppresses_record_and_actions() -> None:
    db = FakeDatabase(lease_acquired=False)
    executor = FakeExecutor(db.calls)
    observed = ObservedPoolState(pool="gpu-main", route_demands=(RouteDemand("route-a", queued=2),), observed_at=NOW)
    reconciler = StaticObservationReconciler(
        [observed, observed],
        db=db,
        config=_config(),
        mode="authoritative",
        action_executor=executor,
        clock=FrozenClock(),
    )

    result = _run(reconciler.reconcile(45))

    assert result.lease_acquired is False
    assert result.outcome["lease_suppressed"] is True
    assert db.intent_rows == []
    assert not any(call.startswith("spawn:") for call in db.calls)


def test_shadow_no_side_effects_and_observer_dedupe_identifiers() -> None:
    db = FakeDatabase()
    executor = FakeExecutor(db.calls)
    observed = ObservedPoolState(pool="gpu-main", route_demands=(RouteDemand("route-a", queued=1),), observed_at=NOW)
    reconciler = StaticObservationReconciler(
        [observed, observed],
        db=db,
        config=_config(),
        mode="shadow",
        observer_id="observer-a",
        action_executor=executor,
        clock=FrozenClock(),
    )

    _run(reconciler.reconcile(1))
    _run(reconciler.reconcile(2))

    assert len(db.intent_rows) == 2
    assert all(row["shadow"] is True for row in db.intent_rows)
    assert {row["observer_id"] for row in db.intent_rows} == {"observer-a"}
    assert db.intent_rows[0]["observation_id"] != db.intent_rows[1]["observation_id"]
    assert "acquire_lease" not in db.calls
    assert not any(call.startswith("spawn:") for call in db.calls)
    assert db.reset_backoffs == []
    assert db.failure_backoffs == []


def test_spawn_idempotency_and_summary_counters_and_cancel_terminal_marking() -> None:
    db = FakeDatabase(existing_worker={"id": "existing-worker"})
    executor = FakeExecutor(db.calls)
    observed = ObservedPoolState(pool="gpu-main", route_demands=(RouteDemand("route-a", queued=1),), observed_at=NOW)
    reconciler = StaticObservationReconciler(
        [observed, observed],
        db=db,
        config=_config(),
        mode="authoritative",
        action_executor=executor,
        clock=FrozenClock(),
    )

    result = _run(reconciler.reconcile(46))
    summary = CycleSummary(workers_spawned=1, workers_terminated=1)
    apply_reconciliation_counters_to_summary(
        summary,
        ResultCounters(spawned=2, cancelled_pending_spawns=1, terminated_idle=1),
    )

    assert result.outcome["skipped_duplicate_spawn_actions"][0]["worker_id"] == "existing-worker"
    assert not any(call.startswith("spawn:") for call in db.calls)
    assert summary.workers_spawned == 3
    assert summary.workers_terminated == 3

    action_db = ActionDatabase()
    action_db.workers["spawning"] = {"id": "spawning", "metadata": {"runpod_id": "pod-1"}}
    runpod = ActionRunPod(terminate_success=False)
    action = CapacityAction(CapacityActionType.CANCEL_PENDING_SPAWN, reason="test", worker_id="spawning", ordinal=7)
    cancel_result = _run(
        CapacityActionExecutor(db=action_db, runpod=runpod, config=SimpleNamespace(), clock=FrozenClock()).cancel_pending_spawn(
            action=action,
            intent_row={"id": "intent-cancel"},
            pool="gpu-main",
        )
    )
    assert cancel_result["success"] is True
    assert action_db.workers["spawning"]["status"] == "terminated"
    assert action_db.workers["spawning"]["metadata"]["runpod_termination_success"] is False


class ActionDatabase:
    def __init__(self) -> None:
        self.workers: dict[str, dict[str, Any]] = {}

    async def get_worker(self, worker_id: str) -> dict[str, Any] | None:
        return self.workers.get(worker_id)

    async def update_worker_status(self, worker_id: str, status: str, metadata_update: dict[str, Any]) -> bool:
        self.workers.setdefault(worker_id, {"id": worker_id, "metadata": {}})
        self.workers[worker_id]["status"] = status
        self.workers[worker_id].setdefault("metadata", {}).update(metadata_update)
        return True


class ActionRunPod:
    gpu_type = "RTX 4090"

    def __init__(self, *, terminate_success: bool) -> None:
        self.terminate_success = terminate_success

    async def terminate_worker(self, runpod_id: str) -> bool:
        return self.terminate_success

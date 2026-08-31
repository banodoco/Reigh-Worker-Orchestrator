"""Tests for pure capacity reconciler planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from gpu_orchestrator.control.capacity_plan import build_capacity_plan
from gpu_orchestrator.control.capacity_types import (
    CapacityActionType,
    IntentRecord,
    ObservedPoolState,
    POOL_FLOOR_ROUTE_KEY,
    RouteDemand,
    WorkerCapacitySnapshot,
)


@dataclass(frozen=True)
class FrozenClock:
    value: datetime

    def now(self) -> datetime:
        return self.value


NOW = datetime(2026, 5, 13, 14, 11, tzinfo=timezone.utc)


def _worker(
    worker_id: str,
    *,
    active: bool = False,
    spawning: bool = False,
    idle: bool = False,
    route_stale: bool = False,
    route_key: str | None = "route-a",
    created_offset_sec: int = 0,
) -> WorkerCapacitySnapshot:
    return WorkerCapacitySnapshot(
        worker_id=worker_id,
        route_key=route_key,
        is_active=active,
        is_spawning=spawning,
        is_idle=idle,
        is_route_stale=route_stale,
        created_at=NOW + timedelta(seconds=created_offset_sec),
    )


def test_desired_formula_is_pool_scoped_and_backoff_only_reduces_eligible_queue() -> None:
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(
            _worker("busy-1", active=True, idle=False),
            _worker("idle-1", active=True, idle=True),
        ),
        route_demands=(
            RouteDemand("route-a", queued=2, spawn_allowed=True),
            RouteDemand("route-b", queued=10, spawn_allowed=False, backoff_until=NOW + timedelta(minutes=5)),
        ),
        observed_at=NOW,
    )

    plan = build_capacity_plan(
        observed,
        min_active_gpus=1,
        max_active_gpus=8,
        machines_to_keep_idle=2,
        orchestrator_poll_sec=30,
        clock=FrozenClock(NOW),
    )

    assert plan.formula_components["eligible_queued_total"] == 2
    assert plan.formula_components["queued_coverage"] == 3
    assert plan.formula_components["busy_buffer"] == 3
    assert plan.desired_capacity == 3
    assert plan.effective_capacity == 2
    assert [action.action_type for action in plan.actions] == [CapacityActionType.SPAWN]
    assert plan.actions[0].route_key == "route-a"
    assert [action.reason for action in plan.suppressed_actions] == ["suppressed_by_backoff"]
    assert plan.suppressed_actions[0].route_key == "route-b"


def test_spawn_budget_allocates_route_demand_before_pool_floor_buffer() -> None:
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(),
        route_demands=(RouteDemand("route-a", queued=1, spawn_allowed=True),),
        observed_at=NOW,
    )

    plan = build_capacity_plan(
        observed,
        min_active_gpus=0,
        max_active_gpus=5,
        machines_to_keep_idle=2,
        orchestrator_poll_sec=30,
        clock=FrozenClock(NOW),
    )

    assert plan.desired_capacity == 2
    assert [(action.action_type, action.reason, action.route_key) for action in plan.actions] == [
        (CapacityActionType.SPAWN, "queued_coverage", "route-a"),
        (CapacityActionType.SPAWN, "pool_floor_buffer", POOL_FLOOR_ROUTE_KEY),
    ]


def test_route_stale_workers_do_not_count_as_effective_capacity() -> None:
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(
            _worker("stale", active=True, idle=True, route_stale=True),
            _worker("fresh-spawning", spawning=True),
        ),
        route_demands=(RouteDemand("route-a", queued=1),),
        observed_at=NOW,
    )

    plan = build_capacity_plan(
        observed,
        min_active_gpus=1,
        max_active_gpus=4,
        machines_to_keep_idle=1,
        orchestrator_poll_sec=30,
        clock=FrozenClock(NOW),
    )

    assert plan.active_count == 0
    assert plan.spawning_count == 1
    assert plan.effective_capacity == 1
    assert plan.actions[0].action_type == CapacityActionType.NOOP


def test_over_capacity_terminates_only_idle_active_workers_first() -> None:
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(
            _worker("busy", active=True, idle=False),
            _worker("idle-old", active=True, idle=True, created_offset_sec=-20),
            _worker("idle-new", active=True, idle=True, created_offset_sec=-10),
        ),
        route_demands=(),
        observed_at=NOW,
    )

    plan = build_capacity_plan(
        observed,
        min_active_gpus=0,
        max_active_gpus=5,
        machines_to_keep_idle=1,
        orchestrator_poll_sec=30,
        clock=FrozenClock(NOW),
    )

    assert plan.desired_capacity == 2
    assert [(action.action_type, action.worker_id) for action in plan.actions] == [
        (CapacityActionType.TERMINATE_IDLE, "idle-old")
    ]


def test_cancel_pending_spawn_requires_wall_clock_hysteresis() -> None:
    previous = IntentRecord(
        desired_capacity=1,
        effective_capacity=2,
        created_at=NOW - timedelta(seconds=120),
        stable_since=NOW - timedelta(seconds=120),
        valid_until=NOW + timedelta(minutes=5),
        shadow=False,
    )
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(
            _worker("busy", active=True, idle=False),
            _worker("spawning-1", spawning=True, created_offset_sec=-20),
            _worker("spawning-2", spawning=True, created_offset_sec=-10),
        ),
        route_demands=(),
        previous_intent=previous,
        observed_at=NOW,
    )

    plan = build_capacity_plan(
        observed,
        min_active_gpus=0,
        max_active_gpus=5,
        machines_to_keep_idle=0,
        orchestrator_poll_sec=30,
        clock=FrozenClock(NOW),
    )

    assert plan.desired_capacity == 2
    assert [(action.action_type, action.worker_id) for action in plan.actions] == [
        (CapacityActionType.CANCEL_PENDING_SPAWN, "spawning-2")
    ]


def test_cancel_pending_spawn_is_not_immediate_and_valid_until_allows_cancel() -> None:
    recent_previous = IntentRecord(
        desired_capacity=1,
        effective_capacity=2,
        stable_since=NOW - timedelta(seconds=20),
        valid_until=NOW + timedelta(minutes=5),
        shadow=False,
    )
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(
            _worker("busy", active=True, idle=False),
            _worker("spawning-1", spawning=True, created_offset_sec=-20),
            _worker("spawning-2", spawning=True, created_offset_sec=-10),
        ),
        previous_intent=recent_previous,
        observed_at=NOW,
    )

    plan = build_capacity_plan(
        observed,
        min_active_gpus=0,
        max_active_gpus=5,
        machines_to_keep_idle=0,
        orchestrator_poll_sec=30,
        clock=FrozenClock(NOW),
    )
    assert plan.actions[0].action_type == CapacityActionType.NOOP

    expired_previous = IntentRecord(
        desired_capacity=2,
        effective_capacity=1,
        stable_since=NOW,
        valid_until=NOW - timedelta(seconds=1),
        shadow=False,
    )
    expired_plan = build_capacity_plan(
        ObservedPoolState(
            pool="gpu-main",
            workers=observed.workers,
            previous_intent=expired_previous,
            observed_at=NOW,
        ),
        min_active_gpus=0,
        max_active_gpus=5,
        machines_to_keep_idle=0,
        orchestrator_poll_sec=30,
        clock=FrozenClock(NOW),
    )
    assert [(action.action_type, action.worker_id) for action in expired_plan.actions] == [
        (CapacityActionType.CANCEL_PENDING_SPAWN, "spawning-2")
    ]


def test_spawning_counts_as_capacity_when_queue_drops_to_zero() -> None:
    observed = ObservedPoolState(
        pool="gpu-main",
        workers=(
            _worker("busy", active=True, idle=False),
            _worker("spawning", spawning=True),
        ),
        route_demands=(RouteDemand("route-a", queued=0, in_progress=5),),
        previous_intent=None,
        observed_at=NOW,
    )

    plan = build_capacity_plan(
        observed,
        min_active_gpus=0,
        max_active_gpus=5,
        machines_to_keep_idle=0,
        orchestrator_poll_sec=30,
        clock=FrozenClock(NOW),
    )

    assert plan.effective_capacity == 2
    assert plan.desired_capacity == 2
    assert plan.actions[0].action_type == CapacityActionType.NOOP

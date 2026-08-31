"""Pure capacity planning for the capacity reconciler."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from gpu_orchestrator.control.capacity_types import (
    CapacityAction,
    CapacityActionType,
    CapacityPlan,
    Clock,
    ObservedPoolState,
    POOL_FLOOR_ROUTE_KEY,
    RouteDemand,
    SystemClock,
    WorkerCapacitySnapshot,
)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _route_key(route_key: str | None) -> str:
    return route_key or POOL_FLOOR_ROUTE_KEY


def _desired_reason(
    *,
    desired: int,
    min_active_gpus: int,
    queued_coverage: int,
    busy_buffer: int,
) -> str:
    if desired == min_active_gpus and min_active_gpus >= queued_coverage and min_active_gpus >= busy_buffer:
        return "min_floor"
    if desired == queued_coverage and queued_coverage >= busy_buffer:
        return "queued_coverage"
    if desired == busy_buffer:
        return "busy_buffer"
    return "max_capacity"


def _stable_since(observed: ObservedPoolState, desired: int, now: datetime) -> datetime:
    previous = observed.previous_intent
    if previous and previous.desired_capacity == desired and previous.stable_since:
        return _as_utc(previous.stable_since) or now
    return now


def _cancel_hysteresis_satisfied(
    observed: ObservedPoolState,
    *,
    desired: int,
    effective: int,
    orchestrator_poll_sec: int,
    now: datetime,
) -> bool:
    previous = observed.previous_intent
    if not previous:
        return False

    valid_until = _as_utc(previous.valid_until)
    if valid_until and valid_until <= now:
        return True

    if previous.desired_capacity > previous.effective_capacity:
        return False
    if desired > effective:
        return False

    stable_since = _as_utc(previous.stable_since) or _as_utc(previous.created_at)
    if not stable_since:
        return False
    return now - stable_since >= timedelta(seconds=2 * orchestrator_poll_sec)


def _sorted_idle_termination_candidates(
    workers: Iterable[WorkerCapacitySnapshot],
) -> list[WorkerCapacitySnapshot]:
    return sorted(
        [worker for worker in workers if worker.can_terminate_idle],
        key=lambda worker: (worker.created_at or datetime.min.replace(tzinfo=timezone.utc), worker.worker_id),
    )


def _sorted_spawn_cancel_candidates(
    workers: Iterable[WorkerCapacitySnapshot],
) -> list[WorkerCapacitySnapshot]:
    return sorted(
        [worker for worker in workers if worker.can_cancel_pending_spawn],
        key=lambda worker: (worker.created_at or datetime.max.replace(tzinfo=timezone.utc), worker.worker_id),
        reverse=True,
    )


def _allocate_spawn_actions(
    route_demands: tuple[RouteDemand, ...],
    *,
    spawn_budget: int,
    workers_covering_tasks: int,
) -> list[CapacityAction]:
    actions: list[CapacityAction] = []
    remaining_budget = max(0, spawn_budget)
    remaining_coverage = max(0, workers_covering_tasks)

    for route in sorted(route_demands, key=lambda demand: demand.route_key):
        if remaining_budget <= 0:
            break
        if not route.spawn_allowed:
            continue

        queued = max(0, route.queued)
        covered_here = min(queued, remaining_coverage)
        remaining_coverage -= covered_here
        uncovered = queued - covered_here
        route_spawns = min(uncovered, remaining_budget)

        for _ in range(route_spawns):
            actions.append(
                CapacityAction(
                    action_type=CapacityActionType.SPAWN,
                    reason="queued_coverage",
                    route_key=_route_key(route.route_key),
                    ordinal=len(actions),
                )
            )
        remaining_budget -= route_spawns

    for _ in range(remaining_budget):
        actions.append(
            CapacityAction(
                action_type=CapacityActionType.SPAWN,
                reason="pool_floor_buffer",
                route_key=POOL_FLOOR_ROUTE_KEY,
                ordinal=len(actions),
            )
        )

    return actions


def build_capacity_plan(
    observed: ObservedPoolState,
    *,
    min_active_gpus: int,
    max_active_gpus: int,
    machines_to_keep_idle: int,
    orchestrator_poll_sec: int,
    clock: Clock | None = None,
    valid_for_seconds: int | None = None,
) -> CapacityPlan:
    """Build one pure, pool-scoped capacity plan.

    Desired capacity composition is exactly:
    ``min(max_active_gpus, max(min_active_gpus, busy + eligible_queued, busy + max(idle_buffer, 1)))``.
    Route backoff only reduces ``eligible_queued``; pool min, max, and buffer are applied once.
    """

    now = _as_utc(observed.observed_at) or (clock or SystemClock()).now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    active_count = observed.active_count
    spawning_count = observed.spawning_count
    idle_count = observed.idle_count
    busy_count = observed.busy_count
    effective_capacity = observed.effective_capacity
    eligible_queued_total = observed.eligible_queued_total
    idle_buffer = max(machines_to_keep_idle, 1)

    queued_coverage = busy_count + eligible_queued_total
    busy_buffer = busy_count + idle_buffer
    desired_unbounded = max(min_active_gpus, queued_coverage, busy_buffer)
    desired = min(max_active_gpus, desired_unbounded)
    desired = max(0, desired)

    stable_since = _stable_since(observed, desired, now)
    valid_until = now + timedelta(seconds=valid_for_seconds) if valid_for_seconds is not None else None
    reason = _desired_reason(
        desired=desired,
        min_active_gpus=min_active_gpus,
        queued_coverage=queued_coverage,
        busy_buffer=busy_buffer,
    )

    actions: list[CapacityAction] = []
    suppressed: list[CapacityAction] = []

    for route in observed.route_demands:
        if not route.spawn_allowed and route.queued > 0:
            suppressed.append(
                CapacityAction(
                    action_type=CapacityActionType.SPAWN,
                    reason="suppressed_by_backoff",
                    route_key=_route_key(route.route_key),
                    ordinal=len(suppressed),
                    metadata={"queued": route.queued, "backoff_until": route.backoff_until.isoformat() if route.backoff_until else None},
                )
            )

    if desired > effective_capacity:
        workers_covering_tasks = idle_count + spawning_count
        actions.extend(
            _allocate_spawn_actions(
                observed.route_demands,
                spawn_budget=desired - effective_capacity,
                workers_covering_tasks=workers_covering_tasks,
            )
        )
    elif effective_capacity > desired:
        excess = effective_capacity - desired

        for worker in _sorted_idle_termination_candidates(observed.workers)[:excess]:
            actions.append(
                CapacityAction(
                    action_type=CapacityActionType.TERMINATE_IDLE,
                    reason="idle_over_desired",
                    route_key=_route_key(worker.route_key),
                    worker_id=worker.worker_id,
                    ordinal=len(actions),
                )
            )

        remaining_excess = excess - sum(1 for action in actions if action.action_type == CapacityActionType.TERMINATE_IDLE)
        if remaining_excess > 0 and _cancel_hysteresis_satisfied(
            observed,
            desired=desired,
            effective=effective_capacity,
            orchestrator_poll_sec=orchestrator_poll_sec,
            now=now,
        ):
            for worker in _sorted_spawn_cancel_candidates(observed.workers)[:remaining_excess]:
                actions.append(
                    CapacityAction(
                        action_type=CapacityActionType.CANCEL_PENDING_SPAWN,
                        reason="cancel_hysteresis_satisfied",
                        route_key=_route_key(worker.route_key),
                        worker_id=worker.worker_id,
                        ordinal=len(actions),
                    )
                )

    if not actions:
        actions.append(
            CapacityAction(
                action_type=CapacityActionType.NOOP,
                reason="desired_equals_effective" if desired == effective_capacity else "no_safe_action",
                ordinal=0,
            )
        )

    return CapacityPlan(
        pool=observed.pool,
        desired_capacity=desired,
        effective_capacity=effective_capacity,
        active_count=active_count,
        spawning_count=spawning_count,
        idle_count=idle_count,
        busy_count=busy_count,
        eligible_queued_total=eligible_queued_total,
        actions=tuple(actions),
        suppressed_actions=tuple(suppressed),
        reason=reason,
        valid_until=valid_until,
        stable_since=stable_since,
        observed_at=now,
        formula_components={
            "min_active_gpus": min_active_gpus,
            "queued_coverage": queued_coverage,
            "busy_buffer": busy_buffer,
            "max_active_gpus": max_active_gpus,
            "eligible_queued_total": eligible_queued_total,
            "idle_buffer": idle_buffer,
        },
    )

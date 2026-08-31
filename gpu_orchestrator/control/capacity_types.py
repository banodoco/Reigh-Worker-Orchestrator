"""Frozen data types for capacity reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol


POOL_FLOOR_ROUTE_KEY = "__pool_floor__"


class Clock(Protocol):
    """Clock protocol used to keep planning deterministic in tests."""

    def now(self) -> datetime: ...


@dataclass(frozen=True)
class SystemClock:
    """Production clock implementation."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RouteDemand:
    """Queued and in-progress task demand for one route key."""

    route_key: str
    queued: int = 0
    in_progress: int = 0
    spawn_allowed: bool = True
    backoff_until: datetime | None = None

    @property
    def eligible_queued(self) -> int:
        return max(0, self.queued) if self.spawn_allowed else 0


@dataclass(frozen=True)
class WorkerCapacitySnapshot:
    """Capacity-relevant view of one worker at observation time."""

    worker_id: str
    route_key: str | None = None
    is_active: bool = False
    is_spawning: bool = False
    is_idle: bool = False
    is_route_stale: bool = False
    should_terminate: bool = False
    excluded_from_capacity_control: bool = False
    created_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def counts_as_effective_capacity(self) -> bool:
        return (
            not self.excluded_from_capacity_control
            and not self.should_terminate
            and not self.is_route_stale
            and (self.is_active or self.is_spawning)
        )

    @property
    def can_terminate_idle(self) -> bool:
        return self.counts_as_effective_capacity and self.is_active and self.is_idle

    @property
    def can_cancel_pending_spawn(self) -> bool:
        return self.counts_as_effective_capacity and self.is_spawning


@dataclass(frozen=True)
class IntentRecord:
    """Previous capacity intent context used for hysteresis decisions."""

    intent_id: str | None = None
    desired_capacity: int = 0
    effective_capacity: int = 0
    created_at: datetime | None = None
    stable_since: datetime | None = None
    valid_until: datetime | None = None
    shadow: bool = True


@dataclass(frozen=True)
class ObservedPoolState:
    """Immutable pool observation passed into pure planning."""

    pool: str
    workers: tuple[WorkerCapacitySnapshot, ...] = ()
    excluded_workers: tuple[Mapping[str, Any], ...] = ()
    route_demands: tuple[RouteDemand, ...] = ()
    previous_intent: IntentRecord | None = None
    observed_at: datetime | None = None
    cycle_id: int | None = None
    route_demand_source: str = "none"
    route_demand_fallback: bool = False

    @property
    def active_count(self) -> int:
        return sum(1 for worker in self.workers if worker.counts_as_effective_capacity and worker.is_active)

    @property
    def spawning_count(self) -> int:
        return sum(1 for worker in self.workers if worker.counts_as_effective_capacity and worker.is_spawning)

    @property
    def idle_count(self) -> int:
        return sum(1 for worker in self.workers if worker.can_terminate_idle)

    @property
    def busy_count(self) -> int:
        return max(0, self.active_count - self.idle_count)

    @property
    def effective_capacity(self) -> int:
        return self.active_count + self.spawning_count

    @property
    def queued_total(self) -> int:
        return sum(max(0, route.queued) for route in self.route_demands)

    @property
    def eligible_queued_total(self) -> int:
        return sum(route.eligible_queued for route in self.route_demands)

    @property
    def in_progress_total(self) -> int:
        return sum(max(0, route.in_progress) for route in self.route_demands)


class CapacityActionType(str, Enum):
    SPAWN = "spawn"
    CANCEL_PENDING_SPAWN = "cancel_pending_spawn"
    TERMINATE_IDLE = "terminate_idle"
    NOOP = "noop"


@dataclass(frozen=True)
class CapacityAction:
    action_type: CapacityActionType
    reason: str
    route_key: str = POOL_FLOOR_ROUTE_KEY
    worker_id: str | None = None
    ordinal: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapacityPlan:
    pool: str
    desired_capacity: int
    effective_capacity: int
    active_count: int
    spawning_count: int
    idle_count: int
    busy_count: int
    eligible_queued_total: int
    actions: tuple[CapacityAction, ...]
    suppressed_actions: tuple[CapacityAction, ...] = ()
    reason: str = "noop"
    valid_until: datetime | None = None
    stable_since: datetime | None = None
    observed_at: datetime | None = None
    formula_components: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmittedCapacityPlan:
    plan: CapacityPlan
    actions: tuple[CapacityAction, ...]
    suppressed_actions: tuple[CapacityAction, ...] = ()


@dataclass(frozen=True)
class ResultCounters:
    spawned: int = 0
    cancelled_pending_spawns: int = 0
    terminated_idle: int = 0
    suppressed_by_backoff: int = 0
    errors: int = 0


@dataclass(frozen=True)
class ReconciliationResult:
    mode: str
    observed: ObservedPoolState
    plan: CapacityPlan
    admitted: AdmittedCapacityPlan
    intent_id: str | None = None
    counters: ResultCounters = field(default_factory=ResultCounters)
    outcome: Mapping[str, Any] = field(default_factory=dict)
    lease_acquired: bool = False
    adopted_worker_ids: tuple[str, ...] = ()

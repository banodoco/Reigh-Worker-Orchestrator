"""Capacity reconciler observation and orchestration entrypoint."""

from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

from gpu_orchestrator.control.capacity_plan import build_capacity_plan
from gpu_orchestrator.control.capacity_types import (
    AdmittedCapacityPlan,
    CapacityAction,
    CapacityActionType,
    CapacityPlan,
    Clock,
    IntentRecord,
    ObservedPoolState,
    ReconciliationResult,
    ResultCounters,
    RouteDemand,
    SystemClock,
    WorkerCapacitySnapshot,
)
from gpu_orchestrator.database import selected_pool_route_filter
from gpu_orchestrator.live_test_workers import partition_capacity_workers
from gpu_orchestrator.worker_state import derive_worker_state

logger = logging.getLogger(__name__)

UNATTRIBUTED_ROUTE_KEY = "__unattributed_route__"


def _as_utc(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _profiles_compatible(worker_profile: str, route_profile: Any) -> bool:
    profile = str(route_profile or "")
    return profile == worker_profile or {profile, worker_profile} == {"default", "1"}


def _route_matches_selected_pool(route: Mapping[str, Any], route_filter: Mapping[str, Any]) -> bool:
    if route.get("selected_backend") != route_filter["worker_backend"]:
        return False
    if route.get("selector_namespace") != route_filter["selector_namespace"]:
        return False
    if str(route.get("selector_version") or "") != str(route_filter.get("selector_version") or ""):
        return False
    if int(route.get("worker_contract_version") or 0) != int(route_filter["worker_contract_version"]):
        return False
    return _profiles_compatible(route_filter["worker_profile"], route.get("selected_profile"))


def _selected_totals(detailed_counts: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not detailed_counts:
        return {}
    return detailed_counts.get("selected_pool_totals") or detailed_counts.get("totals") or {}


def _queued_from_counts(counts: Mapping[str, Any]) -> int:
    if counts.get("potentially_claimable") is not None:
        return int(counts.get("potentially_claimable") or 0)
    return int(counts.get("queued_only") or 0) + int(counts.get("blocked_by_capacity") or 0)


def _active_from_counts(counts: Mapping[str, Any]) -> int:
    return int(counts.get("active_only") or 0)


def _route_key_from_worker_metadata(metadata: Mapping[str, Any]) -> str | None:
    route_contract = metadata.get("route_contract") if isinstance(metadata.get("route_contract"), Mapping) else {}
    return (
        metadata.get("spawn_reason_route_key")
        or metadata.get("route_key")
        or route_contract.get("route_key")
    )


class CapacityReconciler:
    """Pool-scoped capacity reconciler."""

    def __init__(
        self,
        *,
        db: Any,
        config: Any,
        mode: str = "shadow",
        observer_id: str | None = None,
        holder_id: str | None = None,
        lease_ttl_seconds: int | None = None,
        intent_valid_for_seconds: int | None = None,
        action_executor: Any = None,
        clock: Clock | None = None,
        state_deriver: Callable[..., Any] = derive_worker_state,
    ) -> None:
        self.db = db
        self.config = config
        self.mode = mode
        self.observer_id = observer_id or f"capacity-reconciler-{uuid4()}"
        self.holder_id = holder_id or self.observer_id
        poll_sec = int(getattr(config, "orchestrator_poll_sec", 30))
        self.lease_ttl_seconds = lease_ttl_seconds or max(60, poll_sec * 3)
        self.intent_valid_for_seconds = intent_valid_for_seconds
        self.action_executor = action_executor
        self.clock = clock or SystemClock()
        self.state_deriver = state_deriver

    async def reconcile(self, cycle_id: int, *, mode: str | None = None) -> ReconciliationResult:
        """Run observe, plan, admit, record, act, and outcome recording."""

        effective_mode = mode or self.mode
        observed = await self.observe(cycle_id=cycle_id)
        plan = self.plan(observed)
        admitted = self.admit(plan)

        if effective_mode == "authoritative":
            return await self._reconcile_authoritative(cycle_id=cycle_id)

        intent_row = await self.record_pre_act(
            observed=observed,
            plan=plan,
            admitted=admitted,
            shadow=True,
            cycle_id=cycle_id,
            adopted_worker_ids=(),
        )
        counters, outcome = await self.act(
            admitted,
            intent_row=intent_row,
            shadow=True,
            cycle_id=cycle_id,
            adopted_worker_ids=(),
        )
        if intent_row:
            await self.record_outcome(intent_row["id"], outcome=outcome)
        return ReconciliationResult(
            mode=effective_mode,
            observed=observed,
            plan=plan,
            admitted=admitted,
            intent_id=intent_row.get("id") if intent_row else None,
            counters=counters,
            outcome=outcome,
            lease_acquired=False,
        )

    def plan(self, observed: ObservedPoolState) -> CapacityPlan:
        return build_capacity_plan(
            observed,
            min_active_gpus=int(self.config.min_active_gpus),
            max_active_gpus=int(self.config.max_active_gpus),
            machines_to_keep_idle=int(self.config.machines_to_keep_idle),
            orchestrator_poll_sec=int(self.config.orchestrator_poll_sec),
            clock=self.clock,
            valid_for_seconds=self.intent_valid_for_seconds,
        )

    def admit(self, plan: CapacityPlan) -> AdmittedCapacityPlan:
        """Return the admitted plan.

        Backoff suppression is computed from the immutable observation by the
        pure planner. This explicit boundary gives authoritative mode a place
        to re-admit after it holds the pool lease.
        """

        return AdmittedCapacityPlan(
            plan=plan,
            actions=plan.actions,
            suppressed_actions=plan.suppressed_actions,
        )

    async def _reconcile_authoritative(self, *, cycle_id: int) -> ReconciliationResult:
        now = self.clock.now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        lease_acquired = await self.db.acquire_pool_lease(
            pool=self.config.worker_pool,
            holder_id=self.holder_id,
            ttl_seconds=self.lease_ttl_seconds,
            metadata={"cycle_id": cycle_id, "observer_id": self.observer_id},
            now=now,
        )
        if not lease_acquired:
            observed = await self.observe(cycle_id=cycle_id)
            plan = self.plan(observed)
            admitted = self.admit(plan)
            outcome = {"lease_suppressed": True, "holder_id": self.holder_id}
            return ReconciliationResult(
                mode="authoritative",
                observed=observed,
                plan=plan,
                admitted=admitted,
                counters=ResultCounters(errors=1),
                outcome=outcome,
                lease_acquired=False,
            )

        intent_row: Mapping[str, Any] | None = None
        observed = await self.observe(cycle_id=cycle_id)
        plan = self.plan(observed)
        admitted = self.admit(plan)
        adopted_worker_ids = self._legacy_spawning_worker_ids(observed)
        counters = ResultCounters()
        outcome: Mapping[str, Any] = {}
        try:
            intent_row = await self.record_pre_act(
                observed=observed,
                plan=plan,
                admitted=admitted,
                shadow=False,
                cycle_id=cycle_id,
                adopted_worker_ids=adopted_worker_ids,
            )
            if not intent_row:
                outcome = {"record_pre_act_failed": True}
                counters = ResultCounters(errors=1)
            else:
                counters, outcome = await self.act(
                    admitted,
                    intent_row=intent_row,
                    shadow=False,
                    cycle_id=cycle_id,
                    adopted_worker_ids=adopted_worker_ids,
                )
                await self.record_outcome(intent_row["id"], outcome=outcome)
        finally:
            await self.db.release_pool_lease(pool=self.config.worker_pool, holder_id=self.holder_id)

        return ReconciliationResult(
            mode="authoritative",
            observed=observed,
            plan=plan,
            admitted=admitted,
            intent_id=intent_row.get("id") if intent_row else None,
            counters=counters,
            outcome=outcome,
            lease_acquired=True,
            adopted_worker_ids=adopted_worker_ids,
        )

    async def record_pre_act(
        self,
        *,
        observed: ObservedPoolState,
        plan: CapacityPlan,
        admitted: AdmittedCapacityPlan,
        shadow: bool,
        cycle_id: int,
        adopted_worker_ids: tuple[str, ...],
    ) -> Mapping[str, Any] | None:
        return await self.db.insert_worker_capacity_intent(
            pool=observed.pool,
            desired_capacity=plan.desired_capacity,
            reason="adoption" if adopted_worker_ids and not observed.previous_intent else plan.reason,
            observed_queued=observed.queued_total,
            observed_active=observed.active_count,
            observed_spawning=observed.spawning_count,
            observed_idle=observed.idle_count,
            observed_in_progress=observed.in_progress_total,
            observed_route_stale=sum(1 for worker in observed.workers if worker.is_route_stale),
            effective_capacity=plan.effective_capacity,
            cycle_id=cycle_id,
            observer_id=self.observer_id,
            observation_id=str(uuid4()) if shadow else None,
            actions=[self._action_payload(action) for action in admitted.actions],
            suppressed_actions=[self._action_payload(action) for action in admitted.suppressed_actions],
            outcome={
                "phase": "pre_act",
                "adopted_legacy_spawning_worker_ids": list(adopted_worker_ids),
                "route_demand_source": observed.route_demand_source,
                "route_demand_fallback": observed.route_demand_fallback,
            },
            valid_until=plan.valid_until,
            stable_since=plan.stable_since,
            shadow=shadow,
        )

    async def act(
        self,
        admitted: AdmittedCapacityPlan,
        *,
        intent_row: Mapping[str, Any] | None,
        shadow: bool,
        cycle_id: int,
        adopted_worker_ids: tuple[str, ...],
    ) -> tuple[ResultCounters, Mapping[str, Any]]:
        if shadow:
            return await self._act_shadow(admitted, intent_row=intent_row, cycle_id=cycle_id)
        return await self._act_authoritative(
            admitted,
            intent_row=intent_row,
            adopted_worker_ids=adopted_worker_ids,
        )

    async def record_outcome(self, intent_id: str, *, outcome: Mapping[str, Any]) -> bool:
        return await self.db.update_worker_capacity_intent_outcome(intent_id, outcome=dict(outcome))

    async def _act_shadow(
        self,
        admitted: AdmittedCapacityPlan,
        *,
        intent_row: Mapping[str, Any] | None,
        cycle_id: int,
    ) -> tuple[ResultCounters, Mapping[str, Any]]:
        logged = 0
        for action in (*admitted.actions, *admitted.suppressed_actions):
            await self.db.insert_system_log_metadata(
                source_id=self.observer_id,
                message="capacity reconciler shadow intended action",
                metadata={
                    "shadow_intended_action": "true",
                    "pool": admitted.plan.pool,
                    "cycle_id": cycle_id,
                    "intent_id": intent_row.get("id") if intent_row else None,
                    "action": self._action_payload(action),
                },
                cycle_number=cycle_id,
                timestamp=admitted.plan.observed_at,
            )
            logged += 1
        counters = ResultCounters(suppressed_by_backoff=len(admitted.suppressed_actions))
        return counters, {
            "shadow": True,
            "logged_intended_actions": logged,
            "suppressed_by_backoff": len(admitted.suppressed_actions),
        }

    async def _act_authoritative(
        self,
        admitted: AdmittedCapacityPlan,
        *,
        intent_row: Mapping[str, Any] | None,
        adopted_worker_ids: tuple[str, ...],
    ) -> tuple[ResultCounters, Mapping[str, Any]]:
        intent_id = str(intent_row.get("id")) if intent_row else ""
        spawned = 0
        cancelled = 0
        terminated = 0
        errors = 0
        executed: list[dict[str, Any]] = []
        skipped_duplicates: list[dict[str, Any]] = []

        for action in admitted.actions:
            if action.action_type == CapacityActionType.NOOP:
                executed.append({**self._action_payload(action), "result": "noop"})
                continue

            if action.action_type == CapacityActionType.SPAWN:
                existing = await self.db.get_worker_by_capacity_action(intent_id, action.ordinal)
                if existing:
                    skipped_duplicates.append({**self._action_payload(action), "worker_id": existing.get("id")})
                    continue

            success, detail = await self._execute_action(action, intent_row=intent_row)
            executed.append({**self._action_payload(action), "success": success, "detail": detail})

            if action.action_type == CapacityActionType.SPAWN:
                if success:
                    spawned += 1
                    await self.db.reset_route_backoff(admitted.plan.pool, action.route_key, now=admitted.plan.observed_at)
                else:
                    errors += 1
                    await self.db.record_route_spawn_failure(
                        admitted.plan.pool,
                        action.route_key,
                        now=admitted.plan.observed_at,
                        error=str(detail.get("error") or "spawn_failed"),
                    )
            elif action.action_type == CapacityActionType.CANCEL_PENDING_SPAWN:
                if success:
                    cancelled += 1
                else:
                    errors += 1
            elif action.action_type == CapacityActionType.TERMINATE_IDLE:
                if success:
                    terminated += 1
                else:
                    errors += 1

        counters = ResultCounters(
            spawned=spawned,
            cancelled_pending_spawns=cancelled,
            terminated_idle=terminated,
            suppressed_by_backoff=len(admitted.suppressed_actions),
            errors=errors,
        )
        return counters, {
            "shadow": False,
            "executed_actions": executed,
            "skipped_duplicate_spawn_actions": skipped_duplicates,
            "suppressed_by_backoff": len(admitted.suppressed_actions),
            "adopted_legacy_spawning_worker_ids": list(adopted_worker_ids),
            "counters": {
                "spawned": counters.spawned,
                "cancelled_pending_spawns": counters.cancelled_pending_spawns,
                "terminated_idle": counters.terminated_idle,
                "suppressed_by_backoff": counters.suppressed_by_backoff,
                "errors": counters.errors,
            },
        }

    async def _execute_action(
        self,
        action: CapacityAction,
        *,
        intent_row: Mapping[str, Any] | None,
    ) -> tuple[bool, dict[str, Any]]:
        if self.action_executor is None:
            return False, {"error": "no_action_executor"}

        method_name = {
            CapacityActionType.SPAWN: "spawn",
            CapacityActionType.CANCEL_PENDING_SPAWN: "cancel_pending_spawn",
            CapacityActionType.TERMINATE_IDLE: "terminate_idle",
        }.get(action.action_type)
        if method_name is None:
            return True, {}

        method = getattr(self.action_executor, method_name, None)
        if method is None:
            return False, {"error": f"missing_executor_method:{method_name}"}

        try:
            result = method(action=action, intent_row=intent_row, pool=self.config.worker_pool)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # pragma: no cover - exercised by integration tests in later batches
            logger.exception("Capacity action %s failed", action.action_type.value)
            return False, {"error": str(exc)}

        if isinstance(result, Mapping):
            return bool(result.get("success", True)), dict(result)
        return bool(result), {}

    def _legacy_spawning_worker_ids(self, observed: ObservedPoolState) -> tuple[str, ...]:
        if observed.previous_intent is not None:
            return ()
        return tuple(
            worker.worker_id
            for worker in observed.workers
            if worker.is_spawning and not worker.metadata.get("capacity_intent_id")
        )

    @staticmethod
    def _action_payload(action: CapacityAction) -> dict[str, Any]:
        return {
            "type": action.action_type.value,
            "reason": action.reason,
            "route_key": action.route_key,
            "worker_id": action.worker_id,
            "ordinal": action.ordinal,
            "metadata": dict(action.metadata),
        }

    async def observe(self, cycle_id: int | None = None) -> ObservedPoolState:
        """Read workers and tasks once and return an immutable pool observation."""

        now = self.clock.now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        fetched_workers = await self.db.get_workers(status=["spawning", "active", "error"])
        production_workers, excluded_workers = partition_capacity_workers(fetched_workers)
        if excluded_workers:
            logger.info(
                "Ignoring %s worker(s) for capacity reconciliation; cleanup can still inspect them",
                len(excluded_workers),
            )

        detailed_counts = await self.db.get_detailed_task_counts_via_edge_function()
        route_demands, source, fallback = self._route_demands_from_detailed_counts(detailed_counts)
        if not route_demands:
            queued = await self.db.count_available_tasks_via_edge_function(include_active=False)
            total = await self.db.count_available_tasks_via_edge_function(include_active=True)
            active = max(0, total - queued)
            route_demands = (RouteDemand(UNATTRIBUTED_ROUTE_KEY, queued=queued, in_progress=active),)
            source = "fallback_edge_counts"
            fallback = True

        route_demands = await self._apply_route_backoffs(route_demands, now=now)
        queued_for_derivation = sum(max(0, demand.queued) for demand in route_demands)
        worker_states = await self._derive_worker_states(production_workers, queued_for_derivation, now)
        worker_snapshots = tuple(self._worker_snapshot(worker, state) for worker, state in worker_states)
        previous_intent = await self._latest_previous_intent(self.config.worker_pool)

        return ObservedPoolState(
            pool=self.config.worker_pool,
            workers=worker_snapshots,
            excluded_workers=tuple(excluded_workers),
            route_demands=route_demands,
            previous_intent=previous_intent,
            observed_at=now,
            cycle_id=cycle_id,
            route_demand_source=source,
            route_demand_fallback=fallback,
        )

    async def _derive_worker_states(
        self,
        workers: list[dict[str, Any]],
        queued_count: int,
        now: datetime,
    ) -> list[tuple[dict[str, Any], Any]]:
        worker_ids = [worker["id"] for worker in workers]
        if not worker_ids:
            return []
        claimed_map, active_tasks_map = await asyncio.gather(
            self.db.batch_check_ever_claimed(worker_ids),
            self.db.batch_check_active_tasks(worker_ids),
        )
        return [
            (
                worker,
                self.state_deriver(
                    worker=worker,
                    config=self.config,
                    now=now,
                    has_ever_claimed=claimed_map.get(worker["id"], False),
                    has_active_task=active_tasks_map.get(worker["id"], False),
                    queued_count=queued_count,
                ),
            )
            for worker in workers
        ]

    def _worker_snapshot(self, worker: Mapping[str, Any], state: Any) -> WorkerCapacitySnapshot:
        metadata = worker.get("metadata") or {}
        route_key = _route_key_from_worker_metadata(metadata) if isinstance(metadata, Mapping) else None

        return WorkerCapacitySnapshot(
            worker_id=state.worker_id,
            route_key=route_key,
            is_active=bool(state.is_active),
            is_spawning=bool(state.is_spawning),
            is_idle=bool(state.is_active and not state.has_active_task),
            is_route_stale=bool(state.is_route_stale),
            should_terminate=bool(state.should_terminate),
            excluded_from_capacity_control=bool(getattr(state, "excluded_from_capacity_control", False)),
            created_at=getattr(state, "created_at", None),
            metadata=metadata if isinstance(metadata, Mapping) else {},
        )

    def _route_demands_from_detailed_counts(
        self,
        detailed_counts: Mapping[str, Any] | None,
    ) -> tuple[tuple[RouteDemand, ...], str, bool]:
        if not detailed_counts:
            return (), "missing_detailed_counts", True

        route_filter = selected_pool_route_filter()
        exact: list[RouteDemand] = []
        for route in detailed_counts.get("route_totals") or []:
            if not isinstance(route, Mapping):
                continue
            if not _route_matches_selected_pool(route, route_filter):
                continue
            route_key = route.get("route_key")
            if not route_key:
                continue
            exact.append(
                RouteDemand(
                    route_key=str(route_key),
                    queued=_queued_from_counts(route),
                    in_progress=_active_from_counts(route),
                )
            )

        if exact:
            return tuple(exact), "route_totals", False

        totals = _selected_totals(detailed_counts)
        route_keys = [key for key in (totals.get("route_keys") or []) if key]
        queued = _queued_from_counts(totals)
        active = _active_from_counts(totals)
        if len(route_keys) == 1:
            return (RouteDemand(str(route_keys[0]), queued=queued, in_progress=active),), "single_route_key_inferred", False

        if queued > 0 or active > 0:
            return (
                RouteDemand(UNATTRIBUTED_ROUTE_KEY, queued=queued, in_progress=active),
            ), "unattributed_selected_totals", True

        return (), "empty_selected_totals", False

    async def _apply_route_backoffs(
        self,
        route_demands: Iterable[RouteDemand],
        *,
        now: datetime,
    ) -> tuple[RouteDemand, ...]:
        admitted: list[RouteDemand] = []
        for demand in route_demands:
            backoff = await self.db.get_route_backoff(self.config.worker_pool, demand.route_key)
            backoff_until = _as_utc(backoff.get("next_spawn_allowed_at"))
            spawn_allowed = backoff_until is None or backoff_until <= now
            admitted.append(
                RouteDemand(
                    route_key=demand.route_key,
                    queued=demand.queued,
                    in_progress=demand.in_progress,
                    spawn_allowed=spawn_allowed,
                    backoff_until=backoff_until,
                )
            )
        return tuple(admitted)

    async def _latest_previous_intent(self, pool: str) -> IntentRecord | None:
        row = await self.db.get_latest_authoritative_capacity_intent(pool)
        if not row:
            return None
        return IntentRecord(
            intent_id=row.get("id"),
            desired_capacity=int(row.get("desired_capacity") or 0),
            effective_capacity=int(row.get("effective_capacity") or 0),
            created_at=_as_utc(row.get("created_at")),
            stable_since=_as_utc(row.get("stable_since")),
            valid_until=_as_utc(row.get("valid_until")),
            shadow=bool(row.get("shadow", True)),
        )

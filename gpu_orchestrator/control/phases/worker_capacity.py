"""Worker capacity actions: scale execution, idle detection, spawn/terminate, failure-rate guard."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from gpu_orchestrator.worker_state import CycleSummary, DerivedWorkerState, ScalingDecision, TaskCounts

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gpu_orchestrator.control.contracts import ControlHost


class WorkerCapacityPhaseMixin:
    """Execute scaling decisions and manage worker capacity state."""

    async def _execute_scaling(
        self: "ControlHost",
        decision: ScalingDecision,
        worker_states: List[DerivedWorkerState],
        task_counts: TaskCounts,
        summary: CycleSummary,
        detailed_counts: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Spawn or terminate workers based on scaling decision."""
        config = self.config
        await self._handle_stale_route_workers(worker_states, summary)

        if decision.should_scale_down:
            if decision.current_capacity > decision.desired_workers:
                dynamic_timeout = config.overcapacity_idle_timeout_sec
            else:
                dynamic_timeout = config.gpu_idle_timeout_sec

            active_states = [
                ws for ws in worker_states
                if ws.is_active and not ws.has_active_task and not ws.is_route_stale
            ]
            terminatable = []

            for ws in active_states:
                worker = await self.db.get_worker_by_id(ws.worker_id)
                if worker and await self._is_worker_idle_with_timeout(worker, dynamic_timeout):
                    terminatable.append(worker)

            terminatable.sort(key=lambda w: w.get("created_at", ""), reverse=True)

            # The normal policy keeps at least one idle worker, but MAX_ACTIVE_GPUS=0
            # is an explicit emergency/off switch and must override that floor.
            idle_reserve = min(max(config.machines_to_keep_idle, 1), config.max_active_gpus)
            max_by_buffer = max(0, decision.idle_count - idle_reserve)
            max_by_capacity = decision.workers_to_terminate
            max_by_min_active = max(0, decision.active_count - config.min_active_gpus)

            to_terminate = min(len(terminatable), max_by_buffer, max_by_capacity, max_by_min_active)

            logger.info(
                "SCALE_DOWN_EVALUATION: eligible=%s idle_reserve=%s "
                "by_buffer=%s by_capacity=%s by_min=%s selected=%s",
                len(terminatable),
                idle_reserve,
                max_by_buffer,
                max_by_capacity,
                max_by_min_active,
                to_terminate,
            )

            if to_terminate > 0:
                logger.info(
                    f"Terminating {to_terminate} workers "
                    f"(over-capacity: {decision.current_capacity} > {decision.desired_workers})"
                )

                for worker in terminatable[:to_terminate]:
                    reason = (
                        f"scale_down_idle (capacity {decision.current_capacity} > "
                        f"desired {decision.desired_workers})"
                    )
                    if await self._terminate_worker(worker, reason=reason):
                        summary.workers_terminated += 1
                        logger.info(f"Terminated idle worker {worker['id']} (idle > {dynamic_timeout}s)")

        if decision.should_scale_up:
            pool = getattr(config, "worker_pool", None) or "production"
            if await _scale_paused_for_pool(self, pool):
                logger.warning(
                    f"SCALE-UP SUPPRESSED: pause_scaling row active for pool={pool}; "
                    f"skipping spawn of {decision.workers_to_spawn} workers"
                )
                return

            logger.info(
                f"SCALING UP: Spawning {decision.workers_to_spawn} workers "
                f"(current: {decision.current_capacity}, desired: {decision.desired_workers})"
            )

            if detailed_counts:
                queued_tasks = detailed_counts.get("queued_tasks", [])
                active_tasks = detailed_counts.get("active_tasks", [])

                if queued_tasks:
                    logger.debug(f"QUEUED TASKS triggering scale-up ({len(queued_tasks)} tasks):")
                    for i, task in enumerate(queued_tasks[:10], 1):
                        task_id = task.get("task_id", "unknown")[:8]
                        task_type = task.get("task_type", "unknown")
                        user_id = task.get("user_id", "unknown")
                        created_at = task.get("created_at", "unknown")
                        if created_at != "unknown":
                            created_at = created_at[11:19] if len(created_at) > 19 else created_at
                        logger.debug(
                            f"   {i}. {task_id}... | {task_type:25} | "
                            f"user: {user_id} | created: {created_at}"
                        )

                    if len(queued_tasks) > 10:
                        logger.debug(f"   ... and {len(queued_tasks) - 10} more queued tasks")

                if active_tasks:
                    logger.debug(f"ACTIVE TASKS currently being processed ({len(active_tasks)} tasks):")
                    for i, task in enumerate(active_tasks[:5], 1):
                        task_id = task.get("task_id", "unknown")[:8]
                        task_type = task.get("task_type", "unknown")
                        worker_id = task.get("worker_id", "unknown")
                        started_at = task.get("started_at", "unknown")
                        if started_at != "unknown":
                            started_at = started_at[11:19] if len(started_at) > 19 else started_at
                        logger.debug(
                            f"   {i}. {task_id}... | {task_type:25} | "
                            f"worker: {worker_id} | started: {started_at}"
                        )

                    if len(active_tasks) > 5:
                        logger.debug(f"   ... and {len(active_tasks) - 5} more active tasks")

            logger.debug(f"SCALING UP: Creating {decision.workers_to_spawn} new workers")

            for i in range(decision.workers_to_spawn):
                if await self._spawn_worker(queued_count=task_counts.queued):
                    summary.workers_spawned += 1
                    logger.info(f"Worker {i + 1}/{decision.workers_to_spawn} spawned successfully")
                else:
                    logger.error(f"Failed to spawn worker {i + 1}/{decision.workers_to_spawn}")

    async def _is_worker_idle_with_timeout(
        self: "ControlHost",
        worker: Dict[str, Any],
        timeout_seconds: int,
    ) -> bool:
        """Check if a worker is idle and has been idle long enough."""
        has_tasks = await self.db.has_running_tasks(worker["id"])
        if has_tasks:
            return False

        recent_completions = await self._check_recent_task_completions(worker["id"])
        if recent_completions > 0:
            return False

        try:
            result = (
                self.db.supabase.table("tasks")
                .select("generation_processed_at, updated_at")
                .eq("worker_id", worker["id"])
                .eq("status", "Complete")
                .order("generation_processed_at", desc=True)
                .limit(1)
                .execute()
            )

            if result.data:
                last_completion_time = result.data[0].get("generation_processed_at") or result.data[0].get(
                    "updated_at"
                )
                if last_completion_time:
                    last_completion_dt = datetime.fromisoformat(last_completion_time.replace("Z", "+00:00"))
                    if last_completion_dt.tzinfo is None:
                        last_completion_dt = last_completion_dt.replace(tzinfo=timezone.utc)
                    idle_time = (datetime.now(timezone.utc) - last_completion_dt).total_seconds()
                    return idle_time > timeout_seconds

            created_dt = datetime.fromisoformat(worker["created_at"].replace("Z", "+00:00"))
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            worker_age = (datetime.now(timezone.utc) - created_dt).total_seconds()
            return worker_age > timeout_seconds

        except Exception as exc:
            logger.error(f"Error checking idle time for worker {worker['id']}: {exc}")
            return False

    async def _check_recent_task_completions(self: "ControlHost", worker_id: str) -> int:
        """Check how many tasks this worker completed since the last cycle."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.config.orchestrator_poll_sec)
            result = (
                self.db.supabase.table("tasks")
                .select("id", count="exact")
                .eq("worker_id", worker_id)
                .eq("status", "Complete")
                .gte("generation_processed_at", cutoff_time.isoformat())
                .execute()
            )
            return result.count or 0
        except Exception as exc:
            logger.error(f"Error checking recent completions for worker {worker_id}: {exc}")
            return 0

    async def _spawn_worker(self: "ControlHost", queued_count: int = 0) -> bool:
        """Spawn a new worker."""
        try:
            worker_id = self.runpod.generate_worker_id()

            if not await self.db.create_worker_record(worker_id, self.runpod.gpu_type):
                return False

            result = await self.runpod.spawn_worker(worker_id)

            if result and result.get("runpod_id"):
                pod_id = result["runpod_id"]
                status = result.get("status", "spawning")

                if status == "running":
                    final_status = "active"
                elif status == "error":
                    final_status = "terminated"
                else:
                    final_status = "spawning"

                metadata = {"runpod_id": pod_id, **_selected_route_metadata(self)}
                if status == "error":
                    metadata["termination_reason"] = "spawn_failed"
                    metadata["terminated_at"] = datetime.now(timezone.utc).isoformat()
                if "ssh_details" in result:
                    metadata["ssh_details"] = result["ssh_details"]
                if "pod_details" in result:
                    metadata["pod_details"] = result["pod_details"]
                if "ram_tier" in result:
                    metadata["ram_tier"] = result["ram_tier"]
                if "storage_volume" in result:
                    metadata["storage_volume"] = result["storage_volume"]

                update_success = await self.db.update_worker_status(worker_id, final_status, metadata)
                if not update_success:
                    logger.error(
                        f"CRITICAL: Failed to update worker {worker_id} in database "
                        f"but RunPod pod {pod_id} was created!"
                    )
                    try:
                        await self.runpod.terminate_worker(pod_id)
                    except Exception as exc:
                        logger.error(
                            f"MANUAL INTERVENTION REQUIRED: Orphaned pod {pod_id} ({exc})"
                        )
                    return False

                if final_status == "active" and self.config.auto_start_worker_process:
                    await self.runpod.start_worker_process(
                        pod_id,
                        worker_id,
                        has_pending_tasks=(queued_count > 0),
                    )

                return final_status not in ("error", "terminated")

            await self.db.update_worker_status(
                worker_id,
                "terminated",
                {
                    "termination_reason": "spawn_failed",
                    "terminated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return False

        except Exception as exc:
            logger.error(f"Error spawning worker: {exc}")
            return False

    async def _terminate_worker(
        self: "ControlHost",
        worker: Dict[str, Any],
        reason: str = "scale_down",
    ) -> bool:
        """Terminate a worker and update its status."""
        worker_id = worker["id"]
        runpod_id = worker.get("metadata", {}).get("runpod_id")

        logger.info(f"WORKER_LIFECYCLE [Worker {worker_id}] TERMINATING worker (reason: {reason})")

        try:
            if runpod_id:
                success = await self.runpod.terminate_worker(runpod_id)
                if not success:
                    logger.error(f"WORKER_LIFECYCLE [Worker {worker_id}] RunPod termination FAILED")
                    return False

                logger.info(f"WORKER_LIFECYCLE [Worker {worker_id}] RunPod terminated successfully")
            else:
                logger.error(f"WORKER_LIFECYCLE [Worker {worker_id}] NO RUNPOD_ID - cannot terminate pod!")

            termination_metadata = {
                "terminated_at": datetime.now(timezone.utc).isoformat(),
                "termination_reason": reason,
            }
            await self.db.update_worker_status(worker_id, "terminated", termination_metadata)
            return True

        except Exception as exc:
            logger.error(f"WORKER_LIFECYCLE [Worker {worker_id}] TERMINATION FAILED: {exc}")
            return False

    async def _handle_stale_route_workers(
        self: "ControlHost",
        worker_states: List[DerivedWorkerState],
        summary: CycleSummary,
    ) -> None:
        """Drain active route-stale workers and terminate idle route-stale workers."""
        for ws in worker_states:
            if not ws.is_route_stale or ws.is_terminal:
                continue
            if ws.has_active_task:
                logger.info(
                    "Draining route-stale worker %s with active task; no new-route capacity credit",
                    ws.worker_id,
                )
                continue

            worker = await self.db.get_worker_by_id(ws.worker_id)
            if not worker:
                logger.warning("Route-stale worker %s disappeared before termination", ws.worker_id)
                continue

            reason = ws.termination_reason or "STALE_ROUTE_CONTRACT"
            if not reason.startswith("STALE_ROUTE_CONTRACT"):
                reason = f"STALE_ROUTE_CONTRACT: {reason}"
            if await self._terminate_worker(worker, reason=reason):
                summary.workers_terminated += 1
                logger.info("Terminated idle route-stale worker %s", ws.worker_id)

    async def _check_worker_failure_rate(self: "ControlHost") -> bool:
        """Check if the worker failure rate is too high."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=self.config.failure_window_minutes)

            workers = await self.db.get_workers(["spawning", "active", "terminating", "error", "terminated"])

            recent_workers = []
            for worker in workers:
                worker_time = worker.get("updated_at", worker.get("created_at"))
                if worker_time:
                    try:
                        worker_dt = datetime.fromisoformat(worker_time.replace("Z", "+00:00"))
                        if worker_dt > cutoff_time:
                            recent_workers.append(worker)
                    except Exception as exc:
                        logger.debug(
                            "[FAILURE_RATE] skipping worker with unparseable timestamp %s (%s)",
                            worker.get("id"),
                            exc,
                        )

            if len(recent_workers) < self.config.min_workers_for_rate_check:
                return True

            failed_workers = [w for w in recent_workers if w["status"] == "error"]
            terminated_workers = [w for w in recent_workers if w["status"] == "terminated"]

            failure_rate = len(failed_workers) / len(recent_workers)

            logger.info(
                f"[FAILURE_RATE] Recent: {len(recent_workers)}, Errors: {len(failed_workers)}, "
                f"Terminated (scale-down): {len(terminated_workers)}, Failure rate: {failure_rate:.2%}"
            )

            if failure_rate > self.config.max_worker_failure_rate:
                logger.error(
                    f"[FAILURE_RATE] CRITICAL: Rate ({failure_rate:.2%}) exceeds threshold "
                    f"({self.config.max_worker_failure_rate:.2%})"
                )
                return False

            return True

        except Exception as exc:
            logger.error(f"Error checking worker failure rate: {exc}")
            return True


__all__ = ["WorkerCapacityPhaseMixin"]


def _selected_route_metadata(host: "ControlHost") -> dict[str, Any]:
    metadata_fn = getattr(host.runpod, "selected_route_metadata", None)
    if callable(metadata_fn):
        return metadata_fn()
    return {}


async def _scale_paused_for_pool(host: "ControlHost", pool: str) -> bool:
    """Return True iff a public.pause_scaling row for `pool` has `until > now()`.

    Written by the route-contract-sentinel edge function on persistent
    UNCLAIMABLE_WORK / WORKERS_STUCK_INITIALIZING ticks. Soft-fail to False on
    error so a sentinel-side outage never bricks scale-up.
    """
    try:
        result = (
            host.db.supabase.table("pause_scaling")
            .select("until")
            .eq("pool", pool)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return False
        until_raw = rows[0].get("until")
        if not until_raw:
            return False
        until_dt = datetime.fromisoformat(str(until_raw).replace("Z", "+00:00"))
        if until_dt.tzinfo is None:
            until_dt = until_dt.replace(tzinfo=timezone.utc)
        return until_dt > datetime.now(timezone.utc)
    except Exception as exc:
        logger.warning(f"pause_scaling lookup failed for pool={pool}: {exc}")
        return False

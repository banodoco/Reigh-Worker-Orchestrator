"""Periodic maintenance checks (storage, stale workers, zombie reconciliation)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

import runpod
from gpu_orchestrator.control.diagnostics import WorkerDiagnosticsMixin
from gpu_orchestrator.worker_state import CycleSummary, DerivedWorkerState, WorkerLifecycle

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gpu_orchestrator.control.contracts import ControlHost


class PeriodicChecksMixin:
    """Periodic check methods executed every cycle or every N cycles."""

    async def _run_periodic_checks(
        self: "ControlHost",
        worker_states: List[DerivedWorkerState],
        excluded_workers: List[Dict[str, Any]],
        queued_count: int,
        summary: CycleSummary,
    ) -> None:
        """Run checks that only happen every Nth cycle."""
        if not isinstance(self, WorkerDiagnosticsMixin):
            logger.debug("Periodic checks host is missing diagnostics mixin methods")

        periodic_summary: Dict[str, Any] = {
            "actions": {
                "tasks_reset": 0,
                "workers_failed": 0,
                "workers_terminated": 0,
            },
            "storage_health": {},
        }

        active_workers = []
        for ws in worker_states:
            if ws.is_active:
                worker = await self.db.get_worker_by_id(ws.worker_id)
                if worker:
                    active_workers.append(worker)

        await self._check_storage_health(active_workers, periodic_summary)

        # Failsafe applies to ALL workers — including excluded ones — so
        # harness crashes can't leak GPUs.
        excluded_states = await self._derive_all_worker_states(excluded_workers, queued_count) if excluded_workers else []
        await self._failsafe_stale_worker_check(worker_states + excluded_states, periodic_summary)

        summary.tasks_reset += periodic_summary["actions"].get("tasks_reset", 0)
        summary.workers_failed += periodic_summary["actions"].get("workers_failed", 0)
        summary.workers_terminated += periodic_summary["actions"].get("workers_terminated", 0)

    async def _check_storage_health(
        self: "ControlHost",
        active_workers: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> None:
        """Check storage health across all storage volumes."""
        if self.cycle_count - self.last_storage_check_cycle < self.config.storage_check_interval_cycles:
            return

        self.last_storage_check_cycle = self.cycle_count
        logger.info(f"STORAGE_CHECK Starting periodic check (cycle {self.cycle_count})")

        workers_by_storage: Dict[str, List[Dict[str, Any]]] = {}
        for worker in active_workers:
            storage_volume = worker.get("metadata", {}).get("storage_volume")
            if storage_volume:
                if storage_volume not in workers_by_storage:
                    workers_by_storage[storage_volume] = []
                workers_by_storage[storage_volume].append(worker)

        if not workers_by_storage:
            logger.info("STORAGE_CHECK No active workers with storage volume info")
            return

        if "storage_health" not in summary:
            summary["storage_health"] = {}

        for storage_name, storage_workers in workers_by_storage.items():
            try:
                volume_id = self.runpod._get_storage_volume_id(storage_name)
                if not volume_id:
                    continue

                check_worker = None
                for worker in storage_workers:
                    if worker.get("last_heartbeat"):
                        try:
                            heartbeat_time = datetime.fromisoformat(worker["last_heartbeat"].replace("Z", "+00:00"))
                            heartbeat_age = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()
                            if heartbeat_age < 60:
                                runpod_id = worker.get("metadata", {}).get("runpod_id")
                                if runpod_id:
                                    check_worker = worker
                                    break
                        except Exception as exc:
                            logger.debug(
                                "STORAGE_CHECK skipping worker heartbeat parse for %s: %s",
                                worker.get("id"),
                                exc,
                            )
                            continue

                if not check_worker:
                    continue

                runpod_id = check_worker.get("metadata", {}).get("runpod_id")
                health = self.runpod.check_storage_health(
                    storage_name=storage_name,
                    volume_id=volume_id,
                    active_runpod_id=runpod_id,
                    min_free_gb=self.config.storage_min_free_gb,
                    max_percent_used=self.config.storage_max_percent_used,
                )

                summary["storage_health"][storage_name] = health

                if health.get("needs_expansion"):
                    current_size = health.get("total_gb", 100)
                    new_size = current_size + self.config.storage_expansion_increment_gb

                    logger.warning(f"STORAGE_CHECK '{storage_name}' needs expansion: {health.get('message')}")
                    if self.runpod._expand_network_volume(volume_id, new_size):
                        logger.info(f"STORAGE_CHECK Successfully expanded '{storage_name}' to {new_size}GB")
                    else:
                        logger.error(f"STORAGE_CHECK Failed to expand '{storage_name}'!")

            except Exception as exc:
                logger.error(f"STORAGE_CHECK Error checking '{storage_name}': {exc}")

    async def _check_error_worker_cleanup(self: "ControlHost", worker: Dict[str, Any]) -> None:
        """Check workers marked as error and ensure RunPod instances are terminated."""
        worker_id = worker["id"]
        metadata = worker.get("metadata", {})
        runpod_id = metadata.get("runpod_id")
        error_time_str = metadata.get("error_time")

        if error_time_str:
            try:
                error_time = datetime.fromisoformat(error_time_str.replace("Z", "+00:00"))
                error_age = (datetime.now(timezone.utc) - error_time).total_seconds()

                if error_age > self.config.error_cleanup_grace_period_sec:
                    logger.info(
                        f"[ERROR_CLEANUP] Worker {worker_id} past grace period ({error_age:.0f}s), forcing cleanup"
                    )

                    if runpod_id:
                        pod_status = self.runpod.get_pod_status(runpod_id)
                        if pod_status and pod_status.get("desired_status") not in ["TERMINATED", "FAILED"]:
                            try:
                                self.runpod.terminate_worker(runpod_id)
                            except Exception as exc:
                                logger.error(f"Failed to force terminate RunPod {runpod_id}: {exc}")

                    termination_metadata = {"terminated_at": datetime.now(timezone.utc).isoformat()}
                    await self.db.update_worker_status(worker_id, "terminated", termination_metadata)

            except Exception as exc:
                logger.error(f"Error parsing error_time for worker {worker_id}: {exc}")
        else:
            logger.warning(f"[ERROR_CLEANUP] Worker {worker_id} has no error_time, forcing immediate cleanup")
            if runpod_id:
                try:
                    self.runpod.terminate_worker(runpod_id)
                    termination_metadata = {"terminated_at": datetime.now(timezone.utc).isoformat()}
                    await self.db.update_worker_status(worker_id, "terminated", termination_metadata)
                except Exception as exc:
                    logger.error(f"[ERROR_CLEANUP] Failed to cleanup legacy error worker {worker_id}: {exc}")

    async def _failsafe_stale_worker_check(
        self: "ControlHost",
        worker_states: List[DerivedWorkerState],
        summary: Dict[str, Any],
    ) -> None:
        """Failsafe check for workers that should be reaped.

        Catches three failure modes: wall-clock cap exceeded, stuck spawning,
        stale heartbeat. The wall-clock cap fires even with a fresh heartbeat
        so harness pods that poll forever after a crash still get reaped.
        """
        now = datetime.now(timezone.utc)
        stale_workers: List[DerivedWorkerState] = []

        for ws in worker_states:
            if ws.is_terminal:
                continue

            # Hard wall-clock cap — checked BEFORE the recent-heartbeat
            # short-circuit so a healthy-looking but indefinitely-polling
            # worker still gets reaped.
            if ws.max_lifetime_sec is not None and ws.effective_age_sec > ws.max_lifetime_sec:
                stale_workers.append(ws)
                continue

            if ws.in_startup_phase:
                continue

            if ws.should_terminate:
                continue

            # Idle healthy workers are PATH B's responsibility (scaling execution).
            # Excluded workers don't go through PATH B, so the cap above is
            # what bounds their lifetime.
            # Exception: a worker stuck in ACTIVE_INITIALIZING past
            # active_initializing_max_sec is a zombie even with a fresh heartbeat
            # (its launcher subprocess can keep the DB row warm while the main
            # process is dead). Don't short-circuit it.
            stuck_initializing = (
                ws.lifecycle == WorkerLifecycle.ACTIVE_INITIALIZING
                and ws.effective_age_sec > self.config.active_initializing_max_sec
            )
            if (
                ws.is_active
                and not ws.has_active_task
                and ws.heartbeat_is_recent
                and not stuck_initializing
            ):
                continue

            cutoff_sec = self.config.spawning_timeout_sec if ws.is_spawning else self.config.gpu_idle_timeout_sec
            age = ws.heartbeat_age_sec if ws.has_heartbeat else ws.effective_age_sec
            if age is not None and age > cutoff_sec:
                stale_workers.append(ws)

        for ws in stale_workers:
            worker_id = ws.worker_id
            age_sec = ws.heartbeat_age_sec if ws.has_heartbeat else ws.effective_age_sec
            logger.warning(f"FAILSAFE: Worker {worker_id} has stale heartbeat ({age_sec:.0f}s)")

            try:
                worker = await self.db.get_worker_by_id(worker_id)
                if not worker:
                    continue

                runpod_id = ws.runpod_id
                if runpod_id:
                    pod_status = self.runpod.get_pod_status(runpod_id)
                    if pod_status and pod_status.get("desired_status") not in ["TERMINATED", "FAILED"]:
                        self.runpod.terminate_worker(runpod_id)

                reset_count = await self.db.reset_orphaned_tasks([worker_id])
                if reset_count > 0:
                    summary["actions"]["tasks_reset"] = summary["actions"].get("tasks_reset", 0) + reset_count

                metadata_update = {
                    "error_reason": f"FAILSAFE: Stale heartbeat with status {ws.db_status}",
                    "error_time": now.isoformat(),
                }
                await self.db.update_worker_status(worker_id, "terminated", metadata_update)
                summary["actions"]["workers_failed"] = summary["actions"].get("workers_failed", 0) + 1

            except Exception as exc:
                logger.error(f"FAILSAFE: Failed to cleanup stale worker {worker_id}: {exc}")

        await self._efficient_zombie_check(summary)

    async def _efficient_zombie_check(self: "ControlHost", summary: Dict[str, Any]) -> None:
        """Bidirectional zombie detection."""
        if self.cycle_count % 10 != 0:
            return

        try:
            logger.info("Running efficient zombie check (every 10th cycle)")

            runpod.api_key = os.getenv("RUNPOD_API_KEY")

            pods = runpod.get_pods()
            live_pods = [
                pod
                for pod in pods
                if pod.get("desiredStatus") not in ["TERMINATED", "FAILED"]
            ]
            gpu_worker_pods = [
                pod
                for pod in pods
                if pod.get("name", "").startswith("gpu-")
                and pod.get("desiredStatus") not in ["TERMINATED", "FAILED"]
            ]

            if not gpu_worker_pods:
                return

            db_workers = await self.db.get_workers()
            db_worker_names = {w["id"] for w in db_workers if w["status"] != "terminated"}

            for pod in gpu_worker_pods:
                pod_name = pod.get("name")
                runpod_id = pod.get("id")

                if pod_name not in db_worker_names:
                    logger.warning(f"ZOMBIE DETECTED: RunPod pod {pod_name} ({runpod_id}) not in database")
                    try:
                        success = self.runpod.terminate_worker(runpod_id)
                        if success:
                            summary["actions"]["workers_terminated"] = (
                                summary["actions"].get("workers_terminated", 0) + 1
                            )
                    except Exception as exc:
                        logger.error(f"ZOMBIE CLEANUP: Error terminating {runpod_id}: {exc}")

            runpod_pod_ids = {pod.get("id") for pod in live_pods}
            runpod_pod_names = {pod.get("name") for pod in live_pods}

            for worker in db_workers:
                if worker["status"] in ["active", "spawning"]:
                    runpod_id = worker.get("metadata", {}).get("runpod_id")
                    worker_id = worker["id"]

                    if runpod_id and runpod_id not in runpod_pod_ids and worker_id not in runpod_pod_names:
                        logger.warning(
                            f"EXTERNALLY TERMINATED: Worker {worker_id} not found in RunPod"
                        )
                        try:
                            await self._mark_worker_error(
                                worker,
                                "Pod externally terminated - not found in RunPod",
                            )
                            summary["actions"]["workers_failed"] = summary["actions"].get("workers_failed", 0) + 1
                        except Exception as exc:
                            logger.error(
                                f"Error marking externally terminated worker {worker_id}: {exc}"
                            )

        except Exception as exc:
            logger.error(f"Error during efficient zombie check: {exc}")


__all__ = ["PeriodicChecksMixin"]

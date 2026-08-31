"""Authoritative side-effect helpers for capacity reconciliation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping

from gpu_orchestrator.control.capacity_types import CapacityAction, ResultCounters
from gpu_orchestrator.database import normalize_capacity_route_key, selected_pool_route_metadata
from gpu_orchestrator.worker_state import CycleSummary

logger = logging.getLogger(__name__)


def apply_reconciliation_counters_to_summary(
    summary: CycleSummary,
    counters: ResultCounters,
) -> None:
    """Fold reconciler action counters into the existing cycle summary."""

    summary.workers_spawned += counters.spawned
    summary.workers_terminated += counters.cancelled_pending_spawns + counters.terminated_idle
    if counters.errors:
        summary.error = summary.error or f"capacity_reconciler_errors={counters.errors}"


class CapacityActionExecutor:
    """Execute admitted capacity actions against worker rows and RunPod."""

    def __init__(self, *, db: Any, runpod: Any, config: Any, clock: Any | None = None) -> None:
        self.db = db
        self.runpod = runpod
        self.config = config
        self.clock = clock

    async def spawn(
        self,
        *,
        action: CapacityAction,
        intent_row: Mapping[str, Any] | None,
        pool: str,
    ) -> dict[str, Any]:
        """Create the worker row with capacity metadata before launching RunPod."""

        intent_id = _intent_id(intent_row)
        if not intent_id:
            return {"success": False, "error": "missing_intent_id"}

        worker_id = self.runpod.generate_worker_id()
        route_key = normalize_capacity_route_key(action.route_key)
        created = await self.db.create_worker_record(
            worker_id,
            self.runpod.gpu_type,
            capacity_intent_id=intent_id,
            capacity_action_ordinal=action.ordinal,
            spawn_reason_route_key=route_key,
            metadata_update=self._capacity_metadata(action, intent_id, pool),
        )
        if not created:
            return {"success": False, "error": "worker_record_create_failed", "worker_id": worker_id}

        result = await self.runpod.spawn_worker(worker_id)
        if not result or not result.get("runpod_id"):
            await self._mark_terminal(
                worker_id,
                action=action,
                intent_id=intent_id,
                pool=pool,
                reason="spawn_failed",
                extra={"capacity_spawn_result": result or {}},
            )
            return {"success": False, "error": "spawn_failed", "worker_id": worker_id}

        pod_id = result["runpod_id"]
        status = result.get("status", "spawning")
        if status == "running":
            final_status = "active"
        elif status == "error":
            final_status = "terminated"
        else:
            final_status = "spawning"

        metadata = self._capacity_metadata(action, intent_id, pool)
        metadata.update({"runpod_id": pod_id, **self._selected_route_metadata()})
        if final_status == "terminated":
            metadata["termination_reason"] = "spawn_failed"
            metadata["terminated_at"] = self._now_iso()
        for key in ("ssh_details", "pod_details", "ram_tier", "storage_volume", "storage_volume_id"):
            if key in result:
                metadata[key] = result[key]

        updated = await self.db.update_worker_status(worker_id, final_status, metadata)
        if not updated:
            logger.error(
                "CRITICAL: worker %s was launched as RunPod pod %s but database update failed",
                worker_id,
                pod_id,
            )
            try:
                await self.runpod.terminate_worker(pod_id)
            except Exception:
                logger.exception("Manual intervention required for orphaned pod %s", pod_id)
            return {
                "success": False,
                "error": "worker_status_update_failed",
                "worker_id": worker_id,
                "runpod_id": pod_id,
            }

        if final_status == "active" and getattr(self.config, "auto_start_worker_process", False):
            await self.runpod.start_worker_process(pod_id, worker_id, has_pending_tasks=True)

        return {
            "success": final_status != "terminated",
            "worker_id": worker_id,
            "runpod_id": pod_id,
            "status": final_status,
            "route_key": route_key,
        }

    async def cancel_pending_spawn(
        self,
        *,
        action: CapacityAction,
        intent_row: Mapping[str, Any] | None,
        pool: str,
    ) -> dict[str, Any]:
        """Cancel a spawning worker and always try to mark its row terminal."""

        intent_id = _intent_id(intent_row)
        worker_id = action.worker_id
        if not worker_id:
            return {"success": False, "error": "missing_worker_id"}

        worker = await self._get_worker(worker_id)
        runpod_id = (worker.get("metadata") or {}).get("runpod_id") if worker else None
        terminate_success: bool | None = None
        if runpod_id:
            try:
                terminate_success = bool(await self.runpod.terminate_worker(runpod_id))
            except Exception as exc:
                logger.warning("RunPod cancellation failed for pending spawn %s: %s", worker_id, exc)
                terminate_success = False

        metadata = self._capacity_metadata(action, intent_id, pool)
        metadata.update(
            {
                "termination_reason": "reconciler_cancel_pending_spawn",
                "terminated_at": self._now_iso(),
                "runpod_termination_attempted": bool(runpod_id),
                "runpod_termination_success": terminate_success,
            }
        )
        if runpod_id:
            metadata["runpod_id"] = runpod_id

        marked = await self.db.update_worker_status(worker_id, "terminated", metadata)
        return {
            "success": bool(marked),
            "worker_id": worker_id,
            "runpod_id": runpod_id,
            "runpod_termination_success": terminate_success,
            "terminal_marked": bool(marked),
            **({} if marked else {"error": "worker_terminal_mark_failed"}),
        }

    async def terminate_idle(
        self,
        *,
        action: CapacityAction,
        intent_row: Mapping[str, Any] | None,
        pool: str,
    ) -> dict[str, Any]:
        """Terminate an idle active worker for reconciler-directed scale-down."""

        intent_id = _intent_id(intent_row)
        worker_id = action.worker_id
        if not worker_id:
            return {"success": False, "error": "missing_worker_id"}

        worker = await self._get_worker(worker_id)
        if not worker:
            return {"success": False, "error": "worker_not_found", "worker_id": worker_id}

        runpod_id = (worker.get("metadata") or {}).get("runpod_id")
        if runpod_id:
            try:
                terminated = bool(await self.runpod.terminate_worker(runpod_id))
            except Exception as exc:
                logger.warning("RunPod termination failed for idle worker %s: %s", worker_id, exc)
                terminated = False
            if not terminated:
                return {
                    "success": False,
                    "error": "runpod_termination_failed",
                    "worker_id": worker_id,
                    "runpod_id": runpod_id,
                }
        else:
            logger.error("Worker %s has no runpod_id; marking terminated may leave an orphaned pod", worker_id)

        metadata = self._capacity_metadata(action, intent_id, pool)
        metadata.update(
            {
                "termination_reason": "reconciler_terminate_idle",
                "terminated_at": self._now_iso(),
            }
        )
        if runpod_id:
            metadata["runpod_id"] = runpod_id

        marked = await self.db.update_worker_status(worker_id, "terminated", metadata)
        return {
            "success": bool(marked),
            "worker_id": worker_id,
            "runpod_id": runpod_id,
            "terminal_marked": bool(marked),
            **({} if marked else {"error": "worker_terminal_mark_failed"}),
        }

    async def _get_worker(self, worker_id: str) -> Mapping[str, Any] | None:
        get_worker = getattr(self.db, "get_worker", None)
        if get_worker is None:
            return None
        return await get_worker(worker_id)

    async def _mark_terminal(
        self,
        worker_id: str,
        *,
        action: CapacityAction,
        intent_id: str | None,
        pool: str,
        reason: str,
        extra: Mapping[str, Any] | None = None,
    ) -> bool:
        metadata = self._capacity_metadata(action, intent_id, pool)
        metadata.update(
            {
                "termination_reason": reason,
                "terminated_at": self._now_iso(),
            }
        )
        if extra:
            metadata.update(dict(extra))
        return bool(await self.db.update_worker_status(worker_id, "terminated", metadata))

    def _capacity_metadata(
        self,
        action: CapacityAction,
        intent_id: str | None,
        pool: str,
    ) -> dict[str, Any]:
        metadata = {
            "capacity_action_ordinal": action.ordinal,
            "capacity_action_type": action.action_type.value,
            "capacity_action_reason": action.reason,
            "spawn_reason_route_key": normalize_capacity_route_key(action.route_key),
            "capacity_pool": pool,
        }
        if intent_id:
            metadata["capacity_intent_id"] = intent_id
        return metadata

    def _selected_route_metadata(self) -> dict[str, Any]:
        selected = getattr(self.runpod, "selected_route_metadata", None)
        if callable(selected):
            return selected()
        return selected_pool_route_metadata()

    def _now_iso(self) -> str:
        now = self.clock.now() if self.clock is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc).isoformat()


def _intent_id(intent_row: Mapping[str, Any] | None) -> str | None:
    if not intent_row:
        return None
    value = intent_row.get("id")
    return str(value) if value else None

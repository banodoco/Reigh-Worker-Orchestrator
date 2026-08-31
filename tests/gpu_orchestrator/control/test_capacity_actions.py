"""Tests for capacity reconciler side-effect helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from gpu_orchestrator.control.capacity_actions import (
    CapacityActionExecutor,
    apply_reconciliation_counters_to_summary,
)
from gpu_orchestrator.control.capacity_types import CapacityAction, CapacityActionType, ResultCounters
from gpu_orchestrator.worker_state import CycleSummary


NOW = datetime(2026, 5, 13, 14, 11, tzinfo=timezone.utc)


@dataclass(frozen=True)
class FrozenClock:
    def now(self) -> datetime:
        return NOW


class FakeDatabase:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.workers: dict[str, dict[str, Any]] = {}
        self.create_success = True
        self.update_success = True

    async def create_worker_record(self, worker_id: str, instance_type: str, **kwargs: Any) -> bool:
        self.calls.append(("create_worker_record", {"worker_id": worker_id, "instance_type": instance_type, **kwargs}))
        if self.create_success:
            self.workers[worker_id] = {
                "id": worker_id,
                "instance_type": instance_type,
                "status": "inactive",
                "metadata": {
                    "capacity_intent_id": kwargs.get("capacity_intent_id"),
                    "capacity_action_ordinal": kwargs.get("capacity_action_ordinal"),
                    "spawn_reason_route_key": kwargs.get("spawn_reason_route_key"),
                    **(kwargs.get("metadata_update") or {}),
                },
            }
        return self.create_success

    async def update_worker_status(self, worker_id: str, status: str, metadata_update: dict[str, Any]) -> bool:
        self.calls.append(("update_worker_status", {"worker_id": worker_id, "status": status, "metadata_update": metadata_update}))
        if self.update_success:
            worker = self.workers.setdefault(worker_id, {"id": worker_id, "metadata": {}})
            worker["status"] = status
            worker.setdefault("metadata", {}).update(metadata_update)
        return self.update_success

    async def get_worker(self, worker_id: str) -> dict[str, Any] | None:
        self.calls.append(("get_worker", {"worker_id": worker_id}))
        return self.workers.get(worker_id)


class FakeRunPod:
    gpu_type = "RTX 4090"

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.spawn_result: dict[str, Any] | None = {
            "runpod_id": "pod-1",
            "status": "spawning",
            "pod_details": {"desired_status": "RUNNING"},
            "storage_volume_id": "vol-1",
        }
        self.terminate_success = True

    def generate_worker_id(self) -> str:
        self.calls.append(("generate_worker_id", None))
        return "worker-new"

    async def spawn_worker(self, worker_id: str) -> dict[str, Any] | None:
        self.calls.append(("spawn_worker", worker_id))
        return self.spawn_result

    async def terminate_worker(self, runpod_id: str) -> bool:
        self.calls.append(("terminate_worker", runpod_id))
        return self.terminate_success

    async def start_worker_process(self, runpod_id: str, worker_id: str, *, has_pending_tasks: bool) -> bool:
        self.calls.append(("start_worker_process", (runpod_id, worker_id, has_pending_tasks)))
        return True

    def selected_route_metadata(self) -> dict[str, Any]:
        return {"worker_pool": "gpu-main", "route_contract": {"worker_pool": "gpu-main"}}


def _run(value: Any) -> Any:
    return asyncio.run(value)


def _executor(db: FakeDatabase, runpod: FakeRunPod) -> CapacityActionExecutor:
    return CapacityActionExecutor(
        db=db,
        runpod=runpod,
        config=SimpleNamespace(auto_start_worker_process=False),
        clock=FrozenClock(),
    )


def test_spawn_creates_capacity_tagged_worker_before_runpod_launch() -> None:
    db = FakeDatabase()
    runpod = FakeRunPod()
    action = CapacityAction(CapacityActionType.SPAWN, reason="queued_coverage", route_key="route-a", ordinal=3)

    result = _run(_executor(db, runpod).spawn(action=action, intent_row={"id": "intent-1"}, pool="gpu-main"))

    assert result["success"] is True
    assert db.calls[0][0] == "create_worker_record"
    assert runpod.calls[0] == ("generate_worker_id", None)
    assert runpod.calls[1] == ("spawn_worker", "worker-new")
    create_payload = db.calls[0][1]
    assert create_payload["capacity_intent_id"] == "intent-1"
    assert create_payload["capacity_action_ordinal"] == 3
    assert create_payload["spawn_reason_route_key"] == "route-a"
    assert db.workers["worker-new"]["metadata"]["capacity_intent_id"] == "intent-1"
    assert db.workers["worker-new"]["metadata"]["runpod_id"] == "pod-1"


def test_spawn_failure_marks_worker_terminal_with_capacity_metadata() -> None:
    db = FakeDatabase()
    runpod = FakeRunPod()
    runpod.spawn_result = None
    action = CapacityAction(CapacityActionType.SPAWN, reason="pool_floor_buffer", ordinal=0)

    result = _run(_executor(db, runpod).spawn(action=action, intent_row={"id": "intent-2"}, pool="gpu-main"))

    assert result["success"] is False
    assert db.workers["worker-new"]["status"] == "terminated"
    metadata = db.workers["worker-new"]["metadata"]
    assert metadata["capacity_intent_id"] == "intent-2"
    assert metadata["capacity_action_ordinal"] == 0
    assert metadata["termination_reason"] == "spawn_failed"
    assert metadata["spawn_reason_route_key"] == "__pool_floor__"


def test_cancel_pending_spawn_marks_terminal_even_when_runpod_termination_fails() -> None:
    db = FakeDatabase()
    db.workers["worker-spawning"] = {
        "id": "worker-spawning",
        "status": "spawning",
        "metadata": {"runpod_id": "pod-spawning"},
    }
    runpod = FakeRunPod()
    runpod.terminate_success = False
    action = CapacityAction(
        CapacityActionType.CANCEL_PENDING_SPAWN,
        reason="hysteresis_elapsed",
        worker_id="worker-spawning",
        route_key="route-a",
        ordinal=4,
    )

    result = _run(_executor(db, runpod).cancel_pending_spawn(action=action, intent_row={"id": "intent-3"}, pool="gpu-main"))

    assert result["success"] is True
    assert result["runpod_termination_success"] is False
    assert db.workers["worker-spawning"]["status"] == "terminated"
    metadata = db.workers["worker-spawning"]["metadata"]
    assert metadata["termination_reason"] == "reconciler_cancel_pending_spawn"
    assert metadata["capacity_intent_id"] == "intent-3"
    assert metadata["capacity_action_ordinal"] == 4
    assert metadata["runpod_termination_attempted"] is True
    assert metadata["runpod_termination_success"] is False


def test_terminate_idle_requires_runpod_success_before_terminal_mark() -> None:
    db = FakeDatabase()
    db.workers["idle-worker"] = {
        "id": "idle-worker",
        "status": "active",
        "metadata": {"runpod_id": "pod-idle"},
    }
    runpod = FakeRunPod()
    runpod.terminate_success = False
    action = CapacityAction(
        CapacityActionType.TERMINATE_IDLE,
        reason="over_capacity_idle",
        worker_id="idle-worker",
        ordinal=5,
    )

    result = _run(_executor(db, runpod).terminate_idle(action=action, intent_row={"id": "intent-4"}, pool="gpu-main"))

    assert result["success"] is False
    assert result["error"] == "runpod_termination_failed"
    assert db.workers["idle-worker"]["status"] == "active"
    assert [call[0] for call in db.calls] == ["get_worker"]


def test_reconciler_counters_fold_into_cycle_summary() -> None:
    summary = CycleSummary(workers_spawned=1, workers_terminated=2)

    apply_reconciliation_counters_to_summary(
        summary,
        ResultCounters(spawned=3, cancelled_pending_spawns=1, terminated_idle=2, errors=1),
    )

    assert summary.workers_spawned == 4
    assert summary.workers_terminated == 5
    assert summary.error == "capacity_reconciler_errors=1"

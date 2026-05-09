import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from gpu_orchestrator.control.phases import periodic
from tests.scaling_decision_helpers import make_config, make_worker_state
from gpu_orchestrator.worker_state import WorkerLifecycle


def test_periodic_phase_mixin_exists():
    assert hasattr(periodic, "PeriodicChecksMixin")


def test_failsafe_skips_startup_phase_workers():
    startup_state = replace(
        make_worker_state(
            worker_id="worker-startup",
            lifecycle=WorkerLifecycle.SPAWNING_SCRIPT_RUNNING,
            in_startup_phase=True,
        ),
        db_status="active",
        effective_age_sec=9999,
        heartbeat_age_sec=None,
        has_heartbeat=False,
        heartbeat_is_recent=False,
        ready_for_tasks=False,
        vram_total_mb=None,
        vram_used_mb=None,
        vram_reported=False,
    )

    class Host(periodic.PeriodicChecksMixin):
        def __init__(self):
            self.config = make_config(spawning_timeout_sec=60, gpu_idle_timeout_sec=60)
            self.db = SimpleNamespace(
                get_worker_by_id=AsyncMock(return_value=None),
                reset_orphaned_tasks=AsyncMock(return_value=0),
                get_workers=AsyncMock(return_value=[]),
            )
            self.runpod = Mock()
            self.cycle_count = 1
            self.last_storage_check_cycle = 0

    host = Host()
    summary = {"actions": {"tasks_reset": 0, "workers_failed": 0, "workers_terminated": 0}}

    asyncio.run(host._failsafe_stale_worker_check([startup_state], summary))

    host.db.get_worker_by_id.assert_not_awaited()
    host.db.reset_orphaned_tasks.assert_not_awaited()
    host.runpod.get_pod_status.assert_not_called()
    host.runpod.terminate_worker.assert_not_called()
    assert summary["actions"] == {"tasks_reset": 0, "workers_failed": 0, "workers_terminated": 0}


def test_failsafe_reaps_excluded_worker_past_max_lifetime():
    """A live-test worker with a fresh heartbeat but past its wall-clock cap
    must be reaped. This is the regression test for the leaked-pod incident:
    harness pods polled forever after a crash because heartbeats stayed
    current and nothing else bounded their lifetime.
    """
    excluded_state = replace(
        make_worker_state(
            worker_id="worker-livetest",
            lifecycle=WorkerLifecycle.ACTIVE_READY,
            excluded_from_capacity_control=True,
            max_lifetime_sec=7200,
        ),
        effective_age_sec=10_000,  # past the 7200s cap
        heartbeat_age_sec=5,        # heartbeat is fresh
        heartbeat_is_recent=True,
        has_heartbeat=True,
    )

    captured_worker = {
        "id": "worker-livetest",
        "status": "active",
        "metadata": {"runpod_id": "runpod-worker-livetest", "live_test_variant": "fresh"},
    }

    class Host(periodic.PeriodicChecksMixin):
        def __init__(self):
            self.config = make_config()
            self.db = SimpleNamespace(
                get_worker_by_id=AsyncMock(return_value=captured_worker),
                reset_orphaned_tasks=AsyncMock(return_value=0),
                update_worker_status=AsyncMock(return_value=True),
                get_workers=AsyncMock(return_value=[]),
            )
            runpod = Mock()
            runpod.get_pod_status = Mock(return_value={"desired_status": "RUNNING"})
            runpod.terminate_worker = Mock(return_value=True)
            self.runpod = runpod
            self.cycle_count = 1
            self.last_storage_check_cycle = 0

    host = Host()
    summary = {"actions": {"tasks_reset": 0, "workers_failed": 0, "workers_terminated": 0}}

    asyncio.run(host._failsafe_stale_worker_check([excluded_state], summary))

    host.runpod.terminate_worker.assert_called_once_with("runpod-worker-livetest")
    host.db.update_worker_status.assert_awaited_once()
    args, _ = host.db.update_worker_status.await_args
    assert args[0] == "worker-livetest"
    assert args[1] == "terminated"
    assert summary["actions"]["workers_failed"] == 1


def test_zombie_check_does_not_mark_live_test_pod_missing_when_non_gpu_named_pod_exists(monkeypatch):
    live_test_worker = {
        "id": "pod-live",
        "status": "active",
        "metadata": {"runpod_id": "pod-live", "live_test_variant": "fresh"},
    }
    production_worker = {
        "id": "gpu-prod",
        "status": "active",
        "metadata": {"runpod_id": "pod-prod"},
    }

    class Host(periodic.PeriodicChecksMixin):
        def __init__(self):
            self.db = SimpleNamespace(
                get_workers=AsyncMock(return_value=[live_test_worker, production_worker]),
            )
            self.runpod = Mock()
            self.runpod.terminate_worker = Mock(return_value=True)
            self._mark_worker_error = AsyncMock()
            self.cycle_count = 10

    monkeypatch.setattr(
        periodic.runpod,
        "get_pods",
        lambda: [
            {
                "id": "pod-prod",
                "name": "gpu-prod",
                "desiredStatus": "RUNNING",
            },
            {
                "id": "pod-live",
                "name": "reigh-live-test-fresh-20260509t182954z",
                "desiredStatus": "RUNNING",
            }
        ],
    )

    host = Host()
    summary = {"actions": {"workers_failed": 0, "workers_terminated": 0}}

    asyncio.run(host._efficient_zombie_check(summary))

    host._mark_worker_error.assert_not_awaited()
    host.runpod.terminate_worker.assert_not_called()
    assert summary["actions"] == {"workers_failed": 0, "workers_terminated": 0}

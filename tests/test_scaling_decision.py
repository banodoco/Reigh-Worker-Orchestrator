"""
Unit tests for scaling decision logic.

Tests the pure calculate_scaling_decision_pure() function to verify:
- Reactive scaling: Cover queued tasks with idle + spawning workers
- Proactive scaling: Maintain floor/minimum workers
- Double-spawn prevention
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from gpu_orchestrator.worker_state import (
    calculate_scaling_decision_pure,
    DerivedWorkerState,
    WorkerLifecycle,
    TaskCounts,
    ScalingDecision,
)
from gpu_orchestrator.config import OrchestratorConfig


def make_config(
    min_active_gpus: int = 0,
    max_active_gpus: int = 10,
    machines_to_keep_idle: int = 1,
    scale_up_multiplier: float = 1.0,
) -> OrchestratorConfig:
    """Create a config for testing."""
    return OrchestratorConfig(
        min_active_gpus=min_active_gpus,
        max_active_gpus=max_active_gpus,
        machines_to_keep_idle=machines_to_keep_idle,
        tasks_per_gpu_threshold=3,
        scale_up_multiplier=scale_up_multiplier,
        scale_down_multiplier=0.9,
        min_scaling_interval_sec=45,
        spawning_grace_period_sec=180,
        scale_down_grace_period_sec=60,
        spawning_timeout_sec=600,
        gpu_idle_timeout_sec=600,
        overcapacity_idle_timeout_sec=30,
        task_stuck_timeout_sec=1200,
        graceful_shutdown_timeout_sec=600,
        startup_grace_period_sec=600,
        ready_not_claiming_timeout_sec=180,
        gpu_not_detected_timeout_sec=300,
        heartbeat_promotion_threshold_sec=90,
        max_worker_failure_rate=0.8,
        failure_window_minutes=5,
        min_workers_for_rate_check=5,
        storage_check_interval_cycles=10,
        storage_min_free_gb=50,
        storage_max_percent_used=85,
        storage_expansion_increment_gb=50,
        error_cleanup_grace_period_sec=600,
        orchestrator_poll_sec=30,
        auto_start_worker_process=True,
        use_new_scaling_logic=True,
    )


def make_worker_state(
    worker_id: str = "worker-1",
    lifecycle: WorkerLifecycle = WorkerLifecycle.ACTIVE_READY,
    has_active_task: bool = False,
    should_terminate: bool = False,
) -> DerivedWorkerState:
    """Create a mock worker state for testing."""
    now = datetime.now(timezone.utc)
    return DerivedWorkerState(
        worker_id=worker_id,
        runpod_id=f"runpod-{worker_id}",
        lifecycle=lifecycle,
        db_status="active" if lifecycle.value.startswith("active") else "spawning",
        created_at=now - timedelta(minutes=5),
        script_launched_at=now - timedelta(minutes=4),
        effective_age_sec=240,
        heartbeat_age_sec=10 if lifecycle.value.startswith("active") else None,
        has_heartbeat=lifecycle.value.startswith("active"),
        heartbeat_is_recent=lifecycle.value.startswith("active"),
        vram_total_mb=24000 if lifecycle == WorkerLifecycle.ACTIVE_READY else None,
        vram_used_mb=1000 if lifecycle == WorkerLifecycle.ACTIVE_READY else None,
        vram_reported=lifecycle == WorkerLifecycle.ACTIVE_READY,
        ready_for_tasks=lifecycle == WorkerLifecycle.ACTIVE_READY,
        has_ever_claimed_task=has_active_task,
        has_active_task=has_active_task,
        is_gpu_broken=False,
        is_not_claiming=False,
        is_stale=False,
        should_promote_to_active=False,
        should_terminate=should_terminate,
        termination_reason=None,
        error_code=None,
    )


class TestDoubleSpawnPrevention:
    """Tests for the core double-spawn fix."""

    def test_one_queued_one_spawning_no_new_spawn(self):
        """
        CRITICAL: 1 queued task + 1 spawning worker = 0 new spawns.
        This is the exact scenario that caused the double-spawn bug.
        """
        config = make_config(min_active_gpus=0, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("spawning-1", WorkerLifecycle.SPAWNING_SCRIPT_RUNNING),
        ]
        task_counts = TaskCounts(queued=1, active_cloud=0, total=1)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        assert decision.workers_to_spawn == 0, (
            f"Should not spawn any workers when spawning worker can cover queued task. "
            f"Got spawn_for_tasks={decision.spawn_reason_tasks}, spawn_for_floor={decision.spawn_reason_floor}"
        )
        assert decision.uncovered_tasks == 0
        assert decision.workers_covering_tasks == 1  # spawning worker

    def test_two_queued_one_spawning_spawns_one(self):
        """2 queued tasks + 1 spawning worker = 1 new spawn."""
        config = make_config(min_active_gpus=0, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("spawning-1", WorkerLifecycle.SPAWNING_SCRIPT_RUNNING),
        ]
        task_counts = TaskCounts(queued=2, active_cloud=0, total=2)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        assert decision.workers_to_spawn == 1
        assert decision.spawn_reason_tasks == 1
        assert decision.uncovered_tasks == 1
        assert decision.workers_covering_tasks == 1


class TestTerminatingWorkersExcluded:
    """Tests for excluding workers marked for termination."""

    def test_terminating_spawning_worker_does_not_cover_task(self):
        """
        A spawning worker marked for termination (e.g., timeout) should not
        count as covering a queued task.
        """
        config = make_config(min_active_gpus=0, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state(
                "spawning-timeout",
                WorkerLifecycle.SPAWNING_SCRIPT_RUNNING,
                should_terminate=True,  # Timed out, about to be killed
            ),
        ]
        task_counts = TaskCounts(queued=1, active_cloud=0, total=1)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        # The terminating spawning worker should NOT cover the task
        assert decision.workers_to_spawn == 1
        assert decision.workers_covering_tasks == 0
        assert decision.spawning_count == 0  # Excluded from count


class TestIdleWorkerCoversTask:
    """Tests for idle workers covering queued tasks."""

    def test_idle_worker_covers_queued_task(self):
        """1 queued task + 1 idle worker = 0 new spawns."""
        config = make_config(min_active_gpus=0, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("idle-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
        ]
        task_counts = TaskCounts(queued=1, active_cloud=0, total=1)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        assert decision.workers_to_spawn == 0
        assert decision.uncovered_tasks == 0
        assert decision.workers_covering_tasks == 1

    def test_busy_worker_does_not_cover_queued_task(self):
        """1 queued task + 1 busy worker = 1 new spawn."""
        config = make_config(min_active_gpus=0, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("busy-1", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
        ]
        task_counts = TaskCounts(queued=1, active_cloud=0, total=1)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        assert decision.workers_to_spawn == 1
        assert decision.spawn_reason_tasks == 1
        assert decision.uncovered_tasks == 1
        assert decision.workers_covering_tasks == 0  # busy worker doesn't count

    def test_mixed_idle_spawning_cover_tasks(self):
        """3 queued + 1 idle + 1 spawning = 1 new spawn."""
        config = make_config(min_active_gpus=0, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("idle-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
            make_worker_state("spawning-1", WorkerLifecycle.SPAWNING_SCRIPT_RUNNING),
        ]
        task_counts = TaskCounts(queued=3, active_cloud=0, total=3)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        assert decision.workers_to_spawn == 1
        assert decision.uncovered_tasks == 1
        assert decision.workers_covering_tasks == 2


class TestBufferMaintenance:
    """Tests for proactive buffer/floor maintenance."""

    def test_buffer_spawn_when_no_spawning_in_progress(self):
        """Spawn for buffer when no workers are currently spawning."""
        config = make_config(min_active_gpus=0, machines_to_keep_idle=1)
        worker_states = [
            make_worker_state("busy-1", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
        ]
        task_counts = TaskCounts(queued=0, active_cloud=1, total=1)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        # Should spawn 1 for buffer (busy_count=1 + machines_to_keep_idle=1 = 2, current=1)
        assert decision.workers_to_spawn == 1
        assert decision.spawn_reason_floor == 1
        assert decision.spawn_reason_tasks == 0

    def test_buffer_suppressed_when_spawning_in_progress(self):
        """Don't spawn for buffer when workers are already spawning."""
        config = make_config(min_active_gpus=0, machines_to_keep_idle=1)
        worker_states = [
            make_worker_state("busy-1", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
            make_worker_state("spawning-1", WorkerLifecycle.SPAWNING_SCRIPT_RUNNING),
        ]
        task_counts = TaskCounts(queued=0, active_cloud=1, total=1)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        # No spawn for buffer because we're already spawning
        assert decision.workers_to_spawn == 0
        assert decision.spawn_reason_floor == 0

    def test_buffer_suppressed_when_spawning_for_tasks(self):
        """
        Don't double-spawn for buffer when we're already spawning for tasks.
        1 busy + 2 queued + buffer=1 should spawn 2 (for tasks), not 3.
        """
        config = make_config(min_active_gpus=0, machines_to_keep_idle=1)
        worker_states = [
            make_worker_state("busy-1", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
        ]
        task_counts = TaskCounts(queued=2, active_cloud=1, total=3)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        # spawn_for_tasks = 2 (to cover queued tasks)
        # desired_floor = 1 busy + 1 buffer = 2
        # capacity_after_task_spawns = 1 + 2 = 3
        # spawn_for_floor = max(0, 2 - 3) = 0
        # Total: 2 spawns for tasks, 0 for floor
        assert decision.workers_to_spawn == 2
        assert decision.spawn_reason_tasks == 2
        assert decision.spawn_reason_floor == 0


class TestMinimumWorkers:
    """Tests for minimum worker enforcement."""

    def test_minimum_always_spawned(self):
        """Always spawn to reach minimum, even if spawning in progress."""
        config = make_config(min_active_gpus=3, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("spawning-1", WorkerLifecycle.SPAWNING_SCRIPT_RUNNING),
        ]
        task_counts = TaskCounts(queued=0, active_cloud=0, total=0)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        # Need 2 more to reach minimum of 3 (have 1 spawning)
        assert decision.workers_to_spawn == 2
        assert decision.spawn_reason_floor == 2

    def test_task_spawns_count_toward_minimum(self):
        """
        Workers spawned for tasks count toward minimum.
        1 queued task + min=2 should spawn 2 total, not 3.
        """
        config = make_config(min_active_gpus=2, machines_to_keep_idle=0)
        worker_states = []  # No workers at all
        task_counts = TaskCounts(queued=1, active_cloud=0, total=1)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        # 1 for the task + 1 for minimum = 2 total (not 1 + 2 = 3)
        assert decision.workers_to_spawn == 2, (
            f"Expected 2 total spawns (1 task + 1 floor), got {decision.workers_to_spawn} "
            f"(tasks={decision.spawn_reason_tasks}, floor={decision.spawn_reason_floor})"
        )
        assert decision.spawn_reason_tasks == 1
        assert decision.spawn_reason_floor == 1

    def test_minimum_with_active_workers(self):
        """Minimum respected with active workers."""
        config = make_config(min_active_gpus=2, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("active-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
        ]
        task_counts = TaskCounts(queued=0, active_cloud=0, total=0)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        # Need 1 more to reach minimum of 2
        assert decision.workers_to_spawn == 1
        assert decision.spawn_reason_floor == 1

    def test_at_minimum_no_spawn(self):
        """Don't spawn if already at minimum."""
        config = make_config(min_active_gpus=2, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("active-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
            make_worker_state("active-2", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
        ]
        task_counts = TaskCounts(queued=0, active_cloud=0, total=0)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        assert decision.workers_to_spawn == 0


class TestMaxCapacity:
    """Tests for max capacity enforcement."""

    def test_max_capacity_limits_spawning(self):
        """Don't spawn beyond max capacity."""
        config = make_config(min_active_gpus=0, max_active_gpus=2, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("active-1", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
            make_worker_state("active-2", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
        ]
        task_counts = TaskCounts(queued=5, active_cloud=2, total=7)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        assert decision.workers_to_spawn == 0
        assert decision.at_max_capacity is True


class TestFailureRate:
    """Tests for failure rate blocking."""

    def test_failure_rate_blocks_spawning(self):
        """Don't spawn when failure rate is too high."""
        config = make_config(min_active_gpus=0, machines_to_keep_idle=0)
        worker_states = []
        task_counts = TaskCounts(queued=5, active_cloud=0, total=5)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=False
        )

        assert decision.workers_to_spawn == 0
        assert decision.failure_rate_ok is False


class TestScaleDown:
    """Tests for scale-down decisions."""

    def test_no_scale_down_with_active_tasks(self):
        """Don't recommend scale-down for workers with active tasks."""
        config = make_config(min_active_gpus=0, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("busy-1", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
            make_worker_state("idle-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
        ]
        task_counts = TaskCounts(queued=0, active_cloud=1, total=1)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        # Should indicate 1 worker to terminate (the idle one)
        assert decision.workers_to_terminate == 1

    def test_no_scale_down_below_minimum(self):
        """Don't scale down below minimum workers."""
        config = make_config(min_active_gpus=2, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("idle-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
            make_worker_state("idle-2", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
        ]
        task_counts = TaskCounts(queued=0, active_cloud=0, total=0)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        assert decision.workers_to_terminate == 0


class TestObservabilityFields:
    """Tests for new observability fields."""

    def test_all_observability_fields_populated(self):
        """Verify all new fields are set correctly."""
        config = make_config(min_active_gpus=1, machines_to_keep_idle=1)
        worker_states = [
            make_worker_state("busy-1", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
            make_worker_state("idle-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
            make_worker_state("spawning-1", WorkerLifecycle.SPAWNING_SCRIPT_RUNNING),
        ]
        task_counts = TaskCounts(queued=2, active_cloud=1, total=3)

        decision = calculate_scaling_decision_pure(
            worker_states, task_counts, config, failure_rate_ok=True
        )

        # Verify observability fields
        assert decision.workers_covering_tasks == 2  # 1 idle + 1 spawning
        assert decision.uncovered_tasks == 0  # 2 queued - 2 covering = 0
        assert hasattr(decision, 'spawn_reason_tasks')
        assert hasattr(decision, 'spawn_reason_floor')

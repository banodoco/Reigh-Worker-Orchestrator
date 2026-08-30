"""Scaling decision tests for limits, failures, and observability."""

from __future__ import annotations

from gpu_orchestrator.worker_state import TaskCounts, WorkerLifecycle, calculate_scaling_decision_pure
from tests.scaling_decision_helpers import make_config, make_worker_state


class TestMinimumWorkers:
    """Tests for minimum worker enforcement."""

    def test_minimum_always_spawned(self):
        config = make_config(min_active_gpus=3, machines_to_keep_idle=0)
        worker_states = [make_worker_state("spawning-1", WorkerLifecycle.SPAWNING_SCRIPT_RUNNING)]
        task_counts = TaskCounts(queued=0, active_cloud=0, total=0)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=True)

        assert decision.workers_to_spawn == 2
        assert decision.spawn_reason_floor == 2

    def test_task_spawns_count_toward_minimum(self):
        config = make_config(min_active_gpus=2, machines_to_keep_idle=0)
        worker_states = []
        task_counts = TaskCounts(queued=1, active_cloud=0, total=1)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=True)

        assert decision.workers_to_spawn == 2
        assert decision.spawn_reason_tasks == 1
        assert decision.spawn_reason_floor == 1

    def test_minimum_with_active_workers(self):
        config = make_config(min_active_gpus=2, machines_to_keep_idle=0)
        worker_states = [make_worker_state("active-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False)]
        task_counts = TaskCounts(queued=0, active_cloud=0, total=0)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=True)

        assert decision.workers_to_spawn == 1
        assert decision.spawn_reason_floor == 1

    def test_at_minimum_no_spawn(self):
        config = make_config(min_active_gpus=2, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("active-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
            make_worker_state("active-2", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
        ]
        task_counts = TaskCounts(queued=0, active_cloud=0, total=0)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=True)

        assert decision.workers_to_spawn == 0


class TestMaxCapacity:
    """Tests for max capacity enforcement."""

    def test_zero_max_is_a_hard_off_switch(self):
        config = make_config(min_active_gpus=0, max_active_gpus=0, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("active-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
        ]
        task_counts = TaskCounts(queued=5, active_cloud=0, total=5)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=True)

        assert decision.desired_workers == 0
        assert decision.workers_to_spawn == 0
        assert decision.workers_to_terminate == 1

    def test_max_capacity_limits_spawning(self):
        config = make_config(min_active_gpus=0, max_active_gpus=2, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("active-1", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
            make_worker_state("active-2", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
        ]
        task_counts = TaskCounts(queued=5, active_cloud=2, total=7)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=True)

        assert decision.workers_to_spawn == 0
        assert decision.at_max_capacity is True

    def test_route_stale_worker_does_not_consume_new_route_max_capacity(self):
        config = make_config(min_active_gpus=0, max_active_gpus=1, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state(
                "stale-active",
                WorkerLifecycle.ACTIVE_READY,
                has_active_task=False,
                is_route_stale=True,
            ),
        ]
        task_counts = TaskCounts(queued=1, active_cloud=0, total=1)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=True)

        assert decision.current_capacity == 0
        assert decision.at_max_capacity is False
        assert decision.workers_to_spawn == 1


class TestFailureRate:
    """Tests for failure rate blocking."""

    def test_failure_rate_blocks_spawning(self):
        config = make_config(min_active_gpus=0, machines_to_keep_idle=0)
        worker_states = []
        task_counts = TaskCounts(queued=5, active_cloud=0, total=5)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=False)

        assert decision.workers_to_spawn == 0
        assert decision.failure_rate_ok is False


class TestScaleDown:
    """Tests for scale-down decisions."""

    def test_no_scale_down_with_active_tasks(self):
        config = make_config(min_active_gpus=0, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("busy-1", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
            make_worker_state("idle-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
        ]
        task_counts = TaskCounts(queued=0, active_cloud=1, total=1)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=True)

        assert decision.buffer_based_workers == 2
        assert decision.workers_to_terminate == 0

    def test_no_scale_down_below_minimum(self):
        config = make_config(min_active_gpus=2, machines_to_keep_idle=0)
        worker_states = [
            make_worker_state("idle-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
            make_worker_state("idle-2", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
        ]
        task_counts = TaskCounts(queued=0, active_cloud=0, total=0)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=True)

        assert decision.workers_to_terminate == 0


class TestObservabilityFields:
    """Tests for new observability fields."""

    def test_all_observability_fields_populated(self):
        config = make_config(min_active_gpus=1, machines_to_keep_idle=1)
        worker_states = [
            make_worker_state("busy-1", WorkerLifecycle.ACTIVE_READY, has_active_task=True),
            make_worker_state("idle-1", WorkerLifecycle.ACTIVE_READY, has_active_task=False),
            make_worker_state("spawning-1", WorkerLifecycle.SPAWNING_SCRIPT_RUNNING),
        ]
        task_counts = TaskCounts(queued=2, active_cloud=1, total=3)

        decision = calculate_scaling_decision_pure(worker_states, task_counts, config, failure_rate_ok=True)

        assert decision.workers_covering_tasks == 2
        assert decision.uncovered_tasks == 0
        assert hasattr(decision, "spawn_reason_tasks")
        assert hasattr(decision, "spawn_reason_floor")

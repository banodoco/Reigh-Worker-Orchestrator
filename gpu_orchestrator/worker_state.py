"""
Worker state model for the GPU Orchestrator.

Provides a single source of truth for worker state derivation:
- WorkerLifecycle enum: All possible worker states
- DerivedWorkerState: Complete computed state for a worker at a point in time
- derive_worker_state(): Pure function that computes state from worker data

This replaces scattered, inconsistent state checks throughout the codebase.
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import OrchestratorConfig


class WorkerLifecycle(Enum):
    """Lifecycle states for a GPU worker."""

    # Spawning sub-states
    SPAWNING_POD_PENDING = "spawning_pod_pending"        # Pod not ready yet
    SPAWNING_SCRIPT_PENDING = "spawning_script_pending"  # Pod ready, script not launched
    SPAWNING_SCRIPT_RUNNING = "spawning_script_running"  # Script launched, no heartbeat yet

    # Active sub-states (process confirmed running via heartbeat)
    ACTIVE_INITIALIZING = "active_initializing"          # Heartbeat exists, no VRAM data yet
    ACTIVE_GPU_NOT_DETECTED = "active_gpu_not_detected"  # Heartbeat + VRAM=0
    ACTIVE_READY = "active_ready"                        # Heartbeat + VRAM>0
    ACTIVE_STALE = "active_stale"                        # Heartbeat too old

    # Terminal states
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class DerivedWorkerState:
    """
    Complete derived state for a worker at a point in time.
    All timeout checks and health assessments happen during derivation.
    """

    # Identity
    worker_id: str
    runpod_id: Optional[str]
    lifecycle: WorkerLifecycle
    db_status: str  # Original DB status for reference

    # Timing (all in seconds)
    created_at: datetime
    script_launched_at: Optional[datetime]
    effective_age_sec: float          # Time since script launch (or creation if no script)
    heartbeat_age_sec: Optional[float]  # Time since last heartbeat (None if never)

    # Health signals
    has_heartbeat: bool
    heartbeat_is_recent: bool         # Within promotion threshold
    vram_total_mb: Optional[int]
    vram_used_mb: Optional[int]
    vram_reported: bool               # Has VRAM data been reported?
    ready_for_tasks: bool             # Worker explicitly signaled readiness
    has_ever_claimed_task: bool
    has_active_task: bool

    # Derived health flags
    is_gpu_broken: bool               # VRAM=0 for too long
    is_not_claiming: bool             # Ready but not claiming for too long
    is_stale: bool                    # Heartbeat too old

    # Action flags
    should_promote_to_active: bool    # Spawning worker with heartbeat
    should_terminate: bool            # Any termination condition met
    termination_reason: Optional[str]
    error_code: Optional[str]         # Structured error code for analytics

    @property
    def is_healthy(self) -> bool:
        """Worker is in a healthy, productive state."""
        return self.lifecycle == WorkerLifecycle.ACTIVE_READY and not self.should_terminate

    @property
    def is_spawning(self) -> bool:
        """Worker is still in spawning phase."""
        return self.lifecycle in (
            WorkerLifecycle.SPAWNING_POD_PENDING,
            WorkerLifecycle.SPAWNING_SCRIPT_PENDING,
            WorkerLifecycle.SPAWNING_SCRIPT_RUNNING,
        )

    @property
    def is_active(self) -> bool:
        """Worker is in an active phase (has heartbeat)."""
        return self.lifecycle in (
            WorkerLifecycle.ACTIVE_INITIALIZING,
            WorkerLifecycle.ACTIVE_GPU_NOT_DETECTED,
            WorkerLifecycle.ACTIVE_READY,
            WorkerLifecycle.ACTIVE_STALE,
        )

    @property
    def is_terminal(self) -> bool:
        """Worker is in a terminal state."""
        return self.lifecycle in (WorkerLifecycle.ERROR, WorkerLifecycle.TERMINATED)


def derive_worker_state(
    worker: Dict[str, Any],
    config: 'OrchestratorConfig',
    now: datetime,
    has_ever_claimed: bool,
    has_active_task: bool,
    queued_count: int
) -> DerivedWorkerState:
    """
    Single source of truth for worker state derivation.

    All timeout checks happen here. The rest of the control loop
    just pattern-matches on the derived state.

    Args:
        worker: Raw worker dict from database
        config: Orchestrator configuration
        now: Current timestamp (passed in for testability)
        has_ever_claimed: Whether worker has ever claimed any task
        has_active_task: Whether worker currently has an active task
        queued_count: Number of queued tasks (for "not claiming" checks)

    Returns:
        DerivedWorkerState with all health assessments completed
    """
    worker_id = worker['id']
    metadata = worker.get('metadata', {}) or {}
    db_status = worker['status']
    runpod_id = metadata.get('runpod_id')

    # Parse timestamps
    created_at = _parse_timestamp(worker.get('created_at')) or now
    script_launched_at = _parse_timestamp(metadata.get('startup_script_launched_at'))
    last_heartbeat = _parse_timestamp(worker.get('last_heartbeat'))

    # Calculate ages
    age_basis = script_launched_at or created_at
    effective_age_sec = (now - age_basis).total_seconds()
    heartbeat_age_sec = (now - last_heartbeat).total_seconds() if last_heartbeat else None
    worker_age_sec = (now - created_at).total_seconds()

    # VRAM data
    vram_total = metadata.get('vram_total_mb')
    vram_used = metadata.get('vram_used_mb')
    vram_timestamp = metadata.get('vram_timestamp')
    vram_reported = vram_timestamp is not None

    # Ready for tasks flag - explicit signal from worker that it can claim tasks
    ready_for_tasks = metadata.get('ready_for_tasks', False)

    # Heartbeat flags
    # Note: The workers table has DEFAULT NOW() for last_heartbeat, so newly created
    # workers have a "fake" heartbeat at creation time. We only consider a heartbeat
    # "real" if it's at least 5 seconds after the creation time.
    heartbeat_since_creation = (last_heartbeat - created_at).total_seconds() if last_heartbeat else None
    has_real_heartbeat = last_heartbeat is not None and heartbeat_since_creation is not None and heartbeat_since_creation > 5
    has_heartbeat = has_real_heartbeat
    heartbeat_is_recent = has_heartbeat and heartbeat_age_sec < config.heartbeat_promotion_threshold_sec

    # Determine lifecycle state
    lifecycle = _determine_lifecycle(
        db_status=db_status,
        has_heartbeat=has_heartbeat,
        heartbeat_age_sec=heartbeat_age_sec,
        vram_total=vram_total,
        vram_reported=vram_reported,
        ready_for_tasks=ready_for_tasks,
        script_launched=metadata.get('startup_script_launched', False),
        config=config
    )

    # Determine termination conditions
    should_terminate = False
    termination_reason = None
    error_code = None

    is_gpu_broken = False
    is_not_claiming = False
    is_stale = False

    if lifecycle == WorkerLifecycle.ACTIVE_STALE:
        is_stale = True
        if has_active_task:
            should_terminate = True
            termination_reason = f"Stale heartbeat with active tasks ({heartbeat_age_sec:.0f}s old)"
            error_code = "STALE_HEARTBEAT_ACTIVE_TASK"
        elif queued_count > 0:
            should_terminate = True
            termination_reason = f"Stale heartbeat with tasks queued ({heartbeat_age_sec:.0f}s old)"
            error_code = "STALE_HEARTBEAT_TASKS_QUEUED"

    elif lifecycle == WorkerLifecycle.ACTIVE_GPU_NOT_DETECTED:
        if effective_age_sec > config.gpu_not_detected_timeout_sec:
            is_gpu_broken = True
            should_terminate = True
            termination_reason = f"GPU not detected (VRAM=0) after {effective_age_sec:.0f}s"
            error_code = "GPU_NOT_DETECTED"

    elif lifecycle == WorkerLifecycle.ACTIVE_READY:
        if queued_count > 0 and not has_active_task and not has_ever_claimed:
            if effective_age_sec > config.ready_not_claiming_timeout_sec:
                is_not_claiming = True
                should_terminate = True
                termination_reason = f"GPU ready but never claimed tasks ({effective_age_sec:.0f}s)"
                error_code = "GPU_READY_NOT_CLAIMING"

    elif lifecycle == WorkerLifecycle.ACTIVE_INITIALIZING:
        if queued_count > 0 and not has_ever_claimed:
            if effective_age_sec > config.startup_grace_period_sec:
                should_terminate = True
                termination_reason = f"Never initialized after startup grace period ({effective_age_sec:.0f}s)"
                error_code = "STARTUP_NEVER_READY"

    elif lifecycle in (WorkerLifecycle.SPAWNING_POD_PENDING,
                       WorkerLifecycle.SPAWNING_SCRIPT_PENDING,
                       WorkerLifecycle.SPAWNING_SCRIPT_RUNNING):
        if worker_age_sec > config.spawning_timeout_sec:
            should_terminate = True
            termination_reason = f"Spawning timeout ({worker_age_sec:.0f}s)"
            error_code = "SPAWNING_TIMEOUT"

    # Should promote?
    # TODO: Add ready_for_tasks check once worker sends it
    should_promote = (
        db_status == 'spawning' and
        heartbeat_is_recent and
        lifecycle in (WorkerLifecycle.ACTIVE_INITIALIZING,
                      WorkerLifecycle.ACTIVE_GPU_NOT_DETECTED,
                      WorkerLifecycle.ACTIVE_READY)
    )

    return DerivedWorkerState(
        worker_id=worker_id,
        runpod_id=runpod_id,
        lifecycle=lifecycle,
        db_status=db_status,
        created_at=created_at,
        script_launched_at=script_launched_at,
        effective_age_sec=effective_age_sec,
        heartbeat_age_sec=heartbeat_age_sec,
        has_heartbeat=has_heartbeat,
        heartbeat_is_recent=heartbeat_is_recent,
        vram_total_mb=vram_total,
        vram_used_mb=vram_used,
        vram_reported=vram_reported,
        ready_for_tasks=ready_for_tasks,
        has_ever_claimed_task=has_ever_claimed,
        has_active_task=has_active_task,
        is_gpu_broken=is_gpu_broken,
        is_not_claiming=is_not_claiming,
        is_stale=is_stale,
        should_promote_to_active=should_promote,
        should_terminate=should_terminate,
        termination_reason=termination_reason,
        error_code=error_code,
    )


def _determine_lifecycle(
    db_status: str,
    has_heartbeat: bool,
    heartbeat_age_sec: Optional[float],
    vram_total: Optional[int],
    vram_reported: bool,
    ready_for_tasks: bool,
    script_launched: bool,
    config: 'OrchestratorConfig'
) -> WorkerLifecycle:
    """Map signals to lifecycle state."""

    if db_status == 'terminated':
        return WorkerLifecycle.TERMINATED

    if db_status == 'error':
        return WorkerLifecycle.ERROR

    # If a heartbeat exists, the process has run at least once; treat it as active and
    # use heartbeat_age_sec only to decide "stale" vs "recent enough for promotion".
    if has_heartbeat:
        # Check for heartbeat staleness first
        if heartbeat_age_sec is not None and heartbeat_age_sec > config.gpu_idle_timeout_sec:
            return WorkerLifecycle.ACTIVE_STALE

        # Not stale => active. VRAM determines sub-state.
        # TODO: Once worker sends ready_for_tasks=true, add that check here
        if vram_reported:
            if vram_total == 0:
                return WorkerLifecycle.ACTIVE_GPU_NOT_DETECTED
            else:
                return WorkerLifecycle.ACTIVE_READY
        return WorkerLifecycle.ACTIVE_INITIALIZING

    # No heartbeat - still spawning
    if db_status in ('spawning', 'inactive', 'active'):
        # Note: 'active' without heartbeat can happen if manually promoted or legacy data
        if script_launched:
            return WorkerLifecycle.SPAWNING_SCRIPT_RUNNING
        else:
            return WorkerLifecycle.SPAWNING_SCRIPT_PENDING

    # Fallback
    return WorkerLifecycle.SPAWNING_POD_PENDING


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


@dataclass
class TaskCounts:
    """Task count summary for scaling decisions."""
    queued: int
    active_cloud: int
    total: int

    @property
    def total_workload(self) -> int:
        """Total tasks that need workers (queued + active)."""
        return self.queued + self.active_cloud


@dataclass
class ScalingDecision:
    """Result of scaling calculation."""
    desired_workers: int
    current_capacity: int
    active_count: int
    spawning_count: int
    idle_count: int
    busy_count: int

    # Components of desired calculation
    task_based_workers: int
    buffer_based_workers: int
    min_based_workers: int

    # Action recommendations
    workers_to_spawn: int
    workers_to_terminate: int

    # Constraints that may have limited action
    failure_rate_ok: bool
    at_max_capacity: bool
    at_min_capacity: bool

    # New fields for observability (reactive vs proactive scaling)
    spawn_reason_tasks: int = 0      # Workers spawned to cover uncovered tasks
    spawn_reason_floor: int = 0      # Workers spawned for floor maintenance
    uncovered_tasks: int = 0         # Queued tasks without coverage
    workers_covering_tasks: int = 0  # Idle + spawning workers

    @property
    def should_scale_up(self) -> bool:
        return self.workers_to_spawn > 0

    @property
    def should_scale_down(self) -> bool:
        return self.workers_to_terminate > 0


def calculate_scaling_decision_pure(
    worker_states: list,  # List[DerivedWorkerState]
    task_counts: TaskCounts,
    config: 'OrchestratorConfig',
    failure_rate_ok: bool,
) -> ScalingDecision:
    """
    Pure function to calculate scaling decision with no I/O.

    Separates reactive scaling (cover queued tasks) from proactive scaling
    (maintain buffer/minimum) to prevent double-spawning.

    Args:
        worker_states: List of derived worker states
        task_counts: Current task counts
        config: Orchestrator configuration
        failure_rate_ok: Whether failure rate allows spawning

    Returns:
        ScalingDecision with spawn/terminate recommendations
    """
    import math

    # Count workers by state (exclude those marked for termination)
    active_states = [ws for ws in worker_states if ws.is_active and not ws.should_terminate]
    spawning_states = [ws for ws in worker_states if ws.is_spawning and not ws.should_terminate]

    active_count = len(active_states)
    spawning_count = len(spawning_states)
    current_capacity = active_count + spawning_count

    # Count idle vs busy among active workers
    idle_count = len([ws for ws in active_states if not ws.has_active_task])
    busy_count = active_count - idle_count

    # =========================================================================
    # NEW SCALING LOGIC: Separate reactive from proactive
    # =========================================================================

    # 1. Reactive scaling: Cover queued tasks
    # Workers that can take tasks: idle active workers + spawning workers
    workers_covering_tasks = idle_count + spawning_count
    uncovered_tasks = max(0, task_counts.queued - workers_covering_tasks)
    spawn_for_tasks = uncovered_tasks

    # 2. Proactive scaling: Maintain floor (only if not already spawning)
    # Floor = max(min_active_gpus, busy_count + machines_to_keep_idle)
    desired_floor = max(config.min_active_gpus, busy_count + config.machines_to_keep_idle)

    # Account for workers we're about to spawn for tasks
    capacity_after_task_spawns = current_capacity + spawn_for_tasks

    # Only spawn for floor if no workers are currently spawning
    if spawning_count == 0:
        spawn_for_floor = max(0, desired_floor - capacity_after_task_spawns)
    else:
        spawn_for_floor = 0

    # Exception: Always allow spawning to reach minimum (even if spawning in progress)
    min_gap = max(0, config.min_active_gpus - capacity_after_task_spawns)
    spawn_for_floor = max(spawn_for_floor, min_gap)

    workers_to_spawn = spawn_for_tasks + spawn_for_floor

    # =========================================================================
    # Legacy fields for backwards compatibility
    # =========================================================================
    total_workload = task_counts.total_workload
    if total_workload > 0:
        up_scaled = math.ceil(total_workload * config.scale_up_multiplier)
        task_based = max(1, up_scaled)
    else:
        task_based = 0

    buffer_based = busy_count + config.machines_to_keep_idle
    min_based = config.min_active_gpus

    desired = max(min_based, task_based, buffer_based)
    desired = min(desired, config.max_active_gpus)

    # =========================================================================
    # Apply constraints
    # =========================================================================

    # Don't spawn if failure rate is too high
    if not failure_rate_ok:
        workers_to_spawn = 0

    # Don't spawn beyond max capacity
    max_spawnable = config.max_active_gpus - current_capacity
    workers_to_spawn = min(workers_to_spawn, max(0, max_spawnable))

    # Update spawn_for_tasks and spawn_for_floor after constraints
    # (proportionally reduce if constrained)
    if workers_to_spawn < spawn_for_tasks + spawn_for_floor and workers_to_spawn > 0:
        # Prioritize tasks over floor
        actual_spawn_for_tasks = min(spawn_for_tasks, workers_to_spawn)
        actual_spawn_for_floor = workers_to_spawn - actual_spawn_for_tasks
    else:
        actual_spawn_for_tasks = spawn_for_tasks if workers_to_spawn > 0 else 0
        actual_spawn_for_floor = spawn_for_floor if workers_to_spawn > 0 else 0

    # Calculate scale-down (if over capacity)
    workers_to_terminate = 0
    if current_capacity > desired:
        workers_to_terminate = current_capacity - desired

    return ScalingDecision(
        desired_workers=desired,
        current_capacity=current_capacity,
        active_count=active_count,
        spawning_count=spawning_count,
        idle_count=idle_count,
        busy_count=busy_count,
        task_based_workers=task_based,
        buffer_based_workers=buffer_based,
        min_based_workers=min_based,
        workers_to_spawn=workers_to_spawn,
        workers_to_terminate=workers_to_terminate,
        failure_rate_ok=failure_rate_ok,
        at_max_capacity=current_capacity >= config.max_active_gpus,
        at_min_capacity=active_count <= config.min_active_gpus,
        # New observability fields
        spawn_reason_tasks=actual_spawn_for_tasks,
        spawn_reason_floor=actual_spawn_for_floor,
        uncovered_tasks=uncovered_tasks,
        workers_covering_tasks=workers_covering_tasks,
    )


@dataclass
class CycleSummary:
    """Summary of actions taken in a cycle."""
    workers_promoted: int = 0
    workers_failed: int = 0
    workers_spawned: int = 0
    workers_terminated: int = 0
    tasks_reset: int = 0
    startup_scripts_launched: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": {
                "workers_promoted": self.workers_promoted,
                "workers_failed": self.workers_failed,
                "workers_spawned": self.workers_spawned,
                "workers_terminated": self.workers_terminated,
                "tasks_reset": self.tasks_reset,
                "startup_scripts_launched": self.startup_scripts_launched,
            },
            "error": self.error,
        }

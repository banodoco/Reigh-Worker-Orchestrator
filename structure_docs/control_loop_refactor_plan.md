# GPU Orchestrator Control Loop Refactor Plan

## Overview

This document outlines a refactoring plan for `gpu_orchestrator/control_loop.py` to improve maintainability, testability, and debuggability while preserving existing behavior.

**Status**: Proposed  
**Created**: 2026-01-21  
**Author**: AI-assisted planning

---

## Current State Analysis

### What's Strong

- **Phased cycle structure**: Clear order of operations (capacity view → lifecycle updates → health → reconciliation → scaling)
- **Defensive operations**: Multiple backstops (health checks + failsafe + zombie checks) reduce billing leakage and stuck-task risk
- **Good observability intent**: Detailed task-count logging and diagnostics collection

### Main Structural Issues

1. **`run_single_cycle()` is too large (~900+ lines)**: Scaling policy, lifecycle state machine, health policy, DB reconciliation, and log formatting are all intertwined
2. **Status model is inconsistent**: DB `status` and `metadata.orchestrator_status` duality causes "why didn't X happen?" bugs
3. **Async + blocking calls risk**: Some RunPod/Supabase operations inside async loops can stall entire cycles
4. **Multiple truth sources**: Readiness inferred from logs, heartbeat, VRAM, and timestamps inconsistently across different code paths

---

## Goals

1. **Easier to reason about** – each concern in its own place
2. **Consistent state derivation** – one function decides "what state is this worker in?"
3. **Testable** – policy logic separated from I/O
4. **No behavior change** – same timeouts, same decisions, just cleaner structure

---

## Phase 1: Configuration Object

### Problem

Timeout/threshold values scattered across `__init__`, env lookups inline, and hardcoded values.

### Solution

Create an `OrchestratorConfig` dataclass loaded once at startup.

```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class OrchestratorConfig:
    """All orchestrator configuration in one place."""
    
    # Scaling limits
    min_active_gpus: int
    max_active_gpus: int
    machines_to_keep_idle: int
    tasks_per_gpu_threshold: int
    
    # Scaling behavior
    scale_up_multiplier: float
    scale_down_multiplier: float
    min_scaling_interval_sec: int
    spawning_grace_period_sec: int
    scale_down_grace_period_sec: int
    
    # Worker timeouts (seconds)
    spawning_timeout_sec: int
    gpu_idle_timeout_sec: int
    overcapacity_idle_timeout_sec: int
    task_stuck_timeout_sec: int
    graceful_shutdown_timeout_sec: int
    
    # Health check timeouts (seconds)
    startup_grace_period_sec: int
    ready_not_claiming_timeout_sec: int
    gpu_not_detected_timeout_sec: int
    heartbeat_promotion_threshold_sec: int
    
    # Failure protection
    max_worker_failure_rate: float
    failure_window_minutes: int
    min_workers_for_rate_check: int
    
    # Storage monitoring
    storage_check_interval_cycles: int
    storage_min_free_gb: int
    storage_max_percent_used: int
    storage_expansion_increment_gb: int
    
    # Polling
    orchestrator_poll_sec: int
    
    @classmethod
    def from_env(cls) -> 'OrchestratorConfig':
        """Load all config from environment with defaults."""
        poll_sec = int(os.getenv("ORCHESTRATOR_POLL_SEC", "30"))
        
        return cls(
            # Scaling limits
            min_active_gpus=int(os.getenv("MIN_ACTIVE_GPUS", "2")),
            max_active_gpus=int(os.getenv("MAX_ACTIVE_GPUS", "10")),
            machines_to_keep_idle=int(os.getenv("MACHINES_TO_KEEP_IDLE", "0")),
            tasks_per_gpu_threshold=int(os.getenv("TASKS_PER_GPU_THRESHOLD", "3")),
            
            # Scaling behavior
            scale_up_multiplier=float(os.getenv("SCALE_UP_MULTIPLIER", "1.0")),
            scale_down_multiplier=float(os.getenv("SCALE_DOWN_MULTIPLIER", "0.9")),
            min_scaling_interval_sec=int(os.getenv("MIN_SCALING_INTERVAL_SEC", "45")),
            spawning_grace_period_sec=int(os.getenv("SPAWNING_GRACE_PERIOD_SEC", "180")),
            scale_down_grace_period_sec=int(os.getenv("SCALE_DOWN_GRACE_PERIOD_SEC", "60")),
            
            # Worker timeouts
            spawning_timeout_sec=int(os.getenv("SPAWNING_TIMEOUT_SEC", "600")),
            gpu_idle_timeout_sec=int(os.getenv("GPU_IDLE_TIMEOUT_SEC", "600")),
            overcapacity_idle_timeout_sec=int(os.getenv("GPU_OVERCAPACITY_IDLE_TIMEOUT_SEC", "30")),
            task_stuck_timeout_sec=int(os.getenv("TASK_STUCK_TIMEOUT_SEC", "1200")),
            graceful_shutdown_timeout_sec=int(os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT_SEC", "600")),
            
            # Health check timeouts
            startup_grace_period_sec=int(os.getenv("STARTUP_GRACE_PERIOD_SEC", "600")),
            ready_not_claiming_timeout_sec=int(os.getenv("READY_NOT_CLAIMING_TIMEOUT_SEC", "180")),
            gpu_not_detected_timeout_sec=int(os.getenv("GPU_NOT_DETECTED_TIMEOUT_SEC", "300")),
            heartbeat_promotion_threshold_sec=int(os.getenv(
                "HEARTBEAT_PROMOTION_THRESHOLD_SEC",
                str(max(60, poll_sec * 3))
            )),
            
            # Failure protection
            max_worker_failure_rate=float(os.getenv("MAX_WORKER_FAILURE_RATE", "0.8")),
            failure_window_minutes=int(os.getenv("FAILURE_WINDOW_MINUTES", "5")),
            min_workers_for_rate_check=int(os.getenv("MIN_WORKERS_FOR_RATE_CHECK", "5")),
            
            # Storage monitoring
            storage_check_interval_cycles=int(os.getenv("STORAGE_CHECK_INTERVAL_CYCLES", "10")),
            storage_min_free_gb=int(os.getenv("STORAGE_MIN_FREE_GB", "50")),
            storage_max_percent_used=int(os.getenv("STORAGE_MAX_PERCENT_USED", "85")),
            storage_expansion_increment_gb=int(os.getenv("STORAGE_EXPANSION_INCREMENT_GB", "50")),
            
            # Polling
            orchestrator_poll_sec=poll_sec,
        )
    
    def log_config(self, logger):
        """Log all config values at startup for debugging."""
        logger.info("🔧 ORCHESTRATOR CONFIG:")
        logger.info(f"   Scaling: {self.min_active_gpus}-{self.max_active_gpus} GPUs, idle buffer: {self.machines_to_keep_idle}")
        logger.info(f"   Timeouts: spawning={self.spawning_timeout_sec}s, idle={self.gpu_idle_timeout_sec}s, stuck={self.task_stuck_timeout_sec}s")
        logger.info(f"   Health: startup_grace={self.startup_grace_period_sec}s, not_claiming={self.ready_not_claiming_timeout_sec}s, gpu_not_detected={self.gpu_not_detected_timeout_sec}s")
        logger.info(f"   Promotion: heartbeat_threshold={self.heartbeat_promotion_threshold_sec}s")
```

### Benefits

- Single place to see all knobs
- Easy to log at startup
- Easy to test with different configs
- No more hunting for env var names

---

## Phase 2: Worker State Model

### Problem

Worker state is derived differently in different places:
- Heartbeat age calculated multiple times
- VRAM checks duplicated
- "Has ever claimed" checked ad-hoc
- Inconsistent use of `created_at` vs `startup_script_launched_at`

### Solution

Define a clear `WorkerLifecycle` enum and a `derive_worker_state()` function.

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

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
    # NOTE: created_at should always exist; treat missing/invalid as "now" to avoid blowing up derivation.
    created_at = _parse_timestamp(worker.get('created_at')) or now
    script_launched_at = _parse_timestamp(metadata.get('startup_script_launched_at'))
    last_heartbeat = _parse_timestamp(worker.get('last_heartbeat'))
    
    # Calculate ages
    age_basis = script_launched_at or created_at
    effective_age_sec = (now - age_basis).total_seconds()
    heartbeat_age_sec = (now - last_heartbeat).total_seconds() if last_heartbeat else None
    
    # VRAM data
    vram_total = metadata.get('vram_total_mb')
    vram_used = metadata.get('vram_used_mb')
    vram_timestamp = metadata.get('vram_timestamp')
    vram_reported = vram_timestamp is not None
    
    # Heartbeat flags
    has_heartbeat = last_heartbeat is not None
    heartbeat_is_recent = has_heartbeat and heartbeat_age_sec < config.heartbeat_promotion_threshold_sec
    
    # Determine lifecycle state
    lifecycle = _determine_lifecycle(
        db_status=db_status,
        has_heartbeat=has_heartbeat,
        heartbeat_age_sec=heartbeat_age_sec,
        vram_total=vram_total,
        vram_reported=vram_reported,
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
        worker_age = (now - created_at).total_seconds()
        if worker_age > config.spawning_timeout_sec:
            should_terminate = True
            termination_reason = f"Spawning timeout ({worker_age:.0f}s)"
            error_code = "SPAWNING_TIMEOUT"
    
    # Should promote?
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
        if vram_reported:
            if vram_total == 0:
                return WorkerLifecycle.ACTIVE_GPU_NOT_DETECTED
            else:
                return WorkerLifecycle.ACTIVE_READY
        return WorkerLifecycle.ACTIVE_INITIALIZING
    
    # No heartbeat - still spawning
    if db_status in ('spawning', 'inactive'):
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
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
```

### Benefits

- State derivation tested independently (pure function)
- Health checks become simple pattern matches on `DerivedWorkerState`
- No more duplicated datetime parsing / age calculations
- Clear "should_terminate" and "termination_reason" for debugging

---

## Phase 3: Split `run_single_cycle()` into Methods

### Current

~900 lines doing everything inline with deep nesting.

### Proposed Structure

```python
async def run_single_cycle(self) -> Dict[str, Any]:
    """Main orchestration cycle - coordinates all phases."""
    cycle_start = datetime.now(timezone.utc)
    self.cycle_count += 1
    set_current_cycle(self.cycle_count)
    
    summary = CycleSummary()
    
    try:
        # Phase 1: Fetch current state
        workers, task_counts = await self._fetch_current_state()
        
        # Phase 2: Derive state for all workers
        worker_states = await self._derive_all_worker_states(workers, task_counts.queued)
        
        # Phase 3: Handle spawning workers (pod init, script launch, promotion)
        await self._handle_spawning_workers(worker_states, task_counts, summary)
        
        # Phase 4: Health check active workers
        await self._health_check_active_workers(worker_states, task_counts, summary)
        
        # Phase 5: Cleanup error workers
        await self._cleanup_error_workers(worker_states, summary)
        
        # Phase 6: Reconcile orphaned tasks
        await self._reconcile_orphaned_tasks(worker_states, summary)
        
        # Phase 7: Calculate scaling decision
        scaling = self._calculate_scaling_decision(worker_states, task_counts)
        
        # Phase 8: Execute scaling
        await self._execute_scaling(scaling, worker_states, task_counts, summary)
        
        # Phase 9: Periodic checks (storage, zombies)
        await self._run_periodic_checks(worker_states, summary)
        
    except Exception as e:
        logger.error(f"Error in orchestrator cycle: {e}")
        summary.error = str(e)
    
    # Phase 10: Log summary
    self._log_cycle_summary(summary, cycle_start)
    
    return summary.to_dict()
```

### Method Signatures

```python
async def _fetch_current_state(self) -> Tuple[List[Dict], TaskCounts]:
    """Fetch workers and task counts from database."""
    ...

async def _derive_all_worker_states(
    self, 
    workers: List[Dict], 
    queued_count: int
) -> List[DerivedWorkerState]:
    """Derive state for all workers (batches async lookups)."""
    ...

async def _handle_spawning_workers(
    self,
    worker_states: List[DerivedWorkerState],
    task_counts: TaskCounts,
    summary: CycleSummary
) -> None:
    """Handle pod initialization, script launch, and promotion."""
    ...

async def _health_check_active_workers(
    self,
    worker_states: List[DerivedWorkerState],
    task_counts: TaskCounts,
    summary: CycleSummary
) -> None:
    """Check health of active workers, mark errors."""
    ...

async def _cleanup_error_workers(
    self,
    worker_states: List[DerivedWorkerState],
    summary: CycleSummary
) -> None:
    """Terminate RunPod instances for error workers."""
    ...

async def _reconcile_orphaned_tasks(
    self,
    worker_states: List[DerivedWorkerState],
    summary: CycleSummary
) -> None:
    """Reset tasks stuck on failed/terminated workers."""
    ...

def _calculate_scaling_decision(
    self,
    worker_states: List[DerivedWorkerState],
    task_counts: TaskCounts
) -> ScalingDecision:
    """Calculate desired workers and scaling actions."""
    ...

async def _execute_scaling(
    self,
    decision: ScalingDecision,
    worker_states: List[DerivedWorkerState],
    task_counts: TaskCounts,
    summary: CycleSummary
) -> None:
    """Spawn or terminate workers based on scaling decision."""
    ...

async def _run_periodic_checks(
    self,
    worker_states: List[DerivedWorkerState],
    summary: CycleSummary
) -> None:
    """Run checks that only need to happen every Nth cycle."""
    ...
```

### Benefits

- Each method is 50-150 lines focused on one concern
- Easy to find where a specific behavior lives
- Can test individual phases
- Failures easier to isolate

---

## Phase 4: Simplify Status Model

### Problem

`worker['status']` vs `worker['metadata']['orchestrator_status']` creates confusion.

### Solution

Use DB `status` as the **only** lifecycle field:

| DB Status | Meaning | Transitions To |
|-----------|---------|----------------|
| `inactive` | Initial state (legacy) | `spawning` |
| `spawning` | Pod/script starting, worker process not yet running | `active`, `error` |
| `active` | Worker process is running (proven by heartbeat) | `error`, `terminated` |
| `error` | Something went wrong, pending cleanup | `terminated` |
| `terminated` | Fully cleaned up | (terminal) |

### Changes Required

1. **Remove `metadata.orchestrator_status`** (or keep read-only for backward compat)
2. **Promotion updates DB `status` to `active`**
3. **Health assessment is derived every cycle**, not stored
4. **`metadata.diagnostics`** stores historical info for debugging terminated workers

### Migration

```python
# Old code
if metadata.get('orchestrator_status') == 'spawning':
    ...

# New code
if worker['status'] == 'spawning':
    ...

# Or with derived state
if worker_state.lifecycle in (WorkerLifecycle.SPAWNING_POD_PENDING, ...):
    ...
```

---

## Phase 5: Async Hygiene

### Problem

Some RunPod/Supabase calls may block or be slow; one slow worker can delay the whole cycle.

### Solutions

#### 1. Batch Lookups

```python
# Before: N queries for N workers
for worker in workers:
    has_claimed = await self.db.has_worker_ever_claimed_task(worker['id'])

# After: 1 query for N workers
async def batch_check_ever_claimed(self, worker_ids: List[str]) -> Dict[str, bool]:
    """Check if workers have ever claimed tasks in one query."""
    # NOTE: This naive approach can return a lot of rows if tasks is large.
    # Prefer an RPC / SQL function that returns DISTINCT worker_id values for the input list
    # (or a materialized view / index-assisted query), and ensure pagination if needed.
    result = self.supabase.table('tasks') \
        .select('worker_id') \
        .in_('worker_id', worker_ids) \
        .execute()
    
    claimed_workers = {r['worker_id'] for r in (result.data or [])}
    return {wid: wid in claimed_workers for wid in worker_ids}
```

#### 2. Timeouts on External Calls

```python
async def _safe_runpod_call(self, fn, *args, timeout_sec: int = 30, **kwargs):
    """
    Wrap RunPod calls with timeout.
    
    NOTE: In our current codebase, many RunPod client methods are synchronous.
    Use asyncio.to_thread to avoid blocking the event loop.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs),
            timeout=timeout_sec
        )
    except asyncio.TimeoutError:
        logger.warning(f"RunPod call timed out after {timeout_sec}s")
        return None
```

#### 3. Parallel Where Safe

```python
async def _derive_all_worker_states(self, workers, queued_count):
    # Batch queries
    worker_ids = [w['id'] for w in workers]
    
    # Run batch queries in parallel
    claimed_map, active_tasks_map = await asyncio.gather(
        self.db.batch_check_ever_claimed(worker_ids),
        self.db.batch_check_active_tasks(worker_ids)
    )
    
    # Derive state for each (pure computation, no I/O)
    now = datetime.now(timezone.utc)
    return [
        derive_worker_state(
            worker=w,
            config=self.config,
            now=now,
            has_ever_claimed=claimed_map.get(w['id'], False),
            has_active_task=active_tasks_map.get(w['id'], False),
            queued_count=queued_count
        )
        for w in workers
    ]
```

---

## Implementation Order

| Step | What | Risk | Effort | Dependencies |
|------|------|------|--------|--------------|
| 1 | Extract `OrchestratorConfig` | Low | 1 hr | None |
| 2 | Define `WorkerLifecycle` enum + `DerivedWorkerState` | Low | 1 hr | None |
| 3 | Implement `derive_worker_state()` with tests | Medium | 2 hr | Step 2 |
| 4 | Add `batch_check_ever_claimed()` to DB client | Low | 30 min | None |
| 5 | Split cycle into methods (no logic change) | Medium | 2-3 hr | Steps 1-4 |
| 6 | Clean up status model (remove orchestrator_status) | Medium | 1 hr | Step 5 |
| 7 | Add async timeouts | Low | 30 min | Step 5 |

**Total: ~8-10 hours of focused work**

---

## Testing Strategy

### Unit Tests

```python
# test_worker_state.py

def test_derive_state_spawning_no_heartbeat():
    """Spawning worker with no heartbeat stays spawning."""
    config = OrchestratorConfig.from_env()
    worker = {'id': 'test-1', 'status': 'spawning', 'created_at': '...', 'metadata': {}}
    
    state = derive_worker_state(worker, config, now, has_ever_claimed=False, ...)
    
    assert state.lifecycle == WorkerLifecycle.SPAWNING_SCRIPT_PENDING
    assert not state.should_promote_to_active
    assert not state.should_terminate

def test_derive_state_spawning_with_heartbeat_promotes():
    """Spawning worker with recent heartbeat should be promoted."""
    ...
    assert state.should_promote_to_active

def test_derive_state_gpu_not_detected_terminates():
    """Worker with VRAM=0 for too long should terminate."""
    ...
    assert state.should_terminate
    assert state.error_code == "GPU_NOT_DETECTED"
```

### Integration Tests

- Test full cycle with mocked DB/RunPod
- Verify correct transitions between states
- Verify scaling decisions match expected

---

## What This Enables

1. **Easier debugging**: "Why didn't worker X get killed?" → check `derive_worker_state()` output
2. **Policy changes in one place**: want a new timeout? Add to config + update `derive_worker_state()`
3. **Testable**: unit test state derivation without mocking DB/RunPod
4. **Cleaner logs**: log derived state once per worker per cycle
5. **Future features**: easy to add new lifecycle states or health checks

---

## Appendix: File Structure After Refactor

```
gpu_orchestrator/
├── __init__.py
├── main.py
├── config.py                 # NEW: OrchestratorConfig
├── worker_state.py           # NEW: WorkerLifecycle, DerivedWorkerState, derive_worker_state()
├── control_loop.py           # REFACTORED: split into methods
├── database.py               # UPDATED: batch methods
├── runpod_client.py
├── health_monitor.py
├── logging_config.py
└── database_log_handler.py
```

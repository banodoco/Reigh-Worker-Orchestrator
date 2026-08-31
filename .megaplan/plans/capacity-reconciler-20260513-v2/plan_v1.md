# Implementation Plan: Unified Capacity Reconciler

## Overview
The current production path is `gpu_orchestrator/main.py` -> `gpu_orchestrator/control_loop.py::OrchestratorControlLoop`. The brief's diagnosis matches the code shape: `_handle_early_termination()` mutates spawning workers before `_calculate_scaling_decision()` runs `calculate_scaling_decision_pure()`, and the duplicated `gpu_orchestrator/control/phases/` tree is test-only.

I considered the simpler fix of only disabling `_handle_early_termination` and adding RunPod spawn backoff. That would reduce the immediate churn, but it would not satisfy the durable intent, audit, single-writer, cutover adoption, or shadow divergence requirements. The plan below uses the brief as the design target, with one important implementation correction: use a lease table, not Postgres advisory locks, because the Supabase/PostgREST client cannot hold an advisory lock across multiple HTTP calls while RunPod side effects execute.

## Open Questions
1. Shadow-mode ordering conflicts with the current cycle order: the brief says shadow must run after lifecycle probing but before legacy early termination; current code runs early termination before lifecycle probing. Should Phase A intentionally reorder lifecycle probing before legacy capacity actions?
2. The brief asks for `_handle_early_termination` deletion but also says shadow mode should compare against the existing two-controller legacy path. Should this be delivered as two deployable slices: shadow PR keeps legacy early-term for divergence, final cutover PR deletes it?
3. Is manual Supabase SQL execution still the deployment norm? `railway.json` does not run migrations, while `scripts/apply_sql_migrations.py` and `scripts/show_migrations.py` have hardcoded migration lists.

## Assumptions
1. Final desired-capacity formula will be one expression: `desired = min(max_active_gpus, max(min_active_gpus, busy_count + queued_count, busy_count + max(machines_to_keep_idle, 1)))`. This preserves min floor, queued coverage, and busy-worker idle buffer without the old independent early-term formula.
2. `route_key` is nullable and pool-wide for the current orchestrator because selected route metadata exposes pool/contract but the active capacity controller is not yet doing per-route pool balancing.
3. Shadow mode may write duplicate intent rows from multiple replicas; authoritative mode must be single-writer through a lease. Audit queries will dedupe shadow rows by pool/cycle/action timestamp where needed.
4. Final cleanup state deletes `gpu_orchestrator/control/phases/`; any tests that import it are removed or rewritten against the reconciler.

## Phase 1: Schema And DB Boundaries

### Step 1: Add capacity-intent and lease migrations (`sql/20260514000000_create_worker_capacity_intents.sql`)
**Scope:** Medium
1. Create `worker_capacity_intents` with the brief's columns and `worker_capacity_intents_pool_route_created` index.
2. Add a small `orchestrator_leases` table with `pool`, `route_key`, `owner_id`, `expires_at`, and `updated_at`.
3. Add RPC helpers for lease acquire/release using TTL semantics, because a table lease survives multiple Supabase HTTP calls while RunPod side effects run.
4. Keep `system_logs.source_type='orchestrator_gpu'`; do not change the `valid_source_type` constraint.

### Step 2: Register migrations (`scripts/apply_sql_migrations.py`, `scripts/show_migrations.py`)
**Scope:** Small
1. Add the new migration to both hardcoded migration lists.
2. If practical, replace the hardcoded list with sorted `sql/*.sql` discovery while preserving current ordering.
3. Note explicitly that Railway currently starts `python -m gpu_orchestrator.main continuous` and does not apply SQL at boot.

### Step 3: Add DB methods (`gpu_orchestrator/database.py`)
**Scope:** Medium
1. Add methods for inserting an intent row, updating the just-written intent with spawn-result backoff fields, reading latest intent by `(pool, route_key, shadow)`, and acquiring/releasing the lease.
2. Add a direct `insert_system_log()` helper so shadow metadata is written at `metadata.shadow_intended_action`, not nested under `metadata.extra` by the logging handler.
3. Add worker lookup helpers needed for idempotency, especially querying workers by `metadata.capacity_intent_id`.

## Phase 2: Pure Planning Core

### Step 4: Add reconciler types (`gpu_orchestrator/control/capacity_types.py`)
**Scope:** Medium
1. Define frozen dataclasses for `Observed`, `CapacityPlan`, `AdmittedPlan`, `ReconciliationResult`, and action variants: `Spawn`, `CancelPendingSpawn`, `TerminateIdle`, `Noop`.
2. Include `Clock` injection with a production UTC implementation and fake-clock friendly protocol.
3. Include observed counts for active, idle, busy, spawning, route-stale, queued, in-progress, effective capacity, pool, route key, cycle id, and timestamps.

### Step 5: Add pure plan computation (`gpu_orchestrator/control/capacity_plan.py`)
**Scope:** Large
1. Implement `build_capacity_plan(observed, config, previous_intent)` as pure logic with no database, RunPod, or wall-clock calls beyond injected `clock.now()`.
2. Count spawning workers as capacity unless they are terminal, route-stale, over active-initializing timeout, or explicitly represented by a prior admitted cancel.
3. Exclude `is_route_stale=True` workers from fresh-route effective capacity while still allowing existing route-stale drain logic to terminate them when idle.
4. Implement time-based cancel hysteresis: `CancelPendingSpawn` only after desired <= effective across at least two snapshots and at least `2 * orchestrator_poll_sec`, or after prior `valid_until` elapsed.
5. Keep active scale-down limited to idle-timeout candidates; never terminate active busy workers due to capacity pressure.

## Phase 3: Reconciler Runtime

### Step 6: Implement reconciler orchestration (`gpu_orchestrator/control/capacity_reconciler.py`)
**Scope:** Large
1. Implement explicit phases: observe, plan, admit, record, act. Authoritative mode records before side effects; shadow mode records and logs intended actions but does not call `act()`.
2. In `admit()`, suppress `Spawn` actions when latest intent has `next_spawn_allowed_at > now`, and log `suppressed_by_backoff`.
3. In authoritative mode, acquire the lease for `(pool, route_key)` before final admit/record/act and release it afterward.
4. On spawn failure, update the already-written intent row with `consecutive_spawn_failures += 1` and `next_spawn_allowed_at = now + min(2**N * 30s, 600s)`; on successful spawn, reset the counter on that intent row.
5. For spawn idempotency, tag worker metadata with `capacity_intent_id` and action ordinal, and check existing workers for that tag before creating more RunPod pods.
6. On first authoritative observe, if spawning workers exist but no authoritative intent exists for the pool, write an adoption intent and count those workers as effective capacity.

### Step 7: Factor action primitives safely (`gpu_orchestrator/control_loop.py`, `gpu_orchestrator/database.py`)
**Scope:** Medium
1. Keep RunPod primitives in `WorkerSpawnerAdapter`; avoid duplicating RunPod API calls.
2. Add or adjust a spawn helper so the worker row is visible as `status='spawning'` before RunPod launch, with `capacity_intent_id` metadata, preventing crash windows from hiding in-progress spawns.
3. Reuse `_terminate_worker()` semantics for authoritative `TerminateIdle` and `CancelPendingSpawn`, with explicit termination reasons.

## Phase 4: Control Loop Integration And Rollout Flags

### Step 8: Add config flag (`gpu_orchestrator/config.py`)
**Scope:** Small
1. Add `capacity_reconciler_mode` from `ORCHESTRATOR_CAPACITY_RECONCILER_MODE`, accepting `off`, `shadow`, and `authoritative`.
2. Default to `shadow` only when the migration is expected to exist; otherwise consider `off` until migration deployment is confirmed.

### Step 9: Wire shadow mode into the cycle (`gpu_orchestrator/control_loop.py`)
**Scope:** Large
1. Capture an immutable `Observed` snapshot at the agreed point in the cycle.
2. In shadow mode, run `OBSERVE -> PLAN -> ADMIT -> RECORD -> shadow log` and then let legacy capacity behavior continue for rollout comparison.
3. Ensure shadow mode never calls RunPod, never mutates worker rows, and never updates backoff state.
4. Emit shadow logs to `system_logs` with `source_type='orchestrator_gpu'` and root metadata keys like `shadow_intended_action`, `pool`, `route_key`, `desired_capacity`, `effective_capacity`, and `cycle_id`.

### Step 10: Wire authoritative mode and legacy cleanup (`gpu_orchestrator/control_loop.py`)
**Scope:** Large
1. In authoritative mode, run the reconciler instead of `_calculate_scaling_decision()` / `_execute_scaling()` for capacity actions.
2. Keep lifecycle probing, heartbeat promotion, health checks, stale route draining, error cleanup, orphan task reconciliation, and periodic zombie/storage checks.
3. Delete `_handle_early_termination` in the final cutover cleanup slice and remove any call sites.
4. Remove `gpu_orchestrator/control/phases/` in the final cleanup slice and rewrite/delete all tests that import it.

## Phase 5: Debugging And Audit Surfaces

### Step 11: Add divergence SQL (`sql/` or `scripts/`)
**Scope:** Medium
1. Commit a query that compares shadow intended actions from `system_logs` with actual worker mutations from `workers.metadata.termination_reason`, created workers, and status changes.
2. Include per-pool counts for `shadow_desired`, legacy action, backoff suppression, and cancel recommendations.

### Step 12: Extend app debug audit (`/Users/peteromalley/Documents/reigh-workspace/reigh-app/scripts/debug.py`, `reigh-app/scripts/debug/commands/scaling_audit.py`)
**Scope:** Medium
1. Keep the existing `scaling-audit` entry point and extend the command implementation to query `worker_capacity_intents`.
2. Report shadow vs authoritative counts, divergence counts, per-pool desired stability, backoff windows, and latest adoption intents.
3. Preserve current churn/utilization report output so existing debugging workflows still work.

## Phase 6: Tests And Deletion Cleanup

### Step 13: Add cycle-trace tests (`tests/control/test_capacity_reconciler.py`)
**Scope:** Large
1. Test queued snapshot gate: queued drops to zero while one active worker has five in-progress tasks and a spawning replacement exists; no `CancelPendingSpawn`.
2. Test fake-clock backoff: four failures yield 30s, 60s, 120s, 240s cooldowns; fifth success resets counter.
3. Test cancel hysteresis with both `2 * orchestrator_poll_sec` and `valid_until` branches.
4. Test route-stale worker exclusion from fresh capacity.
5. Test cutover adoption for pre-existing spawning workers with no prior intent.
6. Test replay fixture for the 2026-05-13 pathological state and assert no oscillation or premature cancel.

### Step 14: Add integration and concurrency tests (`tests/control/`)
**Scope:** Medium
1. Test shadow mode with mocked DB/RunPod and assert no side-effect methods are awaited.
2. Test record-before-act ordering in authoritative mode.
3. Test two reconciler instances against the same fake DB/lease: only one authoritative actor runs side effects.
4. Test partial act idempotency: if one of two spawns already exists with `capacity_intent_id`, the next reconcile does not double-spawn.

### Step 15: Update existing tests after phase deletion (`tests/gpu_orchestrator/control/phases/`, `tests/test_direct_control_health.py`, sprint fixture tests)
**Scope:** Medium
1. Delete phase-mixin smoke tests that only assert mixin existence.
2. Move valuable behavior tests to the new reconciler tests.
3. Update sprint rollback/staging tests that import `WorkerCapacityPhaseMixin` to use production `OrchestratorControlLoop` helpers or reconciler action primitives.

## Execution Order
1. Land schema/DB helpers first, with no runtime behavior change.
2. Land pure planning types and unit tests before control-loop wiring.
3. Land shadow-mode wiring with side-effect tests and divergence audit.
4. Deploy shadow, collect at least 24h and at least 50 meaningful capacity events.
5. Flip authoritative mode after replay and divergence gates pass.
6. Delete legacy `_handle_early_termination`, delete `control/phases/`, and remove shadow-only cleanup after the one-day post-cutover window.

## Validation Order
1. `python -m pytest tests/control/test_capacity_reconciler.py -q`
2. `python -m pytest tests/test_scaling_decision_spawn.py tests/test_scaling_decision_limits.py tests/test_route_stale_worker_state.py -q`
3. `python -m pytest tests/gpu_orchestrator tests/test_module_import_coverage.py -q`
4. `python -m py_compile /Users/peteromalley/Documents/reigh-workspace/reigh-app/scripts/debug.py /Users/peteromalley/Documents/reigh-workspace/reigh-app/scripts/debug/commands/scaling_audit.py`
5. `python -m pytest -q`
6. Grep gates: `_handle_early_termination` has zero matches; `gpu_orchestrator/control/phases` is absent; no remaining imports of `gpu_orchestrator.control.phases`.

# Implementation Plan: Unified Capacity Reconciler

## Overview
The production path remains `gpu_orchestrator/main.py` -> `gpu_orchestrator/control_loop.py::OrchestratorControlLoop`; the duplicate `gpu_orchestrator/control/phases/` tree is test-only and will be deleted in the final cleanup slice. The critique does not change the root cause: the production loop still has two capacity controllers, with `_handle_early_termination()` mutating spawning workers before the later scaling decision re-derives capacity from a different formula. The flags identify safety boundaries the first plan under-specified: route keys, excluded workers, cutover slice gates, migration deployment, spawn idempotency, and cancel semantics.

The revised approach keeps the durable-intent reconciler design, but makes these decisions explicit:

1. Route-aware intent rows are required. The reconciler writes one intent per `(pool, route_key)` when route keys are present in `selected_pool_totals.route_keys` / `route_totals`; `route_key = NULL` is only a fallback when no route key is available.
2. `excluded_from_capacity_control` remains a hard capacity boundary. Excluded live-test or harness workers are not counted, spawned for, cancelled, or terminated by capacity reconciliation; periodic cleanup still sees them through the existing cleanup path.
3. The rollout is three deployable slices with separate gates: shadow, authoritative cutover, and final cleanup. Grep-zero deletion gates apply only to the final cleanup slice.
4. Phase A intentionally reorders lifecycle probing before legacy capacity actions so shadow observation is post-lifecycle and pre-capacity, as the brief requires.
5. The single-writer mechanism is a lease table rather than advisory locks because Supabase/PostgREST cannot hold a session lock across external RunPod calls.
6. Authoritative mode resolves the ACT/RECORD contradiction in favor of invariant 7: it records the admitted intent before side effects, then updates that same row with action outcomes/backoff after side effects.

## Phase 1: Schema, Migration Registration, And Deployment Gate

### Step 1: Add capacity-intent and lease schema (`sql/20260514000000_create_worker_capacity_intents.sql`)
**Scope:** Medium
1. Create `worker_capacity_intents` with the brief's required columns and `worker_capacity_intents_pool_route_created` index.
2. Treat `(pool, route_key)` as the logical admission/backoff key. Use nullable `route_key` only for legacy or unavailable route data.
3. Add an `orchestrator_leases` table with `pool`, `route_key`, `owner_id`, `expires_at`, and `updated_at`.
4. Add lease acquire/release RPCs with TTL semantics. Do not use advisory locks; the lease must survive multiple Supabase HTTP calls while RunPod side effects execute.
5. Keep `system_logs.valid_source_type` unchanged; shadow and authoritative audit logs continue to use `source_type='orchestrator_gpu'`.

### Step 2: Register and gate the migration (`scripts/apply_sql_migrations.py`, `scripts/show_migrations.py`, release notes or script docs)
**Scope:** Small
1. Update hardcoded migration lists or replace them with sorted `sql/*.sql` discovery.
2. Add an explicit pre-deploy gate: the capacity reconciler can run only after the migration has been applied and verified.
3. Do not add automatic DDL execution to `gpu_orchestrator/railway.json` startup; Railway currently starts the long-running orchestrator, and boot-time schema mutation would add operational risk. Instead, document/run the existing migration path before setting `ORCHESTRATOR_CAPACITY_RECONCILER_MODE=shadow`.
4. Add a cheap schema verification step, either in the migration script output or a small query/check, that confirms `worker_capacity_intents` and `orchestrator_leases` exist.

### Step 3: Add database helpers (`gpu_orchestrator/database.py`)
**Scope:** Medium
1. Add helpers to insert one intent row, update that same row with spawn outcomes/backoff fields, read latest intents by `(pool, route_key, shadow)`, acquire/release leases, and query workers by `metadata.capacity_intent_id`.
2. Add direct `insert_system_log()` support so shadow metadata lands at root keys like `metadata.shadow_intended_action`, not nested under `metadata.extra` by `DatabaseLogHandler`.
3. Add a capacity-specific worker creation helper or extend `create_worker_record()` to accept initial `status`, route metadata, `capacity_intent_id`, `capacity_action_ordinal`, and `route_key` before RunPod launch.

## Phase 2: Route-Aware Observation And Pure Planning

### Step 4: Add reconciler data types (`gpu_orchestrator/control/capacity_types.py`)
**Scope:** Medium
1. Define frozen dataclasses for `Observed`, `RouteObserved`, `CapacityPlan`, `AdmittedPlan`, `ReconciliationResult`, `IntentRecord`, and actions: `Spawn`, `CancelPendingSpawn`, `TerminateIdle`, `Noop`.
2. Include `Clock` injection with a production UTC implementation and fake-clock test implementation. `plan()` and `admit()` must not call `datetime.now()` directly.
3. `Observed` must carry `included_worker_states`, `excluded_worker_states`, route keys, per-route task counts, pool, cycle id, snapshot time, and selected route contract metadata.
4. `RouteObserved` must carry route-scoped queued/in-progress counts, usable active/idle/busy/spawning counts, route-stale counts, and effective capacity.

### Step 5: Preserve the excluded-worker boundary (`gpu_orchestrator/control_loop.py`, `gpu_orchestrator/live_test_workers.py`)
**Scope:** Medium
1. Keep the existing `partition_capacity_workers()` behavior before capacity observation.
2. Excluded workers must not contribute to desired/effective capacity and must never be targets of `Spawn`, `CancelPendingSpawn`, or `TerminateIdle` from the reconciler.
3. Periodic cleanup and failsafe stale-worker checks continue to receive excluded workers through the existing cleanup path.
4. Add a test fixture with an excluded spawning worker to prove the reconciler does not count or mutate it.

### Step 6: Build route-aware observation (`gpu_orchestrator/control/capacity_reconciler.py`, `gpu_orchestrator/database.py`)
**Scope:** Large
1. Derive route keys from `detailed_counts.selected_pool_totals.route_keys` and route-scoped rows in `detailed_counts.route_totals`.
2. When route totals are available, create one `RouteObserved` per selected route key using that route's queued/active/potentially-claimable counts.
3. When route totals are absent but exactly one selected route key exists, attribute aggregate selected-pool demand to that route key and mark the observation as `route_counts_inferred=true` in metadata.
4. When no route key is available, use `route_key=NULL` and emit a warning/audit tag. This is a fallback, not the normal path.
5. Extract worker route affinity from worker metadata / route contract where available; if existing legacy spawning workers have no route key at cutover, adopt them as shared pool capacity and do not cancel them during the first authoritative observe.

### Step 7: Implement pure plan computation (`gpu_orchestrator/control/capacity_plan.py`)
**Scope:** Large
1. Implement `build_capacity_plan(route_observed, config, previous_intent, clock)` as pure logic.
2. Use one desired-capacity formula per route: `desired = min(max_active_gpus, max(min_active_gpus, busy_count + queued_count, busy_count + max(machines_to_keep_idle, 1)))`.
3. Count spawning workers as capacity unless terminal, route-stale for the planned route, over the configured initializing/spawning timeout, or already admitted for cancellation.
4. Exclude route-stale workers from fresh-route effective capacity while preserving existing idle route-stale drain behavior outside the plan.
5. Implement time-based cancel hysteresis: `CancelPendingSpawn` only after desired <= effective across at least two snapshots spanning at least `2 * orchestrator_poll_sec`, or after previous `valid_until` elapsed.
6. Active workers are only candidates for `TerminateIdle` after idle-timeout checks and min/buffer constraints. Busy active workers are never killed by capacity pressure.

## Phase 3: Runtime Reconciler And Side-Effect Safety

### Step 8: Implement reconciler orchestration (`gpu_orchestrator/control/capacity_reconciler.py`)
**Scope:** Large
1. Implement explicit methods for `observe`, `plan`, `admit`, `record_pre_act`, `act`, and `record_outcome`.
2. Shadow mode runs observe -> plan -> admit -> record -> shadow log and never calls RunPod, mutates workers, acquires leases, or updates backoff state.
3. Authoritative mode acquires the `(pool, route_key)` lease before final admit/record/act. If the lease is unavailable, record/log a suppressed action and skip side effects.
4. `admit()` suppresses `Spawn` while `next_spawn_allowed_at > clock.now()` for that exact `(pool, route_key)`.
5. `record_pre_act()` must succeed before any authoritative side effect. `record_outcome()` updates the same row after spawn success/failure or cancel/terminate results.
6. On spawn failure, update `consecutive_spawn_failures += 1` and `next_spawn_allowed_at = now + min(2**N * 30s, 600s)`. On successful spawn for the same `(pool, route_key)`, reset the counter to 0.

### Step 9: Make spawn idempotency explicit (`gpu_orchestrator/control_loop.py`, `gpu_orchestrator/database.py`, `gpu_orchestrator/worker_spawner.py`)
**Scope:** Medium
1. Replace the reconciler's use of legacy `_spawn_worker()` with a capacity-aware spawn helper, or extend `_spawn_worker()` so it accepts `capacity_intent_id`, `capacity_action_ordinal`, and `route_key`.
2. Insert the worker row as `status='spawning'` with intent/action metadata before calling RunPod.
3. Before launching a pod, check whether a worker already exists for the same `capacity_intent_id` and action ordinal; if it does, treat that action as already applied.
4. Preserve existing RunPod primitive calls through `WorkerSpawnerAdapter.spawn_worker()` / `start_worker_process()` rather than duplicating RunPod API code.
5. Keep `CycleSummary.workers_spawned` semantics: increment only after a spawn action successfully creates/adopts a worker that should count as spawned.

### Step 10: Define cancellation and termination semantics (`gpu_orchestrator/control_loop.py`, `gpu_orchestrator/control/capacity_reconciler.py`)
**Scope:** Medium
1. `CancelPendingSpawn` must use a dedicated helper, not plain `_terminate_worker()` semantics.
2. For pending-spawn cancellation, attempt RunPod termination when a `runpod_id` exists, but always mark the worker row terminal with `termination_reason='capacity_reconciler_cancel_pending_spawn'`, `capacity_intent_id`, and `runpod_termination_success=true|false`.
3. If RunPod termination fails, emit an error log with enough metadata for zombie cleanup/manual follow-up, and prevent the worker from continuing to count as capacity in the next cycle.
4. `TerminateIdle` for active idle workers may reuse `_terminate_worker()` or a small wrapper, but it must respect idle timeout, min floor, and idle buffer constraints.

### Step 11: Implement cutover adoption (`gpu_orchestrator/control/capacity_reconciler.py`)
**Scope:** Medium
1. On first authoritative observation for a `(pool, route_key)` with existing spawning workers and no prior authoritative intent, write a synthetic adoption intent.
2. Adopted spawning workers count toward effective capacity and cannot be cancelled in the same observation solely because no prior intent exists.
3. Legacy spawning workers without route-key metadata are counted as shared pool capacity for the first authoritative cycle and receive route attribution only when enough metadata exists.

## Phase 4: Control Loop Integration And Deployable Slices

### Step 12: Add feature flag plumbing (`gpu_orchestrator/config.py`)
**Scope:** Small
1. Add `capacity_reconciler_mode` from `ORCHESTRATOR_CAPACITY_RECONCILER_MODE`, accepting `off`, `shadow`, and `authoritative`.
2. Default to `off` until the migration is applied. Shadow mode is enabled only by explicit env var.
3. Log the mode at startup with the rest of `OrchestratorConfig`.

### Step 13: Shadow slice integration (`gpu_orchestrator/control_loop.py`)
**Scope:** Large
1. Reorder the cycle for the shadow slice to run lifecycle probing before capacity actions: fetch/derive included workers, handle spawning workers, health/error/orphan maintenance as currently appropriate, then capture an immutable post-lifecycle/pre-capacity `Observed` snapshot.
2. Run shadow reconciliation from that frozen snapshot before legacy `_handle_early_termination()` and `_calculate_scaling_decision()`.
3. Let legacy capacity actions remain authoritative in the shadow slice so divergence audit is meaningful.
4. Add tests proving shadow observation happens before legacy capacity mutations and after lifecycle status updates.

### Step 14: Authoritative cutover slice (`gpu_orchestrator/control_loop.py`)
**Scope:** Large
1. In authoritative mode, call the reconciler for capacity actions and skip legacy `_handle_early_termination()`, `_calculate_scaling_decision()`, and `_execute_scaling()`.
2. Keep lifecycle probing, heartbeat promotion, health checks, stale route draining, error cleanup, orphan task reconciliation, and periodic zombie/storage checks intact.
3. Keep the legacy methods present but unreachable in this slice so rollback can set mode back to `shadow` or `off` during the first 24 hours.
4. Require cutover gates: migration verified, shadow ran at least 24 hours, at least 50 meaningful capacity events observed or shadow window extended, replay test passed, no correct-legacy divergence unresolved, and cutover adoption test passed.

### Step 15: Final cleanup slice (`gpu_orchestrator/control_loop.py`, `gpu_orchestrator/control/phases/`, tests)
**Scope:** Medium
1. After the post-cutover observation window, delete `_handle_early_termination()` and all call sites. The grep-zero criterion applies here.
2. Delete `gpu_orchestrator/control/phases/` and remove or rewrite every test importing `gpu_orchestrator.control.phases`.
3. Remove unreachable legacy capacity stubs only after authoritative mode has been stable for the agreed window.
4. Keep shadow intent recording code if it remains useful for audit, but remove shadow-only comparison code that depends on legacy actions.

## Phase 5: Audit And Debug Surfaces

### Step 16: Add divergence SQL (`sql/` or `scripts/`)
**Scope:** Medium
1. Commit a query comparing shadow intended actions from `system_logs` and `worker_capacity_intents` with legacy actual worker mutations from `workers` and existing lifecycle logs.
2. Group by `(pool, route_key)` and report shadow desired, effective capacity, legacy spawns/terminations, backoff suppressions, cancel recommendations, and route-count fallback cases.
3. Dedupe shadow rows where multiple replicas emit the same cycle observation; authoritative rows should be single-writer by lease.

### Step 17: Extend app debug audit (`/Users/peteromalley/Documents/reigh-workspace/reigh-app/scripts/debug.py`, `reigh-app/scripts/debug/commands/scaling_audit.py`)
**Scope:** Medium
1. Keep the existing `scaling-audit` entry point and extend the command implementation to query `worker_capacity_intents`.
2. Report shadow vs authoritative cycles, divergence counts, per-route intent stability, backoff windows, route-key fallback counts, latest adoption intents, and lease-suppressed actions.
3. Preserve the existing churn/utilization report so current debugging workflows still work.

## Phase 6: Tests And Verification

### Step 18: Add route-aware cycle-trace tests (`tests/control/test_capacity_reconciler.py`)
**Scope:** Large
1. Queued snapshot gate: queued drops to zero while one active worker has five in-progress tasks and a spawning replacement exists; no `CancelPendingSpawn`.
2. Spawn backoff with fake clock: four failures on `(pool, route_a)` yield 30s, 60s, 120s, 240s cooldowns; a concurrent `(pool, route_b)` spawn is not suppressed; fifth route_a success resets the counter.
3. Cancel hysteresis: test both `2 * orchestrator_poll_sec` and `valid_until` branches.
4. Route-stale worker exclusion: a route-stale worker does not satisfy fresh-route desired capacity.
5. Cutover adoption: pre-existing spawning workers with no authoritative intent are adopted and not cancelled.
6. Excluded worker boundary: excluded spawning/active workers are not counted or mutated by capacity reconciliation.
7. Replay fixture: synthetic or captured 2026-05-13 incident state produces no oscillation and no premature cancel.

### Step 19: Add runtime safety tests (`tests/control/`)
**Scope:** Medium
1. Shadow mode asserts no RunPod calls, no worker mutations, no backoff updates, and no lease acquisition.
2. Authoritative record-before-act ordering is enforced.
3. Two reconciler instances against the same fake DB/lease cannot both execute side effects for the same `(pool, route_key)`.
4. Partial act idempotency: an existing worker tagged with the same `capacity_intent_id` and action ordinal prevents duplicate spawn.
5. Cancel-pending-spawn marks the worker terminal even when RunPod termination returns false, and logs the failed RunPod termination.
6. Control-loop ordering test proves shadow observation is post-lifecycle and pre-legacy-capacity in the shadow slice.

### Step 20: Update existing tests after phase deletion (`tests/gpu_orchestrator/control/phases/`, `tests/test_direct_control_health.py`, sprint fixture tests)
**Scope:** Medium
1. Delete phase-mixin smoke tests that only assert mixin existence.
2. Move behavior coverage worth keeping to `tests/control/test_capacity_reconciler.py` or production `control_loop.py` tests.
3. Update sprint rollback/staging tests that import `WorkerCapacityPhaseMixin` to use the production loop or reconciler action primitives.
4. Keep existing route-aware task count tests and extend them where needed for route-intent observation.

## Execution Order
1. Apply schema, DB helpers, and migration registration first. Do not enable runtime mode until schema verification passes.
2. Add pure types/planning and route-aware tests before control-loop wiring.
3. Land the shadow slice: lifecycle-before-capacity ordering, post-lifecycle immutable observation, shadow records/logs, legacy capacity still authoritative.
4. Run shadow in production for at least 24 hours and at least 50 meaningful capacity events, extending the window if needed.
5. Land authoritative cutover: reconciler executes capacity actions under leases; legacy capacity actions are skipped but still present for rollback.
6. After the post-cutover observation window, land final cleanup: delete `_handle_early_termination`, delete `control/phases/`, remove rewritten tests/imports.

## Validation Order
1. `python -m pytest tests/control/test_capacity_reconciler.py -q`
2. `python -m pytest tests/test_scaling_decision_spawn.py tests/test_scaling_decision_limits.py tests/test_route_stale_worker_state.py tests/gpu_orchestrator/test_route_aware_task_counts.py -q`
3. `python -m pytest tests/gpu_orchestrator tests/test_module_import_coverage.py -q`
4. `python -m py_compile /Users/peteromalley/Documents/reigh-workspace/reigh-app/scripts/debug.py /Users/peteromalley/Documents/reigh-workspace/reigh-app/scripts/debug/commands/scaling_audit.py`
5. `python -m pytest -q`
6. Shadow-slice gate: schema verified, shadow has no side effects, observation ordering test passes, divergence SQL/debug output includes `(pool, route_key)`.
7. Authoritative-slice gate: record-before-act, lease, idempotency, route backoff, adoption, and replay tests pass.
8. Final-cleanup gate: `rg "_handle_early_termination"` returns zero matches; `test ! -d gpu_orchestrator/control/phases`; `rg "gpu_orchestrator\.control\.phases"` returns zero matches; full tests pass.

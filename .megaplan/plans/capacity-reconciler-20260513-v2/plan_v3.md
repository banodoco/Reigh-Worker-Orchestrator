# Implementation Plan: Unified Pool Capacity Reconciler

## Overview
The production path remains `gpu_orchestrator/main.py` -> `gpu_orchestrator/control_loop.py::OrchestratorControlLoop`; the duplicate `gpu_orchestrator/control/phases/` tree is test-only and will be deleted in the final cleanup slice. The latest critique does not change the root cause: the production loop still has two capacity controllers, with `_handle_early_termination()` mutating spawning workers before the later scaling decision re-derives capacity from a different formula.

The second critique does identify a design correction: route keys are required for backoff and audit, but capacity budgeting is pool-wide in this repository. `min_active_gpus`, `machines_to_keep_idle`, and `max_active_gpus` are loaded once for the orchestrator/pool, so the reconciler must not run independent per-route planners. The revised design is therefore a single pool-level reconciler that observes route-scoped demand and route-scoped backoff state inside one pool budget and one pool lease.

Settled decisions for execution:

1. Capacity planning and authoritative locking are pool-scoped. There is one pool capacity plan per cycle, one authoritative pool lease, and one pool-level max/min/buffer budget.
2. Route keys remain part of admission and audit. Spawn demand is attributed to route keys, and `spawn_failed` backoff is keyed by `(pool, route_key)`, but route demand competes inside the single pool budget.
3. `worker_capacity_intents` writes exactly one pool summary row per cycle, with `route_key=NULL`. Route-level backoff state is stored separately in a small route-backoff table or equivalent migration-owned state, not by multiplying capacity intent rows.
4. `excluded_from_capacity_control` remains a hard boundary. Excluded live-test or harness workers are not counted, spawned for, cancelled, or terminated by capacity reconciliation; periodic cleanup still sees them through the existing cleanup path.
5. The rollout is three deployable slices with separate gates: shadow, authoritative cutover, and final cleanup. Grep-zero deletion gates apply only to the final cleanup slice.
6. Phase A intentionally reorders lifecycle probing before legacy capacity actions so shadow observation is post-lifecycle and pre-capacity, as the brief requires.
7. Authoritative mode resolves the ACT/RECORD contradiction in favor of invariant 7: it records the admitted pool intent before side effects, then updates that same row and route-backoff state with action outcomes after side effects.

## Phase 1: Schema, Migration Registration, And Deployment Gate

### Step 1: Add capacity-intent, route-backoff, and pool-lease schema (`sql/20260514000000_create_worker_capacity_intents.sql`)
**Scope:** Medium
1. Create `worker_capacity_intents` with the brief's required columns and `worker_capacity_intents_pool_route_created` index.
2. Use `worker_capacity_intents` as the pool audit trail: exactly one summary row per pool per cycle, normally with `route_key=NULL`, `desired_capacity` equal to the pool-level desired capacity, and observed counts equal to pool totals.
3. Add a small `worker_capacity_route_backoffs` table keyed by `(pool, route_key)` with `consecutive_spawn_failures`, `next_spawn_allowed_at`, `updated_at`, and last failure/success metadata. Use `route_key=NULL` only for pool-floor spawn actions that are not attributable to queued route demand.
4. Add an `orchestrator_leases` table keyed by `pool`, not route, with `owner_id`, `expires_at`, and `updated_at`.
5. Add lease acquire/release RPCs with TTL semantics. Do not use advisory locks; the lease must survive multiple Supabase HTTP calls while RunPod side effects execute.
6. Keep `system_logs.valid_source_type` unchanged; shadow and authoritative audit logs continue to use `source_type='orchestrator_gpu'`.

### Step 2: Register and gate the migration (`scripts/apply_sql_migrations.py`, `scripts/show_migrations.py`, release notes or script docs)
**Scope:** Small
1. Update hardcoded migration lists or replace them with sorted `sql/*.sql` discovery.
2. Add an explicit pre-deploy gate: the capacity reconciler can run only after the migration has been applied and verified.
3. Do not add automatic DDL execution to `gpu_orchestrator/railway.json` startup; Railway currently starts the long-running orchestrator, and boot-time schema mutation would add operational risk. Instead, document/run the existing migration path before setting `ORCHESTRATOR_CAPACITY_RECONCILER_MODE=shadow`.
4. Add a cheap schema verification step confirming `worker_capacity_intents`, `worker_capacity_route_backoffs`, and `orchestrator_leases` exist.

### Step 3: Add database helpers (`gpu_orchestrator/database.py`)
**Scope:** Medium
1. Add helpers to insert one pool intent row, update that same row with action outcome totals, read latest pool intent by `(pool, shadow)`, and acquire/release pool leases.
2. Add helpers to read/update route backoff state by `(pool, route_key)`.
3. Add direct `insert_system_log()` support so shadow metadata lands at root keys like `metadata.shadow_intended_action`, not nested under `metadata.extra` by `DatabaseLogHandler`.
4. Add a capacity-specific worker creation helper or extend `create_worker_record()` to accept initial `status`, route/action metadata, `capacity_intent_id`, `capacity_action_ordinal`, and `spawn_reason_route_key` before RunPod launch.
5. Add worker lookup helpers needed for idempotency, especially querying workers by `metadata.capacity_intent_id` and `metadata.capacity_action_ordinal`.

## Phase 2: Pool Observation With Route Demand

### Step 4: Add reconciler data types (`gpu_orchestrator/control/capacity_types.py`)
**Scope:** Medium
1. Define frozen dataclasses for `ObservedPool`, `RouteDemand`, `CapacityPlan`, `AdmittedPlan`, `ReconciliationResult`, `IntentRecord`, and actions: `Spawn`, `CancelPendingSpawn`, `TerminateIdle`, `Noop`.
2. Include `Clock` injection with a production UTC implementation and fake-clock test implementation. `plan()` and `admit()` must not call `datetime.now()` directly.
3. `ObservedPool` must carry included worker states, excluded worker states, route demands, pool-level active/idle/busy/spawning counts, route-stale counts, pool, cycle id, snapshot time, and selected route contract metadata.
4. `RouteDemand` must carry route key, queued/potentially-claimable count, in-progress count if available, backoff state, and whether route counts were exact or inferred.

### Step 5: Preserve the excluded-worker boundary (`gpu_orchestrator/control_loop.py`, `gpu_orchestrator/live_test_workers.py`)
**Scope:** Medium
1. Keep the existing `partition_capacity_workers()` behavior before capacity observation.
2. Excluded workers must not contribute to desired/effective capacity and must never be targets of `Spawn`, `CancelPendingSpawn`, or `TerminateIdle` from the reconciler.
3. Periodic cleanup and failsafe stale-worker checks continue to receive excluded workers through the existing cleanup path.
4. Add a test fixture with an excluded spawning worker to prove the reconciler does not count or mutate it.

### Step 6: Build pool observation with route demand (`gpu_orchestrator/control/capacity_reconciler.py`, `gpu_orchestrator/database.py`)
**Scope:** Large
1. Derive route demands from `detailed_counts.selected_pool_totals.route_keys` and route-scoped rows in `detailed_counts.route_totals`.
2. When route totals are available, preserve each selected route's queued/potentially-claimable demand for action attribution and backoff checks.
3. When route totals are absent but exactly one selected route key exists, attribute aggregate selected-pool demand to that route key and mark `route_counts_inferred=true` in audit metadata.
4. When no route key is available, use `route_key=NULL` for demand attribution and emit a warning/audit tag. This is a fallback, not the normal path.
5. Do not require worker route affinity for normal capacity math. The current worker metadata is primarily pool/selector/contract scoped; route keys come from task demand. Existing legacy spawning workers without route-key metadata at cutover are adopted as pool capacity and are not cancelled during first authoritative observe.
6. Compute pool-level effective capacity from included, non-terminal, non-route-stale workers only; route demand is folded into the pool-level queued total for desired-capacity math.

### Step 7: Implement pure pool plan computation (`gpu_orchestrator/control/capacity_plan.py`)
**Scope:** Large
1. Implement `build_capacity_plan(observed_pool, config, previous_pool_intent, route_backoffs, clock)` as pure logic.
2. Apply pool-level min, idle buffer, and max exactly once: `desired = min(max_active_gpus, max(min_active_gpus, busy_count + eligible_queued_total, busy_count + max(machines_to_keep_idle, 1)))`.
3. `eligible_queued_total` is the sum of queued/potentially-claimable route demand that is not suppressed by route backoff. Suppressed routes remain visible in the plan/audit but do not generate spawn demand while cooling down.
4. Count spawning workers as pool capacity unless terminal, route-stale for the selected pool contract, over the configured initializing/spawning timeout, or already admitted for cancellation.
5. Allocate any `Spawn` action budget across route demands after computing the single pool spawn budget. Prioritize uncovered non-backoff route demand, then pool-floor/buffer spawns with `route_key=NULL`.
6. Exclude route-stale workers from fresh-route/pool effective capacity while preserving existing idle route-stale drain behavior outside the plan.
7. Implement time-based cancel hysteresis at the pool level: `CancelPendingSpawn` only after desired <= effective across at least two snapshots spanning at least `2 * orchestrator_poll_sec`, or after previous `valid_until` elapsed.
8. Active workers are only candidates for `TerminateIdle` after idle-timeout checks and min/buffer constraints. Busy active workers are never killed by capacity pressure.

## Phase 3: Runtime Reconciler And Side-Effect Safety

### Step 8: Implement reconciler orchestration (`gpu_orchestrator/control/capacity_reconciler.py`)
**Scope:** Large
1. Implement explicit methods for `observe`, `plan`, `admit`, `record_pre_act`, `act`, and `record_outcome`.
2. Shadow mode runs observe -> plan -> admit -> record -> shadow log and never calls RunPod, mutates workers, acquires leases, or updates route backoff state.
3. Authoritative mode acquires the pool lease before final admit/record/act. If the pool lease is unavailable, record/log a lease-suppressed result and skip side effects.
4. `admit()` suppresses only route-attributed `Spawn` actions whose `(pool, route_key)` has `next_spawn_allowed_at > clock.now()`. Backoff on one route must not suppress other route demand in the same pool.
5. `record_pre_act()` must insert the single pool summary intent before any authoritative side effect. `record_outcome()` updates that row after spawn success/failure or cancel/terminate results.
6. On route-attributed spawn failure, update `worker_capacity_route_backoffs` for that `(pool, route_key)` with `consecutive_spawn_failures += 1` and `next_spawn_allowed_at = now + min(2**N * 30s, 600s)`. On successful spawn for that key, reset the route counter to 0.
7. Pool-floor spawns with `route_key=NULL` use the nullable route-backoff key and are reported separately in audit output.

### Step 9: Make spawn idempotency explicit (`gpu_orchestrator/control_loop.py`, `gpu_orchestrator/database.py`, `gpu_orchestrator/worker_spawner.py`)
**Scope:** Medium
1. Replace the reconciler's use of legacy `_spawn_worker()` with a capacity-aware spawn helper, or extend `_spawn_worker()` so it accepts `capacity_intent_id`, `capacity_action_ordinal`, and `spawn_reason_route_key`.
2. Insert the worker row as `status='spawning'` with intent/action metadata before calling RunPod.
3. Before launching a pod, check whether a worker already exists for the same `capacity_intent_id` and action ordinal; if it does, treat that action as already applied.
4. Preserve existing RunPod primitive calls through `WorkerSpawnerAdapter.spawn_worker()` / `start_worker_process()` rather than duplicating RunPod API code.
5. Keep `CycleSummary.workers_spawned` semantics: increment only after a spawn action successfully creates/adopts a worker that should count as spawned.

### Step 10: Define cancellation, termination, and summary semantics (`gpu_orchestrator/control_loop.py`, `gpu_orchestrator/control/capacity_reconciler.py`)
**Scope:** Medium
1. `CancelPendingSpawn` must use a dedicated helper, not plain `_terminate_worker()` semantics.
2. For pending-spawn cancellation, attempt RunPod termination when a `runpod_id` exists, but always mark the worker row terminal with `termination_reason='capacity_reconciler_cancel_pending_spawn'`, `capacity_intent_id`, and `runpod_termination_success=true|false`.
3. If RunPod termination fails, emit an error log with enough metadata for zombie cleanup/manual follow-up, and prevent the worker from continuing to count as capacity in the next cycle.
4. `TerminateIdle` for active idle workers may reuse `_terminate_worker()` or a small wrapper, but it must respect idle timeout, min floor, and idle buffer constraints.
5. Reconciler results must carry `workers_spawned`, `workers_terminated`, `workers_failed`, `spawns_suppressed_by_backoff`, and `actions_suppressed_by_lease` counts.
6. `control_loop.py` must fold reconciler result counters into `CycleSummary` so health monitoring and structured cycle summaries remain accurate for reconciler-driven `CancelPendingSpawn` and `TerminateIdle`.

### Step 11: Implement cutover adoption (`gpu_orchestrator/control/capacity_reconciler.py`)
**Scope:** Medium
1. On first authoritative observation for a pool with existing spawning workers and no prior authoritative pool intent, write a synthetic adoption intent.
2. Adopted spawning workers count toward effective pool capacity and cannot be cancelled in the same observation solely because no prior intent exists.
3. Legacy spawning workers without route-key metadata are counted as shared pool capacity for the first authoritative cycle; route attribution is only used for new spawn action reasons and route backoff state.

## Phase 4: Control Loop Integration And Deployable Slices

### Step 12: Add feature flag plumbing (`gpu_orchestrator/config.py`)
**Scope:** Small
1. Add `capacity_reconciler_mode` from `ORCHESTRATOR_CAPACITY_RECONCILER_MODE`, accepting `off`, `shadow`, and `authoritative`.
2. Default to `off` until the migration is applied. Shadow mode is enabled only by explicit env var.
3. Log the mode at startup with the rest of `OrchestratorConfig`.

### Step 13: Shadow slice integration (`gpu_orchestrator/control_loop.py`)
**Scope:** Large
1. Reorder the cycle for the shadow slice to run lifecycle probing before capacity actions: fetch/derive included workers, handle spawning workers, health/error/orphan maintenance as currently appropriate, then capture an immutable post-lifecycle/pre-capacity `ObservedPool` snapshot.
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
1. Commit a query comparing shadow intended actions from `system_logs` and pool summary rows in `worker_capacity_intents` with legacy actual worker mutations from `workers` and existing lifecycle logs.
2. Report route-level demand/backoff details from `worker_capacity_route_backoffs` and shadow metadata without treating route rows as separate pool capacity plans.
3. Report pool desired, effective capacity, legacy spawns/terminations, route backoff suppressions, cancel recommendations, lease suppressions, summary counters, and route-count fallback cases.
4. Dedupe shadow rows where multiple replicas emit the same cycle observation; authoritative rows should be single-writer by pool lease.

### Step 17: Extend app debug audit (`/Users/peteromalley/Documents/reigh-workspace/reigh-app/scripts/debug.py`, `reigh-app/scripts/debug/commands/scaling_audit.py`)
**Scope:** Medium
1. Keep the existing `scaling-audit` entry point and extend the command implementation to query `worker_capacity_intents` and route backoff state.
2. Report shadow vs authoritative cycles, divergence counts, pool-level intent stability, route backoff windows, route-key fallback counts, latest adoption intents, summary counters, and lease-suppressed actions.
3. Preserve the existing churn/utilization report so current debugging workflows still work.

## Phase 6: Tests And Verification

### Step 18: Add pool-budget cycle-trace tests (`tests/control/test_capacity_reconciler.py`)
**Scope:** Large
1. Queued snapshot gate: queued drops to zero while one active worker has five in-progress tasks and a spawning replacement exists; no `CancelPendingSpawn`.
2. Spawn backoff with fake clock: four failures on `(pool, route_a)` yield 30s, 60s, 120s, 240s cooldowns; concurrent `(pool, route_b)` demand remains eligible inside the same pool budget; fifth route_a success resets the counter.
3. Pool capacity bounds: multiple selected route keys cannot multiply `min_active_gpus`, `machines_to_keep_idle`, or `max_active_gpus`; the pool desired capacity is capped once.
4. Pool single-writer: two authoritative reconciler instances cannot both execute side effects for the same pool, even if they are acting on different route-attributed demand.
5. Cancel hysteresis: test both `2 * orchestrator_poll_sec` and `valid_until` branches at pool level.
6. Route-stale worker exclusion: a route-stale worker does not satisfy fresh selected-pool capacity.
7. Cutover adoption: pre-existing spawning workers with no authoritative intent are adopted and not cancelled.
8. Excluded worker boundary: excluded spawning/active workers are not counted or mutated by capacity reconciliation.
9. CycleSummary integration: reconciler-driven spawns, pending-spawn cancels, idle terminations, failed spawns, backoff suppressions, and lease suppressions appear in result counters and `CycleSummary` as appropriate.
10. Replay fixture: synthetic or captured 2026-05-13 incident state produces no oscillation and no premature cancel.

### Step 19: Add runtime safety tests (`tests/control/`)
**Scope:** Medium
1. Shadow mode asserts no RunPod calls, no worker mutations, no backoff updates, and no lease acquisition.
2. Authoritative record-before-act ordering is enforced.
3. Partial act idempotency: an existing worker tagged with the same `capacity_intent_id` and action ordinal prevents duplicate spawn.
4. Cancel-pending-spawn marks the worker terminal even when RunPod termination returns false, and logs the failed RunPod termination.
5. Control-loop ordering test proves shadow observation is post-lifecycle and pre-legacy-capacity in the shadow slice.
6. Pool intent invariant: exactly one `worker_capacity_intents` summary row is written per pool per cycle; route backoff updates do not create extra pool intent rows.

### Step 20: Update existing tests after phase deletion (`tests/gpu_orchestrator/control/phases/`, `tests/test_direct_control_health.py`, sprint fixture tests)
**Scope:** Medium
1. Delete phase-mixin smoke tests that only assert mixin existence.
2. Move behavior coverage worth keeping to `tests/control/test_capacity_reconciler.py` or production `control_loop.py` tests.
3. Update sprint rollback/staging tests that import `WorkerCapacityPhaseMixin` to use the production loop or reconciler action primitives.
4. Keep existing route-aware task count tests and extend them where needed for route demand/backoff observation.

## Execution Order
1. Apply schema, DB helpers, and migration registration first. Do not enable runtime mode until schema verification passes.
2. Add pure types/planning and pool-budget tests before control-loop wiring.
3. Land the shadow slice: lifecycle-before-capacity ordering, post-lifecycle immutable observation, shadow records/logs, legacy capacity still authoritative.
4. Run shadow in production for at least 24 hours and at least 50 meaningful capacity events, extending the window if needed.
5. Land authoritative cutover: reconciler executes capacity actions under a pool lease; legacy capacity actions are skipped but still present for rollback.
6. After the post-cutover observation window, land final cleanup: delete `_handle_early_termination`, delete `control/phases/`, remove rewritten tests/imports.

## Validation Order
1. `python -m pytest tests/control/test_capacity_reconciler.py -q`
2. `python -m pytest tests/test_scaling_decision_spawn.py tests/test_scaling_decision_limits.py tests/test_route_stale_worker_state.py tests/gpu_orchestrator/test_route_aware_task_counts.py -q`
3. `python -m pytest tests/gpu_orchestrator tests/test_module_import_coverage.py -q`
4. `python -m py_compile /Users/peteromalley/Documents/reigh-workspace/reigh-app/scripts/debug.py /Users/peteromalley/Documents/reigh-workspace/reigh-app/scripts/debug/commands/scaling_audit.py`
5. `python -m pytest -q`
6. Shadow-slice gate: schema verified, shadow has no side effects, observation ordering test passes, divergence SQL/debug output includes pool summaries and route backoff details.
7. Authoritative-slice gate: record-before-act, pool lease, idempotency, route backoff, pool capacity bounds, summary counters, adoption, and replay tests pass.
8. Final-cleanup gate: `rg "_handle_early_termination"` returns zero matches; `test ! -d gpu_orchestrator/control/phases`; `rg "gpu_orchestrator\.control\.phases"` returns zero matches; full tests pass.

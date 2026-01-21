"""
Main orchestrator control loop for the Runpod GPU Worker Orchestrator.
Handles scaling decisions, health checks, and worker lifecycle management.

REFACTORED: This module now uses:
- OrchestratorConfig for centralized configuration
- WorkerLifecycle/DerivedWorkerState for consistent state derivation
- Split methods for each cycle phase
"""

import os
import asyncio
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

from .config import OrchestratorConfig
from .worker_state import (
    WorkerLifecycle,
    DerivedWorkerState,
    derive_worker_state,
    TaskCounts,
    ScalingDecision,
    CycleSummary,
    calculate_scaling_decision_pure,
)
from .database import DatabaseClient
from .runpod_client import create_runpod_client, spawn_runpod_gpu, terminate_runpod_gpu
from .health_monitor import OrchestratorHealthMonitor
from .logging_config import set_current_worker

logger = logging.getLogger(__name__)


class OrchestratorControlLoop:
    """Main orchestrator control loop."""

    def __init__(self):
        load_dotenv()

        self.db = DatabaseClient()
        self.runpod = create_runpod_client()

        # Load all configuration from environment
        self.config = OrchestratorConfig.from_env()

        # Cycle counter for continuous mode
        self.cycle_count = 0
        # Track recent scale actions to apply grace periods
        self.last_scale_down_at: Optional[datetime] = None
        self.last_scale_up_at: Optional[datetime] = None

        # Storage monitoring tracking
        self.last_storage_check_cycle = 0

        # Health monitor
        self.health_monitor = OrchestratorHealthMonitor()

        # Log configuration at startup
        logger.info(f"OrchestratorControlLoop initialized with scaling: {self.config.min_active_gpus}-{self.config.max_active_gpus} GPUs, idle buffer: {self.config.machines_to_keep_idle}")
        self.config.log_config()

        # Log SSH configuration
        self._log_ssh_config()

    def _log_ssh_config(self):
        """Log SSH configuration for debugging."""
        ssh_private_key = os.getenv("RUNPOD_SSH_PRIVATE_KEY")
        ssh_public_key = os.getenv("RUNPOD_SSH_PUBLIC_KEY")
        ssh_private_key_path = os.getenv("RUNPOD_SSH_PRIVATE_KEY_PATH")

        logger.info("ENV_VALIDATION SSH Configuration:")
        logger.info(f"   - RUNPOD_SSH_PRIVATE_KEY: {'SET' if ssh_private_key else 'MISSING'}")
        logger.info(f"   - RUNPOD_SSH_PUBLIC_KEY: {'SET' if ssh_public_key else 'MISSING'}")
        logger.info(f"   - RUNPOD_SSH_PRIVATE_KEY_PATH: {'SET' if ssh_private_key_path else 'MISSING'}")

        if not ssh_private_key and not ssh_private_key_path:
            logger.error("ENV_VALIDATION CRITICAL: No SSH private key configured!")
            logger.error("This will cause SSH authentication failures and worker terminations")
        elif ssh_private_key:
            logger.info(f"Using SSH key from environment variable ({len(ssh_private_key)} chars)")
        else:
            logger.info(f"Using SSH key from file path: {ssh_private_key_path}")

    # =========================================================================
    # MAIN CYCLE: Coordinates all phases
    # =========================================================================

    async def run_single_cycle(self) -> Dict[str, Any]:
        """
        Run a single orchestrator cycle.
        Returns summary of actions taken.
        """
        cycle_start = datetime.now(timezone.utc)
        self.cycle_count += 1

        # Set cycle number in logging context
        try:
            from .logging_config import set_current_cycle
            set_current_cycle(self.cycle_count)
        except Exception:
            pass

        # Visual separator for continuous mode
        print("\n" + "=" * 80)
        print(f"ORCHESTRATOR CYCLE #{self.cycle_count}")
        print("=" * 80 + "\n")

        logger.info("Starting orchestrator cycle")

        summary = CycleSummary()

        try:
            # Phase 1: Fetch current state
            workers, task_counts, detailed_counts = await self._fetch_current_state()

            # Phase 2: Derive state for all workers
            worker_states = await self._derive_all_worker_states(workers, task_counts.queued)

            # Phase 3: Handle early termination of over-capacity spawning workers
            worker_states = await self._handle_early_termination(worker_states, task_counts, summary)

            # Phase 4: Handle spawning workers (pod init, script launch, promotion)
            await self._handle_spawning_workers(worker_states, task_counts, summary)

            # Phase 5: Health check active workers
            await self._health_check_active_workers(worker_states, task_counts, summary)

            # Phase 6: Cleanup error workers
            await self._cleanup_error_workers(worker_states, summary)

            # Phase 7: Reconcile orphaned tasks
            await self._reconcile_orphaned_tasks(worker_states, summary)

            # Phase 8: Calculate scaling decision
            scaling = await self._calculate_scaling_decision(worker_states, task_counts)

            # Phase 9: Execute scaling
            await self._execute_scaling(scaling, worker_states, task_counts, summary, detailed_counts)

            # Phase 10: Periodic checks (storage, zombies)
            await self._run_periodic_checks(worker_states, summary)

        except Exception as e:
            logger.error(f"Error in orchestrator cycle: {e}")
            summary.error = str(e)

        # Phase 11: Log summary
        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        logger.info(f"Orchestrator cycle completed in {cycle_duration:.2f}s: {summary.to_dict()['actions']}")

        # Health monitoring
        total_workers = len([ws for ws in worker_states if not ws.is_terminal]) if 'worker_states' in dir() else 0
        self.health_monitor.check_scaling_anomaly(
            task_count=task_counts.total if 'task_counts' in dir() else 0,
            worker_count=total_workers,
            workers_spawned=summary.workers_spawned
        )
        self.health_monitor.log_health_summary(self.cycle_count)

        if self.cycle_count % 10 == 0:
            if not self.health_monitor.check_logging_health():
                logger.error("Database logging is degraded - check logs above")

        return {
            "timestamp": cycle_start.isoformat(),
            **summary.to_dict()
        }

    # =========================================================================
    # PHASE 1: Fetch current state
    # =========================================================================

    async def _fetch_current_state(self) -> Tuple[List[Dict], TaskCounts, Optional[Dict]]:
        """Fetch workers and task counts from database. Returns (workers, task_counts, detailed_counts)."""
        # Get ALL workers for comprehensive health checks
        workers = await self.db.get_workers()

        # Get detailed task counts from edge function
        detailed_counts = await self.db.get_detailed_task_counts_via_edge_function()

        if detailed_counts:
            totals = detailed_counts.get('totals', {})
            total_tasks = totals.get('queued_plus_active', 0)
            queued_count = totals.get('queued_only', 0)
            active_cloud_count = totals.get('active_only', 0)

            logger.critical(f"TASK COUNT (Cycle #{self.cycle_count}): Queued={queued_count}, Active={active_cloud_count}, Total={total_tasks}")
            self._log_detailed_task_breakdown(detailed_counts, queued_count, active_cloud_count, total_tasks)
        else:
            logger.warning("Failed to get detailed task counts, falling back to simple counting")
            total_tasks = await self.db.count_available_tasks_via_edge_function(include_active=True)
            queued_count = await self.db.count_available_tasks_via_edge_function(include_active=False)
            active_cloud_count = total_tasks - queued_count
            logger.warning(f"Fallback returned: {total_tasks} total, {queued_count} queued")

        return workers, TaskCounts(queued=queued_count, active_cloud=active_cloud_count, total=total_tasks), detailed_counts

    def _log_detailed_task_breakdown(self, detailed_counts: Dict, queued: int, active: int, total: int):
        """Log detailed task breakdown for debugging."""
        import sys

        print(f"\n{'=' * 80}", file=sys.stderr)
        print(f"ORCHESTRATOR CYCLE #{self.cycle_count} - TASK COUNT", file=sys.stderr)
        print(f"  Queued only: {queued}", file=sys.stderr)
        print(f"  Active (cloud): {active}", file=sys.stderr)
        print(f"  Total workload: {total}", file=sys.stderr)
        print(f"{'=' * 80}\n", file=sys.stderr)

        logger.info("DETAILED TASK BREAKDOWN:")
        logger.info(f"   Queued only: {queued}")
        logger.info(f"   Active (cloud-claimed): {active}")
        logger.info(f"   Total (queued + active): {total}")

        if 'global_task_breakdown' in detailed_counts:
            breakdown = detailed_counts['global_task_breakdown']
            logger.info(f"   In Progress (total): {breakdown.get('in_progress_total', 0)}")
            logger.info(f"   In Progress (cloud): {breakdown.get('in_progress_cloud', 0)}")
            logger.info(f"   In Progress (local): {breakdown.get('in_progress_local', 0)}")
            logger.info(f"   Orchestrator tasks: {breakdown.get('orchestrator_tasks', 0)}")

        users = detailed_counts.get('users', [])
        if users:
            logger.info(f"   Users with tasks: {len(users)}")
            at_limit_users = [u for u in users if u.get('at_limit', False)]
            if at_limit_users:
                logger.info(f"   Users at concurrency limit: {len(at_limit_users)}")

    # =========================================================================
    # PHASE 2: Derive state for all workers
    # =========================================================================

    async def _derive_all_worker_states(
        self,
        workers: List[Dict],
        queued_count: int
    ) -> List[DerivedWorkerState]:
        """Derive state for all workers using batch queries."""
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

    # =========================================================================
    # PHASE 3: Early termination of over-capacity spawning workers
    # =========================================================================

    async def _handle_early_termination(
        self,
        worker_states: List[DerivedWorkerState],
        task_counts: TaskCounts,
        summary: CycleSummary
    ) -> List[DerivedWorkerState]:
        """
        Terminate excess spawning workers before they finish starting up.
        Returns updated worker_states list with terminated workers removed.
        """
        now = datetime.now(timezone.utc)
        config = self.config

        spawning_states = [ws for ws in worker_states if ws.is_spawning]
        active_states = [ws for ws in worker_states if ws.is_active and not ws.is_terminal]

        current_capacity = len(active_states) + len(spawning_states)

        # Calculate early desired workers
        if task_counts.total > 0:
            down_scaled = math.ceil(task_counts.total * config.scale_down_multiplier)
            task_based = max(1, down_scaled)
        else:
            task_based = 0

        early_desired = max(config.min_active_gpus, task_based, config.machines_to_keep_idle)
        early_desired = min(early_desired, config.max_active_gpus)

        if current_capacity <= early_desired:
            return worker_states

        # Skip if there's queued work
        if task_counts.queued > 0:
            logger.debug("Skipping early termination because queued work exists")
            return worker_states

        # Check grace period
        time_since_last_down = None
        if self.last_scale_down_at:
            time_since_last_down = (now - self.last_scale_down_at).total_seconds()

        if time_since_last_down is not None and time_since_last_down < config.scale_down_grace_period_sec:
            logger.debug(f"Skipping early termination due to scale-down grace period: {time_since_last_down:.0f}s < {config.scale_down_grace_period_sec}s")
            return worker_states

        # Find eligible spawning workers (past grace period)
        excess = current_capacity - early_desired
        eligible = []
        for ws in sorted(spawning_states, key=lambda x: x.created_at, reverse=True):
            age = (now - ws.created_at).total_seconds()
            if age > config.spawning_grace_period_sec:
                eligible.append(ws)

        to_terminate = min(excess, len(eligible))

        if to_terminate == 0:
            logger.debug("No spawning workers eligible for early termination (within grace period)")
            return worker_states

        logger.info(f"Early termination: {current_capacity} workers > {early_desired} desired, terminating {to_terminate} spawning workers")

        terminated_ids = set()
        for ws in eligible[:to_terminate]:
            if ws.runpod_id:
                success = self.runpod.terminate_worker(ws.runpod_id)
                if success:
                    logger.info(f"Terminated spawning worker {ws.worker_id} (RunPod {ws.runpod_id})")
                else:
                    logger.warning(f"Failed to terminate spawning worker {ws.worker_id}")

            metadata_update = {
                'error_reason': f'Early termination: over-capacity ({current_capacity} > {early_desired})',
                'error_time': now.isoformat(),
                'terminated_at': now.isoformat()
            }
            await self.db.update_worker_status(ws.worker_id, 'terminated', metadata_update)
            summary.workers_terminated += 1
            terminated_ids.add(ws.worker_id)

        self.last_scale_down_at = now

        # Return updated list without terminated workers
        return [ws for ws in worker_states if ws.worker_id not in terminated_ids]

    # =========================================================================
    # PHASE 4: Handle spawning workers
    # =========================================================================

    async def _handle_spawning_workers(
        self,
        worker_states: List[DerivedWorkerState],
        task_counts: TaskCounts,
        summary: CycleSummary
    ) -> None:
        """Handle pod initialization, script launch, and promotion."""
        # Include workers that are either:
        # 1. Still in spawning phase (is_spawning)
        # 2. Ready for promotion (DB says 'spawning' but heartbeat detected)
        # The second case catches workers where lifecycle is ACTIVE_* due to heartbeat
        # but DB status is still 'spawning'
        spawning_states = [
            ws for ws in worker_states
            if ws.is_spawning or ws.should_promote_to_active
        ]

        for ws in spawning_states:
            worker_id = ws.worker_id
            runpod_id = ws.runpod_id

            if not runpod_id:
                await self._mark_worker_error_by_id(worker_id, 'No Runpod ID')
                summary.workers_failed += 1
                continue

            # PRIORITY 1: Check for promotion first (worker signaled ready_for_tasks)
            # This must come before RunPod status checks to avoid the if/elif trap
            if ws.should_promote_to_active:
                logger.info(f"Worker {worker_id} signaled ready_for_tasks - promoting to active")
                metadata_update = {
                    'promoted_to_active_at': datetime.now(timezone.utc).isoformat(),
                    'promotion_reason': 'ready_for_tasks'
                }
                await self.db.update_worker_status(worker_id, 'active', metadata_update)
                summary.workers_promoted += 1
                continue

            # PRIORITY 2: Check if pod is ready and initialize
            status_update = self.runpod.check_and_initialize_worker(worker_id, runpod_id)

            if status_update['status'] == 'error':
                await self._mark_worker_error_by_id(worker_id, status_update.get('error', 'Unknown error'))
                summary.workers_failed += 1
                continue

            if ws.should_terminate:
                # Spawning timeout
                await self._mark_worker_error_by_id(worker_id, ws.termination_reason or 'Spawning timeout')
                summary.workers_failed += 1
                continue

            if status_update.get('ready') and status_update['status'] == 'spawning':
                # Pod ready - launch startup script if not already done
                if ws.lifecycle == WorkerLifecycle.SPAWNING_SCRIPT_PENDING:
                    if self.config.auto_start_worker_process:
                        if self.runpod.start_worker_process(runpod_id, worker_id, has_pending_tasks=(task_counts.queued > 0)):
                            logger.info(f"Launched startup script for {worker_id}")
                            metadata_update = {
                                'ssh_details': status_update.get('ssh_details'),
                                'startup_script_launched': True,
                                'startup_script_launched_at': datetime.now(timezone.utc).isoformat()
                            }
                            await self.db.update_worker_status(worker_id, 'spawning', metadata_update)
                            summary.startup_scripts_launched += 1
                        else:
                            logger.warning(f"Failed to launch startup script for {worker_id}")
            
            # Note: Promotion now happens via ready_for_tasks flag in heartbeat,
            # not via log-based detection. Worker stays in 'spawning' until it
            # explicitly signals ready_for_tasks=true.

    # =========================================================================
    # PHASE 5: Health check active workers
    # =========================================================================

    async def _health_check_active_workers(
        self,
        worker_states: List[DerivedWorkerState],
        task_counts: TaskCounts,
        summary: CycleSummary
    ) -> None:
        """Check health of active workers based on derived state."""
        active_states = [ws for ws in worker_states if ws.is_active]

        for ws in active_states:
            try:
                set_current_worker(ws.worker_id)
                worker_id = ws.worker_id

                # Check termination conditions from derived state
                if ws.should_terminate:
                    logger.warning(f"[WORKER_HEALTH] Worker {worker_id}: {ws.termination_reason}")
                    # Fetch raw worker for _mark_worker_error
                    worker = await self.db.get_worker_by_id(worker_id)
                    if worker:
                        await self._mark_worker_error(
                            worker,
                            ws.termination_reason or 'Health check failed',
                            context={
                                'error_code': ws.error_code,
                                'lifecycle': ws.lifecycle.value,
                                'effective_age_sec': ws.effective_age_sec,
                                'heartbeat_age_sec': ws.heartbeat_age_sec,
                                'vram_total_mb': ws.vram_total_mb,
                                'has_ever_claimed': ws.has_ever_claimed_task,
                                'has_active_task': ws.has_active_task,
                            }
                        )
                        summary.workers_failed += 1
                    continue

                # If worker has active task - validate heartbeat
                if ws.has_active_task:
                    if not ws.has_heartbeat:
                        logger.warning(f"[WORKER_HEALTH] Worker {worker_id}: No heartbeat with active tasks")
                        worker = await self.db.get_worker_by_id(worker_id)
                        if worker:
                            await self._mark_worker_error(worker, 'No heartbeat with active tasks')
                            summary.workers_failed += 1
                        continue

                    logger.debug(f"Worker {worker_id} has active tasks and heartbeat ({ws.heartbeat_age_sec:.0f}s old)")

                # Worker has no active task - check idle conditions
                elif not ws.has_active_task:
                    worker = await self.db.get_worker_by_id(worker_id)
                    if not worker:
                        continue

                    # CRITICAL: Check if worker is idle while tasks are queued
                    if task_counts.queued > 0:
                        # Worker should be claiming tasks but isn't
                        if ws.has_heartbeat and ws.heartbeat_age_sec is not None:
                            # Has recent heartbeat but not claiming - check timeout
                            if ws.heartbeat_age_sec > self.config.gpu_idle_timeout_sec:
                                logger.warning(f"[WORKER_HEALTH] Worker {worker_id}: Idle with stale heartbeat ({ws.heartbeat_age_sec:.0f}s) while {task_counts.queued} tasks queued")
                                await self._mark_worker_error(worker, f'Idle with tasks queued (heartbeat age: {ws.heartbeat_age_sec:.0f}s)')
                                summary.workers_failed += 1
                                continue
                            else:
                                # Fresh heartbeat but not claiming - perform basic health check
                                if not await self._perform_basic_health_check(worker):
                                    logger.warning(f"[WORKER_HEALTH] Worker {worker_id}: Failed basic health check with tasks queued")
                                    await self._mark_worker_error(worker, 'Failed basic health check with tasks queued')
                                    summary.workers_failed += 1
                                    continue
                        else:
                            # No heartbeat while tasks are queued
                            worker_age = (datetime.now(timezone.utc) - datetime.fromisoformat(worker['created_at'].replace('Z', '+00:00'))).total_seconds()
                            if worker_age > self.config.gpu_idle_timeout_sec:
                                logger.warning(f"[WORKER_HEALTH] Worker {worker_id}: No heartbeat ({worker_age:.0f}s old) with {task_counts.queued} tasks queued")
                                await self._mark_worker_error(worker, 'No heartbeat or activity with tasks queued')
                                summary.workers_failed += 1
                                continue

                    # No tasks queued - check idle timeout for potential scale-down
                    else:
                        if await self._is_worker_idle_with_timeout(worker, self.config.gpu_idle_timeout_sec):
                            # Check if we can terminate (above minimum)
                            stable_active = len([s for s in active_states if s.is_healthy])
                            if stable_active > self.config.min_active_gpus:
                                logger.info(f"[WORKER_HEALTH] Worker {worker_id}: Idle timeout with no work, terminating")
                                await self._mark_worker_error(worker, f'Idle timeout ({self.config.gpu_idle_timeout_sec}s) with no work available')
                                summary.workers_failed += 1
                                continue

                # Log healthy worker state
                if ws.lifecycle == WorkerLifecycle.ACTIVE_READY:
                    logger.debug(f"Worker {worker_id}: healthy (VRAM: {ws.vram_total_mb}MB, ready_for_tasks=True)")
                elif ws.lifecycle == WorkerLifecycle.ACTIVE_INITIALIZING:
                    # Worker is heartbeating but not yet ready_for_tasks
                    vram_status = f"VRAM: {ws.vram_total_mb}MB" if ws.vram_reported else "no VRAM"
                    ready_status = "ready_for_tasks=True" if ws.ready_for_tasks else "waiting for ready_for_tasks"
                    logger.info(f"Worker {worker_id}: initializing ({ws.effective_age_sec:.0f}s, {vram_status}, {ready_status})")
                elif ws.lifecycle == WorkerLifecycle.ACTIVE_GPU_NOT_DETECTED:
                    logger.info(f"Worker {worker_id}: GPU not detected yet ({ws.effective_age_sec:.0f}s / {self.config.gpu_not_detected_timeout_sec}s)")

                # Perform basic health check for idle workers with fresh heartbeat
                if not ws.has_active_task and ws.heartbeat_is_recent:
                    if not await self._perform_basic_health_check(worker if 'worker' in dir() else await self.db.get_worker_by_id(worker_id)):
                        logger.warning(f"[WORKER_HEALTH] Worker {worker_id}: Failed basic health check")

                # Check for stuck tasks
                await self._check_stuck_tasks(ws, summary)

            except Exception as e:
                logger.error(f"Error checking health of worker {ws.worker_id}: {e}")
            finally:
                set_current_worker(None)

    async def _check_stuck_tasks(self, ws: DerivedWorkerState, summary: CycleSummary) -> None:
        """Check if worker has stuck tasks."""
        if not ws.has_active_task:
            return

        running_tasks = await self.db.get_running_tasks_for_worker(ws.worker_id)
        for task in running_tasks:
            task_type = task.get('task_type', '')

            # Skip orchestrator tasks
            if '_orchestrator' in task_type.lower():
                continue

            if task.get('generation_started_at'):
                task_start = datetime.fromisoformat(task['generation_started_at'].replace('Z', '+00:00'))
                if task_start.tzinfo is None:
                    task_start = task_start.replace(tzinfo=timezone.utc)

                task_age = (datetime.now(timezone.utc) - task_start).total_seconds()
                if task_age > self.config.task_stuck_timeout_sec:
                    worker = await self.db.get_worker_by_id(ws.worker_id)
                    if worker:
                        await self._mark_worker_error(worker, f'Stuck task {task["id"]} (timeout: {self.config.task_stuck_timeout_sec}s)')
                        summary.workers_failed += 1
                    return

    async def _perform_basic_health_check(self, worker: Dict[str, Any]) -> bool:
        """
        Perform a basic health check on an idle worker.

        Relies on heartbeat to determine if worker is alive.
        If the worker is actively heartbeating, it means the worker script
        is running and healthy - no need for SSH or other checks.
        """
        if not worker:
            return False

        worker_id = worker['id']
        logger.debug(f"HEALTH_CHECK [Worker {worker_id}] Starting basic health check")

        try:
            # Check if worker has a recent heartbeat
            if worker.get('last_heartbeat'):
                heartbeat_time = datetime.fromisoformat(worker['last_heartbeat'].replace('Z', '+00:00'))
                heartbeat_age = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()

                # If heartbeat is recent (< 60s), worker is healthy
                if heartbeat_age < 60:
                    logger.debug(f"HEALTH_CHECK [Worker {worker_id}] Recent heartbeat ({heartbeat_age:.1f}s old) - worker is healthy")
                    return True
                else:
                    logger.warning(f"HEALTH_CHECK [Worker {worker_id}] Stale heartbeat ({heartbeat_age:.1f}s old)")
                    return False
            else:
                # No heartbeat ever received
                logger.warning(f"HEALTH_CHECK [Worker {worker_id}] No heartbeat received")
                return False

        except Exception as e:
            logger.error(f"HEALTH_CHECK [Worker {worker_id}] Health check failed: {e}")
            return False

    # =========================================================================
    # PHASE 6: Cleanup error workers
    # =========================================================================

    async def _cleanup_error_workers(
        self,
        worker_states: List[DerivedWorkerState],
        summary: CycleSummary
    ) -> None:
        """Terminate RunPod instances for error workers."""
        error_states = [ws for ws in worker_states if ws.lifecycle == WorkerLifecycle.ERROR]

        for ws in error_states:
            try:
                worker = await self.db.get_worker_by_id(ws.worker_id)
                if worker:
                    await self._check_error_worker_cleanup(worker)
                    summary.workers_failed += 1
            except Exception as e:
                logger.error(f"Error during error worker cleanup for {ws.worker_id}: {e}")

    # =========================================================================
    # PHASE 7: Reconcile orphaned tasks
    # =========================================================================

    async def _reconcile_orphaned_tasks(
        self,
        worker_states: List[DerivedWorkerState],
        summary: CycleSummary
    ) -> None:
        """Reset tasks stuck on failed/terminated workers."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)

        # Find recently failed workers
        failed_ids = []
        for ws in worker_states:
            if ws.lifecycle in (WorkerLifecycle.ERROR, WorkerLifecycle.TERMINATED):
                # Check if recently failed (would need to re-fetch worker for updated_at)
                # For now, include all error/terminated workers
                failed_ids.append(ws.worker_id)

        if failed_ids:
            reset_count = await self.db.reset_orphaned_tasks(failed_ids[:100])  # Limit batch size
            summary.tasks_reset = reset_count

        # Also check for unassigned orphaned tasks
        unassigned_reset = await self.db.reset_unassigned_orphaned_tasks(timeout_minutes=15)
        summary.tasks_reset += unassigned_reset

        # Check API worker orphaned tasks
        api_reset = await self.db.reset_api_worker_orphaned_tasks(timeout_minutes=7)
        summary.tasks_reset += api_reset

        # Check stale assigned tasks
        stale_reset = await self.db.reset_stale_assigned_tasks(timeout_minutes=30)
        summary.tasks_reset += stale_reset

    # =========================================================================
    # PHASE 8: Calculate scaling decision
    # =========================================================================

    async def _calculate_scaling_decision(
        self,
        worker_states: List[DerivedWorkerState],
        task_counts: TaskCounts
    ) -> ScalingDecision:
        """Calculate desired workers and scaling actions."""
        config = self.config

        # Check failure rate (async operation)
        failure_rate_ok = await self._check_worker_failure_rate()

        if config.use_new_scaling_logic:
            # New scaling logic: uses pure function that separates reactive from proactive
            decision = calculate_scaling_decision_pure(
                worker_states, task_counts, config, failure_rate_ok
            )
        else:
            # Legacy scaling logic (for rollback)
            decision = self._calculate_scaling_decision_legacy(
                worker_states, task_counts, failure_rate_ok
            )

        self._log_scaling_decision(decision, task_counts)
        return decision

    def _calculate_scaling_decision_legacy(
        self,
        worker_states: List[DerivedWorkerState],
        task_counts: TaskCounts,
        failure_rate_ok: bool,
    ) -> ScalingDecision:
        """Legacy scaling logic (before double-spawn fix)."""
        config = self.config

        # Count workers by state
        active_states = [ws for ws in worker_states if ws.is_active and not ws.should_terminate]
        spawning_states = [ws for ws in worker_states if ws.is_spawning]

        active_count = len(active_states)
        spawning_count = len(spawning_states)
        current_capacity = active_count + spawning_count

        # Count idle vs busy
        idle_count = len([ws for ws in active_states if not ws.has_active_task])
        busy_count = active_count - idle_count

        # Calculate desired workers
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

        # Calculate actions
        workers_to_spawn = 0
        workers_to_terminate = 0

        if current_capacity < desired and failure_rate_ok:
            workers_to_spawn = min(desired - current_capacity, config.max_active_gpus - current_capacity)
        elif current_capacity > desired:
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
        )

    def _log_scaling_decision(self, decision: ScalingDecision, task_counts: TaskCounts):
        """Log scaling decision details."""
        import sys

        logger.critical(f"SCALING DECISION (Cycle #{self.cycle_count}): Current={decision.active_count}+{decision.spawning_count}, Desired={decision.desired_workers}")
        logger.info("SCALING DECISION ANALYSIS:")
        logger.info(f"   Task-based: {decision.task_based_workers} workers (workload: {task_counts.total_workload})")
        logger.info(f"   Buffer-based: {decision.buffer_based_workers} workers (busy: {decision.busy_count}, buffer: {self.config.machines_to_keep_idle})")
        logger.info(f"   Min-based: {decision.min_based_workers} workers")
        logger.info(f"   FINAL DESIRED: {decision.desired_workers} workers")
        logger.info(f"   Current: {decision.idle_count} idle + {decision.busy_count} busy = {decision.active_count} active, {decision.spawning_count} spawning")
        logger.info(f"   Failure rate check: {'PASS' if decision.failure_rate_ok else 'FAIL (blocking scale-up)'}")

        # Log new scaling logic breakdown if using new logic
        if self.config.use_new_scaling_logic:
            logger.info("   NEW SCALING LOGIC:")
            logger.info(f"     Queued tasks: {task_counts.queued}")
            logger.info(f"     Workers covering tasks: {decision.workers_covering_tasks} (idle + spawning)")
            logger.info(f"     Uncovered tasks: {decision.uncovered_tasks}")
            logger.info(f"     Spawn for tasks: {decision.spawn_reason_tasks}")
            logger.info(f"     Spawn for floor: {decision.spawn_reason_floor}")

        print(f"\n{'=' * 80}", file=sys.stderr)
        print(f"SCALING DECISION (Cycle #{self.cycle_count})", file=sys.stderr)
        print(f"  Tasks: {task_counts.queued} queued + {task_counts.active_cloud} active = {task_counts.total_workload} total", file=sys.stderr)
        print(f"  Current: {decision.active_count} active + {decision.spawning_count} spawning = {decision.current_capacity} total", file=sys.stderr)

        # Show new breakdown for stderr output
        if self.config.use_new_scaling_logic:
            print(f"  Task coverage: {decision.workers_covering_tasks} workers can cover {task_counts.queued} queued ({decision.uncovered_tasks} uncovered)", file=sys.stderr)
            print(f"  Spawn breakdown: {decision.spawn_reason_tasks} for tasks + {decision.spawn_reason_floor} for floor = {decision.workers_to_spawn} total", file=sys.stderr)
        else:
            print(f"  Desired: {decision.desired_workers} workers", file=sys.stderr)

        if decision.should_scale_up:
            print(f"  Decision: SCALE UP by {decision.workers_to_spawn}", file=sys.stderr)
        elif decision.should_scale_down:
            print(f"  Decision: SCALE DOWN by {decision.workers_to_terminate}", file=sys.stderr)
        else:
            print("  Decision: MAINTAIN (at capacity)", file=sys.stderr)
        print(f"{'=' * 80}\n", file=sys.stderr)

    # =========================================================================
    # PHASE 9: Execute scaling
    # =========================================================================

    async def _execute_scaling(
        self,
        decision: ScalingDecision,
        worker_states: List[DerivedWorkerState],
        task_counts: TaskCounts,
        summary: CycleSummary,
        detailed_counts: Optional[Dict] = None
    ) -> None:
        """Spawn or terminate workers based on scaling decision."""
        config = self.config
        now = datetime.now(timezone.utc)

        # Scale down: terminate idle workers
        if decision.should_scale_down:
            # Dynamic timeout based on capacity
            if decision.current_capacity > decision.desired_workers:
                dynamic_timeout = config.overcapacity_idle_timeout_sec
            else:
                dynamic_timeout = config.gpu_idle_timeout_sec

            # Find terminatable workers
            active_states = [ws for ws in worker_states if ws.is_active and not ws.has_active_task]
            terminatable = []

            for ws in active_states:
                worker = await self.db.get_worker_by_id(ws.worker_id)
                if worker and await self._is_worker_idle_with_timeout(worker, dynamic_timeout):
                    terminatable.append(worker)

            # Sort by creation time (newest first)
            terminatable.sort(key=lambda w: w.get('created_at', ''), reverse=True)

            # Calculate how many to terminate
            max_by_buffer = max(0, decision.idle_count - config.machines_to_keep_idle)
            max_by_capacity = decision.workers_to_terminate
            max_by_min_active = max(0, decision.active_count - config.min_active_gpus)

            to_terminate = min(len(terminatable), max_by_buffer, max_by_capacity, max_by_min_active)

            if to_terminate > 0:
                logger.info(f"Terminating {to_terminate} workers (over-capacity: {decision.current_capacity} > {decision.desired_workers})")

                for worker in terminatable[:to_terminate]:
                    if await self._terminate_worker(worker):
                        summary.workers_terminated += 1
                        logger.info(f"Terminated idle worker {worker['id']} (idle > {dynamic_timeout}s)")

        # Scale up: spawn new workers
        if decision.should_scale_up:
            logger.critical(f"SCALING UP: Spawning {decision.workers_to_spawn} workers (current: {decision.current_capacity}, desired: {decision.desired_workers})")

            # Log detailed task information that triggered the scale-up
            if detailed_counts:
                queued_tasks = detailed_counts.get('queued_tasks', [])
                active_tasks = detailed_counts.get('active_tasks', [])

                if queued_tasks:
                    logger.info(f"QUEUED TASKS triggering scale-up ({len(queued_tasks)} tasks):")
                    for i, task in enumerate(queued_tasks[:10], 1):
                        task_id = task.get('task_id', 'unknown')[:8]
                        task_type = task.get('task_type', 'unknown')
                        user_id = task.get('user_id', 'unknown')
                        created_at = task.get('created_at', 'unknown')
                        if created_at != 'unknown':
                            created_at = created_at[11:19] if len(created_at) > 19 else created_at
                        logger.info(f"   {i}. {task_id}... | {task_type:25} | user: {user_id} | created: {created_at}")

                    if len(queued_tasks) > 10:
                        logger.info(f"   ... and {len(queued_tasks) - 10} more queued tasks")

                if active_tasks:
                    logger.info(f"ACTIVE TASKS currently being processed ({len(active_tasks)} tasks):")
                    for i, task in enumerate(active_tasks[:5], 1):
                        task_id = task.get('task_id', 'unknown')[:8]
                        task_type = task.get('task_type', 'unknown')
                        worker_id = task.get('worker_id', 'unknown')
                        started_at = task.get('started_at', 'unknown')
                        if started_at != 'unknown':
                            started_at = started_at[11:19] if len(started_at) > 19 else started_at
                        logger.info(f"   {i}. {task_id}... | {task_type:25} | worker: {worker_id} | started: {started_at}")

                    if len(active_tasks) > 5:
                        logger.info(f"   ... and {len(active_tasks) - 5} more active tasks")

            import sys
            print(f"\nSCALING UP: Creating {decision.workers_to_spawn} new workers\n", file=sys.stderr)

            for i in range(decision.workers_to_spawn):
                if await self._spawn_worker(queued_count=task_counts.queued):
                    summary.workers_spawned += 1
                    logger.critical(f"Worker {i + 1}/{decision.workers_to_spawn} spawned successfully")
                else:
                    logger.error(f"Failed to spawn worker {i + 1}/{decision.workers_to_spawn}")

    # =========================================================================
    # PHASE 10: Periodic checks
    # =========================================================================

    async def _run_periodic_checks(
        self,
        worker_states: List[DerivedWorkerState],
        summary: CycleSummary
    ) -> None:
        """Run checks that only happen every Nth cycle."""
        # Create a mutable dict to track periodic check results
        # We use a single dict so mutations persist across method calls
        periodic_summary = {
            'actions': {
                'tasks_reset': 0,
                'workers_failed': 0,
                'workers_terminated': 0,
            },
            'storage_health': {}
        }

        # Storage health check
        active_workers = []
        for ws in worker_states:
            if ws.is_active:
                worker = await self.db.get_worker_by_id(ws.worker_id)
                if worker:
                    active_workers.append(worker)

        await self._check_storage_health(active_workers, periodic_summary)

        # Failsafe stale worker check
        all_workers = []
        for ws in worker_states:
            worker = await self.db.get_worker_by_id(ws.worker_id)
            if worker:
                all_workers.append(worker)

        await self._failsafe_stale_worker_check(all_workers, periodic_summary)

        # Copy results back to the main summary
        summary.tasks_reset += periodic_summary['actions'].get('tasks_reset', 0)
        summary.workers_failed += periodic_summary['actions'].get('workers_failed', 0)
        summary.workers_terminated += periodic_summary['actions'].get('workers_terminated', 0)

    # =========================================================================
    # HELPER METHODS (preserved from original)
    # =========================================================================

    async def _mark_worker_error_by_id(self, worker_id: str, reason: str):
        """Mark a worker as error by ID."""
        worker = await self.db.get_worker_by_id(worker_id)
        if worker:
            await self._mark_worker_error(worker, reason)
        else:
            logger.error(f"Could not find worker {worker_id} to mark as error")

    async def _mark_worker_error(self, worker: Dict[str, Any], reason: str, context: Dict[str, Any] | None = None):
        """Mark a worker as error and attempt to terminate it."""
        worker_id = worker['id']
        error_code = self._classify_error_code(reason)
        set_current_worker(worker_id)

        logger.error(f"WORKER_LIFECYCLE [Worker {worker_id}] MARKING AS ERROR: {reason}")

        # Collect diagnostics
        diagnostics = await self._collect_worker_diagnostics(worker, reason)
        diagnostics['error_code'] = error_code
        if context:
            diagnostics['context'] = context
        diagnostics['timeouts'] = {
            'startup_grace_period_sec': self.config.startup_grace_period_sec,
            'ready_not_claiming_timeout_sec': self.config.ready_not_claiming_timeout_sec,
            'gpu_not_detected_timeout_sec': self.config.gpu_not_detected_timeout_sec,
            'gpu_idle_timeout_sec': self.config.gpu_idle_timeout_sec,
            'spawning_timeout_sec': self.config.spawning_timeout_sec,
        }

        # Emit startup tails to system_logs
        try:
            if diagnostics.get('startup_log_tail'):
                self._emit_large_worker_log_to_system_logs(worker_id, "startup/apt tails", diagnostics.get('startup_log_tail', ''))
            if diagnostics.get('startup_log_stderr'):
                self._emit_large_worker_log_to_system_logs(worker_id, "startup stderr", diagnostics.get('startup_log_stderr', ''))
        except Exception:
            pass

        # Prepare metadata
        try:
            metadata_update = {
                'error_reason': reason,
                'error_code': error_code,
                'error_time': datetime.now(timezone.utc).isoformat(),
                'diagnostics': diagnostics
            }
        except Exception as e:
            logger.error(f"Failed to prepare diagnostic metadata: {e}")
            metadata_update = {
                'error_reason': reason,
                'error_code': error_code,
                'error_time': datetime.now(timezone.utc).isoformat()
            }

        # Reset orphaned tasks
        reset_count = await self.db.reset_orphaned_tasks([worker_id])
        if reset_count > 0:
            logger.info(f"WORKER_LIFECYCLE [Worker {worker_id}] Reset {reset_count} orphaned tasks")

        # Update database
        success = await self.db.update_worker_status(worker_id, 'error', metadata_update)
        if success:
            logger.info(f"WORKER_LIFECYCLE [Worker {worker_id}] Database updated with error status")
        else:
            logger.error(f"WORKER_LIFECYCLE [Worker {worker_id}] Failed to update database")

        worker['status'] = 'error'

        # Terminate RunPod instance
        runpod_id = worker.get('metadata', {}).get('runpod_id')
        if runpod_id:
            logger.info(f"WORKER_LIFECYCLE [Worker {worker_id}] Terminating RunPod instance: {runpod_id}")
            try:
                success = self.runpod.terminate_worker(runpod_id)
                if success:
                    logger.info(f"WORKER_LIFECYCLE [Worker {worker_id}] RunPod terminated successfully")
                else:
                    logger.error(f"WORKER_LIFECYCLE [Worker {worker_id}] Failed to terminate RunPod")
            except Exception as e:
                logger.error(f"WORKER_LIFECYCLE [Worker {worker_id}] Exception terminating RunPod: {e}")

        set_current_worker(None)

    def _classify_error_code(self, reason: str) -> str:
        """Map a human reason string to a stable error code."""
        r = (reason or "").lower()
        if r.startswith("gpu ready but never claimed"):
            return "GPU_READY_NOT_CLAIMING"
        if r.startswith("gpu not detected"):
            return "GPU_NOT_DETECTED"
        if r.startswith("never initialized"):
            return "STARTUP_NEVER_READY"
        if "stale heartbeat with active tasks" in r:
            return "STALE_HEARTBEAT_ACTIVE_TASK"
        if r.startswith("idle with tasks queued"):
            return "IDLE_WITH_TASKS_QUEUED"
        if r.startswith("no heartbeat or activity"):
            return "NO_HEARTBEAT"
        if r.startswith("early termination: over-capacity"):
            return "OVERCAPACITY_EARLY_TERMINATION"
        if r.startswith("failsafe:"):
            return "FAILSAFE"
        return "UNKNOWN"

    async def _collect_worker_diagnostics(self, worker: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Collect comprehensive diagnostic information from a failing worker."""
        worker_id = worker['id']
        runpod_id = worker.get('metadata', {}).get('runpod_id')

        diagnostics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'worker_id': worker_id,
            'runpod_id': runpod_id,
            'error_reason': reason,
            'collection_success': False
        }

        logger.info(f"DIAGNOSTICS [Worker {worker_id}] Collecting diagnostic information")

        try:
            metadata = worker.get('metadata', {})
            diagnostics['last_heartbeat'] = worker.get('last_heartbeat')
            diagnostics['worker_status'] = worker.get('status')
            diagnostics['worker_created_at'] = worker.get('created_at')

            if isinstance(metadata, dict):
                if metadata.get('startup_script_launched_at'):
                    diagnostics['startup_script_launched_at'] = metadata.get('startup_script_launched_at')

            if 'vram_total_mb' in metadata:
                diagnostics['vram_total_mb'] = metadata['vram_total_mb']
                diagnostics['vram_used_mb'] = metadata.get('vram_used_mb')
                diagnostics['vram_timestamp'] = metadata.get('vram_timestamp')

            # Get running tasks
            try:
                running_tasks = await self.db.get_running_tasks_for_worker(worker_id)
                diagnostics['running_tasks_count'] = len(running_tasks)
                diagnostics['running_tasks'] = [
                    {
                        'id': str(task['id']),
                        'task_type': task.get('task_type'),
                        'started_at': task.get('generation_started_at'),
                    }
                    for task in running_tasks[:5]
                ]
            except Exception as e:
                diagnostics['running_tasks_error'] = str(e)

            # Get pod status
            if runpod_id:
                try:
                    pod_status = self.runpod.get_pod_status(runpod_id)
                    if pod_status:
                        diagnostics['pod_status'] = {
                            'desired_status': pod_status.get('desired_status'),
                            'actual_status': pod_status.get('actual_status'),
                            'uptime_seconds': pod_status.get('uptime_seconds'),
                        }
                except Exception as e:
                    diagnostics['pod_status_error'] = str(e)

            diagnostics['collection_success'] = True

        except Exception as e:
            logger.error(f"DIAGNOSTICS [Worker {worker_id}] Error collecting: {e}")
            diagnostics['collection_error'] = str(e)

        return diagnostics

    def _redact_for_system_logs(self, text: str) -> str:
        """Best-effort redaction for secrets."""
        try:
            import re
            t = text or ""
            t = re.sub(r"eyJ[a-zA-Z0-9_\\-]{20,}\\.[a-zA-Z0-9_\\-]{20,}\\.[a-zA-Z0-9_\\-]{20,}", "[REDACTED_JWT]", t)
            t = re.sub(r"Bearer\\s+[A-Za-z0-9_\\-\\.]{20,}", "Bearer [REDACTED]", t, flags=re.IGNORECASE)
            t = re.sub(r"(SUPABASE_(SERVICE_ROLE_KEY|SERVICE_KEY|ANON_KEY)=)\"?[^\n\" ]+\"?", r"\\1[REDACTED]", t)
            return t
        except Exception:
            return text

    def _emit_large_worker_log_to_system_logs(self, worker_id: str, title: str, content: str, max_chunk: int = 3500):
        """Emit large text content to system_logs."""
        if not content:
            return
        safe = self._redact_for_system_logs(content)
        safe = safe.replace("\r\n", "\n")
        total_cap = 16000
        if len(safe) > total_cap:
            safe = safe[-total_cap:]
        chunks = [safe[i:i + max_chunk] for i in range(0, len(safe), max_chunk)]
        for idx, chunk in enumerate(chunks, start=1):
            logger.info(f"[WORKER_LOG_TAIL] {title} ({idx}/{len(chunks)})\n{chunk}")

    async def _is_worker_idle_with_timeout(self, worker: Dict[str, Any], timeout_seconds: int) -> bool:
        """Check if a worker is idle and has been idle long enough."""
        has_tasks = await self.db.has_running_tasks(worker['id'])
        if has_tasks:
            return False

        recent_completions = await self._check_recent_task_completions(worker['id'])
        if recent_completions > 0:
            return False

        try:
            result = self.db.supabase.table('tasks').select('generation_processed_at, updated_at') \
                .eq('worker_id', worker['id']) \
                .eq('status', 'Complete') \
                .order('generation_processed_at', desc=True) \
                .limit(1) \
                .execute()

            if result.data:
                last_completion_time = result.data[0].get('generation_processed_at') or result.data[0].get('updated_at')
                if last_completion_time:
                    last_completion_dt = datetime.fromisoformat(last_completion_time.replace('Z', '+00:00'))
                    if last_completion_dt.tzinfo is None:
                        last_completion_dt = last_completion_dt.replace(tzinfo=timezone.utc)
                    idle_time = (datetime.now(timezone.utc) - last_completion_dt).total_seconds()
                    return idle_time > timeout_seconds

            created_dt = datetime.fromisoformat(worker['created_at'].replace('Z', '+00:00'))
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            worker_age = (datetime.now(timezone.utc) - created_dt).total_seconds()
            return worker_age > timeout_seconds

        except Exception as e:
            logger.error(f"Error checking idle time for worker {worker['id']}: {e}")
            return False

    async def _check_recent_task_completions(self, worker_id: str) -> int:
        """Check how many tasks this worker completed since the last cycle."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.config.orchestrator_poll_sec)
            result = self.db.supabase.table('tasks').select('id', count='exact') \
                .eq('worker_id', worker_id) \
                .eq('status', 'Complete') \
                .gte('generation_processed_at', cutoff_time.isoformat()) \
                .execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Error checking recent completions for worker {worker_id}: {e}")
            return 0

    async def _spawn_worker(self, queued_count: int = 0) -> bool:
        """Spawn a new worker."""
        try:
            worker_id = self.runpod.generate_worker_id()

            if not await self.db.create_worker_record(worker_id, self.runpod.gpu_type):
                return False

            result = self.runpod.spawn_worker(worker_id)

            if result and result.get("runpod_id"):
                pod_id = result["runpod_id"]
                status = result.get('status', 'spawning')

                if status == 'running':
                    final_status = 'active'
                elif status == 'error':
                    final_status = 'error'
                else:
                    final_status = 'spawning'

                metadata = {'runpod_id': pod_id}
                if 'ssh_details' in result:
                    metadata['ssh_details'] = result['ssh_details']
                if 'pod_details' in result:
                    metadata['pod_details'] = result['pod_details']
                if 'ram_tier' in result:
                    metadata['ram_tier'] = result['ram_tier']
                if 'storage_volume' in result:
                    metadata['storage_volume'] = result['storage_volume']

                update_success = await self.db.update_worker_status(worker_id, final_status, metadata)
                if not update_success:
                    logger.error(f"CRITICAL: Failed to update worker {worker_id} in database but RunPod pod {pod_id} was created!")
                    try:
                        self.runpod.terminate_worker(pod_id)
                    except Exception:
                        logger.error(f"MANUAL INTERVENTION REQUIRED: Orphaned pod {pod_id}")
                    return False

                if final_status == 'active' and self.config.auto_start_worker_process:
                    self.runpod.start_worker_process(pod_id, worker_id, has_pending_tasks=(queued_count > 0))

                return final_status != 'error'
            else:
                await self.db.mark_worker_error(worker_id, 'Failed to spawn Runpod instance')
                return False

        except Exception as e:
            logger.error(f"Error spawning worker: {e}")
            return False

    async def _terminate_worker(self, worker: Dict[str, Any]) -> bool:
        """Terminate a worker and update its status."""
        worker_id = worker['id']
        runpod_id = worker.get('metadata', {}).get('runpod_id')

        logger.info(f"WORKER_LIFECYCLE [Worker {worker_id}] TERMINATING worker")

        try:
            if runpod_id:
                success = self.runpod.terminate_worker(runpod_id)
                if not success:
                    logger.error(f"WORKER_LIFECYCLE [Worker {worker_id}] RunPod termination FAILED")
                    return False

                logger.info(f"WORKER_LIFECYCLE [Worker {worker_id}] RunPod terminated successfully")
            else:
                # BUG FIX: Don't silently skip - this leaves orphaned pods!
                logger.error(f"WORKER_LIFECYCLE [Worker {worker_id}] NO RUNPOD_ID - cannot terminate pod!")

            termination_metadata = {'terminated_at': datetime.now(timezone.utc).isoformat()}
            await self.db.update_worker_status(worker_id, 'terminated', termination_metadata)
            return True

        except Exception as e:
            logger.error(f"WORKER_LIFECYCLE [Worker {worker_id}] TERMINATION FAILED: {e}")
            return False

    async def _check_worker_failure_rate(self) -> bool:
        """Check if the worker failure rate is too high."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=self.config.failure_window_minutes)

            workers = await self.db.get_workers(['spawning', 'active', 'terminating', 'error', 'terminated'])

            recent_workers = []
            for worker in workers:
                worker_time = worker.get('updated_at', worker.get('created_at'))
                if worker_time:
                    try:
                        worker_dt = datetime.fromisoformat(worker_time.replace('Z', '+00:00'))
                        if worker_dt > cutoff_time:
                            recent_workers.append(worker)
                    except Exception:
                        pass

            if len(recent_workers) < self.config.min_workers_for_rate_check:
                return True

            failed_workers = [w for w in recent_workers if w['status'] in ['error', 'terminated']]
            failure_rate = len(failed_workers) / len(recent_workers)

            logger.info(f"[FAILURE_RATE] Recent: {len(recent_workers)}, Failed: {len(failed_workers)}, Rate: {failure_rate:.2%}")

            if failure_rate > self.config.max_worker_failure_rate:
                logger.error(f"[FAILURE_RATE] CRITICAL: Rate ({failure_rate:.2%}) exceeds threshold ({self.config.max_worker_failure_rate:.2%})")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking worker failure rate: {e}")
            return True

    async def _check_storage_health(self, active_workers: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        """Check storage health across all storage volumes."""
        if self.cycle_count - self.last_storage_check_cycle < self.config.storage_check_interval_cycles:
            return

        self.last_storage_check_cycle = self.cycle_count
        logger.info(f"STORAGE_CHECK Starting periodic check (cycle {self.cycle_count})")

        workers_by_storage: Dict[str, List[Dict[str, Any]]] = {}
        for worker in active_workers:
            storage_volume = worker.get('metadata', {}).get('storage_volume')
            if storage_volume:
                if storage_volume not in workers_by_storage:
                    workers_by_storage[storage_volume] = []
                workers_by_storage[storage_volume].append(worker)

        if not workers_by_storage:
            logger.info("STORAGE_CHECK No active workers with storage volume info")
            return

        if 'storage_health' not in summary:
            summary['storage_health'] = {}

        for storage_name, storage_workers in workers_by_storage.items():
            try:
                volume_id = self.runpod._get_storage_volume_id(storage_name)
                if not volume_id:
                    continue

                check_worker = None
                for w in storage_workers:
                    if w.get('last_heartbeat'):
                        try:
                            heartbeat_time = datetime.fromisoformat(w['last_heartbeat'].replace('Z', '+00:00'))
                            heartbeat_age = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()
                            if heartbeat_age < 60:
                                runpod_id = w.get('metadata', {}).get('runpod_id')
                                if runpod_id:
                                    check_worker = w
                                    break
                        except Exception:
                            continue

                if not check_worker:
                    continue

                runpod_id = check_worker.get('metadata', {}).get('runpod_id')
                health = self.runpod.check_storage_health(
                    storage_name=storage_name,
                    volume_id=volume_id,
                    active_runpod_id=runpod_id,
                    min_free_gb=self.config.storage_min_free_gb,
                    max_percent_used=self.config.storage_max_percent_used
                )

                summary['storage_health'][storage_name] = health

                if health.get('needs_expansion'):
                    current_size = health.get('total_gb', 100)
                    new_size = current_size + self.config.storage_expansion_increment_gb

                    logger.warning(f"STORAGE_CHECK '{storage_name}' needs expansion: {health.get('message')}")
                    if self.runpod._expand_network_volume(volume_id, new_size):
                        logger.info(f"STORAGE_CHECK Successfully expanded '{storage_name}' to {new_size}GB")
                    else:
                        logger.error(f"STORAGE_CHECK Failed to expand '{storage_name}'!")

            except Exception as e:
                logger.error(f"STORAGE_CHECK Error checking '{storage_name}': {e}")

    async def _check_error_worker_cleanup(self, worker: Dict[str, Any]) -> None:
        """Check workers marked as error and ensure RunPod instances are terminated."""
        worker_id = worker['id']
        metadata = worker.get('metadata', {})
        runpod_id = metadata.get('runpod_id')
        error_time_str = metadata.get('error_time')

        if error_time_str:
            try:
                error_time = datetime.fromisoformat(error_time_str.replace('Z', '+00:00'))
                error_age = (datetime.now(timezone.utc) - error_time).total_seconds()

                if error_age > self.config.error_cleanup_grace_period_sec:
                    logger.info(f"[ERROR_CLEANUP] Worker {worker_id} past grace period ({error_age:.0f}s), forcing cleanup")

                    if runpod_id:
                        pod_status = self.runpod.get_pod_status(runpod_id)
                        if pod_status and pod_status.get('desired_status') not in ['TERMINATED', 'FAILED']:
                            try:
                                self.runpod.terminate_worker(runpod_id)
                            except Exception as e:
                                logger.error(f"Failed to force terminate RunPod {runpod_id}: {e}")

                    termination_metadata = {'terminated_at': datetime.now(timezone.utc).isoformat()}
                    await self.db.update_worker_status(worker_id, 'terminated', termination_metadata)

            except Exception as e:
                logger.error(f"Error parsing error_time for worker {worker_id}: {e}")
        else:
            logger.warning(f"[ERROR_CLEANUP] Worker {worker_id} has no error_time, forcing immediate cleanup")
            if runpod_id:
                try:
                    self.runpod.terminate_worker(runpod_id)
                    termination_metadata = {'terminated_at': datetime.now(timezone.utc).isoformat()}
                    await self.db.update_worker_status(worker_id, 'terminated', termination_metadata)
                except Exception as e:
                    logger.error(f"[ERROR_CLEANUP] Failed to cleanup legacy error worker {worker_id}: {e}")

    async def _failsafe_stale_worker_check(self, workers: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        """Failsafe check for workers with very stale heartbeats."""
        now = datetime.now(timezone.utc)
        active_cutoff = now - timedelta(seconds=self.config.gpu_idle_timeout_sec)
        spawning_cutoff = now - timedelta(seconds=self.config.spawning_timeout_sec)

        stale_workers = []

        for worker in workers:
            if worker['status'] == 'terminated':
                continue

            cutoff = spawning_cutoff if worker['status'] == 'spawning' else active_cutoff

            last_heartbeat = worker.get('last_heartbeat')
            if last_heartbeat:
                try:
                    heartbeat_dt = datetime.fromisoformat(last_heartbeat.replace('Z', '+00:00'))
                    if heartbeat_dt < cutoff:
                        stale_workers.append(worker)
                except Exception:
                    pass
            else:
                try:
                    created_dt = datetime.fromisoformat(worker['created_at'].replace('Z', '+00:00'))
                    if created_dt < cutoff:
                        stale_workers.append(worker)
                except Exception:
                    pass

        for worker in stale_workers:
            worker_id = worker['id']
            logger.warning(f"FAILSAFE: Worker {worker_id} has stale heartbeat")

            try:
                runpod_id = worker.get('metadata', {}).get('runpod_id')
                if runpod_id:
                    pod_status = self.runpod.get_pod_status(runpod_id)
                    if pod_status and pod_status.get('desired_status') not in ['TERMINATED', 'FAILED']:
                        self.runpod.terminate_worker(runpod_id)

                reset_count = await self.db.reset_orphaned_tasks([worker_id])
                if reset_count > 0:
                    summary['actions']['tasks_reset'] = summary['actions'].get('tasks_reset', 0) + reset_count

                metadata_update = {
                    'error_reason': f"FAILSAFE: Stale heartbeat with status {worker['status']}",
                    'error_time': now.isoformat()
                }
                await self.db.update_worker_status(worker_id, 'terminated', metadata_update)
                summary['actions']['workers_failed'] = summary['actions'].get('workers_failed', 0) + 1

            except Exception as e:
                logger.error(f"FAILSAFE: Failed to cleanup stale worker {worker_id}: {e}")

        # Efficient zombie check every 10th cycle
        await self._efficient_zombie_check(summary)

    async def _efficient_zombie_check(self, summary: Dict[str, Any]) -> None:
        """Bidirectional zombie detection."""
        if self.cycle_count % 10 != 0:
            return

        try:
            logger.info("Running efficient zombie check (every 10th cycle)")

            import runpod
            runpod.api_key = os.getenv('RUNPOD_API_KEY')

            pods = runpod.get_pods()
            gpu_worker_pods = [
                pod for pod in pods
                if pod.get('name', '').startswith('gpu-') and
                pod.get('desiredStatus') not in ['TERMINATED', 'FAILED']
            ]

            if not gpu_worker_pods:
                return

            db_workers = await self.db.get_workers()
            db_worker_names = {w['id'] for w in db_workers if w['status'] != 'terminated'}

            # Forward check: RunPod pods not in DB
            for pod in gpu_worker_pods:
                pod_name = pod.get('name')
                runpod_id = pod.get('id')

                if pod_name not in db_worker_names:
                    logger.warning(f"ZOMBIE DETECTED: RunPod pod {pod_name} ({runpod_id}) not in database")
                    try:
                        success = self.runpod.terminate_worker(runpod_id)
                        if success:
                            summary['actions']['workers_terminated'] = summary['actions'].get('workers_terminated', 0) + 1
                    except Exception as e:
                        logger.error(f"ZOMBIE CLEANUP: Error terminating {runpod_id}: {e}")

            # Reverse check: DB workers not in RunPod
            runpod_pod_ids = {pod.get('id') for pod in gpu_worker_pods}
            runpod_pod_names = {pod.get('name') for pod in gpu_worker_pods}

            for worker in db_workers:
                if worker['status'] in ['active', 'spawning']:
                    runpod_id = worker.get('metadata', {}).get('runpod_id')
                    worker_id = worker['id']

                    if runpod_id and runpod_id not in runpod_pod_ids and worker_id not in runpod_pod_names:
                        logger.warning(f"EXTERNALLY TERMINATED: Worker {worker_id} not found in RunPod")
                        try:
                            await self._mark_worker_error(worker, 'Pod externally terminated - not found in RunPod')
                            summary['actions']['workers_failed'] = summary['actions'].get('workers_failed', 0) + 1
                        except Exception as e:
                            logger.error(f"Error marking externally terminated worker {worker_id}: {e}")

        except Exception as e:
            logger.error(f"Error during efficient zombie check: {e}")

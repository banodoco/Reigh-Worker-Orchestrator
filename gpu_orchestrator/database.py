"""
Database client for the Runpod GPU Worker Orchestrator.
Handles all Supabase database operations using the existing schema.
"""

import os
import logging
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Max worker IDs per .in_() filter to stay within PostgREST URL length limits.
# Worker IDs are ~40 chars; 10 keeps the query well under the limit.
_IN_CHUNK_SIZE = 10


def _env_str(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def _env_optional_str(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _profiles_compatible(worker_profile: str, route_profile: Any) -> bool:
    profile = str(route_profile or "")
    return profile == worker_profile or {profile, worker_profile} == {"default", "1"}


def selected_pool_route_filter() -> Dict[str, Any]:
    worker_backend = _env_str("REIGH_BACKEND", "WORKER_BACKEND", default="wgp").lower()
    selector_namespace = _env_str("REIGH_SELECTOR_NAMESPACE", "ROUTE_SELECTOR_NAMESPACE", default="production")
    worker_pool = _env_str(
        "REIGH_WORKER_POOL",
        "WORKER_POOL",
        default=f"gpu-{worker_backend}-{selector_namespace}",
    )
    return {
        "worker_backend": worker_backend,
        "worker_profile": _env_str("REIGH_WORKER_PROFILE", "WGP_PROFILE", default="1"),
        "worker_pool": worker_pool,
        "selector_namespace": selector_namespace,
        "selector_version": _env_optional_str("REIGH_SELECTOR_VERSION", "ROUTE_SELECTOR_VERSION"),
        "worker_contract_version": _env_int("REIGH_WORKER_CONTRACT_VERSION", 1),
        "worker_run_id": _env_optional_str("REIGH_WORKER_RUN_ID", "WORKER_RUN_ID", "ROUTE_RUN_ID"),
    }


def selected_pool_route_metadata(route_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    route = route_filter or selected_pool_route_filter()
    contract = {
        "selected_backend": route["worker_backend"],
        "selected_profile": route["worker_profile"],
        "worker_backend": route["worker_backend"],
        "worker_profile": route["worker_profile"],
        "worker_pool": route["worker_pool"],
        "selector_namespace": route["selector_namespace"],
        "selector_version": route.get("selector_version"),
        "worker_contract_version": route["worker_contract_version"],
        "route_run_id": route.get("worker_run_id"),
    }
    return {
        "worker_backend": route["worker_backend"],
        "worker_profile": route["worker_profile"],
        "worker_pool": route["worker_pool"],
        "selector_namespace": route["selector_namespace"],
        "selector_version": route.get("selector_version"),
        "worker_contract_version": route["worker_contract_version"],
        "worker_run_id": route.get("worker_run_id"),
        "route_contract": contract,
    }


def apply_selected_pool_totals(data: Dict[str, Any]) -> Dict[str, Any]:
    route_totals = data.get("route_totals")
    if not isinstance(route_totals, list):
        return data

    route_filter = selected_pool_route_filter()
    selected = {
        "queued_only": 0,
        "active_only": 0,
        "queued_plus_active": 0,
        "blocked_by_capacity": 0,
        "blocked_by_deps": 0,
        "blocked_by_settings": 0,
        "potentially_claimable": 0,
        "route_keys": [],
        "route_filter": route_filter,
        "aggregate_fallback": False,
    }

    for route in route_totals:
        if not isinstance(route, dict):
            continue
        if route.get("selected_backend") != route_filter["worker_backend"]:
            continue
        if route.get("selector_namespace") != route_filter["selector_namespace"]:
            continue
        if str(route.get("selector_version") or "") != str(route_filter.get("selector_version") or ""):
            continue
        if int(route.get("worker_contract_version") or 0) != int(route_filter["worker_contract_version"]):
            continue
        if not _profiles_compatible(route_filter["worker_profile"], route.get("selected_profile")):
            continue

        selected["queued_only"] += int(route.get("queued_only") or 0)
        selected["active_only"] += int(route.get("active_only") or 0)
        selected["blocked_by_capacity"] += int(route.get("blocked_by_capacity") or 0)
        selected["potentially_claimable"] += int(route.get("potentially_claimable") or route.get("queued_only") or 0)
        route_key = route.get("route_key")
        if route_key and route_key not in selected["route_keys"]:
            selected["route_keys"].append(route_key)

    selected["queued_plus_active"] = selected["queued_only"] + selected["active_only"]
    data["selected_pool_totals"] = selected
    return data

class DatabaseClient:
    """Database client for orchestrator operations."""
    
    def __init__(self):
        load_dotenv()
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    # Task operations (minimal - most task management is done by headless workers)
    async def count_available_tasks_via_edge_function(self, include_active: bool = True) -> int:
        """
        Get available task count using the new task-counts endpoint.
        This ensures the orchestrator sees exactly the same task count as workers.
        
        Args:
            include_active: If True, includes both Queued and In Progress tasks.
                           If False, only includes Queued tasks.
        
        Returns:
            Number of available tasks that workers can see.
        """
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            
            if not supabase_url or not supabase_key:
                logger.error("Missing Supabase edge-function configuration")
                return 0
            
            task_counts_url = f"{supabase_url}/functions/v1/task-counts"
            
            headers = {
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "run_type": "gpu",
                "include_active": include_active,
                **selected_pool_route_filter(),
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(task_counts_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Extract count from new endpoint format
                        if "totals" in data:
                            data = apply_selected_pool_totals(data)
                            totals = data.get("selected_pool_totals") or data["totals"]
                            task_count = totals.get("queued_plus_active" if include_active else "queued_only", 0)
                        else:
                            task_count = data.get("available_tasks", 0)
                        logger.debug(f"Task counts endpoint returned {task_count} available tasks (include_active={include_active})")
                        return task_count
                    else:
                        logger.error(f"Task counts endpoint returned status {response.status}: {await response.text()}")
                        return 0
                        
        except Exception as e:
            logger.error(f"Failed to get task count: {e}")
            return 0

    async def get_detailed_task_counts_via_edge_function(self) -> Dict[str, Any]:
        """
        Get detailed task count breakdown using the new task-counts endpoint.
        This provides comprehensive debugging information for scaling decisions.
        
        Returns:
            Dict with detailed task counts and user breakdown, or empty dict on error.
        """
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            
            if not supabase_url or not supabase_key:
                logger.error("Missing Supabase edge-function configuration")
                return {}
            
            task_counts_url = f"{supabase_url}/functions/v1/task-counts"
            
            headers = {
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "run_type": "gpu",
                **selected_pool_route_filter(),
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(task_counts_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        data = apply_selected_pool_totals(data)
                        logger.info(f"✅ Task counts endpoint returned detailed breakdown: {data.get('totals', {})}")
                        
                        # Log response for debugging
                        totals = data.get('totals', {})
                        logger.info(f"🔍 EDGE_FUNCTION_DEBUG Full response keys: {list(data.keys())}")
                        logger.info(f"🔍 EDGE_FUNCTION_DEBUG Totals: {totals}")
                        
                        # Check if new breakdown fields are present (from updated edge function)
                        potentially_claimable = totals.get('potentially_claimable')
                        if potentially_claimable is not None:
                            # New edge function with proper breakdown
                            queued_only = totals.get('queued_only', 0)  # Claimable right now
                            blocked_capacity = totals.get('blocked_by_capacity', 0)
                            blocked_deps = totals.get('blocked_by_deps', 0)
                            blocked_settings = totals.get('blocked_by_settings', 0)
                            logger.info(f"✅ New edge function breakdown: queued_only={queued_only}, "
                                       f"capacity_blocked={blocked_capacity}, deps_blocked={blocked_deps}, "
                                       f"settings_blocked={blocked_settings}, potentially_claimable={potentially_claimable}")
                        else:
                            # Legacy edge function - apply workaround for backward compatibility
                            logger.info("⚠️ Legacy edge function detected (no potentially_claimable field)")
                            reported_queued = totals.get('queued_only', 0)
                            reported_active = totals.get('active_only', 0)
                            
                            # Compute actual counts from arrays if available
                            queued_tasks_list = data.get('queued_tasks', [])
                            active_tasks_list = data.get('active_tasks', [])
                            actual_queued = len(queued_tasks_list)
                            actual_active = len(active_tasks_list) if active_tasks_list else reported_active
                            
                            # LEGACY WORKAROUND: Override if totals don't match array
                            if actual_queued > 0 and reported_queued == 0:
                                logger.warning(f"⚠️ LEGACY_WORKAROUND: totals.queued_only={reported_queued} but "
                                             f"queued_tasks has {actual_queued} items! Overriding.")
                                data['totals']['queued_only'] = actual_queued
                                data['totals']['queued_plus_active'] = actual_queued + actual_active
                        
                        return data
                    else:
                        response_text = await response.text()
                        logger.error(f"❌ Task counts endpoint returned status {response.status}: {response_text}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Failed to get detailed task counts: {e}")
            return {}
    
    # Note: Task claiming is handled by the edge function at /functions/v1/claim-next-task
    # Workers should use that endpoint instead of calling the database directly
    
    # Worker operations
    async def get_workers(self, status: List[str] = None) -> List[Dict[str, Any]]:
        """Get GPU workers by status using only database status field.
        
        Excludes API workers (instance_type='api') which are managed separately
        and should not be subject to GPU worker heartbeat monitoring.
        """
        try:
            # Order by created_at DESC to get most recent workers first
            # This ensures we see active/spawning workers within the 1000 row Supabase limit
            # Exclude API workers - they have their own lifecycle and don't need GPU monitoring
            query = self.supabase.table('workers').select('*').neq('instance_type', 'api').order('created_at', desc=True)
            
            if status:
                # Filter directly by database status - no mapping needed
                query = query.in_('status', status)
            
            result = query.execute()
            
            # Return workers using only database status
            workers = []
            for worker in (result.data or []):
                metadata = worker.get('metadata') or {}
                
                mapped_worker = {
                    'id': worker['id'],
                    'instance_type': worker['instance_type'],
                    'status': worker['status'],  # Use database status only
                    'created_at': worker['created_at'],
                    'last_heartbeat': worker.get('last_heartbeat'),
                    'metadata': metadata
                }
                workers.append(mapped_worker)
            
            return workers
            
        except Exception as e:
            logger.error(f"Failed to get workers: {e}")
            return []
    
    async def get_worker_by_id(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific worker by ID."""
        try:
            result = self.supabase.table('workers').select('*').eq('id', worker_id).single().execute()
            
            if result.data:
                worker = result.data
                metadata = worker.get('metadata') or {}
                
                return {
                    'id': worker['id'],
                    'instance_type': worker['instance_type'],
                    'status': worker['status'],  # Use database status only
                    'created_at': worker['created_at'],
                    'last_heartbeat': worker.get('last_heartbeat'),
                    'metadata': metadata
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get worker {worker_id}: {e}")
            return None
    
    async def create_worker_record(self, worker_id: str, instance_type: str, runpod_id: str = None) -> bool:
        """Create a new worker record (optimistic registration)."""
        try:
            metadata = {
                'orchestrator_status': 'spawning',
                **selected_pool_route_metadata(),
            }
            if runpod_id:
                metadata['runpod_id'] = runpod_id
            
            self.supabase.table('workers').insert({
                'id': worker_id,
                'instance_type': instance_type,
                'status': 'inactive',  # DB status
                'metadata': metadata,
                'created_at': datetime.now(timezone.utc).isoformat()
            }).execute()
            
            logger.debug(f"Created worker record: {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create worker record {worker_id}: {e}")
            return False
    
    async def update_worker_status(self, worker_id: str, status: str, metadata_update: Dict[str, Any] = None) -> bool:
        """Update worker status and metadata."""
        try:
            # Get current metadata directly from database
            try:
                result = self.supabase.table('workers').select('metadata').eq('id', worker_id).single().execute()
                metadata = result.data.get('metadata') or {} if result.data else {}
            except Exception:
                metadata = {}
            
            # Mirror the main status into orchestrator_status for consistency
            metadata['orchestrator_status'] = status

            # Merge in any additional metadata (caller wins)
            if metadata_update:
                metadata.update(metadata_update)
            
            # Update in database with the provided status directly
            result = self.supabase.table('workers').update({
                'status': status,  # Use status directly, no mapping
                'metadata': metadata
            }).eq('id', worker_id).execute()
            
            logger.debug(f"Updated worker {worker_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update worker {worker_id} status: {e}")
            return False
    
    async def mark_worker_error(self, worker_id: str, reason: str) -> bool:
        """Mark a worker as error with reason."""
        try:
            metadata_update = {
                'error_reason': reason,
                'error_time': datetime.now(timezone.utc).isoformat()
            }
            
            return await self.update_worker_status(worker_id, 'error', metadata_update)
            
        except Exception as e:
            logger.error(f"Failed to mark worker {worker_id} as error: {e}")
            return False
    
    async def update_worker_heartbeat(self, worker_id: str, vram_total_mb: int = None, vram_used_mb: int = None) -> bool:
        """Update worker heartbeat and optionally VRAM metrics."""
        try:
            params = {'worker_id_param': worker_id}
            
            if vram_total_mb is not None:
                params['vram_total_mb_param'] = vram_total_mb
                params['vram_used_mb_param'] = vram_used_mb or 0
            
            self.supabase.rpc('func_update_worker_heartbeat', params).execute()
            
            logger.debug(f"Updated heartbeat for worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for worker {worker_id}: {e}")
            return False
    
    # Helper functions
    async def has_running_tasks(self, worker_id: str) -> bool:
        """Check if a worker has any running tasks."""
        try:
            result = self.supabase.table('tasks').select('id').eq('worker_id', worker_id).eq('status', 'In Progress').execute()
            
            return len(result.data or []) > 0
            
        except Exception as e:
            logger.error(f"Failed to check running tasks for worker {worker_id}: {e}")
            return False
    
    async def has_worker_ever_claimed_task(self, worker_id: str) -> bool:
        """Check if a worker has ever been assigned any task (any status).
        
        Used to detect workers that started but never successfully claimed work,
        indicating potential GPU initialization failures or other startup issues.
        """
        try:
            # Check if ANY task has ever been assigned to this worker
            result = self.supabase.table('tasks').select('id').eq('worker_id', worker_id).limit(1).execute()
            
            return len(result.data or []) > 0
            
        except Exception as e:
            logger.error(f"Failed to check if worker {worker_id} ever claimed tasks: {e}")
            # Return True on error to avoid false terminations
            return True
    
    async def get_running_tasks_for_worker(self, worker_id: str) -> List[Dict[str, Any]]:
        """Get all running tasks for a worker."""
        try:
            result = self.supabase.table('tasks').select('*').eq('worker_id', worker_id).eq('status', 'In Progress').execute()
            
            # Map to expected format
            tasks = []
            for task in (result.data or []):
                mapped_task = {
                    'id': task['id'],
                    'status': 'Running',  # Map back to orchestrator status
                    'generation_started_at': task.get('generation_started_at'),
                    'task_type': task.get('task_type')
                }
                tasks.append(mapped_task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to get running tasks for worker {worker_id}: {e}")
            return []

    @staticmethod
    def _build_task_reset_payload(error_message: str, attempts: Optional[int] = None) -> Dict[str, Any]:
        """Build a consistent payload for task reset updates."""
        payload: Dict[str, Any] = {
            'status': 'Queued',
            'worker_id': None,
            'generation_started_at': None,
            'generation_processed_at': None,
            'error_message': error_message,
        }
        if attempts is not None:
            payload['attempts'] = attempts
        return payload
    
    async def reset_orphaned_tasks(self, failed_worker_ids: List[str]) -> int:
        """Reset tasks from failed workers back to queued status (includes orchestrator tasks with assigned workers)."""
        try:
            if not failed_worker_ids:
                return 0
            
            # Find all tasks from these workers that need to be reset
            # Since we're explicitly looking for tasks assigned to failed workers,
            # we include orchestrator tasks - if they have a worker_id, they should be reset
            tasks_result = self.supabase.table('tasks').select('id, task_type').eq('status', 'In Progress').in_('worker_id', failed_worker_ids).lt('attempts', 3).execute()
            
            if not tasks_result.data:
                if len(failed_worker_ids) <= 5:
                    logger.debug(f"No orphaned tasks found for workers: {failed_worker_ids}")
                else:
                    logger.debug(f"No orphaned tasks found for {len(failed_worker_ids)} workers")
                return 0
            
            # Reset all orphaned tasks (including orchestrator tasks)
            task_ids = [task['id'] for task in tasks_result.data]
            orchestrator_task_count = sum(1 for task in tasks_result.data if '_orchestrator' in task.get('task_type', '').lower())
            
            reset_result = self.supabase.table('tasks').update(
                self._build_task_reset_payload('Reset - orphaned from failed worker')
            ).in_('id', task_ids).execute()
            
            count = len(reset_result.data) if reset_result.data else 0
            
            # Logging
            if count > 0:
                if len(failed_worker_ids) <= 5:
                    log_msg = f"Reset {count} orphaned tasks from workers: {failed_worker_ids}"
                else:
                    log_msg = f"Reset {count} orphaned tasks from {len(failed_worker_ids)} workers"
                
                if orchestrator_task_count > 0:
                    log_msg += f" (including {orchestrator_task_count} orchestrator tasks)"
                
                logger.info(log_msg)
            elif failed_worker_ids:
                # If we checked workers but found no orphaned tasks, log at debug level only
                if len(failed_worker_ids) <= 5:
                    logger.debug(f"Checked {len(failed_worker_ids)} workers for orphaned tasks: {failed_worker_ids}")
                else:
                    logger.debug(f"Checked {len(failed_worker_ids)} workers for orphaned tasks (list truncated)")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to reset orphaned tasks: {e}")
            return 0
    
    async def reset_unassigned_orphaned_tasks(self, timeout_minutes: int = 15) -> int:
        """Reset tasks stuck in 'In Progress' with no worker_id assigned (excludes orchestrator tasks)."""
        try:
            # Find tasks that are in progress but have no worker assigned
            # and have been stuck for longer than the timeout
            # EXCLUDE orchestrator tasks since they can legitimately run for extended periods
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
            
            result = self.supabase.table('tasks').select('id, task_type').eq('status', 'In Progress').is_('worker_id', 'null').lt('generation_started_at', cutoff_time.isoformat()).execute()
            
            if not result.data:
                return 0
            
            # Filter out orchestrator tasks
            non_orchestrator_tasks = [
                task for task in result.data 
                if '_orchestrator' not in task.get('task_type', '').lower()
            ]
            
            if not non_orchestrator_tasks:
                logger.debug(f"Found {len(result.data)} unassigned tasks, but all are orchestrator tasks - skipping reset")
                return 0
            
            task_ids = [task['id'] for task in non_orchestrator_tasks]
            
            # Reset these tasks back to Queued status (only if attempts < 3)
            reset_result = self.supabase.table('tasks').update(
                self._build_task_reset_payload('Reset - stuck in progress with no worker assigned')
            ).in_('id', task_ids).lt('attempts', 3).execute()
            
            count = len(reset_result.data) if reset_result.data else 0
            
            if count > 0:
                logger.warning(f"Reset {count} unassigned orphaned tasks that were stuck in progress (excluded orchestrator tasks)")
                # Log task IDs for debugging
                if count <= 5:
                    logger.warning(f"Reset unassigned orphaned task IDs: {task_ids}")
                else:
                    logger.warning(f"Reset {count} unassigned orphaned tasks (IDs truncated)")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to reset unassigned orphaned tasks: {e}")
            return 0
    
    async def reset_api_worker_orphaned_tasks(self, timeout_minutes: int = 5) -> int:
        """Reset tasks stuck in 'In Progress' assigned to API workers (api-worker-*).
        
        API workers (like api-worker-main) are not tracked in the workers table,
        so when the API orchestrator crashes/restarts, their in-flight tasks
        become orphaned with no cleanup mechanism.
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
            
            # Find tasks that are:
            # - In Progress
            # - Assigned to an api-worker-* (not a GPU worker)
            # - Have been stuck for longer than the timeout
            # - Have attempts < 3 (allow retry)
            result = self.supabase.table('tasks').select('id, task_type, worker_id, attempts').eq('status', 'In Progress').like('worker_id', 'api-worker-%').lt('generation_started_at', cutoff_time.isoformat()).lt('attempts', 3).execute()
            
            if not result.data:
                return 0
            
            # Reset each task individually to properly increment attempts
            count = 0
            task_ids = []
            for task in result.data:
                task_id = task['id']
                current_attempts = task.get('attempts', 0)
                new_attempts = current_attempts + 1
                
                try:
                    reset_result = self.supabase.table('tasks').update(
                        self._build_task_reset_payload(
                            f'Reset - orphaned from API worker timeout (attempt {new_attempts}/3)',
                            attempts=new_attempts,
                        )
                    ).eq('id', task_id).execute()
                    
                    if reset_result.data:
                        count += 1
                        task_ids.append(task_id)
                except Exception as e:
                    logger.error(f"Failed to reset orphaned task {task_id}: {e}")
            
            if count > 0:
                logger.warning(f"Reset {count} orphaned API worker tasks stuck in progress for >{timeout_minutes}m")
                if count <= 5:
                    logger.warning(f"Reset API orphaned task IDs: {task_ids}")
                else:
                    logger.warning(f"Reset {count} API orphaned tasks (IDs truncated)")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to reset API worker orphaned tasks: {e}")
            return 0
    
    async def reset_stale_assigned_tasks(self, timeout_minutes: int = 30) -> int:
        """Reset tasks stuck in 'In Progress' that haven't been updated recently.
        
        This is a safety net that catches tasks where:
        - The worker is alive and heartbeating
        - But the task processing itself is stuck (deadlock, infinite loop, etc.)
        - And updated_at isn't being refreshed
        
        Uses updated_at (last activity) rather than generation_started_at (task start time)
        to catch tasks that made some progress but then stalled.
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
            
            # Find tasks that are:
            # - In Progress with a worker assigned (not api-worker-*, those are handled separately)
            # - Haven't been updated in timeout_minutes
            # - Have attempts < 3 (allow retry)
            result = self.supabase.table('tasks').select('id, task_type, worker_id') \
                .eq('status', 'In Progress') \
                .not_.is_('worker_id', 'null') \
                .not_.like('worker_id', 'api-worker-%') \
                .lt('updated_at', cutoff_time.isoformat()) \
                .lt('attempts', 3) \
                .execute()
            
            if not result.data:
                return 0
            
            # Orchestrator tasks get a longer timeout (2 hours) since they
            # coordinate multi-step pipelines, but they're not exempt — a stuck
            # orchestrator blocks the queue just like any other task.
            #
            # Long-running individual generation types (VACE / Wan22 segments,
            # animate_character) also need the 2h window: their inference runs
            # 20–60 minutes on a single GPU and the worker doesn't poke
            # tasks.updated_at during inference, so the default 30m cap
            # incorrectly classifies them as stalled.
            orchestrator_cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
            long_running_task_types = {
                'travel_segment',
                'individual_travel_segment',
                'join_clips_segment',
                'animate_character',
                'video_enhance',
                'edit_video',
            }

            task_ids = []
            for task in result.data:
                task_type = task.get('task_type', '').lower()
                is_orchestrator = '_orchestrator' in task_type
                is_long_running = task_type in long_running_task_types
                if is_orchestrator or is_long_running:
                    # Check against the longer 2-hour cutoff
                    task_updated = task.get('updated_at', '')
                    if task_updated and task_updated < orchestrator_cutoff.isoformat():
                        task_ids.append(task['id'])
                else:
                    task_ids.append(task['id'])
            
            if not task_ids:
                return 0

            # Reset these tasks back to Queued status
            reset_result = self.supabase.table('tasks').update({
                'status': 'Queued',
                'worker_id': None,
                'generation_started_at': None,
                'generation_processed_at': None,
                'error_message': f'Reset - no activity for {timeout_minutes} minutes (stale task)'
            }).in_('id', task_ids).execute()
            
            count = len(reset_result.data) if reset_result.data else 0
            
            if count > 0:
                logger.warning(f"Reset {count} stale assigned tasks with no activity for >{timeout_minutes}m")
                if count <= 5:
                    logger.warning(f"Reset stale task IDs: {task_ids}")
                else:
                    logger.warning(f"Reset {count} stale tasks (IDs truncated)")
            
            return count

        except Exception as e:
            logger.error(f"Failed to reset stale assigned tasks: {e}")
            return 0

    async def get_worker_task_failure_streak(
        self, 
        worker_id: str, 
        window_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Get recent task outcomes for a worker to detect failure streaks.
        
        Queries system_logs for "Finished task (Success: True/False)" messages
        to determine if a worker is repeatedly failing tasks.
        
        Args:
            worker_id: The worker ID to check
            window_minutes: How far back to look for task outcomes
            
        Returns:
            Dict with:
                - consecutive_failures: Number of consecutive failures from most recent
                - total_outcomes: Total task outcomes in window
                - total_failures: Total failures in window
                - failure_rate: failures / total (0.0 if no outcomes)
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            
            # Query system_logs for task completion messages from this worker
            # "Finished task (Success: True)" or "Finished task (Success: False)"
            result = self.supabase.table('system_logs') \
                .select('message, timestamp') \
                .eq('source_id', worker_id) \
                .ilike('message', '%Finished task%') \
                .gte('timestamp', cutoff_time.isoformat()) \
                .order('timestamp', desc=True) \
                .limit(50) \
                .execute()
            
            if not result.data:
                return {
                    'consecutive_failures': 0,
                    'total_outcomes': 0,
                    'total_failures': 0,
                    'failure_rate': 0.0
                }
            
            # Parse outcomes (newest first)
            outcomes = []
            for log in result.data:
                msg = log.get('message', '')
                success = 'Success: True' in msg
                outcomes.append(success)
            
            # Count consecutive failures from most recent
            consecutive_failures = 0
            for success in outcomes:
                if not success:
                    consecutive_failures += 1
                else:
                    break  # Hit a success, stop counting
            
            total_failures = sum(1 for s in outcomes if not s)
            total_outcomes = len(outcomes)
            
            return {
                'consecutive_failures': consecutive_failures,
                'total_outcomes': total_outcomes,
                'total_failures': total_failures,
                'failure_rate': total_failures / total_outcomes if total_outcomes > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to get task failure streak for worker {worker_id}: {e}")
            # Return safe defaults on error (don't falsely flag worker)
            return {
                'consecutive_failures': 0,
                'total_outcomes': 0,
                'total_failures': 0,
                'failure_rate': 0.0
            }

    # Batch operations for efficiency
    async def batch_check_ever_claimed(self, worker_ids: List[str]) -> Dict[str, bool]:
        """
        Check if multiple workers have ever claimed tasks in one query.

        Returns a dict mapping worker_id -> has_ever_claimed (bool).
        Much more efficient than N separate queries.

        Chunks the query to avoid exceeding PostgREST URL length limits.
        """
        if not worker_ids:
            return {}

        try:
            claimed_workers: set = set()
            # Chunk to avoid PostgREST URL length limits on .in_() filters
            for i in range(0, len(worker_ids), _IN_CHUNK_SIZE):
                chunk = worker_ids[i:i + _IN_CHUNK_SIZE]
                result = self.supabase.table('tasks') \
                    .select('worker_id') \
                    .in_('worker_id', chunk) \
                    .execute()
                claimed_workers.update(
                    r['worker_id'] for r in (result.data or []) if r.get('worker_id')
                )

            return {wid: wid in claimed_workers for wid in worker_ids}

        except Exception as e:
            logger.error(f"Failed to batch check ever_claimed: {e}")
            # Return True on error to avoid false terminations
            return {wid: True for wid in worker_ids}

    async def batch_check_active_tasks(self, worker_ids: List[str]) -> Dict[str, bool]:
        """
        Check if multiple workers have active (In Progress) tasks in one query.

        Returns a dict mapping worker_id -> has_active_task (bool).
        Much more efficient than N separate queries.

        Chunks the query to avoid exceeding PostgREST URL length limits.
        """
        if not worker_ids:
            return {}

        try:
            active_workers: set = set()
            for i in range(0, len(worker_ids), _IN_CHUNK_SIZE):
                chunk = worker_ids[i:i + _IN_CHUNK_SIZE]
                result = self.supabase.table('tasks') \
                    .select('worker_id') \
                    .in_('worker_id', chunk) \
                    .eq('status', 'In Progress') \
                    .execute()
                active_workers.update(
                    r['worker_id'] for r in (result.data or []) if r.get('worker_id')
                )

            return {wid: wid in active_workers for wid in worker_ids}

        except Exception as e:
            logger.error(f"Failed to batch check active_tasks: {e}")
            # Return False on error to be conservative
            return {wid: False for wid in worker_ids}

"""
Database client for the Runpod GPU Worker Orchestrator.
Handles all Supabase database operations using the existing schema.
"""

import os
import logging
import json
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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
                logger.error("Missing Supabase credentials for edge function call")
                return 0
            
            task_counts_url = f"{supabase_url}/functions/v1/task-counts"
            
            headers = {
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "run_type": "gpu",
                "include_active": include_active
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(task_counts_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Extract count from new endpoint format
                        if "totals" in data:
                            task_count = data["totals"].get("queued_plus_active" if include_active else "queued_only", 0)
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
                logger.error("Missing Supabase credentials for edge function call")
                return {}
            
            task_counts_url = f"{supabase_url}/functions/v1/task-counts"
            
            headers = {
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "run_type": "gpu"
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(task_counts_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Task counts endpoint returned detailed breakdown: {data.get('totals', {})}")
                        
                        # Log the FULL response for debugging the discrepancy
                        logger.info(f"🔍 EDGE_FUNCTION_DEBUG Full response keys: {list(data.keys())}")
                        if 'users' in data:
                            total_user_queued = sum(u.get('queued_tasks', 0) for u in data['users'])
                            total_user_progress = sum(u.get('in_progress_tasks', 0) for u in data['users'])
                            logger.info(f"🔍 EDGE_FUNCTION_DEBUG User totals: {total_user_queued} queued, {total_user_progress} in progress")
                        
                        logger.info(f"🔍 EDGE_FUNCTION_DEBUG Totals from Edge Function: {data.get('totals', 'MISSING')}")
                        
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
        """Get workers by status using only database status field."""
        try:
            # Order by created_at DESC to get most recent workers first
            # This ensures we see active/spawning workers within the 1000 row Supabase limit
            query = self.supabase.table('workers').select('*').order('created_at', desc=True)
            
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
            metadata = {'orchestrator_status': 'spawning'}
            if runpod_id:
                metadata['runpod_id'] = runpod_id
            
            result = self.supabase.table('workers').insert({
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
            except:
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
            
            reset_result = self.supabase.table('tasks').update({
                'status': 'Queued',
                'worker_id': None,
                'generation_started_at': None,
                'generation_processed_at': None,
                'error_message': 'Reset - orphaned from failed worker'
            }).in_('id', task_ids).execute()
            
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
            reset_result = self.supabase.table('tasks').update({
                'status': 'Queued',
                'worker_id': None,
                'generation_started_at': None,
                'generation_processed_at': None,
                'error_message': 'Reset - stuck in progress with no worker assigned'
            }).in_('id', task_ids).lt('attempts', 3).execute()
            
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
                    reset_result = self.supabase.table('tasks').update({
                        'status': 'Queued',
                        'worker_id': None,
                        'generation_started_at': None,
                        'generation_processed_at': None,
                        'attempts': new_attempts,
                        'error_message': f'Reset - orphaned from API worker timeout (attempt {new_attempts}/3)'
                    }).eq('id', task_id).execute()
                    
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
            
            # Filter out orchestrator tasks - they can run for extended periods
            non_orchestrator_tasks = [
                task for task in result.data 
                if '_orchestrator' not in task.get('task_type', '').lower()
            ]
            
            if not non_orchestrator_tasks:
                logger.debug(f"Found {len(result.data)} stale tasks, but all are orchestrator tasks - skipping reset")
                return 0
            
            task_ids = [task['id'] for task in non_orchestrator_tasks]
            
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

    # Batch operations for efficiency
    async def batch_check_ever_claimed(self, worker_ids: List[str]) -> Dict[str, bool]:
        """
        Check if multiple workers have ever claimed tasks in one query.

        Returns a dict mapping worker_id -> has_ever_claimed (bool).
        Much more efficient than N separate queries.

        NOTE: We don't use .limit() here because it could cause false negatives.
        If worker A has 100 tasks, limit(N) might return all rows for A and miss B.
        We deduplicate into a set anyway, so no limit is needed.
        """
        if not worker_ids:
            return {}

        try:
            # Get worker_ids that have ever had a task assigned
            # No limit - we deduplicate into a set, and the in_() filter
            # ensures we only get rows for workers we care about
            result = self.supabase.table('tasks') \
                .select('worker_id') \
                .in_('worker_id', worker_ids) \
                .execute()

            # Build set of workers that have claimed tasks (deduplication happens here)
            claimed_workers = {r['worker_id'] for r in (result.data or []) if r.get('worker_id')}

            # Return dict for all requested workers
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

        NOTE: No limit needed - "In Progress" tasks are naturally limited (one per worker
        typically), and we deduplicate into a set anyway.
        """
        if not worker_ids:
            return {}

        try:
            # Get worker_ids that have active tasks
            result = self.supabase.table('tasks') \
                .select('worker_id') \
                .in_('worker_id', worker_ids) \
                .eq('status', 'In Progress') \
                .execute()

            # Build set of workers with active tasks (deduplication happens here)
            active_workers = {r['worker_id'] for r in (result.data or []) if r.get('worker_id')}

            # Return dict for all requested workers
            return {wid: wid in active_workers for wid in worker_ids}

        except Exception as e:
            logger.error(f"Failed to batch check active_tasks: {e}")
            # Return False on error to be conservative
            return {wid: False for wid in worker_ids}
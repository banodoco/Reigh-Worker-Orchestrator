"""Minimal database client for API orchestrator logging."""

import os
import logging
from datetime import datetime, timezone
from supabase import create_client, Client

logger = logging.getLogger(__name__)


class DatabaseClient:
    """Minimal database client for centralized logging support."""
    
    def __init__(self):
        """Initialize Supabase client."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)

    def register_worker(self, worker_id: str) -> bool:
        """Register or update an API worker in the workers table.

        Uses upsert to handle both new registration and reconnection cases.
        """
        try:
            self.supabase.table('workers').upsert({
                'id': worker_id,
                'instance_type': 'api',
                'status': 'active',
                'last_heartbeat': datetime.now(timezone.utc).isoformat(),
                'metadata': {'type': 'api_orchestrator'},
                'created_at': datetime.now(timezone.utc).isoformat()
            }, on_conflict='id').execute()

            logger.info(f"Registered API worker: {worker_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register API worker {worker_id}: {e}")
            return False

    def update_heartbeat(self, worker_id: str) -> bool:
        """Refresh only last_heartbeat for an already-registered API worker.

        register_worker() is called once at startup, so without this the
        api worker's last_heartbeat goes stale and health monitors (e.g. the
        route-contract sentinel) mis-classify it as a stuck/dead worker.
        """
        try:
            self.supabase.table('workers').update({
                'last_heartbeat': datetime.now(timezone.utc).isoformat(),
            }).eq('id', worker_id).execute()
            return True
        except Exception as e:
            logger.warning(f"Failed to update heartbeat for {worker_id}: {e}")
            return False

    def reset_orphaned_tasks(self, worker_id: str) -> int:
        """Reset any tasks stuck 'In Progress' on this worker.

        Called at startup — if we just booted, any in-progress task on our
        worker_id is orphaned from a previous crash/redeploy.
        """
        try:
            result = self.supabase.table('tasks').select('id, attempts').eq(
                'status', 'In Progress'
            ).eq('worker_id', worker_id).execute()

            if not result.data:
                return 0

            count = 0
            for task in result.data:
                task_id = task['id']
                attempts = task.get('attempts', 0) + 1
                try:
                    self.supabase.table('tasks').update({
                        'status': 'Queued',
                        'worker_id': None,
                        'generation_started_at': None,
                        'generation_processed_at': None,
                        'attempts': attempts,
                        'error_message': f'Reset on startup - orphaned from worker restart (attempt {attempts}/3)',
                    }).eq('id', task_id).lt('attempts', 3).execute()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to reset orphaned task {task_id}: {e}")

            if count:
                logger.warning(f"[STARTUP] Reset {count} orphaned tasks stuck on {worker_id}")
            return count

        except Exception as e:
            logger.error(f"Failed to reset orphaned tasks: {e}")
            return 0





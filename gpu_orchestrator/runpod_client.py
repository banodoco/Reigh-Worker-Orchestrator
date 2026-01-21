#!/usr/bin/env python3
"""
Runpod client for the GPU worker orchestrator.
Handles spawning and terminating GPU workers on Runpod infrastructure.
Based on the runpod_repo_setup_agent codebase patterns.
"""
import os
import time
import json
import requests
import paramiko
from dotenv import load_dotenv
import runpod
import logging
from typing import Optional, Dict, Any
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------
# Helper Functions & Classes
# ---------------------------

def get_network_volumes(api_key: str):
    """Return a list of your RunPod network volumes."""
    runpod.api_key = api_key
    try:
        # Try using the SDK's get_network_volumes method if available
        if hasattr(runpod, 'get_network_volumes'):
            volumes = runpod.get_network_volumes()
            return volumes if isinstance(volumes, list) else []
        
        # Otherwise try the REST API with different endpoints
        endpoints = [
            "https://api.runpod.io/v1/networkvolumes",
            "https://api.runpod.io/graphql",  # GraphQL endpoint might be needed
        ]
        
        headers = {"Authorization": f"Bearer {api_key}"}
        
        for url in endpoints:
            try:
                if "graphql" in url:
                    # Try GraphQL query for network volumes
                    query = """
                    query {
                        myself {
                            networkVolumes {
                                id
                                name
                                size
                                dataCenterId
                            }
                        }
                    }
                    """
                    response = requests.post(url, json={"query": query}, headers=headers, timeout=30)
                else:
                    response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Handle GraphQL response
                    if "data" in data and "myself" in data["data"]:
                        return data["data"]["myself"].get("networkVolumes", [])
                    
                    # Handle REST response
                    if isinstance(data, list):
                        return data
                        
            except Exception:
                continue
        
        logger.warning("Could not fetch network volumes from any endpoint")
        return []
        
    except Exception as e:
        logger.error(f"Error fetching network volumes: {e}")
        return []


def find_gpu_type(gpu_display_name: str, api_key: str):
    """Find a GPU type by its display name (or ID) and ensure it's available."""
    runpod.api_key = api_key
    try:
        gpus = runpod.get_gpus()
    except Exception as e:
        logger.error(f"Error retrieving GPU list from RunPod: {e}")
        return None

    for gpu in gpus:
        if gpu_display_name in (gpu.get("displayName"), gpu.get("id")):
            return gpu
    return None


def create_pod_and_wait(api_key: str, gpu_type_id: str, image_name: str, name: str = "worker-pod", 
                       network_volume_id: str | None = None, volume_mount_path: str = "/workspace", 
                       disk_in_gb: int = 20, container_disk_in_gb: int = 10, wait_timeout: int = 600, 
                       public_key_string: str | None = None, env_vars: Dict[str, str] = None,
                       min_vcpu_count: int = 8, min_memory_in_gb: int = 32,
                       template_id: str | None = None):
    """Create a RunPod pod and wait until it is running."""
    runpod.api_key = api_key

    params = {
        "name": name,
        "image_name": image_name,
        "gpu_type_id": gpu_type_id,
        "gpu_count": 1,
        "cloud_type": "SECURE",
        "volume_in_gb": disk_in_gb,
        "container_disk_in_gb": container_disk_in_gb,
        "min_vcpu_count": min_vcpu_count,
        "min_memory_in_gb": min_memory_in_gb,
        "ports": "22/tcp,8888/http",
    }
    
    # Use template if provided (includes Jupyter auto-start)
    if template_id:
        params["template_id"] = template_id

    if network_volume_id:
        params["network_volume_id"] = network_volume_id
        params["volume_mount_path"] = volume_mount_path

    # Environment variables for the worker
    pod_env = {}
    if env_vars:
        pod_env.update(env_vars)
    
    # Inject PUBLIC_KEY env var if provided so the pod image adds the key to ~/.ssh/authorized_keys
    if public_key_string:
        pod_env["PUBLIC_KEY"] = public_key_string

    if pod_env:
        params["env"] = pod_env

    try:
        pod = runpod.create_pod(**params)
    except Exception as e:
        logger.error(f"Error creating pod: {e}")
        return None

    # Handle nested response structure
    if isinstance(pod, dict) and 'data' in pod:
        pod_data = pod['data'].get('podFindAndDeployOnDemand', {})
        pod_id = pod_data.get('id')
    else:
        pod_id = pod.get("id")
    
    if not pod_id:
        logger.error("Pod creation failed (no pod ID returned)")
        return None

    logger.info(f"Pod created with ID: {pod_id}")
    
    # Return immediately with the pod ID so we can track it
    # The orchestrator will handle monitoring the pod status
    return {
        'id': pod_id,
        'desiredStatus': 'PROVISIONING',
        'name': name,
        'gpu_type_id': gpu_type_id,
        'created': True
    }


def get_pod_ssh_details(pod_id: str, api_key: str):
    """Return SSH connection details (ip, port, password) for a running pod."""
    runpod.api_key = api_key
    
    # Try the RunPod SDK first
    try:
        status = runpod.get_pod(pod_id)
        if status and isinstance(status, dict):
            runtime = status.get("runtime", {})
            if runtime and isinstance(runtime, dict):
                for port_map in runtime.get("ports", []):
                    if port_map.get("privatePort") == 22:
                        return {
                            "ip": port_map.get("ip"),
                            "port": port_map.get("publicPort"),
                            "password": runtime.get("sshPassword", "runpod"),
                        }
    except Exception as e:
        logger.warning(f"RunPod SDK get_pod failed for {pod_id}: {e}")
    
    # Fallback to direct GraphQL API call
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}"}
        query = f'''
        {{
          pod(input: {{podId: "{pod_id}"}}) {{
            id
            desiredStatus
            runtime {{
              ports {{
                ip
                publicPort
                privatePort
                type
              }}
            }}
          }}
        }}
        '''
        
        response = requests.post('https://api.runpod.io/graphql', 
                                json={'query': query}, 
                                headers=headers, 
                                timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            pod = data.get('data', {}).get('pod')
            if pod:
                runtime = pod.get('runtime', {})
                if runtime:
                    for port_map in runtime.get('ports', []):
                        if port_map.get('privatePort') == 22:
                            return {
                                "ip": port_map.get("ip"),
                                "port": port_map.get("publicPort"),
                                "password": "runpod",  # Default password
                            }
        else:
            logger.warning(f"GraphQL API failed for pod {pod_id}: {response.status_code}")
            
    except Exception as e:
        logger.warning(f"GraphQL fallback failed for pod {pod_id}: {e}")
    
    # If both methods fail, log the issue but don't error out completely
    logger.error(f"Could not get SSH details for pod {pod_id} via SDK or GraphQL API")
    return None


def terminate_pod(pod_id: str, api_key: str):
    """Terminate a RunPod pod to stop billing."""
    runpod.api_key = api_key
    try:
        runpod.terminate_pod(pod_id)
    except Exception as e:
        logger.error(f"Error terminating pod: {e}")


class SSHClient:
    """Minimal paramiko wrapper for executing commands over SSH."""

    def __init__(self, hostname: str, port: int, username: str, password: str | None = None, 
                 private_key_path: str | None = None, private_key_content: str | None = None, timeout: int = 10):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.private_key_path = private_key_path
        self.private_key_content = private_key_content
        self.timeout = timeout
        self.client: paramiko.SSHClient | None = None

    def connect(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = {
            "hostname": self.hostname,
            "port": self.port,
            "username": self.username,
            "timeout": self.timeout,
            "allow_agent": False,
            "look_for_keys": False,
        }

        # Try private key from environment variable first (for Railway)
        if self.private_key_content:
            try:
                from io import StringIO
                # Try Ed25519 first (more common for new keys)
                try:
                    pkey = paramiko.Ed25519Key.from_private_key(StringIO(self.private_key_content))
                except Exception:
                    # Fallback to RSA
                    try:
                        pkey = paramiko.RSAKey.from_private_key(StringIO(self.private_key_content))
                    except Exception:
                        # Fallback to other key types
                        pkey = paramiko.ECDSAKey.from_private_key(StringIO(self.private_key_content))
                connect_kwargs["pkey"] = pkey
            except Exception as e:
                raise RuntimeError(f"Failed to load private key from environment variable: {e}") from e
        # Fallback to key file if path is provided and exists
        elif self.private_key_path and os.path.exists(os.path.expanduser(self.private_key_path)):
            expanded_key = os.path.expanduser(self.private_key_path)
            try:
                # Try different key types
                try:
                    pkey = paramiko.Ed25519Key.from_private_key_file(expanded_key)
                except Exception:
                    try:
                        pkey = paramiko.RSAKey.from_private_key_file(expanded_key)
                    except Exception:
                        pkey = paramiko.ECDSAKey.from_private_key_file(expanded_key)
                connect_kwargs["pkey"] = pkey
            except Exception as e:
                raise RuntimeError(f"Failed to load private key {expanded_key}: {e}") from e
        else:
            connect_kwargs["password"] = self.password

        self.client.connect(**connect_kwargs)

    def execute_command(self, command: str, timeout: int = 600):
        if not self.client:
            raise RuntimeError("SSH client not connected. Call connect() first.")
        
        import time
        
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        channel = stdout.channel
        
        # Wait for command to finish with actual timeout
        # recv_exit_status() blocks forever, so we poll with timeout instead
        start_time = time.time()
        while not channel.exit_status_ready():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                # Command timed out - close the channel and return error
                channel.close()
                return -1, "", f"Command timed out after {timeout} seconds"
            time.sleep(0.1)
        
        exit_status = channel.recv_exit_status()
        out = stdout.read().decode()
        err = stderr.read().decode()
        return exit_status, out, err

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None

# ---------------------------
# End helper section
# ---------------------------


class RunpodClient:
    """Client for managing Runpod GPU instances using the exact patterns from the user's example."""
    
    def __init__(self, api_key: str):
        """Initialize Runpod client with API key."""
        self.api_key = api_key
        runpod.api_key = api_key
        
        # Configuration from environment (matching user's example)
        self.gpu_type = os.getenv("RUNPOD_GPU_TYPE", "NVIDIA GeForce RTX 4090")
        self.worker_image = os.getenv("RUNPOD_WORKER_IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
        # Template ID for auto-starting Jupyter (use template instead of raw image)
        self.template_id = os.getenv("RUNPOD_TEMPLATE_ID", "runpod-torch-v240")
        self.storage_name = os.getenv("RUNPOD_STORAGE_NAME")  # Like "Peter" in your example
        self.volume_mount_path = os.getenv("RUNPOD_VOLUME_MOUNT_PATH", "/workspace")
        self.disk_size_gb = int(os.getenv("RUNPOD_DISK_SIZE_GB", "50"))
        self.container_disk_gb = int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "50"))
        self.min_vcpu_count = int(os.getenv("RUNPOD_MIN_VCPU_COUNT", "8"))
        self.min_memory_gb = int(os.getenv("RUNPOD_MIN_MEMORY_GB", "32"))
        
        # Storage volume fallback: Try multiple storage volumes (may have different instance availability)
        # Hardcoded list of storage volumes to try in order
        self.storage_volumes = ["Peter", "EU-NO-1", "EU-CZ-1", "EUR-IS-1"]  # Add your storage volume names here
        
        # RAM tier fallback strategy: Try to get highest RAM instances, fall back if unavailable
        # Based on testing: 72GB is max available, then 60GB, 48GB, 32GB are common tiers
        # Strategy: Try each RAM tier across ALL storages before falling back to next tier
        # This prioritizes 72GB (lowest failure rate) over 60GB (higher failure rate)
        self.ram_tiers_enabled = os.getenv("RUNPOD_RAM_TIER_FALLBACK", "true").lower() == "true"
        self.ram_tiers = [72, 60, 48, 32, 16]  # Ordered by preference (72GB has lowest failure rate)
        
        # SSH configuration for worker access (both keys like user's example)
        self.ssh_public_key_path = os.getenv("RUNPOD_SSH_PUBLIC_KEY_PATH")
        self.ssh_private_key_path = os.getenv("RUNPOD_SSH_PRIVATE_KEY_PATH")
        
        # Cache storage volume ID (looked up by name)
        self._storage_volume_id = None
        
        # Cache GPU type info
        self._gpu_type_info = None
    
    def _expand_network_volume(self, volume_id: str, new_size_gb: int) -> bool:
        """
        Expand a network volume to a new size using RunPod REST API.
        
        Args:
            volume_id: Network volume ID
            new_size_gb: New size in GB (must be larger than current)
        
        Returns:
            True if successful, False otherwise
        """
        import requests
        
        try:
            url = f"https://rest.runpod.io/v1/networkvolumes/{volume_id}"
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'size': new_size_gb
            }
            
            logger.info(f"📦 Expanding network volume {volume_id} to {new_size_gb} GB...")
            response = requests.patch(url, json=data, headers=headers)
            
            if response.status_code == 200:
                logger.info(f"✅ Successfully expanded volume to {new_size_gb} GB")
                return True
            else:
                logger.error(f"❌ Failed to expand volume: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error expanding volume: {e}")
            return False
    
    def _check_and_expand_storage(self, storage_name: str, volume_id: str, min_free_gb: int = 50) -> bool:
        """
        Check storage space and expand if needed.
        
        Args:
            storage_name: Name of the storage volume
            volume_id: Network volume ID
            min_free_gb: Minimum free space required in GB
        
        Returns:
            True if storage is adequate or was successfully expanded
        """
        try:
            # Get volume info
            volumes = get_network_volumes(self.api_key)
            volume_info = next((v for v in volumes if v.get('id') == volume_id), None)
            
            if not volume_info:
                logger.warning(f"⚠️  Could not find volume info for {volume_id}")
                return True  # Continue anyway
            
            current_size_gb = volume_info.get('size', 0)
            logger.info(f"📊 Storage '{storage_name}': {current_size_gb} GB total")
            
            # Try to check actual free space via a test pod's df command
            # For now, we'll use a heuristic: if total size < 100GB, expand it
            if current_size_gb < 100:
                new_size = current_size_gb + min_free_gb
                logger.warning(f"⚠️  Storage '{storage_name}' is only {current_size_gb} GB")
                logger.info(f"🔧 Expanding to {new_size} GB to ensure adequate space...")
                
                if self._expand_network_volume(volume_id, new_size):
                    logger.info(f"✅ Storage expansion successful")
                    return True
                else:
                    logger.warning(f"⚠️  Storage expansion failed, continuing anyway...")
                    return True  # Don't block worker spawn on expansion failure
            else:
                logger.info(f"✅ Storage '{storage_name}' has adequate capacity ({current_size_gb} GB)")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️  Error checking storage space: {e}, continuing anyway...")
            return True  # Don't block worker spawn on storage check failure
    
    def check_storage_health(
        self, 
        storage_name: str, 
        volume_id: str, 
        active_runpod_id: str,
        min_free_gb: int = 50,
        max_percent_used: int = 85
    ) -> Dict[str, Any]:
        """
        Check storage health by SSHing to a worker and checking actual disk usage.
        
        Args:
            storage_name: Name of the storage volume
            volume_id: Network volume ID  
            active_runpod_id: RunPod ID of an active worker to SSH to
            min_free_gb: Minimum free space required in GB
            max_percent_used: Maximum usage percentage before flagging
            
        Returns:
            Dict with health info:
                - healthy: bool
                - needs_expansion: bool
                - total_gb: int
                - used_gb: int
                - free_gb: int
                - percent_used: int
                - message: str
                - raw_df: str (raw df output)
        """
        try:
            logger.info(f"📦 STORAGE_HEALTH Checking '{storage_name}' via pod {active_runpod_id}")
            
            # Get volume info from RunPod API for total size
            volumes = get_network_volumes(self.api_key)
            volume_info = next((v for v in volumes if v.get('id') == volume_id), None)
            
            api_total_gb = None
            if volume_info:
                api_total_gb = volume_info.get('size', 0)
                logger.info(f"📦 STORAGE_HEALTH API reports '{storage_name}': {api_total_gb} GB total")
            
            # SSH to worker to check actual disk usage
            check_command = """
            echo "=== WORKSPACE STORAGE ==="
            df -h /workspace 2>/dev/null | tail -1
            echo ""
            echo "=== WORKSPACE USAGE DETAILS ==="
            df -BG /workspace 2>/dev/null | tail -1
            echo ""
            echo "=== LARGEST DIRECTORIES ==="
            du -sh /workspace/*/ 2>/dev/null | sort -rh | head -10
            """
            
            result = self.execute_command_on_worker(active_runpod_id, check_command, timeout=30)
            
            if not result or result[0] != 0:
                logger.error(f"📦 STORAGE_HEALTH SSH failed for '{storage_name}': {result}")
                return {
                    'healthy': False,
                    'needs_expansion': False,
                    'message': f'SSH check failed: {result}',
                    'error': True
                }
            
            raw_output = result[1] or ''
            logger.info(f"📦 STORAGE_HEALTH Raw output for '{storage_name}':\n{raw_output}")
            
            # Parse df output to get actual usage
            # Format: "mfs#... 671T 507T 165T 76% /workspace"
            total_gb = 0
            used_gb = 0
            free_gb = 0
            percent_used = 0
            
            for line in raw_output.split('\n'):
                if '/workspace' in line and not line.startswith('==='):
                    parts = line.split()
                    if len(parts) >= 5:
                        # Try to parse the size values
                        try:
                            # Handle different units (G, T, etc.)
                            def parse_size(s):
                                s = s.strip()
                                if s.endswith('T'):
                                    return int(float(s[:-1]) * 1024)
                                elif s.endswith('G'):
                                    return int(float(s[:-1]))
                                elif s.endswith('M'):
                                    return int(float(s[:-1]) / 1024)
                                elif s.endswith('K'):
                                    return 0
                                else:
                                    return int(s)
                            
                            # df -BG output: "Size Used Avail Use% Mounted"
                            if 'G' in parts[1] or 'T' in parts[1]:
                                total_gb = parse_size(parts[1])
                                used_gb = parse_size(parts[2])
                                free_gb = parse_size(parts[3])
                                percent_str = parts[4].rstrip('%')
                                percent_used = int(percent_str) if percent_str.isdigit() else 0
                                break
                        except (ValueError, IndexError) as e:
                            logger.warning(f"📦 STORAGE_HEALTH Could not parse df output: {e}")
            
            # Determine health status
            healthy = True
            needs_expansion = False
            message = f"OK: {free_gb}GB free ({100-percent_used}% available)"
            
            if percent_used >= max_percent_used:
                healthy = False
                needs_expansion = True
                message = f"CRITICAL: {percent_used}% used, only {free_gb}GB free!"
                logger.error(f"📦 STORAGE_HEALTH ❌ '{storage_name}' {message}")
            elif free_gb < min_free_gb:
                healthy = False
                needs_expansion = True
                message = f"LOW SPACE: Only {free_gb}GB free (need {min_free_gb}GB)"
                logger.warning(f"📦 STORAGE_HEALTH ⚠️  '{storage_name}' {message}")
            else:
                logger.info(f"📦 STORAGE_HEALTH ✅ '{storage_name}' {message}")
            
            return {
                'healthy': healthy,
                'needs_expansion': needs_expansion,
                'total_gb': total_gb,
                'used_gb': used_gb,
                'free_gb': free_gb,
                'percent_used': percent_used,
                'message': message,
                'raw_df': raw_output,
                'api_total_gb': api_total_gb
            }
            
        except Exception as e:
            logger.error(f"📦 STORAGE_HEALTH Error checking '{storage_name}': {e}")
            return {
                'healthy': False,
                'needs_expansion': False,
                'message': f'Error: {e}',
                'error': True
            }
    
    def _get_storage_volume_id(self, storage_name: Optional[str] = None) -> Optional[str]:
        """Get storage volume ID by name, optionally cached."""
        # If no storage name provided, use the configured default
        if storage_name is None:
            storage_name = self.storage_name
            # Return cached value for default storage
            if self._storage_volume_id is not None:
                return self._storage_volume_id
        
        if not storage_name:
            logger.info("No storage name configured")
            return None
        
        logger.info(f"Looking up storage volume: {storage_name}")
        volumes = get_network_volumes(self.api_key)
        
        for vol in volumes:
            if vol.get('name') == storage_name:
                volume_id = vol.get('id')
                dc_info = vol.get('dataCenter', {})
                logger.info(f"Found storage '{storage_name}' (ID: {volume_id})")
                logger.info(f"  Size: {vol.get('size')}GB")
                logger.info(f"  Location: {dc_info.get('name')} ({dc_info.get('location')})")
                
                # Cache only if this is the default storage
                if storage_name == self.storage_name:
                    self._storage_volume_id = volume_id
                
                return volume_id
        
        logger.warning(f"Storage '{storage_name}' not found. Available volumes:")
        for vol in volumes:
            dc_info = vol.get('dataCenter', {})
            logger.warning(f"  • {vol.get('name')} (ID: {vol.get('id')}) - {vol.get('size')}GB")
            logger.warning(f"    Location: {dc_info.get('name')} ({dc_info.get('location')})")
        
        return None
    
    def _get_gpu_type_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU type information, cached."""
        if self._gpu_type_info is not None:
            return self._gpu_type_info
            
        self._gpu_type_info = find_gpu_type(self.gpu_type, self.api_key)
        if self._gpu_type_info:
            logger.info(f"Found GPU type: {self._gpu_type_info.get('displayName')} (ID: {self._gpu_type_info.get('id')})")
        else:
            logger.error(f"GPU type '{self.gpu_type}' not found")
        
        return self._gpu_type_info
    
    def _get_public_key_content(self) -> Optional[str]:
        """Get SSH public key content from environment variable or file path."""
        # First try to get from environment variable (for Railway deployment)
        public_key_env = os.getenv("RUNPOD_SSH_PUBLIC_KEY")
        if public_key_env:
            logger.info(f"[SSH_DEBUG] Using SSH public key from RUNPOD_SSH_PUBLIC_KEY environment variable")
            logger.info(f"[SSH_DEBUG] Key preview: {public_key_env[:50]}...{public_key_env[-20:]}")
            return public_key_env.strip()
        
        # Fallback to file path (for local development)
        if not self.ssh_public_key_path:
            logger.warning("No SSH public key configured. Set RUNPOD_SSH_PUBLIC_KEY environment variable or RUNPOD_SSH_PUBLIC_KEY_PATH")
            return None
            
        pub_path = os.path.expanduser(self.ssh_public_key_path)
        if not os.path.exists(pub_path):
            logger.warning(f"SSH public key not found at {pub_path}")
            return None
            
        try:
            with open(pub_path, "r", encoding="utf-8") as f:
                logger.debug(f"Using SSH public key from file: {pub_path}")
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading SSH public key: {e}")
            return None
    
    def spawn_worker(self, worker_id: str, worker_env: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Spawn a new GPU worker on Runpod using storage + RAM tiered fallback strategy.
        
        Strategy:
        1. Try all storage volumes with HIGH RAM (60+ GB) first
        2. If all fail, try all storage volumes with LOWER RAM tiers
        
        This maximizes chance of getting high-RAM instances across different datacenter locations.
        """
        gpu_info = self._get_gpu_type_info()
        if not gpu_info:
            logger.error("Cannot spawn worker: GPU type not available")
            return None
        
        # Prepare environment variables for the worker
        supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        env_vars = {
            "WORKER_ID": worker_id,
            "SUPABASE_URL": supabase_url,
            "SUPABASE_SERVICE_ROLE_KEY": os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
            "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY", ""),
            "REPLICATE_API_TOKEN": os.getenv("REPLICATE_API_TOKEN", ""),
            # Pass the correct edge function URLs to GPU workers
            "SUPABASE_EDGE_COMPLETE_TASK_URL": f"{supabase_url}/functions/v1/complete_task" if supabase_url else "",
            "SUPABASE_EDGE_MARK_FAILED_URL": f"{supabase_url}/functions/v1/mark-task-failed" if supabase_url else "",
        }
        
        # Merge in any additional environment variables
        if worker_env:
            env_vars.update(worker_env)
        
        # Get public key content for injection (like user's example)
        public_key_content = self._get_public_key_content()
        if public_key_content:
            logger.info(f"[SSH_DEBUG] Will inject SSH public key into worker {worker_id}")
        else:
            logger.error(f"[SSH_DEBUG] No SSH public key available for worker {worker_id} - authentication will fail!")
        
        # Determine RAM tier strategy
        # New strategy: Try each RAM tier across ALL storages before falling back to next tier
        # This prioritizes 72GB machines (2.3% failure rate) over 60GB (16.7% failure rate)
        if self.ram_tiers_enabled:
            # Filter RAM tiers based on configured minimum
            ram_tiers = [tier for tier in self.ram_tiers if tier >= self.min_memory_gb]
            
            if not ram_tiers:
                ram_tiers = [self.min_memory_gb]
            
            logger.info(f"🎯 RAM tier fallback enabled (tries each tier across all storages)")
            logger.info(f"   RAM tiers (in order): {ram_tiers} GB")
            logger.info(f"   Storage volumes: {self.storage_volumes}")
        else:
            # Simple mode: just try configured minimum
            ram_tiers = [self.min_memory_gb]
        
        # Try each RAM tier across ALL storage volumes before moving to next tier
        # This ensures we get 72GB machines whenever possible (lowest failure rate)
        last_error = None
        for ram_tier in ram_tiers:
            logger.info(f"🔍 Trying {ram_tier}GB RAM across all storage volumes...")
            
            for storage_name in self.storage_volumes:
                storage_volume_id = self._get_storage_volume_id(storage_name)
                if not storage_volume_id:
                    logger.warning(f"⚠️  Storage '{storage_name}' not found, skipping...")
                    continue
                
                # Check and expand storage if needed (adds +50GB if total < 100GB)
                # Only do this once per storage (on first RAM tier attempt)
                if ram_tier == ram_tiers[0]:
                    self._check_and_expand_storage(storage_name, storage_volume_id, min_free_gb=50)
                
                try:
                    logger.info(f"🚀 Creating worker: {worker_id} (Storage: {storage_name}, RAM: {ram_tier} GB)")
                    
                    pod_details = create_pod_and_wait(
                        api_key=self.api_key,
                        gpu_type_id=gpu_info["id"],
                        image_name=self.worker_image,
                        name=worker_id,
                        network_volume_id=storage_volume_id,
                        volume_mount_path=self.volume_mount_path,
                        disk_in_gb=self.disk_size_gb,
                        container_disk_in_gb=self.container_disk_gb,
                        min_vcpu_count=self.min_vcpu_count,
                        min_memory_in_gb=ram_tier,
                        public_key_string=public_key_content,
                        env_vars=env_vars,
                        template_id=self.template_id,
                    )
                    
                    if pod_details and 'id' in pod_details:
                        pod_id = pod_details['id']
                        logger.info(f"✅ SUCCESS: {worker_id} -> {pod_id} (Storage: {storage_name}, RAM: {ram_tier} GB)")
                        
                        return {
                            "worker_id": worker_id,
                            "runpod_id": pod_id,
                            "gpu_type": gpu_info["displayName"],
                            "status": "spawning",
                            "created_at": time.time(),
                            "pod_details": pod_details,
                            "ram_tier": ram_tier,
                            "storage_volume": storage_name,
                        }
                    else:
                        logger.warning(f"⚠️  Storage: {storage_name}, RAM: {ram_tier} GB - No ID returned")
                        last_error = "Pod creation returned no ID"
                        
                except Exception as e:
                    error_msg = str(e)
                    if "no longer any instances available" in error_msg.lower():
                        logger.warning(f"⚠️  Storage: {storage_name}, RAM: {ram_tier} GB - No instances available")
                        last_error = f"No instances available"
                    else:
                        logger.warning(f"⚠️  Storage: {storage_name}, RAM: {ram_tier} GB - {error_msg}")
                        last_error = error_msg
                    continue
            
            # If we get here, this RAM tier failed across all storages - try next tier
            if ram_tier != ram_tiers[-1]:
                logger.warning(f"⚠️  {ram_tier}GB RAM not available in any storage, trying {ram_tiers[ram_tiers.index(ram_tier)+1]}GB...")
        
        # All attempts failed
        logger.error(f"❌ Failed to create pod for worker {worker_id}")
        logger.error(f"   Tried storages: {self.storage_volumes}")
        logger.error(f"   Tried RAM tiers: {ram_tiers}")
        logger.error(f"   Last error: {last_error}")
        return None
    

    
    def start_worker_process(self, runpod_id: str, worker_id: str, has_pending_tasks: bool = False) -> bool:
        """
        Start the actual worker process in the background.
        This runs the worker.py script with Supabase configuration.
        
        Args:
            runpod_id: The RunPod pod ID
            worker_id: The worker ID
            has_pending_tasks: If True, skip model preloading so worker can claim tasks faster
        """
        # Ensure environment variables are loaded
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get Supabase credentials from environment
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", "")
        supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        
        logger.info(f"Starting worker process on {runpod_id} with worker_id: {worker_id}")
        logger.debug(f"Environment: SUPABASE_URL={supabase_url}, SERVICE_KEY={'***' if supabase_service_key else 'EMPTY'}")
        
        # Create a robust startup script that handles all the steps
        startup_script = f"""#!/bin/bash
set -e  # Exit on any error

# Set environment variables
export WORKER_ID="{worker_id}"
export SUPABASE_URL="{supabase_url}"
export SUPABASE_ANON_KEY="{supabase_anon_key}"
export SUPABASE_SERVICE_ROLE_KEY="{supabase_service_key}"
export SUPABASE_SERVICE_KEY="{supabase_service_key}"
export REPLICATE_API_TOKEN="{os.getenv('REPLICATE_API_TOKEN', '')}"

# Non-interactive apt
export DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# EARLY LOGGING (must exist before any apt/network operations)
# ------------------------------------------------------------
# Write all stdout/stderr to a single per-worker log from the very beginning.
LOG_FILE="/tmp/worker_startup_{worker_id}.log"
APT_UPDATE_LOG="/tmp/apt-update-{worker_id}.log"
APT_INSTALL_LOG="/tmp/apt-install-{worker_id}.log"
mkdir -p /tmp
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "🚀 WORKER STARTUP SCRIPT EXECUTION BEGIN"
echo "========================================="
echo "Worker ID: $WORKER_ID"
echo "Timestamp: $(date -Iseconds)"
echo "Initial PWD: $(pwd)"
echo "USER: $(whoami)"
echo "Kernel: $(uname -a | head -c 200)"
echo "Log file: $LOG_FILE"
echo

# Error handling with useful tails
trap 'echo "❌ SCRIPT FAILED at line $LINENO with exit code $? at $(date -Iseconds)"; \
      echo "--- tail $APT_UPDATE_LOG"; tail -200 "$APT_UPDATE_LOG" 2>/dev/null || true; \
      echo "--- tail $APT_INSTALL_LOG"; tail -200 "$APT_INSTALL_LOG" 2>/dev/null || true; \
      exit 1' ERR

# Apt helper with timeouts/retries (prevents indefinite hangs)
apt_retry() {{
    local name="$1"
    local timeout_sec="$2"
    shift 2
    local attempt rc
    for attempt in 1 2 3; do
        echo "📦 $name (attempt $attempt/3, timeout ${{timeout_sec}}s): $*"
        rm -f "$APT_UPDATE_LOG" "$APT_INSTALL_LOG" 2>/dev/null || true
        if timeout "$timeout_sec" "$@" ; then
            echo "✅ $name succeeded"
            return 0
        fi
        rc=$?
        echo "⚠️  $name failed (rc=$rc)"
        # Show recent output if we captured any
        tail -80 "$APT_UPDATE_LOG" 2>/dev/null || true
        tail -80 "$APT_INSTALL_LOG" 2>/dev/null || true
        sleep $((attempt * 5))
    done
    echo "❌ $name failed after 3 attempts"
    return 1
}}

# Pick worker repo directory (supports both legacy + renamed repo)
WORKSPACE_DIR="/workspace"
PRIMARY_DIR="$WORKSPACE_DIR/Headless-Wan2GP"
FALLBACK_DIR="$WORKSPACE_DIR/Reigh-Worker"

if [ -d "$PRIMARY_DIR" ]; then
    WORKDIR="$PRIMARY_DIR"
    echo "✅ Using worker directory: $WORKDIR"
elif [ -d "$FALLBACK_DIR" ]; then
    WORKDIR="$FALLBACK_DIR"
    echo "⚠️  Headless-Wan2GP not found; using fallback worker directory: $WORKDIR"
else
    echo "📦 Neither Headless-Wan2GP nor Reigh-Worker found. Cloning Headless-Wan2GP into $PRIMARY_DIR..."
    cd "$WORKSPACE_DIR" || exit 1
    git clone https://github.com/peteromallet/Headless-Wan2GP || exit 1
    WORKDIR="$PRIMARY_DIR"
fi

# Ensure a stable log location in the repo (symlink to /tmp early log)
LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"
ln -sf "$LOG_FILE" "$LOG_DIR/{worker_id}.log" || true
echo "🔗 Log symlink: $LOG_DIR/{worker_id}.log -> $LOG_FILE"

# Start Jupyter Lab immediately so it's available while worker initializes
echo "🚀 Starting Jupyter Lab on port 8888 (root: /workspace)..."
cd /workspace
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --ServerApp.root_dir=/workspace > /var/log/jupyter.log 2>&1 &
JUPYTER_PID=$!
echo "✅ Jupyter Lab started (PID: $JUPYTER_PID)"

cd "$WORKDIR" || exit 1

# Ensure core system dependencies exist (needed for venv + video processing)
echo "Installing system dependencies (python3.10-venv ffmpeg git curl wget)..."
# Use explicit logs for postmortem; command output also goes to $LOG_FILE via exec/tee.
apt_retry "apt-get update" 300 bash -lc "apt-get -o Dpkg::Use-Pty=0 -o Acquire::Retries=3 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 update > '$APT_UPDATE_LOG' 2>&1"
apt_retry "apt-get install" 600 bash -lc "apt-get -o Dpkg::Use-Pty=0 -o Acquire::Retries=3 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 install -y python3.10-venv ffmpeg git curl wget > '$APT_INSTALL_LOG' 2>&1"
echo "✅ System dependencies installed"

# Check if venv exists, if not create it and install dependencies
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "⚠️  Virtual environment missing or incomplete, building..."

    echo "Creating virtual environment..."
    python3.10 -m venv venv || exit 1

    echo "Activating venv and installing PyTorch..."
    source venv/bin/activate || exit 1
    pip install --no-cache-dir torch==2.6.0 torchvision torchaudio -f https://download.pytorch.org/whl/cu124 || exit 1

    if [ -f Wan2GP/requirements.txt ]; then
        echo "Installing Wan2GP requirements..."
        pip install --no-cache-dir -r Wan2GP/requirements.txt || exit 1
    fi

    echo "Installing worker requirements..."
    pip install --no-cache-dir -r requirements.txt || exit 1

    echo "✅ Virtual environment build complete"
else
    echo "✅ Virtual environment exists, activating..."
    source venv/bin/activate || exit 1

    # Check if dependencies are installed by testing for a key package
    if ! python -c "import torch, dotenv" 2>/dev/null; then
        echo "⚠️  Dependencies missing in venv, installing..."

        echo "Installing PyTorch..."
        pip install --no-cache-dir torch==2.6.0 torchvision torchaudio -f https://download.pytorch.org/whl/cu124 || exit 1

        if [ -f Wan2GP/requirements.txt ]; then
            echo "Installing Wan2GP requirements..."
            pip install --no-cache-dir -r Wan2GP/requirements.txt || exit 1
        fi

        echo "Installing worker requirements..."
        pip install --no-cache-dir -r requirements.txt || exit 1

        echo "✅ Dependencies installed"
    else
        echo "✅ Dependencies already installed"
    fi
fi

echo "✅ Entering worker directory: $WORKDIR"

# Change to worker directory
cd "$WORKDIR"

echo "✅ Now in directory: $(pwd)" >> "$LOG_FILE" 2>&1
echo "✅ Directory contents:" >> "$LOG_FILE" 2>&1
ls -la >> "$LOG_FILE" 2>&1

echo "Worker ID: $WORKER_ID" >> "$LOG_FILE" 2>&1

# Try git pull (but don't fail if it times out)
echo "=== GIT PULL ===" >> "$LOG_FILE" 2>&1

# Capture commit before pull
BEFORE_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
echo "Before commit: $BEFORE_COMMIT" >> "$LOG_FILE" 2>&1

# Perform pull with timeout and record exit status
# Use || true to prevent set -e from exiting on failure (divergent branches, conflicts, etc.)
timeout 30 git pull --ff-only origin main >> "$LOG_FILE" 2>&1 || {{
    GIT_PULL_EXIT=$?
    echo "Git pull failed (exit $GIT_PULL_EXIT), trying git reset --hard to sync with remote..." >> "$LOG_FILE" 2>&1
    # Reset to remote state to handle divergent branches
    git fetch origin main >> "$LOG_FILE" 2>&1 || true
    git reset --hard origin/main >> "$LOG_FILE" 2>&1 || {{
        echo "Git reset also failed, continuing with existing code" >> "$LOG_FILE" 2>&1
    }}
}}

# Capture commit after pull to detect if code actually changed
AFTER_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
echo "After commit:  $AFTER_COMMIT" >> "$LOG_FILE" 2>&1

# Verify critical dependencies
echo "=== VERIFYING DEPENDENCIES ===" >> $LOG_FILE 2>&1
if command -v ffmpeg >/dev/null 2>&1; then
    echo "✅ FFmpeg found: $(which ffmpeg)" >> $LOG_FILE 2>&1
    echo "✅ FFmpeg version: $(ffmpeg -version 2>&1 | head -1)" >> $LOG_FILE 2>&1
else
    echo "❌ ERROR: FFmpeg not found! Worker cannot process videos" >> $LOG_FILE 2>&1
fi

if command -v git >/dev/null 2>&1; then
    echo "✅ Git found: $(which git)" >> $LOG_FILE 2>&1
else
    echo "❌ WARNING: Git not found!" >> $LOG_FILE 2>&1
fi

if command -v python3.10 >/dev/null 2>&1; then
    echo "✅ Python 3.10 found: $(which python3.10)" >> $LOG_FILE 2>&1
else
    echo "❌ WARNING: Python 3.10 not found!" >> $LOG_FILE 2>&1
fi

# Activate virtual environment
echo "=== ACTIVATING VIRTUAL ENV ===" >> $LOG_FILE 2>&1
source venv/bin/activate
echo "Virtual env activated: $VIRTUAL_ENV" >> $LOG_FILE 2>&1
echo "Python path: $(which python)" >> $LOG_FILE 2>&1
echo "Python version: $(python --version)" >> $LOG_FILE 2>&1

# Update Python dependencies only if requirements.txt changed (hash-based check)
echo "=== DEPENDENCY UPDATE (hash-based) ===" >> $LOG_FILE 2>&1

# Compute current hash of requirements files
MAIN_REQS_HASH=$(md5sum requirements.txt 2>/dev/null | cut -d' ' -f1 || echo "none")
SUB_REQS_HASH=$(md5sum Wan2GP/requirements.txt 2>/dev/null | cut -d' ' -f1 || echo "none")
CURRENT_HASH="${{MAIN_REQS_HASH}}_${{SUB_REQS_HASH}}"

# Load cached hash (stored in venv to persist across restarts)
CACHED_HASH=$(cat venv/.requirements_hash 2>/dev/null || echo "")

echo "Current requirements hash: $CURRENT_HASH" >> $LOG_FILE 2>&1
echo "Cached requirements hash: $CACHED_HASH" >> $LOG_FILE 2>&1

if [ "$CURRENT_HASH" != "$CACHED_HASH" ]; then
    echo "Requirements changed! Installing/upgrading Python deps..." >> $LOG_FILE 2>&1
    python -m pip install --upgrade -r requirements.txt >> $LOG_FILE 2>&1 || echo "WARNING: pip install failed" >> $LOG_FILE 2>&1
    
    # Also install subfolder requirements if present
    if [ -f Wan2GP/requirements.txt ]; then
        echo "Installing subfolder requirements from Wan2GP/requirements.txt" >> $LOG_FILE 2>&1
        python -m pip install --upgrade -r Wan2GP/requirements.txt >> $LOG_FILE 2>&1 || echo "WARNING: subfolder pip install failed" >> $LOG_FILE 2>&1
    fi
    
    # Save new hash
    echo "$CURRENT_HASH" > venv/.requirements_hash
    echo "✅ Dependencies updated, hash saved" >> $LOG_FILE 2>&1
else
    echo "✅ Requirements unchanged, skipping pip install" >> $LOG_FILE 2>&1
fi

# Install all known missing packages from Headless-Wan2GP requirements.txt
echo "=== INSTALLING CRITICAL PACKAGES ===" >> $LOG_FILE 2>&1
python -m pip install --quiet \
    mmgp==3.6.10 \
    GitPython \
    smplfitter \
    s3tokenizer \
    conformer \
    >> $LOG_FILE 2>&1 || echo "WARNING: some critical package installs failed" >> $LOG_FILE 2>&1

# Validate all critical imports before starting worker
echo "=== VALIDATING IMPORTS ===" >> $LOG_FILE 2>&1
python << 'VALIDATE_EOF' >> $LOG_FILE 2>&1
import sys
failed = []
packages = [
    ('mmgp', 'mmgp'),
    ('mmgp.fp8_quanto_bridge', 'mmgp'),
    ('git', 'GitPython'),
    ('smplfitter', 'smplfitter'),
    ('s3tokenizer', 's3tokenizer'),
    ('conformer', 'conformer'),
]
for mod, pkg in packages:
    try:
        __import__(mod)
        print(f"OK: {{mod}}")
    except ImportError as e:
        print(f"MISSING: {{mod}} (pip install {{pkg}})")
        failed.append(pkg)

if failed:
    print(f"Missing packages: {{', '.join(failed)}}")
    print("Attempting to install...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet'] + list(set(failed)))
    # Re-check
    still_missing = []
    for mod, pkg in packages:
        try:
            __import__(mod)
        except ImportError:
            still_missing.append(pkg)
    if still_missing:
        print(f"FATAL: Still missing after install: {{', '.join(still_missing)}}")
        sys.exit(1)
    print("All packages installed successfully on retry")
else:
    print("All critical imports validated")
VALIDATE_EOF

if [ $? -ne 0 ]; then
    echo "❌ Import validation failed!" >> $LOG_FILE 2>&1
    exit 1
fi

# Verify worker.py exists
echo "=== CHECKING FILES ===" >> $LOG_FILE 2>&1
ls -la worker.py >> $LOG_FILE 2>&1

# Final pre-flight checks before starting worker
echo "=== PRE-FLIGHT CHECKS ===" >> $LOG_FILE 2>&1
echo "✅ Virtual env: $VIRTUAL_ENV" >> $LOG_FILE 2>&1
echo "✅ Python: $(which python) ($(python --version 2>&1))" >> $LOG_FILE 2>&1

if [ -f worker.py ]; then
    echo "✅ worker.py exists ($(wc -l < worker.py) lines)" >> $LOG_FILE 2>&1
else
    echo "❌ ERROR: worker.py not found!" >> $LOG_FILE 2>&1
    exit 1
fi

echo "✅ Checking environment variables..." >> $LOG_FILE 2>&1
echo "WORKER_ID: $WORKER_ID" >> $LOG_FILE 2>&1
echo "SUPABASE_URL: ${{SUPABASE_URL:0:30}}..." >> $LOG_FILE 2>&1
echo "SUPABASE_ANON_KEY: ${{SUPABASE_ANON_KEY:0:20}}..." >> $LOG_FILE 2>&1
echo "SUPABASE_SERVICE_ROLE_KEY: ${{SUPABASE_SERVICE_ROLE_KEY:0:20}}..." >> $LOG_FILE 2>&1

# Start the actual worker process
echo "=== STARTING MAIN WORKER ===" >> $LOG_FILE 2>&1
PRELOAD_FLAG="{'' if has_pending_tasks else '--preload-model wan_2_2_i2v_lightning_baseline_2_2_2'}"
WORKER_CMD="python worker.py --supabase-url $SUPABASE_URL --supabase-access-token $SUPABASE_SERVICE_ROLE_KEY --worker $WORKER_ID --debug $PRELOAD_FLAG --wgp-profile 1"
echo "Command: $WORKER_CMD" >> $LOG_FILE 2>&1
echo "Preload model: {'NO (tasks pending)' if has_pending_tasks else 'YES (no tasks pending)'}" >> $LOG_FILE 2>&1
echo "Starting at: $(date)" >> $LOG_FILE 2>&1

# Start worker in background with comprehensive logging
nohup $WORKER_CMD >> $LOG_FILE 2>&1 &
WORKER_PID=$!

echo "✅ Worker process started with PID: $WORKER_PID at $(date)" >> $LOG_FILE 2>&1

# Give the worker a moment to start and check if it's still running
sleep 2
if kill -0 $WORKER_PID 2>/dev/null; then
    echo "✅ Worker process $WORKER_PID is still running after 2 seconds" >> $LOG_FILE 2>&1
else
    echo "❌ ERROR: Worker process $WORKER_PID died immediately!" >> $LOG_FILE 2>&1
    echo "Exit status was: $?" >> $LOG_FILE 2>&1
fi

echo "=========================================" >> $LOG_FILE 2>&1
echo "🏁 STARTUP SCRIPT COMPLETED SUCCESSFULLY" >> $LOG_FILE 2>&1
echo "=========================================" >> $LOG_FILE 2>&1
"""

        # Write the script to a temporary file and execute it
        script_path = f"/tmp/start_worker_{worker_id}.sh"
        
        logger.info(f"Creating startup script at {script_path} for worker {worker_id}")
        
        # First, create the script file
        create_script_command = f"cat > {script_path} << 'SCRIPT_EOF'\n{startup_script}\nSCRIPT_EOF"
        
        result = self.execute_command_on_worker(runpod_id, create_script_command, timeout=10)
        if not result or result[0] != 0:
            logger.error(f"Failed to create startup script for worker {worker_id}: {result}")
            return False
        
        logger.info(f"Startup script created successfully, launching in background...")
        
        # Launch script in background so it doesn't block the orchestrator
        # The script will run for ~20 minutes installing dependencies
        launch_command = f"""
        WORKSPACE_DIR="/workspace"
        PRIMARY_DIR="$WORKSPACE_DIR/Headless-Wan2GP"
        FALLBACK_DIR="$WORKSPACE_DIR/Reigh-Worker"
        if [ -d "$PRIMARY_DIR" ]; then
            WORKDIR="$PRIMARY_DIR"
        elif [ -d "$FALLBACK_DIR" ]; then
            WORKDIR="$FALLBACK_DIR"
        else
            WORKDIR="$PRIMARY_DIR"
        fi
        mkdir -p "$WORKDIR/logs"
        chmod +x {script_path}
        nohup {script_path} > "$WORKDIR/logs/{worker_id}_startup.log" 2>&1 &
        echo $!
        """
        
        result = self.execute_command_on_worker(runpod_id, launch_command, timeout=10)
        
        if result:
            exit_code, stdout, stderr = result
            if exit_code == 0:
                pid = stdout.strip()
                logger.info(f"✅ Worker {worker_id} startup script launched in background (PID: {pid})")
                logger.info(f"   Script will run for ~20 minutes to install dependencies")
                logger.info(f"   Worker will be checked periodically for completion")
                return True
            else:
                logger.error(f"Failed to launch startup script: exit code {exit_code}")
                if stderr and stderr.strip():
                    logger.error(f"Error: {stderr.strip()}")
                return False
        else:
            logger.error(f"Failed to execute worker startup launch command")
            return False
    
    def check_worker_startup_status(self, worker_id: str, runpod_id: str) -> Dict[str, Any]:
        """Check status of a worker that's currently starting up.
        
        Returns:
            Dict with keys:
                - status: 'initializing', 'active', 'failed'
                - message: Human-readable status message
                - logs: Startup logs if available
        """
        try:
            # Check if worker is logging to Supabase (means worker.py started)
            from gpu_orchestrator.database import DatabaseClient
            db = DatabaseClient()
            
            logs = db.supabase.table('system_logs').select('id').eq('source_type', 'worker').eq('worker_id', worker_id).limit(1).execute()
            
            if logs.data:
                # Worker is actively logging - retrieve and log startup logs
                logger.info(f"✅ Worker {worker_id} is now logging - retrieving startup logs")
                
                log_retrieval_command = f"""
                WORKSPACE_DIR="/workspace"
                PRIMARY_DIR="$WORKSPACE_DIR/Headless-Wan2GP"
                FALLBACK_DIR="$WORKSPACE_DIR/Reigh-Worker"
                if [ -d "$PRIMARY_DIR" ]; then
                    WORKDIR="$PRIMARY_DIR"
                elif [ -d "$FALLBACK_DIR" ]; then
                    WORKDIR="$FALLBACK_DIR"
                else
                    WORKDIR="$PRIMARY_DIR"
                fi
                if [ -f "$WORKDIR/logs/gpu_{worker_id}.log" ]; then
                    echo "=== WORKER STARTUP LOG ==="
                    tail -100 "$WORKDIR/logs/gpu_{worker_id}.log"
                fi
                if [ -f "$WORKDIR/logs/{worker_id}_startup.log" ]; then
                    echo "=== INITIALIZATION LOG ==="
                    tail -200 "$WORKDIR/logs/{worker_id}_startup.log"
                fi
                """
                
                result = self.execute_command_on_worker(runpod_id, log_retrieval_command, timeout=15)
                if result and result[1]:
                    startup_logs = result[1]
                    logger.info(f"📋 Startup logs for {worker_id}:\n{startup_logs}")
                    
                    return {
                        'status': 'active',
                        'message': 'Worker successfully started and logging',
                        'logs': startup_logs
                    }
                else:
                    return {
                        'status': 'active',
                        'message': 'Worker is logging but could not retrieve startup logs',
                        'logs': None
                    }
            else:
                # Worker not logging yet - check for errors in startup log AND disk space
                check_command = f"""
                echo "=== DISK SPACE ==="
                df -h / /tmp /var 2>/dev/null | head -10
                echo ""
                WORKSPACE_DIR="/workspace"
                PRIMARY_DIR="$WORKSPACE_DIR/Headless-Wan2GP"
                FALLBACK_DIR="$WORKSPACE_DIR/Reigh-Worker"
                if [ -d "$PRIMARY_DIR" ]; then
                    WORKDIR="$PRIMARY_DIR"
                elif [ -d "$FALLBACK_DIR" ]; then
                    WORKDIR="$FALLBACK_DIR"
                else
                    WORKDIR="$PRIMARY_DIR"
                fi
                if [ -f "$WORKDIR/logs/{worker_id}_startup.log" ]; then
                    echo "=== STARTUP LOG ERRORS ==="
                    tail -50 "$WORKDIR/logs/{worker_id}_startup.log" | grep -i "error\\|fail\\|exception\\|no space" || echo "No errors found in recent logs"
                else
                    echo "=== STARTUP LOG ==="
                    echo "No startup log file yet"
                fi
                """
                
                result = self.execute_command_on_worker(runpod_id, check_command, timeout=10)
                if result and result[1]:
                    output = result[1]
                    
                    # Check for disk space issues
                    if '100%' in output or 'No space left' in output.lower():
                        logger.error(f"❌ DISK SPACE ISSUE on {worker_id}:\n{output}")
                        return {
                            'status': 'initializing',
                            'message': 'Disk space critically low!',
                            'logs': output
                        }
                    
                    # Log disk space info for debugging
                    logger.info(f"📊 Worker {worker_id} system status:\n{output}")
                    
                    if 'error' in output.lower() or 'fail' in output.lower():
                        logger.warning(f"⚠️ Potential errors in {worker_id} startup:\n{output}")
                        return {
                            'status': 'initializing',
                            'message': 'Worker still initializing, potential errors detected',
                            'logs': output
                        }
                
                return {
                    'status': 'initializing',
                    'message': 'Worker still installing dependencies',
                    'logs': None
                }
                
        except Exception as e:
            logger.error(f"Error checking worker {worker_id} startup status: {e}")
            return {
                'status': 'unknown',
                'message': f'Error checking status: {e}',
                'logs': None
            }
    
    def terminate_worker(self, runpod_id: str) -> bool:
        """Terminate a worker pod on Runpod."""
        try:
            logger.info(f"Terminating pod: {runpod_id}")
            terminate_pod(runpod_id, self.api_key)
            logger.info(f"Pod terminated: {runpod_id}")
            return True
        except Exception as e:
            logger.error(f"Error terminating pod {runpod_id}: {e}")
            return False
    
    def get_pod_status(self, runpod_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a pod."""
        runpod.api_key = self.api_key
        try:
            status = runpod.get_pod(runpod_id)
            if not status:
                return None
            
            # Extract key status information
            runtime = status.get("runtime", {})
            return {
                "runpod_id": runpod_id,
                "desired_status": status.get("desiredStatus"),
                "actual_status": status.get("actualStatus"),
                "ip": runtime.get("ip"),
                "ports": runtime.get("ports", []),
                "ssh_password": runtime.get("sshPassword"),
                "created_at": status.get("createdAt"),
                "last_status_change": status.get("lastStatusChange"),
                "uptime_seconds": runtime.get("uptimeInSeconds", 0),
                "cost_per_hr": status.get("costPerHr"),
            }
        except Exception as e:
            logger.error(f"Error getting pod status for {runpod_id}: {e}")
            return None
    
    def get_ssh_client(self, runpod_id: str) -> Optional[SSHClient]:
        """Get an SSH client for connecting to a worker pod."""
        logger.info(f"🔐 SSH_AUTH [Pod {runpod_id}] Getting SSH client - starting authentication flow")
        
        ssh_details = get_pod_ssh_details(runpod_id, self.api_key)
        if not ssh_details:
            logger.error(f"🔐 SSH_AUTH [Pod {runpod_id}] ❌ FAILED: Could not get SSH details from RunPod API")
            return None
        
        logger.info(f"🔐 SSH_AUTH [Pod {runpod_id}] SSH details obtained - IP: {ssh_details['ip']}, Port: {ssh_details['port']}")
        
        # Check environment variables for SSH keys
        private_key_env = os.getenv("RUNPOD_SSH_PRIVATE_KEY")
        public_key_env = os.getenv("RUNPOD_SSH_PUBLIC_KEY")
        private_key_path_env = os.getenv("RUNPOD_SSH_PRIVATE_KEY_PATH")
        
        logger.info(f"🔐 SSH_AUTH [Pod {runpod_id}] Environment check:")
        logger.info(f"🔐 SSH_AUTH [Pod {runpod_id}]   - RUNPOD_SSH_PRIVATE_KEY: {'✅ SET' if private_key_env else '❌ MISSING'}")
        logger.info(f"🔐 SSH_AUTH [Pod {runpod_id}]   - RUNPOD_SSH_PUBLIC_KEY: {'✅ SET' if public_key_env else '❌ MISSING'}")
        logger.info(f"🔐 SSH_AUTH [Pod {runpod_id}]   - RUNPOD_SSH_PRIVATE_KEY_PATH: {'✅ SET' if private_key_path_env else '❌ MISSING'}")
        
        # Try private key from environment variable first (for Railway)
        if private_key_env:
            logger.info(f"🔐 SSH_AUTH [Pod {runpod_id}] ✅ Using PRIVATE KEY from environment variable")
            logger.info(f"🔐 SSH_AUTH [Pod {runpod_id}] Private key length: {len(private_key_env)} chars")
            logger.info(f"🔐 SSH_AUTH [Pod {runpod_id}] Private key starts with: {private_key_env[:50]}...")
            return SSHClient(
                hostname=ssh_details['ip'],
                port=ssh_details['port'],
                username='root',
                private_key_content=private_key_env,
            )
        
        # Fallback to private key file path (for local development)
        if self.ssh_private_key_path and os.path.exists(os.path.expanduser(self.ssh_private_key_path)):
            logger.info(f"🔐 SSH_AUTH [Pod {runpod_id}] ✅ Using PRIVATE KEY from file path: {self.ssh_private_key_path}")
            return SSHClient(
                hostname=ssh_details['ip'],
                port=ssh_details['port'],
                username='root',
                private_key_path=self.ssh_private_key_path,
            )
        else:
            logger.warning(f"🔐 SSH_AUTH [Pod {runpod_id}] ⚠️  FALLING BACK to PASSWORD authentication")
            logger.warning(f"🔐 SSH_AUTH [Pod {runpod_id}] This will likely FAIL as RunPod requires key-based auth")
            logger.warning(f"🔐 SSH_AUTH [Pod {runpod_id}] Password from RunPod: {ssh_details.get('password', 'runpod')}")
            return SSHClient(
                hostname=ssh_details['ip'],
                port=ssh_details['port'],
                username='root',
                password=ssh_details.get('password', 'runpod'),
            )
    
    def execute_command_on_worker(self, runpod_id: str, command: str, timeout: int = 600) -> Optional[tuple]:
        """Execute a command on a worker via SSH."""
        logger.info(f"🔐 SSH_EXEC [Pod {runpod_id}] Executing command: {command[:100]}...")
        
        ssh_client = self.get_ssh_client(runpod_id)
        if not ssh_client:
            logger.error(f"🔐 SSH_EXEC [Pod {runpod_id}] ❌ FAILED: Could not get SSH client")
            return None
        
        try:
            logger.info(f"🔐 SSH_EXEC [Pod {runpod_id}] Attempting SSH connection...")
            ssh_client.connect()
            logger.info(f"🔐 SSH_EXEC [Pod {runpod_id}] ✅ SSH connection successful!")
            
            exit_code, stdout, stderr = ssh_client.execute_command(command, timeout)
            logger.info(f"🔐 SSH_EXEC [Pod {runpod_id}] Command completed - Exit code: {exit_code}")
            if stderr:
                logger.warning(f"🔐 SSH_EXEC [Pod {runpod_id}] Command stderr: {stderr[:200]}...")
            return exit_code, stdout, stderr
        except Exception as e:
            logger.error(f"🔐 SSH_EXEC [Pod {runpod_id}] ❌ SSH EXECUTION FAILED: {e}")
            logger.error(f"🔐 SSH_EXEC [Pod {runpod_id}] This indicates SSH authentication or connection issues")
            return None
        finally:
            if ssh_client:
                ssh_client.disconnect()
    
    def get_network_volumes(self) -> list:
        """Get list of available network volumes."""
        return get_network_volumes(self.api_key)
    
    def generate_worker_id(self) -> str:
        """Generate a unique worker ID for Runpod."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"gpu-{timestamp}-{str(uuid.uuid4())[:8]}"

    def check_and_initialize_worker(self, worker_id: str, runpod_id: str) -> Dict[str, Any]:
        """
        Check if a spawning worker is ready for worker process startup.
        Returns status update for the worker.
        """
        try:
            # Check pod status
            runpod.api_key = self.api_key
            pod_status = runpod.get_pod(runpod_id)
            
            # More robust None checking
            if pod_status is None:
                logger.warning(f"Pod {runpod_id} status returned None (pod may be provisioning)")
                return {"status": "spawning", "message": "Pod status not available yet"}
            
            if not isinstance(pod_status, dict):
                logger.error(f"Pod {runpod_id} status returned unexpected type: {type(pod_status)}")
                return {"status": "error", "error": f"Invalid pod status response: {type(pod_status)}"}
            
            desired_status = pod_status.get("desiredStatus")
            runtime = pod_status.get("runtime", {})
            
            # Ensure runtime is a dict
            if runtime is None:
                runtime = {}
            elif not isinstance(runtime, dict):
                logger.warning(f"Pod {runpod_id} runtime is not a dict: {type(runtime)}")
                runtime = {}
            
            # Check if pod is running and has SSH access
            if desired_status == "RUNNING" and runtime.get("ports"):
                logger.info(f"Pod {runpod_id} is running, checking SSH access...")
                
                # Get SSH details with better error handling
                ssh_details = get_pod_ssh_details(runpod_id, self.api_key)
                
                if ssh_details and ssh_details.get('ip') and ssh_details.get('port'):
                    logger.info(f"SSH available for {worker_id}: {ssh_details['ip']}:{ssh_details['port']}")
                    
                    # Pod is ready for startup script launch, but worker should stay "spawning"
                    # until worker.py actually starts and begins logging
                    return {
                        "status": "spawning",
                        "ssh_details": ssh_details,
                        "ready": True,
                        "message": "Pod ready, startup script can be launched"
                    }
                else:
                    # Pod is running but SSH details are incomplete/missing
                    # Check if we have basic port info from runtime
                    ssh_port = None
                    ssh_ip = None
                    for port_map in runtime.get("ports", []):
                        if port_map.get("privatePort") == 22:
                            ssh_ip = port_map.get("ip")
                            ssh_port = port_map.get("publicPort")
                            break
                    
                    if ssh_ip and ssh_port:
                        logger.info(f"SSH details found in runtime for {worker_id}: {ssh_ip}:{ssh_port}")
                        # Pod is ready for startup script launch, but worker should stay "spawning"
                        return {
                            "status": "spawning",
                            "ssh_details": {
                                "ip": ssh_ip,
                                "port": ssh_port,
                                "password": "runpod"
                            },
                            "ready": True,
                            "message": "Pod ready, startup script can be launched"
                        }
                    else:
                        logger.warning(f"Pod {runpod_id} is running but SSH details incomplete - waiting...")
                        return {"status": "spawning", "message": "Waiting for complete SSH access"}
                    
            elif desired_status in ["FAILED", "TERMINATED"]:
                return {"status": "error", "error": f"Pod {desired_status.lower()}"}
            else:
                # Still provisioning
                return {"status": "spawning", "message": f"Pod status: {desired_status}"}
                
        except Exception as e:
            logger.error(f"Error checking worker {worker_id} (pod {runpod_id}): {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": "error", "error": f"Exception in worker check: {str(e)}"}


# Convenience functions for use in orchestrator
def create_runpod_client() -> RunpodClient:
    """Create a Runpod client using environment configuration."""
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY environment variable is required")
    
    return RunpodClient(api_key)


async def spawn_runpod_gpu(worker_id: str) -> Optional[str]:
    """
    Spawn a GPU worker on Runpod.
    
    Args:
        worker_id: Unique identifier for the worker
        
    Returns:
        Runpod pod ID if successful, None otherwise
    """
    client = create_runpod_client()
    result = client.spawn_worker(worker_id)
    
    if result:
        return result["runpod_id"]
    return None


async def terminate_runpod_gpu(runpod_id: str) -> bool:
    """
    Terminate a GPU worker on Runpod.
    
    Args:
        runpod_id: Runpod pod ID to terminate
        
    Returns:
        True if successful, False otherwise
    """
    client = create_runpod_client()
    return client.terminate_worker(runpod_id)


async def get_runpod_status(runpod_id: str) -> Optional[Dict[str, Any]]:
    """
    Get status of a Runpod GPU worker.
    
    Args:
        runpod_id: Runpod pod ID to check
        
    Returns:
        Status information or None if error
    """
    client = create_runpod_client()
    return client.get_pod_status(runpod_id) 
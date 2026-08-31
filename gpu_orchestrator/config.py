"""
Orchestrator configuration management.

All orchestrator configuration values in one place, loaded from environment variables
with sensible defaults. This replaces scattered os.getenv() calls throughout the codebase.
"""

from dataclasses import dataclass
import os
import logging

logger = logging.getLogger(__name__)


VALID_WORKER_BACKENDS = {"wgp", "vibecomfy"}
VALID_CAPACITY_RECONCILER_MODES = {"off", "shadow", "authoritative"}


def _env_str(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _env_optional_str(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_backend(*names: str, default: str = "wgp") -> str:
    value = _env_str(*names, default=default).lower()
    if value not in VALID_WORKER_BACKENDS:
        raise ValueError(
            f"Unsupported worker backend {value!r}; expected one of {sorted(VALID_WORKER_BACKENDS)}"
        )
    return value


def _env_capacity_reconciler_mode() -> str:
    value = _env_str("ORCHESTRATOR_CAPACITY_RECONCILER_MODE", default="off").lower()
    if value not in VALID_CAPACITY_RECONCILER_MODES:
        raise ValueError(
            "Unsupported ORCHESTRATOR_CAPACITY_RECONCILER_MODE "
            f"{value!r}; expected one of {sorted(VALID_CAPACITY_RECONCILER_MODES)}"
        )
    return value


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
    script_running_timeout_sec: int  # Longer timeout for SPAWNING_SCRIPT_RUNNING (install phase)
    gpu_idle_timeout_sec: int
    overcapacity_idle_timeout_sec: int
    task_stuck_timeout_sec: int
    graceful_shutdown_timeout_sec: int
    # Wall-clock cap applied to workers excluded from production capacity
    # control (e.g. live-test). Production workers default to None.
    excluded_worker_max_lifetime_sec: int

    # Health check timeouts (seconds)
    startup_grace_period_sec: int
    ready_not_claiming_timeout_sec: int
    gpu_not_detected_timeout_sec: int
    heartbeat_promotion_threshold_sec: int
    active_initializing_max_sec: int

    # Failure protection
    max_worker_failure_rate: float
    failure_window_minutes: int
    min_workers_for_rate_check: int
    
    # Per-worker task failure detection
    max_consecutive_task_failures: int  # Restart worker after N consecutive task failures
    task_failure_window_minutes: int    # Time window to look back for task failures

    # Storage monitoring
    storage_check_interval_cycles: int
    storage_min_free_gb: int
    storage_max_percent_used: int
    storage_expansion_increment_gb: int

    # Selected route / pool contract defaults
    worker_backend: str
    worker_profile: str
    worker_pool: str
    selector_namespace: str
    selector_version: str | None
    worker_contract_version: int
    worker_run_id: str | None

    # Error cleanup
    error_cleanup_grace_period_sec: int

    # Polling
    orchestrator_poll_sec: int

    # Auto-start worker process
    auto_start_worker_process: bool

    # Feature flags
    use_new_scaling_logic: bool
    capacity_reconciler_mode: str = "off"

    @classmethod
    def from_env(cls) -> 'OrchestratorConfig':
        """Load all config from environment with defaults."""
        poll_sec = int(os.getenv("ORCHESTRATOR_POLL_SEC", "30"))

        # Calculate min_active_gpus and max_active_gpus
        min_active = int(os.getenv("MIN_ACTIVE_GPUS", "2"))
        max_active = int(os.getenv("MAX_ACTIVE_GPUS", "10"))
        machines_idle = int(os.getenv("MACHINES_TO_KEEP_IDLE", "0"))

        # Validate and clamp machines_to_keep_idle
        if machines_idle < 0:
            logger.warning("MACHINES_TO_KEEP_IDLE cannot be negative, setting to 0")
            machines_idle = 0
        elif machines_idle > max_active:
            logger.warning(
                f"MACHINES_TO_KEEP_IDLE ({machines_idle}) exceeds MAX_ACTIVE_GPUS ({max_active}), "
                f"clamping to {max_active}"
            )
            machines_idle = max_active

        worker_backend = _env_backend("REIGH_BACKEND", "WORKER_BACKEND", default="wgp")
        worker_profile = _env_str("REIGH_WORKER_PROFILE", "WGP_PROFILE", default="1")
        selector_namespace = _env_str("REIGH_SELECTOR_NAMESPACE", "ROUTE_SELECTOR_NAMESPACE", default="production")
        worker_pool = _env_str(
            "REIGH_WORKER_POOL",
            "WORKER_POOL",
            default=f"gpu-{worker_backend}-{selector_namespace}",
        )

        return cls(
            # Scaling limits
            min_active_gpus=min_active,
            max_active_gpus=max_active,
            machines_to_keep_idle=machines_idle,
            tasks_per_gpu_threshold=int(os.getenv("TASKS_PER_GPU_THRESHOLD", "3")),

            # Scaling behavior
            scale_up_multiplier=float(os.getenv("SCALE_UP_MULTIPLIER", "1.0")),
            scale_down_multiplier=float(os.getenv("SCALE_DOWN_MULTIPLIER", "0.9")),
            min_scaling_interval_sec=int(os.getenv("MIN_SCALING_INTERVAL_SEC", "45")),
            spawning_grace_period_sec=int(os.getenv("SPAWNING_GRACE_PERIOD_SEC", "180")),
            scale_down_grace_period_sec=int(os.getenv("SCALE_DOWN_GRACE_PERIOD_SEC", "60")),

            # Worker timeouts
            spawning_timeout_sec=int(os.getenv("SPAWNING_TIMEOUT_SEC", "600")),
            script_running_timeout_sec=int(os.getenv("SCRIPT_RUNNING_TIMEOUT_SEC", "1800")),
            gpu_idle_timeout_sec=int(os.getenv("GPU_IDLE_TIMEOUT_SEC", "600")),
            overcapacity_idle_timeout_sec=int(os.getenv("GPU_OVERCAPACITY_IDLE_TIMEOUT_SEC", "30")),
            task_stuck_timeout_sec=int(os.getenv("TASK_STUCK_TIMEOUT_SEC", "1200")),
            graceful_shutdown_timeout_sec=int(os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT_SEC", "600")),
            excluded_worker_max_lifetime_sec=int(os.getenv("EXCLUDED_WORKER_MAX_LIFETIME_SEC", "7200")),

            # Health check timeouts
            startup_grace_period_sec=int(os.getenv("STARTUP_GRACE_PERIOD_SEC", "1200")),
            ready_not_claiming_timeout_sec=int(os.getenv("READY_NOT_CLAIMING_TIMEOUT_SEC", "600")),
            gpu_not_detected_timeout_sec=int(os.getenv("GPU_NOT_DETECTED_TIMEOUT_SEC", "300")),
            heartbeat_promotion_threshold_sec=int(os.getenv(
                "HEARTBEAT_PROMOTION_THRESHOLD_SEC",
                str(max(60, poll_sec * 3))
            )),
            active_initializing_max_sec=int(os.getenv("ACTIVE_INITIALIZING_MAX_SEC", "1800")),

            # Failure protection
            max_worker_failure_rate=float(os.getenv("MAX_WORKER_FAILURE_RATE", "0.8")),
            failure_window_minutes=int(os.getenv("FAILURE_WINDOW_MINUTES", "5")),
            min_workers_for_rate_check=int(os.getenv("MIN_WORKERS_FOR_RATE_CHECK", "5")),
            
            # Per-worker task failure detection
            max_consecutive_task_failures=int(os.getenv("MAX_CONSECUTIVE_TASK_FAILURES", "3")),
            task_failure_window_minutes=int(os.getenv("TASK_FAILURE_WINDOW_MINUTES", "30")),

            # Storage monitoring
            storage_check_interval_cycles=int(os.getenv("STORAGE_CHECK_INTERVAL_CYCLES", "10")),
            storage_min_free_gb=int(os.getenv("STORAGE_MIN_FREE_GB", "50")),
            storage_max_percent_used=int(os.getenv("STORAGE_MAX_PERCENT_USED", "85")),
            storage_expansion_increment_gb=int(os.getenv("STORAGE_EXPANSION_INCREMENT_GB", "50")),

            # Selected route / pool contract
            worker_backend=worker_backend,
            worker_profile=worker_profile,
            worker_pool=worker_pool,
            selector_namespace=selector_namespace,
            selector_version=_env_optional_str("REIGH_SELECTOR_VERSION", "ROUTE_SELECTOR_VERSION"),
            worker_contract_version=int(os.getenv("REIGH_WORKER_CONTRACT_VERSION", "1")),
            worker_run_id=_env_optional_str("REIGH_WORKER_RUN_ID", "WORKER_RUN_ID"),

            # Error cleanup
            error_cleanup_grace_period_sec=int(os.getenv("ERROR_CLEANUP_GRACE_PERIOD_SEC", "600")),

            # Polling
            orchestrator_poll_sec=poll_sec,

            # Auto-start
            auto_start_worker_process=os.getenv("AUTO_START_WORKER_PROCESS", "true").lower() == "true",

            # Feature flags
            use_new_scaling_logic=os.getenv("USE_NEW_SCALING_LOGIC", "true").lower() == "true",
            capacity_reconciler_mode=_env_capacity_reconciler_mode(),
        )

    def log_config(self):
        """Log all config values at startup for debugging."""
        logger.info("ORCHESTRATOR CONFIG:")
        logger.info(f"   Scaling: {self.min_active_gpus}-{self.max_active_gpus} GPUs, idle buffer: {self.machines_to_keep_idle}")
        logger.info(f"   Scaling multipliers: up={self.scale_up_multiplier}, down={self.scale_down_multiplier}")
        logger.info(f"   Scaling intervals: min={self.min_scaling_interval_sec}s, spawning_grace={self.spawning_grace_period_sec}s, scale_down_grace={self.scale_down_grace_period_sec}s")
        logger.info(f"   Worker timeouts: spawning={self.spawning_timeout_sec}s, script_running={self.script_running_timeout_sec}s, idle={self.gpu_idle_timeout_sec}s, overcapacity_idle={self.overcapacity_idle_timeout_sec}s, excluded_max_lifetime={self.excluded_worker_max_lifetime_sec}s")
        logger.info(f"   Task timeout: stuck={self.task_stuck_timeout_sec}s")
        logger.info(f"   Health: startup_grace={self.startup_grace_period_sec}s, not_claiming={self.ready_not_claiming_timeout_sec}s, gpu_not_detected={self.gpu_not_detected_timeout_sec}s")
        logger.info(f"   Promotion: heartbeat_threshold={self.heartbeat_promotion_threshold_sec}s")
        logger.info(f"   Failure protection: max_rate={self.max_worker_failure_rate:.0%}, window={self.failure_window_minutes}m, min_workers={self.min_workers_for_rate_check}")
        logger.info(f"   Task failure detection: max_consecutive={self.max_consecutive_task_failures}, window={self.task_failure_window_minutes}m")
        logger.info(f"   Storage: check_interval={self.storage_check_interval_cycles} cycles, min_free={self.storage_min_free_gb}GB, max_used={self.storage_max_percent_used}%")
        logger.info(f"   Route contract: backend={self.worker_backend}, profile={self.worker_profile}, pool={self.worker_pool}, selector={self.selector_namespace}/{self.selector_version or 'default'}, contract={self.worker_contract_version}, run_id={self.worker_run_id or 'unset'}")
        logger.info(f"   Polling: {self.orchestrator_poll_sec}s")
        logger.info(f"   Auto-start worker: {self.auto_start_worker_process}")
        logger.info(
            "   Feature flags: "
            f"use_new_scaling_logic={self.use_new_scaling_logic}, "
            f"capacity_reconciler_mode={self.capacity_reconciler_mode}"
        )

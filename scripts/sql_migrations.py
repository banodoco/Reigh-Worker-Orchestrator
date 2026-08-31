"""Shared SQL migration registry and verification helpers."""

from __future__ import annotations

from pathlib import Path


CAPACITY_POOL_FLOOR_ROUTE_KEY = "__pool_floor__"

MIGRATION_FILES: tuple[str, ...] = (
    "20250115000000_create_system_logs.sql",
    "20250121000001_fix_heartbeat_default.sql",
    "20250202000000_add_missing_columns.sql",
    "20250202000001_create_rpc_functions_existing.sql",
    "20250202000002_create_monitoring_views_existing.sql",
    "20250202000003_add_legacy_functions.sql",
    "20260514000000_create_worker_capacity_intents.sql",
)

CAPACITY_MIGRATION_FILE = "20260514000000_create_worker_capacity_intents.sql"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sql_dir() -> Path:
    return project_root() / "sql"


def migration_paths(base_dir: Path | None = None) -> list[Path]:
    root = base_dir or sql_dir()
    return [root / filename for filename in MIGRATION_FILES]


def read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def capacity_schema_verification_sql() -> str:
    """Return a fail-fast SQL block for the capacity-control schema."""

    return f"""
DO $$
BEGIN
    IF to_regclass('public.worker_capacity_intents') IS NULL THEN
        RAISE EXCEPTION 'missing table public.worker_capacity_intents';
    END IF;

    IF to_regclass('public.worker_capacity_route_backoffs') IS NULL THEN
        RAISE EXCEPTION 'missing table public.worker_capacity_route_backoffs';
    END IF;

    IF to_regclass('public.orchestrator_leases') IS NULL THEN
        RAISE EXCEPTION 'missing table public.orchestrator_leases';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'worker_capacity_route_backoffs'
          AND column_name = 'route_key'
          AND is_nullable = 'NO'
          AND column_default LIKE '%{CAPACITY_POOL_FLOOR_ROUTE_KEY}%'
    ) THEN
        RAISE EXCEPTION 'worker_capacity_route_backoffs.route_key must be non-null and default to {CAPACITY_POOL_FLOOR_ROUTE_KEY}';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public'
          AND t.relname = 'worker_capacity_route_backoffs'
          AND c.contype = 'u'
          AND pg_get_constraintdef(c.oid) = 'UNIQUE (pool, route_key)'
    ) THEN
        RAISE EXCEPTION 'worker_capacity_route_backoffs must enforce UNIQUE (pool, route_key)';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public'
          AND t.relname = 'system_logs'
          AND c.conname = 'valid_source_type'
          AND pg_get_constraintdef(c.oid) LIKE '%orchestrator_gpu%'
          AND pg_get_constraintdef(c.oid) LIKE '%orchestrator_api%'
          AND pg_get_constraintdef(c.oid) LIKE '%worker%'
          AND pg_get_constraintdef(c.oid) NOT LIKE '%capacity_reconciler%'
    ) THEN
        RAISE EXCEPTION 'system_logs.valid_source_type was changed or is missing expected values';
    END IF;
END $$;
""".strip()

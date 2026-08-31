-- =====================================================================
-- Capacity Reconciler Durable Intent Schema
-- Created: 2026-05-14
-- Purpose: Audit and coordinate GPU worker capacity decisions.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS public.worker_capacity_intents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    pool text NOT NULL,
    route_key text,
    desired_capacity int NOT NULL CHECK (desired_capacity >= 0),
    reason text NOT NULL,

    observed_queued int CHECK (observed_queued IS NULL OR observed_queued >= 0),
    observed_active int CHECK (observed_active IS NULL OR observed_active >= 0),
    observed_spawning int CHECK (observed_spawning IS NULL OR observed_spawning >= 0),
    observed_idle int CHECK (observed_idle IS NULL OR observed_idle >= 0),
    observed_in_progress int CHECK (observed_in_progress IS NULL OR observed_in_progress >= 0),
    observed_route_stale int CHECK (observed_route_stale IS NULL OR observed_route_stale >= 0),
    effective_capacity int CHECK (effective_capacity IS NULL OR effective_capacity >= 0),

    cycle_id bigint,
    observer_id text,
    observation_id uuid DEFAULT gen_random_uuid(),
    actions jsonb NOT NULL DEFAULT '[]'::jsonb,
    suppressed_actions jsonb NOT NULL DEFAULT '[]'::jsonb,
    outcome jsonb NOT NULL DEFAULT '{}'::jsonb,

    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    valid_until timestamptz,
    stable_since timestamptz,

    consecutive_spawn_failures int NOT NULL DEFAULT 0 CHECK (consecutive_spawn_failures >= 0),
    next_spawn_allowed_at timestamptz,
    shadow boolean NOT NULL DEFAULT true,

    CONSTRAINT worker_capacity_intents_pool_not_blank CHECK (btrim(pool) <> ''),
    CONSTRAINT worker_capacity_intents_reason_not_blank CHECK (btrim(reason) <> '')
);

CREATE INDEX IF NOT EXISTS worker_capacity_intents_pool_route_created
    ON public.worker_capacity_intents (pool, route_key, created_at DESC);

CREATE INDEX IF NOT EXISTS worker_capacity_intents_pool_cycle_created
    ON public.worker_capacity_intents (pool, cycle_id, created_at DESC)
    WHERE cycle_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS worker_capacity_intents_authoritative_pool_cycle_unique
    ON public.worker_capacity_intents (pool, cycle_id)
    WHERE shadow = false AND cycle_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS worker_capacity_intents_shadow_observer_observation
    ON public.worker_capacity_intents (observer_id, observation_id, created_at DESC)
    WHERE shadow = true;

CREATE TABLE IF NOT EXISTS public.worker_capacity_route_backoffs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    pool text NOT NULL,
    route_key text NOT NULL DEFAULT '__pool_floor__',
    consecutive_spawn_failures int NOT NULL DEFAULT 0 CHECK (consecutive_spawn_failures >= 0),
    next_spawn_allowed_at timestamptz,
    last_spawn_failed_at timestamptz,
    last_spawn_succeeded_at timestamptz,
    last_error text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT worker_capacity_route_backoffs_pool_not_blank CHECK (btrim(pool) <> ''),
    CONSTRAINT worker_capacity_route_backoffs_route_not_blank CHECK (btrim(route_key) <> ''),
    CONSTRAINT worker_capacity_route_backoffs_pool_route_unique UNIQUE (pool, route_key)
);

CREATE INDEX IF NOT EXISTS worker_capacity_route_backoffs_next_spawn_allowed
    ON public.worker_capacity_route_backoffs (pool, route_key, next_spawn_allowed_at);

CREATE TABLE IF NOT EXISTS public.orchestrator_leases (
    lease_key text PRIMARY KEY,
    pool text,
    holder_id text NOT NULL,
    acquired_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT orchestrator_leases_key_not_blank CHECK (btrim(lease_key) <> ''),
    CONSTRAINT orchestrator_leases_holder_not_blank CHECK (btrim(holder_id) <> '')
);

CREATE INDEX IF NOT EXISTS orchestrator_leases_pool_expires
    ON public.orchestrator_leases (pool, expires_at);

CREATE INDEX IF NOT EXISTS orchestrator_leases_expired
    ON public.orchestrator_leases (expires_at);

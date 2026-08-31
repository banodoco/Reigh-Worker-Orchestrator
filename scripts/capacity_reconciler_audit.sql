-- Capacity reconciler audit queries.
-- Intended for operator review during shadow and authoritative rollout.

-- 1. Shadow intended actions from system_logs.
SELECT
    sl.timestamp,
    sl.metadata->>'pool' AS pool,
    sl.metadata->>'cycle_id' AS cycle_id,
    sl.metadata->>'intent_id' AS intent_id,
    sl.metadata->'action'->>'type' AS shadow_action_type,
    sl.metadata->'action'->>'route_key' AS route_key,
    sl.metadata->'action'->>'worker_id' AS worker_id
FROM public.system_logs sl
WHERE sl.source_type = 'orchestrator_gpu'
  AND sl.metadata ? 'shadow_intended_action'
  AND sl.timestamp > now() - interval '24 hours'
ORDER BY sl.timestamp DESC;

-- 2. Pool summary intents with route-demand fallback, lease suppression,
-- adoption, and action counts.
SELECT
    wci.created_at,
    wci.pool,
    wci.cycle_id,
    wci.shadow,
    wci.observer_id,
    wci.observation_id,
    wci.desired_capacity,
    wci.effective_capacity,
    wci.reason,
    wci.observed_queued,
    wci.observed_active,
    wci.observed_spawning,
    wci.observed_idle,
    wci.observed_in_progress,
    wci.outcome->>'route_demand_source' AS route_demand_source,
    COALESCE((wci.outcome->>'route_demand_fallback')::boolean, false) AS route_demand_fallback,
    COALESCE((wci.outcome->>'lease_suppressed')::boolean, false) AS lease_suppressed,
    COALESCE(jsonb_array_length(wci.outcome->'adopted_legacy_spawning_worker_ids'), 0) AS adopted_workers,
    jsonb_array_length(wci.actions) AS action_count,
    jsonb_array_length(wci.suppressed_actions) AS suppressed_action_count
FROM public.worker_capacity_intents wci
WHERE wci.created_at > now() - interval '24 hours'
ORDER BY wci.created_at DESC;

-- 3. Shadow observer dedupe: duplicate observation ids should be absent.
SELECT
    observer_id,
    observation_id,
    count(*) AS rows_for_observation,
    min(created_at) AS first_seen,
    max(created_at) AS last_seen
FROM public.worker_capacity_intents
WHERE shadow = true
  AND created_at > now() - interval '24 hours'
GROUP BY observer_id, observation_id
HAVING count(*) > 1
ORDER BY rows_for_observation DESC, last_seen DESC;

-- 4. Authoritative exact-one-row invariant by pool/cycle.
SELECT
    pool,
    cycle_id,
    count(*) AS authoritative_rows,
    array_agg(id ORDER BY created_at) AS intent_ids
FROM public.worker_capacity_intents
WHERE shadow = false
  AND cycle_id IS NOT NULL
  AND created_at > now() - interval '24 hours'
GROUP BY pool, cycle_id
HAVING count(*) <> 1
ORDER BY max(cycle_id) DESC;

-- 5. Route-scoped backoff windows.
SELECT
    pool,
    route_key,
    consecutive_spawn_failures,
    next_spawn_allowed_at,
    last_spawn_failed_at,
    last_spawn_succeeded_at,
    last_error,
    now() < next_spawn_allowed_at AS currently_suppressed
FROM public.worker_capacity_route_backoffs
ORDER BY pool, route_key;

-- 6. Shadow-vs-legacy comparison by cycle. Legacy mutations are inferred from
-- worker rows updated by the legacy controller; capacity actions are read from
-- shadow intent rows.
WITH shadow_actions AS (
    SELECT
        wci.pool,
        wci.cycle_id,
        action->>'type' AS action_type,
        count(*) AS shadow_count
    FROM public.worker_capacity_intents wci
    CROSS JOIN LATERAL jsonb_array_elements(wci.actions) AS action
    WHERE wci.shadow = true
      AND wci.created_at > now() - interval '24 hours'
    GROUP BY wci.pool, wci.cycle_id, action->>'type'
),
legacy_worker_mutations AS (
    SELECT
        COALESCE(w.metadata->>'worker_pool', w.metadata->'route_contract'->>'worker_pool', 'unknown') AS pool,
        w.metadata->>'capacity_intent_id' AS capacity_intent_id,
        CASE
            WHEN w.status IN ('terminated', 'error') THEN 'legacy_terminal_worker'
            WHEN w.status IN ('active', 'spawning') THEN 'legacy_live_worker'
            ELSE 'legacy_other_worker'
        END AS action_type,
        count(*) AS legacy_count
    FROM public.workers w
    WHERE w.created_at > now() - interval '24 hours'
       OR COALESCE(w.last_heartbeat, w.created_at) > now() - interval '24 hours'
    GROUP BY 1, 2, 3
)
SELECT
    sa.pool,
    sa.cycle_id,
    sa.action_type AS shadow_action_type,
    sa.shadow_count,
    COALESCE(sum(lwm.legacy_count), 0) AS nearby_legacy_worker_mutations
FROM shadow_actions sa
LEFT JOIN legacy_worker_mutations lwm
  ON lwm.pool = sa.pool
GROUP BY sa.pool, sa.cycle_id, sa.action_type, sa.shadow_count
ORDER BY sa.cycle_id DESC NULLS LAST, sa.pool, sa.action_type;

-- 7. Current lease holders and expired lease debris.
SELECT
    lease_key,
    pool,
    holder_id,
    acquired_at,
    expires_at,
    now() > expires_at AS expired,
    metadata
FROM public.orchestrator_leases
ORDER BY expires_at DESC;

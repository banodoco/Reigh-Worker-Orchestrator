---
id: 01KRHJWXBEY4WR107VGZBYJ1G3
title: Delete legacy capacity controller (T9/T11/T12 follow-up to capacity-reconciler-20260513)
status: open
source: human
tags:
- reigh-worker-orchestrator
- capacity-reconciler
- follow-up
codebase_id: null
created_at: '2026-05-13T21:11:47.822208+00:00'
last_edited_at: '2026-05-13T21:11:47.822208+00:00'
epics: []
---

Follow-up to PR #6 (capacity-reconciler-20260513 sprint). Three megaplan tasks were deferred because they delete the legacy capacity controller, which must happen only after the new reconciler proves out in production.

## Preconditions (must be met before resuming)

1. PR #6 merged to main
2. Migration applied: `sql/20260514000000_create_worker_capacity_intents.sql`
3. Shadow deploy: `ORCHESTRATOR_CAPACITY_RECONCILER_MODE=shadow` running ≥24h with ≥50 meaningful capacity events; intent rows compared against legacy controller decisions show no surprising divergence
4. Cutover deploy: `ORCHESTRATOR_CAPACITY_RECONCILER_MODE=authoritative`
5. Post-cutover monitoring: `scaling-audit --hours 6` watched for 4h, 24h rollback window elapsed, no rollback triggered

## Scope (resumes the megaplan)

Resume the existing plan to land T9, T11, T12:

```
cd /Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator-capacity-reconciler/
# Mark U5 as completed in user_actions.md, then:
PYENV_VERSION=3.11.11 megaplan auto --plan capacity-reconciler-20260513-v2 --on-escalate fail --confirm-destructive
```

What the executor will do:
- **T9**: delete `gpu_orchestrator/_handle_early_termination` and all call sites in `control_loop.py`; delete `gpu_orchestrator/control/phases/`; remove imports of `gpu_orchestrator.control.phases`; remove legacy capacity stubs and shadow-comparison code
- **T11**: update existing tests after the phase deletion (remove phase-mixin smoke tests that no longer apply)
- **T12**: write throwaway script reproducing the 2026-05-13 queued-zero spawning-worker churn against the reconciler, prove no premature cancel or oscillation; then run the full validation suite (`pytest tests/control/test_capacity_reconciler.py`, scaling-decision tests, full pytest); final cleanup checks (`rg "_handle_early_termination"`, `test ! -d gpu_orchestrator/control/phases`, `rg "gpu_orchestrator\.control\.phases"`)

## Rollback path

If anything regresses after cutover but before this PR lands, `git revert` the capacity-reconciler PR(s) and redeploy. The migration table can stay (it's additive).

## References

- PR #6: https://github.com/banodoco/reigh-worker-orchestrator/pull/6
- Plan dir: `/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator-capacity-reconciler/.megaplan/plans/capacity-reconciler-20260513-v2/`
- Brief: `/Users/peteromalley/Documents/reigh-workspace/briefs/brief-unify-capacity-control-20260513.md`
- Approach memo: `/Users/peteromalley/Documents/reigh-workspace/docs/scaling-churn-approach-20260513.md`
- User action gating in plan: `.megaplan/plans/capacity-reconciler-20260513-v2/user_actions.md` (U5)

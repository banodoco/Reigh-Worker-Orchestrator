# User Actions

## Before Execute

- **U5**: Monitor `scaling-audit --hours 6` for 4 hours post-cutover and wait through the agreed 24-hour rollback window before approving the final cleanup slice.
  Rationale: The final deletion of legacy code should happen only after authoritative mode has been stable enough for rollback confidence.

## After Execute

- **U1**: Apply the new SQL migration through the existing approved database migration process and confirm schema verification passes before enabling `ORCHESTRATOR_CAPACITY_RECONCILER_MODE=shadow`.
  Rationale: The plan deliberately avoids Railway boot-time DDL; production or staging DB mutation is an operator action after code is ready.
- **U2**: Deploy the shadow slice with `ORCHESTRATOR_CAPACITY_RECONCILER_MODE=shadow`.
  Rationale: Environment changes and production deploys are outside repo editing.
- **U3**: Observe shadow mode for at least 24 hours and at least 50 meaningful capacity events, extending the window if needed; review `scaling-audit` and divergence SQL for unresolved correct-legacy divergence.
  Rationale: Cutover requires production traffic evidence and operator approval.
- **U4**: After shadow gates pass, deploy the authoritative cutover with `ORCHESTRATOR_CAPACITY_RECONCILER_MODE=authoritative`.
  Rationale: The feature-flag flip is an operational production action.

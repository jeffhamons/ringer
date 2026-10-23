# Dashboard data identity and recovery

The October 23, 2026 investigation found a persistent dashboard using a deleted
October 19 temporary HOME. Its HTTP endpoint was healthy but returned no runs;
the normal state directory still held the run records. Reusing any HTTP 200 as a
healthy dashboard, and omitting the selected configuration from detached startup,
allowed that process to be reused. The exact originating test/session remains
unproven. The ask-command unit fixture could launch such a persistent process and
now mocks that boundary.

The dashboard exposes `/api/identity` with its protocol and canonical data paths.
Both launch entry points require an exact compatible identity before reuse.
Legacy, mismatched, malformed, or unresponsive listeners fail clearly; they are
never killed or automatically replaced. Detached startup forwards the selected
configuration and explicit effective state/model-log paths, waits for compatible
readiness, and reports failure instead of printing a successful dashboard URL.
Only a newly spawned child is terminated if its readiness fails.

Recovery requires identifying the existing dashboard process, stopping only that
process, and launching `ringer --config /absolute/path/config.toml hud --no-open`.
Verify `/api/identity`, `/api/runs`, and the visible Live view against the intended
state directory. Restarting the dashboard does not require restarting workers.
An older running dashboard must be restarted manually once after this upgrade.

Focused regressions use ephemeral localhost ports and temporary state; ask unit
tests mock persistent startup. No production listeners are started by those tests.
Implementation and focused test results belong to the current change; deployment
and live verification must be recorded separately by the operator.

# Singleton readiness/probe reference

One fixed six-slot sibling recipe has two positive traces: the local probe
reports `available` or `async_required`. Both require the semantic readiness
attempt H, one physical readiness leaf R, actual local probe P, generation G,
submission D, and independent assessor A. Six Starts and six canonical attempts
lead to two physical dispatches; neither submission, H nor P adds a charge.
The table fixes request roles/outcomes and candidate text separately from the
production request and activation reducers.

Four event-driven cancellation traces cross the readiness/probe stage with
caught/propagated submitting-task cancellation. Repeated owner cancellation
cannot detach the actual work before release. Self-cancelling P is a separate
regression: its canonical attempt must close and generation must not run.

Negative controls cover failed readiness/probe, undeclared continuation,
service rejection before invocation identity, foreign or overlapping slots,
readiness role/owner/leaf/terminal mismatches, and extra ordinary charges on H,
R or P. Unknown usage is permitted only on the independently admitted generation
role; readiness remains exact. Positive controls execute through the runtime,
normal candidate publication, independent assessment and full qualification.

Manifest SHA-256: `7a7fc89bf56a3cd67f92ff5267a80ecbe97851992d76fe6ed4d16e1d5d400042`.
The first draft used `candidate:source`; the reused input fixture explicitly
supplies `target-0`, so that literal was corrected to `candidate:target-0`.
Count, role and cancellation expectations were unchanged. Older reference tables
remain byte-unchanged. Optional unused retry tails are still rejected.

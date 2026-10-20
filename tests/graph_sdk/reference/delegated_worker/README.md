# X1 worker reference

This literal manifest was written before the X1 implementation. It is an
independent bounded witness, not a reducer-derived oracle. Its positive row adds
a worker/borrowed-loop/to_thread/run_coroutine_threadsafe execution of the
existing readiness recipe; all earlier generation, readiness and 20 proposed
cross-loop expectations remain unchanged.

The cancellation test matrix crosses caught/propagated foreign observation,
confirmed/unknown stop, and owner/foreign-observer cancellation. Confirmed stop
expects `cancelled`; unknown stop expects `lost`. Both retain one physical charge,
one late completed settlement, complete canonical closure and zero qualification.
The first terminal is captured before releasing the event-driven transport and
compared unchanged after settlement. Two subsequent observer cancellations cannot
finalize while that transport remains live.

Additional literal controls reject a forged issued object even under its own
context, missing context, an undeclared loop, expired context, simultaneous
callbacks, an unregistered worker profile, host/stopped loop descriptors and a
blocking physical transport. A full bounded-pool occupancy case cancels the
queued submission and expects zero callbacks/dispatches and only the parent Start.
These are local deterministic transports; no foreign socket, DD provider,
owned-client shutdown, performance or live-quality claim follows.

The final caller-pool controls reject a closed pool, keep unrelated pool work
alive after a cancelled invocation, and reuse the caller pool and borrowed loop
after a completed invocation. The worker pool is a live scheduling resource,
not a stateless provider handle. Neither request counters nor a global pool
occupancy count establish completion of this invocation's retained worker.

The worker parent is separately prepared with `resource_lifetime="caller_owned"`
and an exact caller-owned submitter lease. Fixed additional controls reject a
missing/mismatched/SDK-owned/closed lease and a stateless parent declaration.
The positive expects one caller `left_open` accounting cleanup fact. These
resource-profile expectations were fixed before that comparison; they do not
change the existing physical request counts or reference tables.

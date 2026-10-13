<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Foreign resource integration reference

This additional literal manifest defines the fixed public readiness/probe/
generation fixture, its finite request and activation counts, exact candidate,
cleanup outcomes and withholding expectations. The public fixture is prepared
and admitted normally. It uses the existing worker borrowing route through
`to_thread` and `run_coroutine_threadsafe`; the original client body records each
actual send on its creation loop. Assertions compare the canonical record,
publication and qualified result with independent expected values, without
constructing positive canonical facts.

The initial draft copied the candidate text as `candidate` rather than the
existing readiness and worker reference value `candidate:target-0`. The draft is
preserved in campaign evidence; only that literal was corrected from those
pre-existing independent tables. All old references and review repros remain
unchanged. Confirmed-stop expectations reuse the established request contract;
unknown stop, self-cancelled stop and live stop-hook cases keep separate assertions.

Local resource use is witnessed by actual retained call/hook Futures and worker
completion, separately from request ledger counts. The confirmed-stop case clears
request counts while the original call still runs and proves that resource close
is withheld. Close failure and self-cancellation withhold qualification. Public
loop stop/resume, repeated observers, atomic claims and unrelated caller-pool
work have direct event-driven witnesses. Permanent loop loss has no finite
successful completion witness.

The registered profile requires a fresh loop-created client, exclusive adapter
use for one invocation, and a caller-maintained live loop. Claiming an adapter
cannot establish safety for undisclosed raw-client aliases. This fixture neither
uses providers nor enables DD routes, opaque execution or optional request slots.

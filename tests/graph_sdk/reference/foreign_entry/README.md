<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Foreign physical-entry reference

The literal manifest was frozen before the X2 implementation. Its alphabet,
finite bounds and count columns describe one request crossing one host loop and
one borrowed loop. Expectations come from the physical-entry contract: a rejected
entry has no send, Dispatch or Start; an accepted entry has exactly one of each.
They do not derive from the production reducer. The historical twenty-case
cross-loop proposal and existing worker/readiness tables are unchanged.

The harness uses one registered activation and the existing request authority.
It measures transport entry inside the original transport coroutine, separately
from committed Starts and authoritative Dispatches. It closes the activation
through the existing terminal reducer. Additional unavailable-loop, copied
adapter and cancelled-observer controls reuse these literal zero/one outcomes.
The delayed gate is invoked after timeout or cancellation to prove invalidation.
A deterministic barrier after acceptance exposes the retained forwarding
obligation before actual inner entry; the completed test must drain that work.

Known stop and unknown stop have separate first-terminal expectations. Late
completion may settle usage without replacing that terminal. A stopped loop
cannot complete drain until the same loop resumes. Permanent borrowed-loop loss
has no successful finite witness. The profile stays closed at public service
admission until X3 resource and owner integration; these tests claim no full
Protected graph execution or DD route.

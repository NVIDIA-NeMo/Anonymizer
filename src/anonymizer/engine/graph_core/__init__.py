# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private graph kernel and protection semantics (private); not part of the public ``anonymizer.graph`` surface.

Declare datums and relations, compose operations into a workflow, :func:`prepare` (admission before effects), then
:func:`protect` (synchronous, on the calling thread). Routes register the artifact kinds they need with
:func:`register_kind`; the kernel itself knows none.
"""

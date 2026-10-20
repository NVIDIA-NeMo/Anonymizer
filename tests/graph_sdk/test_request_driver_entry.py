# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation at the actual task-entry dispatch boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from anonymizer.engine.graph_sdk._request_driver import _RequestAuthority, drive_request
from anonymizer.engine.graph_sdk.requests import Dispatch, StopUnknown
from tests.graph_sdk.test_blocking import Control, reserved

TABLE = json.loads((Path(__file__).parent / "reference/delegated_execution/driver.json").read_text())


@pytest.mark.parametrize("prepared", [False, True], ids=TABLE["variants"])
@pytest.mark.parametrize("boundary, expected", TABLE["cases"].items())
def test_dispatch_cannot_precede_callable_entry(monkeypatch, prepared, boundary, expected):
    async def run():
        authority, request = reserved()
        control = Control(cancelled=boundary == "before_driver")
        counts = {"invoke": 0, "cancel": 0, "commit": 0}
        apply = _RequestAuthority.apply
        create_task = asyncio.create_task

        def apply_boundary(owner, event):
            state = apply(owner, event)
            if owner is authority and isinstance(event, Dispatch) and boundary == "after_dispatch":
                control.cancelled = True
            return state

        def schedule(coro, *args, **kwargs):
            task = create_task(coro, *args, **kwargs)
            if boundary == "before_task_entry":
                control.cancelled = True
            return task

        async def invoke():
            counts["invoke"] += 1
            if boundary == "after_dispatch":
                await asyncio.Event().wait()
            return None

        async def cancel(identity):
            assert identity == request
            counts["cancel"] += 1
            return StopUnknown()

        def prepare():
            assert request not in authority.state.dispatched

            def commit():
                assert request in authority.state.dispatched
                counts["commit"] += 1

            return commit

        monkeypatch.setattr(_RequestAuthority, "apply", apply_boundary)
        monkeypatch.setattr(asyncio, "create_task", schedule)
        completion = await drive_request(
            authority, request, control, invoke, cancel, prepare_dispatch=prepare if prepared else None
        )
        assert [len(authority.state.dispatches), counts["invoke"], counts["cancel"]] == expected
        assert counts["commit"] == int(prepared and expected[0] == 1)
        assert not authority.state.reserved
        if not expected[0]:
            assert completion.cancelled and completion.stop_confirmed
            assert not authority.state.remote_outstanding
            assert [item.category for item in authority.state.terminals] == ["cancelled"]
        elif boundary == "after_dispatch":
            assert [item.category for item in authority.state.terminals] == ["lost"]
            assert authority.state.remote_outstanding == frozenset({request})

    asyncio.run(run())

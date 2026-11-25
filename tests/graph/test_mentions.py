# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mentions carry exact offsets on their target; mismatched, overlapping or off-target spans are rejected before callbacks."""

from __future__ import annotations

import pytest

from anonymizer.engine.graph_core.authoring import (
    ContextScope,
    ContextView,
    GraphScope,
    Mention,
    Operation,
    ProtectionGraph,
)
from anonymizer.engine.graph_core.kernel import GraphRejection
from tests.graph.helpers import Recorder, run

TEXT = "Alice met Bob."


def test_mention_offsets_reach_the_operation_exactly() -> None:
    scope = GraphScope()
    t = scope.datum(TEXT)
    seen: list[tuple] = []
    op = Operation("op", lambda text, mentions: seen.append(mentions) or text, {"text": "text", "mentions": "mentions"})
    run(ProtectionGraph(scope, (t,), mentions=(Mention(t, 0, 5, "first_name", "Alice"),)), op)
    m = seen[0][0]
    assert (m.start, m.end) == (0, 5)
    assert TEXT[m.start : m.end] == "Alice"


def test_mention_text_not_matching_slice_is_rejected() -> None:
    scope = GraphScope()
    t = scope.datum(TEXT)
    rec = Recorder()
    with pytest.raises(GraphRejection) as info:
        run(ProtectionGraph(scope, (t,), mentions=(Mention(t, 1, 6, "first_name", "Alice"),)), rec.transform())
    assert info.value.code == "mention_slice_mismatch"
    assert rec.count() == 0


def test_overlapping_mentions_are_rejected() -> None:
    scope = GraphScope()
    t = scope.datum(TEXT)
    rec = Recorder()
    overlap = (Mention(t, 0, 5, "first_name"), Mention(t, 3, 9, "other"))
    with pytest.raises(GraphRejection) as info:
        run(ProtectionGraph(scope, (t,), mentions=overlap), rec.transform())
    assert info.value.code == "overlapping_output_ownership"
    assert rec.count() == 0


def test_overlapping_context_views_are_accepted() -> None:
    scope = GraphScope()
    t = scope.datum(TEXT)
    views = ContextScope(t, (ContextView(t, t, 0, 9), ContextView(t, t, 6, 14)))
    g = ProtectionGraph(scope, (t,), mentions=(Mention(t, 0, 5, "first_name"),), context=(views,))
    assert run(g, Recorder().transform()).outcome(t).terminal == "success"


def test_mention_on_context_artifact_is_rejected() -> None:
    scope = GraphScope()
    t = scope.datum(TEXT)
    ctx = scope.datum("Bob met Alice.")
    rec = Recorder()
    with pytest.raises(GraphRejection) as info:
        g = ProtectionGraph(
            scope, (t,), mentions=(Mention(ctx, 8, 13, "first_name"),), context=(ContextScope(t, (ctx,)),)
        )
        run(g, rec.transform())
    assert info.value.code == "mention_not_on_target"
    assert rec.count() == 0

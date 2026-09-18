# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MkDocs hooks for the NeMo Anonymizer docs build.

Workaround for pymdownx bug: Highlight.highlight() passes filename=title to
pygments BlockHtmlFormatter, but title defaults to None. Pygments 2.20 added
explicit filename processing and crashes on None instead of ignoring it.

Remove this patch once pymdownx fixes the None guard on their end.
"""

import pymdownx.highlight as _hl

_orig_highlight = _hl.Highlight.highlight


def _patched_highlight(self, src, language, *args, title=None, **kwargs):
    return _orig_highlight(self, src, language, *args, title=title or "", **kwargs)


_hl.Highlight.highlight = _patched_highlight

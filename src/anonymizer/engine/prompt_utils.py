# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-pass placeholder substitution for LLM prompt templates."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("anonymizer.prompt_utils")

# \w+ matches [a-zA-Z0-9_] only; hyphens in placeholder names are not supported.
_PLACEHOLDER_RE = re.compile(r"<<\w+>>")


def substitute_placeholders(
    template: str,
    replacements: dict[str, str],
    *,
    strict: bool = True,
) -> str:
    """Single-pass placeholder substitution.

    All ``<<PLACEHOLDER>>`` markers are replaced simultaneously so that
    user-controlled values inserted for one placeholder cannot collide
    with markers intended for a later placeholder.

    When *strict* is True (default):

    - Raises ``ValueError`` if any key does not match the ``<<...>>`` format.
    - Raises ``ValueError`` if any ``<<...>>`` placeholders remain after substitution.
    - Logs a warning if the dict contains keys not present in the template.
    """
    if not replacements:
        if strict:
            remaining = _PLACEHOLDER_RE.findall(template)
            if remaining:
                raise ValueError(f"Unresolved placeholders in prompt: {remaining}")
        return template

    if strict:
        bad_keys = [k for k in replacements if not _PLACEHOLDER_RE.fullmatch(k)]
        if bad_keys:
            raise ValueError(f"Replacement keys must use <<...>> format, got: {bad_keys}")

        unused = [k for k in replacements if k not in template]
        if unused:
            logger.warning("Replacement keys not found in template: %s", unused)

        # Check for unresolved placeholders by comparing original template
        # placeholders against provided keys.  Scanning the *result* would
        # false-positive when a replacement value itself contains <<...>>.
        original_placeholders = set(_PLACEHOLDER_RE.findall(template))
        unresolved = sorted(original_placeholders - set(replacements))
        if unresolved:
            raise ValueError(f"Unresolved placeholders in prompt: {unresolved}")

    pattern = re.compile("|".join(re.escape(k) for k in replacements))
    return pattern.sub(lambda m: replacements[m.group(0)], template)

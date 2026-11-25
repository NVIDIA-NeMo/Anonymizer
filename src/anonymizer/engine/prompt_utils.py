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
    """Single-pass placeholder substitution for dynamic prompt values.

    Used for dynamic prompt values that aren't DataFrame column
    references. All `<<PLACEHOLDER>>` markers are replaced simultaneously
    so that user-controlled values inserted for one placeholder cannot
    collide with markers intended for a later placeholder.

    When `strict` is True (default):

    - Raises `ValueError` if any key does not match the `<<...>>` format.
    - Raises `ValueError` if any `<<...>>` placeholders remain after substitution.
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


def jinja_literal(text: str) -> str:
    """``text`` wrapped so a Jinja render emits it verbatim.

    Prompts built with :func:`substitute_placeholders` are rendered as Jinja afterwards (by DataDesigner, or by the
    validation and augmentation renderers), so user text such as ``instructions`` or ``data_summary`` placed in one
    would otherwise be read as template code. The text goes in a ``{% raw %}`` block; each ``{%`` inside it closes the
    block, emits ``{%`` as a string expression and reopens it, so the text cannot end the block itself.

    Text with no ``{`` cannot open a Jinja tag, expression or comment, so it is returned unchanged; prompts built from
    plain text (the default privacy goal and labels, most summaries) stay byte-identical.
    """
    if "{" not in text:
        return text
    return "{% raw %}" + text.replace("{%", "{% endraw %}{{ '{%' }}{% raw %}") + "{% endraw %}"


def user_instructions_section(instructions: str | None, *, jinja: bool) -> str:
    """The ``<user_instructions>`` block of the rewrite and repair prompts, or ``""`` when none are given.

    ``jinja`` is True for a prompt that is rendered as Jinja afterwards, so the text is wrapped by
    :func:`jinja_literal`; the repair prompt is sent as built and takes the text as is.
    """
    if not instructions or not instructions.strip():
        return ""
    text = instructions.strip()
    return (
        "\n<user_instructions>\n"
        "Additional instructions from the user. Follow them when rewriting, unless they conflict with the privacy "
        "goal or the protection decisions; privacy takes precedence.\n"
        f"{jinja_literal(text) if jinja else text}\n"
        "</user_instructions>\n"
    )

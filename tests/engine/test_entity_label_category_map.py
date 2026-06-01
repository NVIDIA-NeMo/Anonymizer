# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI guard: every label in ``DEFAULT_ENTITY_LABELS`` has a category mapping.

The disposition reconstructor (``engine/rewrite/disposition_derivation.py``)
falls back to the entity-label-to-category map when small models stuff an
entity label into the ``category`` slot, and the pessimistic fallback
disposition (``sensitivity_disposition._pessimistic_fallback_disposition``)
uses it to pick ``replace`` vs ``generalize``. A new ``DEFAULT_ENTITY_LABELS``
entry without a corresponding category mapping silently degrades both paths
to the conservative ``quasi_identifier`` -> generalize bucket — fine for
attributes, wrong for any new direct identifier (which should be ``replace``).

This regression makes the gap visible: a contributor adding a new label
must extend the category map at the same time, or this test fails with a
diff.
"""

from __future__ import annotations

from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS
from anonymizer.engine.schemas.rewrite import _ENTITY_LABEL_TO_CATEGORY


def test_entity_label_to_category_covers_default_labels() -> None:
    missing = sorted(set(DEFAULT_ENTITY_LABELS) - set(_ENTITY_LABEL_TO_CATEGORY))
    assert not missing, (
        "Labels in DEFAULT_ENTITY_LABELS without a category mapping in "
        "anonymizer.engine.schemas.rewrite._ENTITY_LABEL_TO_CATEGORY:\n  "
        + "\n  ".join(missing)
        + "\n\nAdd each label to one of _DIRECT_ID_LABELS or _QUASI_ID_LABELS in "
        "src/anonymizer/engine/schemas/rewrite.py. Direct identifiers (names, "
        "ids, contact info) -> _DIRECT_ID_LABELS; everything else -> "
        "_QUASI_ID_LABELS (the conservative protect-cautiously bucket)."
    )


def test_entity_label_to_category_values_are_valid_categories() -> None:
    """Every value in the map must be a member of the current ``EntityCategory``
    enum. Catches the case where ``EntityCategory`` is shrunk (cf. #150
    removing ``sensitive_attribute``) without updating the map."""
    from anonymizer.engine.schemas.rewrite import EntityCategory

    valid = {c.value for c in EntityCategory}
    invalid = {label: cat for label, cat in _ENTITY_LABEL_TO_CATEGORY.items() if cat not in valid}
    assert not invalid, (
        f"_ENTITY_LABEL_TO_CATEGORY has values not in EntityCategory: {invalid}. "
        "Update src/anonymizer/engine/schemas/rewrite.py to drop these or remap them."
    )

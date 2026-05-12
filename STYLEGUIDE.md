<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Style Guide

Code and documentation conventions for NeMo Anonymizer that ruff and ty cannot enforce. Architecture boundaries and agent workflow rules live in [AGENTS.md](AGENTS.md). Read before adding a new module, workflow, or config class.

NeMo Anonymizer wraps [DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) (NDD) for LLM column generation. References to NDD below mean that library.

For architecture and pipeline identity, see [AGENTS.md](AGENTS.md).
For contribution workflow and branch naming, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Pydantic vs Dataclasses

**Pydantic** for config, validation, and serialization. **Dataclasses** for simple typed containers in the engine.

| Need | Use |
|------|-----|
| User-facing config, validation, JSON schema | `BaseModel` |
| Private/internal frozen value object (e.g. `WorkflowRunResult`, `_RiskToleranceBundle`) | `@dataclass(frozen=True)` |

```python
# Config — Pydantic
class Detect(BaseModel):
    gliner_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

# Internal result — dataclass
@dataclass(frozen=True)
class WorkflowRunResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]
```

Use `Field()` only when you need constraints (`ge`, `le`), descriptions, or `default_factory`. Use bare defaults for simple flags and strings.

---

## Error Handling

Wrap exceptions from NDD and other third-party calls at module boundaries into canonical types from `interface/errors.py`. Callers should never see raw NDD exceptions.

Preserve the traceback:

```python
# Good
try:
    run_results = self._data_designer.create(...)
except Exception as exc:
    raise AnonymizerWorkflowError(f"Workflow failed: {exc}") from exc

# Bad — swallows the traceback
except Exception as exc:
    raise AnonymizerWorkflowError("Workflow failed")
```

Don't use defensive `try/except` on trusted internal calls that shouldn't fail — only catch at module boundaries. `RewriteWorkflow._run_final_judge` is the intentional exception: it's explicitly non-critical and catches broadly, logging with `exc_info=True` and substituting safe defaults.

**Error messages** must identify the actual bad value. Use `!r` to make interpolated values unambiguous:

```python
# Good
raise ValueError(f"Unsupported strategy: {strategy!r}")

# Bad
raise ValueError("Invalid strategy")
```

**No `assert` for validation in production/library code** — `assert` statements are stripped when Python runs with `-O`. Use `if/raise` instead. Pytest assertions in tests are fine.

```python
# Good
if not isinstance(config, AnonymizerConfig):
    raise TypeError(f"Expected AnonymizerConfig, got {type(config)!r}")

# Bad
assert isinstance(config, AnonymizerConfig)
```

---

## Column Names

All column names are constants in `engine/constants.py`. Never use string literals for column names.

```python
# Good
df[COL_DETECTED_ENTITIES]

# Bad
df["_detected_entities"]
```

Internal (intermediate) columns are prefixed with `_`. User-facing output columns use clean names (`final_entities`, `utility_score`). The input text column is always `COL_TEXT` internally and renamed to the user's original column name in `Anonymizer._rename_output_columns()`.

---

## Prompt Construction

**`_jinja(col, key=None)`** from `engine/constants.py` — use for **shared DataFrame column references** in NDD prompt templates. Never format shared column names directly into prompt strings; `_jinja` keeps them grep-able. Local Jinja loop variables (e.g. `entity.value` inside `{% for entity in ... %}`) are scoped to the prompt and don't need `_jinja`.

```python
# Good
f"The text is: {_jinja(COL_TEXT)}"

# Bad
f"The text is: {{{{ {COL_TEXT} }}}}"
```

**`substitute_placeholders(template, replacements)`** from `engine/prompt_utils.py` — use for dynamic prompt values. The `<<PLACEHOLDER>>` format avoids collisions with Jinja2 syntax. Never use f-strings or `.format()` for prompt templates with dynamic values; single-pass substitution prevents a replacement value from being interpreted as a placeholder.

Prompts live as inline triple-quoted strings in the workflow file that uses them. There is no separate prompt registry.

---

## Type Annotations

Type annotations are required on all functions, methods, and class attributes including tests.

Use `TYPE_CHECKING` blocks for imports needed *only* in type annotations. This prevents circular imports and avoids loading heavy libraries at import time:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
```

If a module uses `pandas` at runtime — calls `pd.DataFrame`, indexes a DataFrame in a function body, etc. — import it at the top level. A `TYPE_CHECKING` import raises `NameError` if you reference it at runtime. `pandas` is import-time expensive, so keep top-level imports of it limited to modules that genuinely need it.

---

## Import Style

- **ALWAYS** use absolute imports, never relative imports (enforced by `TID`)
- Place imports at module level, not inside functions (exception: unavoidable for performance reasons)
- Import sorting is handled by `ruff`'s `isort` — imports should be grouped and sorted:
  1. Standard library imports
  2. Third-party imports
  3. First-party imports (`anonymizer`)
- Use standard import conventions (enforced by `ICN`)

```python
# Good
from anonymizer.config.anonymizer_config import AnonymizerConfig

# Bad - relative import (will cause linter errors)
from .anonymizer_config import AnonymizerConfig

# Good - imports at module level
from pathlib import Path

def process_file(filename: str) -> None:
    path = Path(filename)

# Bad - import inside function
def process_file(filename: str) -> None:
    from pathlib import Path
    path = Path(filename)
```

---

## Code Organization

- When adding new symbols, prefer public functions and methods before private (`_`-prefixed) ones within a module or class
- Define helpers at module or class level — avoid nested functions. Nested functions hide logic, make testing harder, and complicate stack traces. The only acceptable use is a closure that genuinely needs to capture local state.

---

## Naming

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Function names start with a verb: `run_workflow`, `build_entity_id`, not `entity_id` or `workflow`

---

## Comments

Only add a comment when the WHY is non-obvious — a hidden constraint, a subtle invariant, a workaround for a specific bug. Don't narrate what the code already says:

```python
# Good — explains a non-obvious invariant
# uuid5 is deterministic so input/output IDs match for missing-record tracking.

# Bad — narrates what the code does
# Loop through the records and append to list
for record in records:
    results.append(record)
```

---

## Future Annotations

Every Python file must include `from __future__ import annotations` after the license header. This defers annotation evaluation, enables forward references, and keeps behavior consistent across the codebase.

---

## License Headers

Every Python and Markdown file requires an SPDX header at the top (enforced by `tools/codestyle/copyright_fixer.py --check`, run via `make copyright-check`). Files listed in `.copyrightignore` are exempt.

---

## Docstrings

Google style (`Args:`, `Returns:`, `Raises:`). Public API classes and methods get docstrings; private helpers (`_`-prefixed) only when the logic is non-obvious. Don't restate the signature — explain why or what, not what the type annotation already says.

---

## Design Principles

**DRY**

- Extract shared logic into pure helper functions rather than duplicating across similar call sites
- Rule of thumb: tolerate duplication until the third occurrence, then extract

**KISS**

- Prefer flat, obvious code over clever abstractions — two similar lines is better than a premature helper
- When in doubt between DRY and KISS, favor readability over deduplication

**YAGNI**

- Don't add parameters, config, or abstraction layers for hypothetical future use cases
- Don't generalize until the third caller appears

**SOLID**

- Wrap third-party exceptions at module boundaries — callers depend on canonical error types, not leaked internals
- Use `Protocol` for contracts between layers
- One function, one job — separate logic from I/O

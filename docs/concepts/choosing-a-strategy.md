<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Choosing a strategy

This guide walks through the decisions you make when configuring an `AnonymizerConfig` for a real dataset. Use it to go from "I have data + a goal" to a concrete starting config in a few minutes.

It is also the primary reference for AI agents that drive the Anonymizer skill: every decision below is something the agent has to make on the user's behalf.

---

## Decision flow

```
1. (Detection) Describe the data?    → AnonymizerInput(data_summary=...)
2. (Detection) Detection knobs?      → Detect(entity_labels=..., gliner_threshold=...)
3. Replace or Rewrite?               → AnonymizerConfig(replace=...) vs AnonymizerConfig(rewrite=...)
4. (Replace) Which strategy?         → Substitute(...) | Redact(...) | Annotate(...) | Hash(...)
5. (Rewrite) Privacy goal?           → Rewrite(privacy_goal=PrivacyGoal(protect=..., preserve=...))
6. (Rewrite) Risk tolerance?         → Rewrite(risk_tolerance="minimal" | "low" | "moderate" | "high")
```

Steps 1–2 govern the [detection](detection.md) stage that runs first in **both** modes — usually the highest-leverage way to improve overall quality on a new dataset. Steps 3–6 shape the mode-specific transformation that follows detection.

---

## 1. (Detection) `data_summary`

`AnonymizerInput.data_summary` is an optional one-line description that flows into LLM prompts. It is the single cheapest quality lever you have. It improves **detection** — which runs first in both Replace and Rewrite modes, so it's a precursor to any transformation. In Rewrite mode it additionally provides context on what the data contains, helping the rewriter preserve meaning.

```python
from anonymizer import AnonymizerInput

data = AnonymizerInput(
    source="patient_notes.csv",
    text_column="note",
    data_summary="De-identified inpatient progress notes from a US oncology service",
)
```

What to include:

- The domain (clinical, legal, financial, customer support, etc.)
- The genre (notes, transcripts, opinions, biographies)
- Anything about the source the engine couldn't infer from a single record (e.g. "transcribed phone calls — expect disfluencies")
- `data_summary` is the only way to provide a soft do-not-tag list for the augmenter when `entity_labels=None` — the augmenter is free to invent labels beyond `DEFAULT_ENTITY_LABELS`, so use it to tell the LLM what *not* to tag (e.g. "do not tag generic anatomical terms, medication class names, or job titles as PII").

What to leave out:

- Lists of entity types **you want detected** (those go in `Detect.entity_labels`)
- Privacy/utility goals (those go in `Rewrite.privacy_goal`)
- Substitute behavior instructions (e.g. "names should remain Portuguese", "preserve numeric magnitude") — those go in `Substitute(instructions)`
- Generic phrasing ("text data" adds no signal)

---

## 2. (Detection) Detection knobs

For most datasets the [detection](detection.md) defaults work. The main reason to adjust `entity_labels` is when your data has **domain-specific entities that can be described in plain English** — GLiNER is zero-shot, so any concept you can name (e.g. `"clinical_facility"`, `"internal_project_codename"`) becomes an entity it can find. Match the snake_case convention of `DEFAULT_ENTITY_LABELS`. If the entities you care about aren't in the default list, write them down and add them. Adjust `gliner_threshold` only when you see a specific recall or precision problem in preview.

### `entity_labels`

| Setting | Behavior | Use when |
|---|---|---|
| `None` (default) | Detect all `DEFAULT_ENTITY_LABELS`; the augmenter LLM can also infer new labels not in the default set | General-purpose — almost always the right starting point |
| Explicit list | **Strict mode** — only the labels you list are detected, augmenter cannot invent new ones | You have a domain-specific entity that the defaults don't cover, or you want to *narrow* detection to a known short list |

Common ways to extend the default list:

- Healthcare: `clinical_facility`, `diagnosis_code`, `medication_name`, `lab_test_code`
- Legal: `case_number`, `docket_number`, `statute_citation`, `judge_name`
- Customer support: `ticket_id`, `internal_user_id`, `transaction_id`
- Internal: `cost_center`, `internal_project_codename`, `experiment_id`

```python
from anonymizer import DEFAULT_ENTITY_LABELS, Detect

detect = Detect(entity_labels=[*DEFAULT_ENTITY_LABELS, "clinical_facility", "diagnosis_code", "medication_name"])
```

### `gliner_threshold`

Default `0.3`. The validator catches false positives downstream, so erring low is safe.

| Symptom | Move | Try |
|---|---|---|
| Entities are being missed | Lower | `0.2` or even `0.15` |
| Validator is slow / expensive — it's being handed a huge candidate list | Raise | `0.4`–`0.5` |

The trade-off is symmetric. **Lowering** the threshold doesn't hurt accuracy — the validator runs in batches of `validation_max_entities_per_call` (default `100`, tunable on `Detect`), so a long candidate list becomes more validator calls but not a worse validator. The cost of `gliner_threshold=0.2` is latency and tokens, not precision. **Raising** the threshold trades that cost for *recall risk*: GLiNER stops surfacing borderline candidates and you're relying on the augmenter LLM alone to fill the gap. Default `0.3` errs low; raise only when validator cost is hurting you, and verify with an `Annotate` preview before trusting a high-threshold setup.

---

## 3. Replace vs Rewrite

Both modes start from the same [detection](detection.md) pipeline. The difference is what happens after entities are detected.

| Question | Replace | Rewrite |
|---|---|---|
| Is the goal "scrub the entities and keep everything else"? | ✅ | — |
| Is the goal "produce a privacy-safe version of this text that downstream models can train on"? | — | ✅ |
| Are there inferable / latent identifiers that aren't explicitly stated (e.g. "during her third round of chemo" → cancer treatment)? | ❌ leaves them | ✅ removes them |
| Additional LLM calls (beyond shared detection) | ~1 (Substitute) or 0 (Redact/Annotate/Hash) | Many (domain → disposition → QA → rewrite → evaluate → repair → judge) |
| Output text length | ≈ same as input | Often shorter / restructured |
| Best for | Structured records, log scrubbing, known-list redaction | Free-text data with implicit identifiers (clinical notes, biographies, depositions, support transcripts) |

**Picking between them.** If your data has inferable identifiers that survive entity-only scrubbing (clinical notes, biographies, depositions), Rewrite is the right fit. For structured records, logs, or single-cell PII, Replace is faster and preserves shape. If you're unsure, walk through a few sample rows before deciding.

---

## 4. (Replace) Which strategy

The four strategies are summarised in [Replace](replace.md#strategy-comparison). The decision rule:

| You want… | Use | Why |
|---|---|---|
| Realistic-looking text safe for sharing or training | **Substitute** | LLM-generated synthetic values preserve readability |
| Clear visual marking that an entity was removed | **Redact** | `[REDACTED_FIRST_NAME]` is unambiguous |
| To inspect what was detected without losing the original | **Annotate** | Original text is preserved next to the label — **not privacy-safe on its own** |
| Deterministic re-identification across documents (same person → same token) | **Hash** | Same input always produces the same hash digest |

**If you're not sure which to pick, use `Substitute`.** It's the most general-purpose choice and matches the bulk of production usage.

### Writing `Substitute.instructions`

`Substitute` accepts free-form `instructions` that are passed to the replacement-generator LLM. Use them when the default behavior produces values that don't match your domain or downstream constraints.

| Pattern | When to use | Example |
|---|---|---|
| Format constraint | The original has a structural shape that must be preserved | `"Replacement IDs must keep the same prefix as the original (e.g. ACME-12345 → ACME-XXXXX)."` |
| Domain hint | Entities are domain-specific and need plausible domain values | `"Replacement names should be plausible Brazilian Portuguese names."` |
| Negative constraint | Avoid certain values | `"Do not use any name that appears in the original text."` |

Keep instructions short (one or two sentences). Long instructions compete with the per-entity context and degrade quality.

!!! note "Substitute is per-row, not per-dataset"

    Within a single row, repeated mentions of the same value get one consistent replacement (entities are grouped by value before the LLM call). **Across rows the LLM has no shared memory** — each row is an independent call, so "Alice" in row 1 and "Alice" in row 47 will likely get different replacements. If you need stable cross-row mappings (e.g. to re-join records by an identifier), use `Hash` instead, or post-process `result.trace_dataframe["_replacement_map"]`.

---

## 5. (Rewrite) Privacy goal

`Rewrite` ships with sensible defaults for `protect` and `preserve` (auto-populated when you pass `Rewrite()` with no arguments). Override them when you can be more specific than the generic defaults.

### How to write `protect`

`protect` answers: **"What should not appear in the output, even by inference?"**

| Pattern | Example |
|---|---|
| Direct identifiers + quasi-identifiers | `"All patient names, medical record numbers, dates of birth, and any combinations of attributes that could re-identify an individual"` |
| Explicit category list | `"Names, addresses, phone numbers, employer names, and any references to specific institutions"` |
| Inferable signals to suppress | `"Direct identifiers and any contextual phrases that could imply a specific medical condition or diagnosis"` |
| Domain-specific identifiers | `"Case numbers, court names, judge names, and any geographic identifiers below the state level"` |

### How to write `preserve`

`preserve` answers: **"What does the rewritten text need to keep so it's still useful?"**

| Pattern | Example |
|---|---|
| Domain content | `"Clinical findings, treatment plans, and medical terminology"` |
| Structural properties | `"The narrative flow, approximate timeline, and emotional tone of the conversation"` |
| Statistical properties | `"The age range and approximate location at country level so downstream demographics analysis remains valid"` |
| Task-relevant signals | `"Argument structure, citations to legal precedent, and the procedural posture of the case"` |

!!! tip "Be specific, but stay short"

    Both fields must be 10–1000 characters and at least 3 words. Aim for 1–3 sentences. The more concrete you are, the more reliably the rewriter targets the right things.

### When to set `strict_entity_protection=True`

**Default: `False`.** Only set `True` when explicitly required by compliance or audit policy — not just because the data is medical, legal, or financial.

By default, low-risk quasi-identifiers may be left unchanged when the engine judges them safe in context. Set `strict_entity_protection=True` to force every detected entity into an active protection method.

Use it when:

- A documented compliance or audit policy *mandates* that every detected entity be actively protected (e.g. HIPAA Safe Harbor with strict interpretation, internal "zero unchanged identifiers" rule)
- You're producing data for external sharing where any unchanged identifier is a compliance risk
- Audit requires "every entity was actively protected"

Being in a regulated domain (medical / legal / financial) is **not** by itself a reason to set this to `True` — most regulated-domain processing tolerates the default behavior. Don't use it when utility matters more than blanket protection — it tends to increase modifications and can lower `utility_score`.

---

## 6. (Rewrite) Risk tolerance

`risk_tolerance` is a rewrite-only knob — it selects a coherent bundle of repair and review thresholds. The full table is in [Rewrite > Risk tolerance](rewrite.md#risk-tolerance); the choice rule is below.

| Goal | Pick |
|---|---|
| "Medical / legal / financial / external release" | `minimal` |
| Default for most privacy-sensitive data | `low` |
| "I want utility prioritized, this is internal-only" | `moderate` |
| "I just want to see the system run, will fix things by hand" | `high` |

Notes:

- `minimal` and `low` differ mostly in how aggressively repair triggers. Both auto-repair on any high-sensitivity leak.
- `high` does **not** auto-repair single high-sensitivity leaks. Use only when you have downstream review.
- `max_repair_iterations` (default 3) caps cost. Set to 0 to skip repair entirely while still computing leakage / utility metrics — useful for audits.

---

## Goal → starting config cheat sheet

A starting point for common scenarios. Always run `preview` and iterate from here.

| Goal | Mode | Strategy / config |
|---|---|---|
| "Scrub PII from logs for retention" | Replace | `Redact()` |
| "De-identify clinical notes for research sharing" | Rewrite | `Rewrite(privacy_goal=PrivacyGoal(protect="all PHI and any context that could imply a specific patient or facility", preserve="clinical findings, treatments, and outcomes"), risk_tolerance="minimal", strict_entity_protection=True)` |
| "Produce realistic-looking biographies for demos" | Replace | `Substitute(instructions="Names and locations should remain plausible for the original cultural context.")` |
| "Anonymize survey responses before sharing the dataset" | Replace | `Substitute()` |
| "Anonymize customer support transcripts for fine-tuning a model" | Replace | `Substitute(instructions="Preserve domain-specific terminology and locale.")` |
| "Anonymize legal opinions for an SFT dataset" | Rewrite | `Rewrite(privacy_goal=PrivacyGoal(protect="party names, case numbers, judge names, and locations below the state level", preserve="argument structure and procedural posture"), risk_tolerance="low")` |
| "Allow re-joining records by identifier without keeping the identifier" | Replace | `Hash(algorithm="sha256", digest_length=16)` |
| "I just want to see what the detector finds" | Replace | `Annotate()` (preview only — never ship Annotate output as anonymized data) |

Once you have a starting config, run `anonymizer validate <config>`, then `anonymizer preview --num-records 5 <config>`, then iterate. See [Troubleshooting](../troubleshooting.md) for what to change when preview shows a problem.

---
name: entity label examples
overview: "Support configured examples for default and non-default labels in one detection-only implementation for issue #259. Configured examples merge additively with built-in examples; non-default example keys must be declared in an explicit `entity_labels` set."
todos:
  - id: configure-label-examples
    content: Add normalized, isolated configured examples for default and non-default labels with cross-field validation.
    status: completed
  - id: resolve-effective-ontology
    content: Resolve additive built-in examples and strict non-default-label membership without global mutation.
    status: completed
  - id: propagate-detection-examples
    content: Pass effective labels to GLiNER, full resolved examples to validator, and configured examples only to augmenter.
    status: completed
  - id: test-example-behavior
    content: Cover validation, explicit membership, prompts, workflow parity, isolation, prompt growth, and detection-only boundaries.
    status: completed
  - id: document-example-api
    content: Document default/non-default label semantics, built-in/configured example semantics, the issue divergence, costs, risks, and update the agent skill.
    status: completed
isProject: false
---

# Entity Label Examples

## Goal and boundaries

- Add `Detect.entity_label_examples: dict[str, list[str]]` as per-run positive detection guidance.
- Pass the full resolved example ontology to the validator and only user-configured examples to the augmenter. Do not change substitution, evaluation, or `AnonymizerResult`/`PreviewResult`.
  - Example: a detected non-default `vendor_api_key` is still substituted using the existing generic non-default-label fallback; its configured detection examples do not guide replacement generation.
  - Evaluation example: suppose detection uses `entity_labels=["vendor_api_key"]` with `entity_label_examples={"vendor_api_key": ["acme_live_abc123"]}`, finds `acme_live_xyz789`, and `Substitute()` produces `acme_live_qrs456`. Existing evaluation may report `entity_coverage=1.0` when the original value is covered, `detection_valid=True` when the optional detection judge accepts the value/label from context, and `type_fidelity_valid=True` when the synthetic value preserves a plausible class and structure. These outcomes are model-dependent, and none of these judges receive `acme_live_abc123`; a non-default label falls back to generic/contextual judgment rather than configured-example comparison.
- Examples improve model interpretation but are not format allowlists or guaranteed exclusions.

## Terminology

- **Explicit label set:** labels supplied through `entity_labels`; it may include default and non-default labels.
- **Default label:** a label present in `DEFAULT_ENTITY_LABELS`.
- **Non-default label:** a label absent from `DEFAULT_ENTITY_LABELS`.
- **Built-in examples:** examples shipped with Anonymizer in `ENTITY_LABEL_EXAMPLES`.
- **Configured examples:** user-supplied positive examples in `entity_label_examples`; they may target default or non-default labels.

## Intentional divergence and known limitations

- Follow issue #259 by requiring every non-excluded example key to belong to the active label set. A non-default label must therefore be declared in an explicit `entity_labels` set; examples never activate it implicitly.
- Document one intentional difference from issue #259: the augmenter receives only configured examples rather than the full resolved mapping of built-in and configured examples.
- Keep evaluation out of scope. Configured examples are not persisted on `AnonymizerResult`/`PreviewResult` or passed to evaluation, but an explicit label set containing non-default labels is persisted and scopes entity coverage consistently.
- Treat example values as potentially sensitive configuration. They are embedded in validator/augmenter prompts, included in exported detection builders, and sent to configured model providers.
  - Example: use synthetic `acme_live_abc123`, never a real production credential or customer identifier.
- Do not include example values in telemetry, logs, warning text, or measurement attributes.

## Examples for default labels

### Behavior

- When a configured example key identifies a default label, keep the label’s built-in examples and append the configured examples.
  - Example: `{"api_key": ["sk-ant-api03-abc123"]}` retains built-in examples such as `sk-abc123def456`.
- Stable-deduplicate the merged list without mutating `[ENTITY_LABEL_EXAMPLES](src/anonymizer/engine/constants.py)`.
- Configured examples for a default label are valid with `entity_labels=None` because that label is already active through defaults.
- If `entity_labels` is explicit, require the default-label example key to be present in that explicit label set.
  - Example: `entity_labels=["email"]` with examples for `api_key` is an error.
- Let `excluded_entity_labels` take precedence: warn and ignore examples for an excluded default label; error only when no effective labels remain.
  - Example: excluded `api_key` examples are omitted while other defaults remain active. If `api_key` is the only explicit label, exclusion leaves an empty set and raises.

### Resolution example

```python
Detect(
    entity_label_examples={
        "api_key": ["sk-ant-api03-abc123"],
    },
)
```

Resolves to the default label set and a run-local `api_key` example list containing built-in examples followed by `sk-ant-api03-abc123`.

### Default-label example flow

```mermaid
flowchart TB
    subgraph builtConfig [NEW OR MODIFIED configuration behavior]
        BuiltStart["Detect configuration"] --> BuiltExamples{"Examples for a default label configured?"}
        BuiltExamples -->|"No"| BuiltCurrent["Resolve labels as today"]
        BuiltExamples -->|"Yes"| BuiltNormalize["Normalize keys and values"]
        BuiltNormalize --> BuiltExplicit{"entity_labels explicitly set?"}
        BuiltExplicit -->|"No"| BuiltDefaults["Use default labels"]
        BuiltExplicit -->|"Yes"| BuiltListed{"Example key in explicit list?"}
        BuiltListed -->|"No"| BuiltMismatch["Raise explicit-set mismatch"]
        BuiltListed -->|"Yes"| BuiltSelected["Use explicit labels"]
        BuiltDefaults --> BuiltExcluded{"Example label excluded?"}
        BuiltSelected --> BuiltExcluded
        BuiltExcluded -->|"Yes"| BuiltWarn["Warn and omit label examples"]
        BuiltWarn --> BuiltRemaining{"Any labels remain?"}
        BuiltRemaining -->|"No"| BuiltEmpty["Raise empty-set error"]
        BuiltRemaining -->|"Yes"| BuiltResolve["Resolve remaining labels"]
        BuiltExcluded -->|"No"| BuiltMerge["Copy defaults and append additions"]
        BuiltMerge --> BuiltResolve
        BuiltCurrent --> BuiltResolve
        BuiltResolve --> BuiltOntology["Effective labels and resolved examples"]
    end

    subgraph builtPipeline [Detection pipeline]
        BuiltOntology --> BuiltGliner(["UNCHANGED: GLiNER gets label names only"])
        BuiltGliner --> BuiltCandidates(["UNCHANGED: seed candidates"])
        BuiltOntology -->|"Labels and built-in examples"| BuiltValidator(["UNCHANGED STAGE: validator"])
        BuiltOntology --> BuiltConfigured["NEW INPUT: configured example additions"]
        BuiltConfigured --> BuiltValidator
        BuiltCandidates --> BuiltValidator
        BuiltValidator --> BuiltDecisions(["UNCHANGED: keep, drop, or reclass"])
        BuiltOntology -->|"All effective label names"| BuiltAugmenter(["UNCHANGED STAGE: augmenter"])
        BuiltConfigured --> BuiltAugmenter
        BuiltDecisions --> BuiltAugmenter
        BuiltAugmenter --> BuiltFinal(["UNCHANGED: merge and finalize"])
        BuiltFinal --> BuiltDownstream(["UNCHANGED: substitution and evaluation"])
    end
```

- Diagram key: rectangles and decision diamonds are new or modified behavior; rounded nodes are unchanged stages from `main`.
- Without `entity_label_examples`, label resolution and prompt behavior remain unchanged.
- With configured examples for default labels, GLiNER still receives only active label names. The validator receives built-in examples plus configured examples, while the augmenter receives only configured examples.
- An explicit `entity_labels` set remains authoritative: a configured default-label example key outside that set is an error.
- Exclusions remove the label and its examples. Processing continues with a warning when labels remain and fails only when the effective set becomes empty.

## Examples for non-default labels

### Behavior

- When a configured example key identifies a non-default label, require an explicit `entity_labels` set containing that label.
  - Example: `entity_label_examples={"vendor_api_key": [...]}` with `entity_labels=None` raises a clear inactive-label error.
- Keep explicit label sets strict: every non-excluded configured example key must already be listed.
  - Example: `entity_labels=["vendor_api_key"]` with matching examples detects only that non-default label.
  - Example: `entity_labels=["email"]` with examples for `vendor_api_key` is an error.
- To detect defaults plus a non-default label, require `entity_labels=[*DEFAULT_ENTITY_LABELS, "vendor_api_key"]`.
- Let exclusions take precedence. Warn and omit configured examples for an excluded non-default label; error only if all explicitly selected labels are excluded.
  - Example: with `entity_labels=[*DEFAULT_ENTITY_LABELS, "vendor_api_key"]`, excluding `vendor_api_key` leaves the defaults active; `entity_labels=["vendor_api_key"]` plus the same exclusion is an empty-set error.
- Reject unknown normalized keys rather than treating them as intentional labels.
  - Example: `vendor_api_ky` raises unless that exact label also appears in the explicit label set.
- An explicit label set keeps the augmenter strict.

### Resolution examples

Defaults plus a non-default label:

```python
from anonymizer import DEFAULT_ENTITY_LABELS

Detect(
    entity_labels=[*DEFAULT_ENTITY_LABELS, "vendor_api_key"],
    entity_label_examples={
        "vendor_api_key": ["acme_live_abc123"],
    },
)
```

Non-default only:

```python
Detect(
    entity_labels=["vendor_api_key"],
    entity_label_examples={
        "vendor_api_key": ["acme_live_abc123"],
    },
)
```

### Non-default-label example flow

```mermaid
flowchart TB
    subgraph nonDefaultConfig [NEW OR MODIFIED non-default-label behavior]
        NonDefaultStart["Non-default-label example key"] --> NonDefaultNormalize["Normalize key and values"]
        NonDefaultNormalize --> NonDefaultKnown{"Key identifies a default label?"}
        NonDefaultKnown -->|"Yes"| NonDefaultBuiltIn["Use default-label flow"]
        NonDefaultKnown -->|"No"| NonDefaultExcluded{"Non-default label excluded?"}
        NonDefaultExcluded -->|"Yes"| NonDefaultWarn["Warn and omit configured examples"]
        NonDefaultWarn --> NonDefaultRemaining{"Any labels remain?"}
        NonDefaultRemaining -->|"No"| NonDefaultEmpty["Raise empty-set error"]
        NonDefaultRemaining -->|"Yes"| NonDefaultResolve["Resolve remaining labels"]
        NonDefaultExcluded -->|"No"| NonDefaultExplicit{"entity_labels explicitly set?"}
        NonDefaultExplicit -->|"No"| NonDefaultInactive["Raise inactive-label error"]
        NonDefaultExplicit -->|"Yes"| NonDefaultListed{"Key in explicit label set?"}
        NonDefaultListed -->|"No"| NonDefaultMismatch["Raise explicit-set mismatch"]
        NonDefaultListed -->|"Yes"| NonDefaultSelected["Use explicit labels exactly"]
        NonDefaultSelected --> NonDefaultExamples["Create run-local configured examples"]
        NonDefaultExamples --> NonDefaultResolve
        NonDefaultResolve --> NonDefaultOntology["Effective labels and resolved examples"]
    end

    subgraph nonDefaultPipeline [Detection pipeline]
        NonDefaultOntology --> NonDefaultGliner(["UNCHANGED STAGE: GLiNER gets all effective label names"])
        NonDefaultGliner --> NonDefaultCandidates(["UNCHANGED STAGE: seed candidates"])
        NonDefaultOntology -->|"Labels and existing built-in examples"| NonDefaultValidator(["UNCHANGED STAGE: validator"])
        NonDefaultOntology --> NonDefaultConfigured["NEW INPUT: configured non-default-label examples"]
        NonDefaultConfigured --> NonDefaultValidator
        NonDefaultCandidates --> NonDefaultValidator
        NonDefaultValidator --> NonDefaultDecisions(["UNCHANGED: keep, drop, or reclass"])
        NonDefaultOntology -->|"All effective label names"| NonDefaultAugmenter(["UNCHANGED STAGE: augmenter"])
        NonDefaultConfigured --> NonDefaultAugmenter
        NonDefaultDecisions --> NonDefaultAugmenter
        NonDefaultAugmenter --> NonDefaultFinal(["UNCHANGED: recover misses and finalize"])
        NonDefaultFinal --> NonDefaultDownstream(["UNCHANGED: generic substitution and evaluation"])
    end
```

- Diagram key: rectangles and decision diamonds are new or modified behavior; rounded nodes are unchanged stages from `main`.
- A configured example key never activates a non-default label implicitly.
- An explicit label set can select non-default-only detection or defaults plus non-default labels, but every non-excluded example key must already be listed.
- A misspelled non-default key raises a mismatch instead of becoming an active label.
- Default and non-default labels use the same detection sequence. GLiNER-found candidates receive validator review; the augmenter then searches for misses using all active label names and only user-configured examples.
- Augmented findings are not independently revalidated. If validator cost later justifies augmenter-only non-default labels, that should be a separate feature with explicit scopes and documented quality tradeoffs.

## Combined alternative flow

This diagram combines the default-label and non-default-label resolution branches before they enter the shared detection pipeline. It is an alternative to the two separate diagrams above; those diagrams remain available for a more detailed view of each case.

```mermaid
flowchart TB
    Start["Configured entity label examples"] --> Normalize["Normalize keys and values"]
    Normalize --> LabelKind{"Key identifies a default label?"}

    subgraph defaultResolution [Default-label resolution]
        LabelKind -->|"Yes"| DefaultExplicit{"entity_labels explicitly set?"}
        DefaultExplicit -->|"No"| DefaultActive["Label already active through defaults"]
        DefaultExplicit -->|"Yes"| DefaultListed{"Key in explicit label set?"}
        DefaultListed -->|"No"| DefaultMismatch["Raise explicit-set mismatch"]
        DefaultListed -->|"Yes"| DefaultSelected["Use explicit label set"]
        DefaultActive --> DefaultMerge["Copy built-in examples and append configured examples"]
        DefaultSelected --> DefaultMerge
    end

    subgraph nonDefaultResolution [Non-default-label resolution]
        LabelKind -->|"No"| NonDefaultExcluded{"Non-default label excluded?"}
        NonDefaultExcluded -->|"Yes"| Omit
        NonDefaultExcluded -->|"No"| NonDefaultExplicit{"entity_labels explicitly set?"}
        NonDefaultExplicit -->|"No"| NonDefaultInactive["Raise inactive-label error"]
        NonDefaultExplicit -->|"Yes"| NonDefaultListed{"Key in explicit label set?"}
        NonDefaultListed -->|"No"| NonDefaultMismatch["Raise explicit-set mismatch"]
        NonDefaultListed -->|"Yes"| NonDefaultSelected["Use explicit label set"]
        NonDefaultSelected --> NonDefaultExamples["Use configured examples"]
    end

    DefaultMerge --> Excluded{"Example label excluded?"}
    Excluded -->|"Yes"| Omit["Warn, remove label, and omit its examples"]
    Omit --> Remaining{"Any labels remain?"}
    Remaining -->|"No"| Empty["Raise empty-set error"]
    Remaining -->|"Yes"| Ontology["Effective labels and resolved examples"]
    Excluded -->|"No"| Ontology
    NonDefaultExamples --> Ontology

    subgraph sharedPipeline [Shared detection pipeline]
        Ontology --> Gliner(["GLiNER gets effective label names"])
        Gliner --> Candidates(["Seed candidates"])
        Ontology -->|"Full resolved examples"| Validator(["Validator"])
        Candidates --> Validator
        Validator --> Decisions(["Keep, drop, or reclass"])
        Ontology -->|"Effective names and configured examples only"| Augmenter(["Augmenter"])
        Decisions --> Augmenter
        Augmenter --> Final(["Merge and finalize"])
        Final --> Downstream(["Substitution and evaluation unchanged"])
    end
```

## Validation shared by both cases

- Extend `[src/anonymizer/config/anonymizer_config.py](src/anonymizer/config/anonymizer_config.py)` with an isolated `default_factory=dict`.
- Normalize keys consistently with `entity_labels` using `strip().casefold()`, trim values without changing case, and copy nested input state.
- Reject blank/non-string keys, empty or non-list collections, and blank/non-string examples.
- Merge normalized duplicate keys in input order and stable-deduplicate their values, emitting a warning.
  - Example: `{" Email ": ["alice@example.test"], "email": ["bob@example.test"]}` becomes one ordered `email` list.
- Require every non-excluded configured example key to belong to `entity_labels` when explicit, or `DEFAULT_ENTITY_LABELS` when `entity_labels=None`.
- Compute the effective label set after exclusions and reject when that final set is empty.
- Ensure separate configs, caller-owned lists, and sequential runs cannot contaminate one another.

## Shared detection pipeline

- Add a pure run-local resolver near `[src/anonymizer/engine/detection/detection_workflow.py](src/anonymizer/engine/detection/detection_workflow.py)` that resolves effective labels and examples once without modifying module defaults.
- Thread the resolved state through `[src/anonymizer/interface/anonymizer.py](src/anonymizer/interface/anonymizer.py)` for normal runs, previews, rewrite-mode detection, dataframe exports, and seed-builder exports.
- Use the same pipeline for default and non-default labels:
  - GLiNER receives effective label names only; it has no per-label examples input.
  - Validator receives every effective label with its full resolved examples: built-in examples plus configured examples for default labels, and configured examples for non-default labels.
  - Augmenter receives every effective label name as it does today, plus examples only for labels explicitly present in `entity_label_examples`.

## Prompt construction and growth

- Update `_format_label_examples()` and `_get_validation_prompt()` to consume the full resolved mapping. Update `_get_augment_prompt()` separately to receive only normalized user-configured examples, filtered to effective active labels.
- Encode configured values safely as prompt data so punctuation and template-like strings cannot alter rendering.
- Add no runtime size limits in this phase. Document that the full ontology repeats per validator chunk, while only the smaller configured-example block is added once per augmenter row.
  - Example: prefer two representative synthetic vendor-key formats over dozens of real credentials.
- Keep all active labels/examples in validator prompts because candidate-only examples weaken reclassification into labels absent from a chunk.
- Add non-blocking prompt-size regression measurements to inform future limits or retrieval-based designs.

## Telemetry semantics

- Never record configured example values.
- Continue recording the raw label configuration:
  - `entity_labels` contains the explicit label set when supplied and remains `None` when defaults are used;
  - `excluded_entity_labels` records exclusions separately;
  - `entity_label_count` retains its existing pre-exclusion meaning for compatibility.
- Add `effective_entity_labels` containing the resolved default or explicit label set after exclusions. This list includes every non-default label associated with configured examples because those labels now require explicit membership in `entity_labels`.
- Add `effective_entity_label_count` as the length of `effective_entity_labels`.
- Define `effective_entity_labels` as the labels passed into the configured detection ontology; in permissive mode, the augmenter may still emit additional labels not present in this list.
- Add both effective fields as optional in measurement records and downstream ingestion/reporting models so historical records without them remain valid.
- Update aggregate reporting to prefer `effective_entity_label_count` and fall back to the existing `entity_label_count` for historical records.

## Tests

- In `[tests/config/test_anonymizer_config.py](tests/config/test_anonymizer_config.py)`, cover shared normalization/isolation plus built-in-example merging, rejection of undeclared non-default labels, explicit non-default-only and defaults-plus-non-default cases, typo behavior, excluded-example warnings, empty effective sets, and serialization.
- In `[tests/engine/test_detection_workflow.py](tests/engine/test_detection_workflow.py)`, separately cover:
  - configured examples merging with built-in examples without global mutation;
  - explicitly declared non-default names reaching GLiNER, full resolved examples reaching validator, and configured-only examples reaching augmenter;
  - built-in examples remaining absent from the augmenter unless the user supplied configured examples for that label;
  - validator keep/drop/reclass and augmenter recovery;
  - effective-label filtering, strict/permissive behavior, safe special characters, and cross-run isolation.
- In `[tests/engine/test_detection_config_serialization.py](tests/engine/test_detection_config_serialization.py)` and interface tests, verify parity across run, preview, rewrite detection, dataframe export, seed export, and round-trip reconstruction.
- Add regressions proving replacement prompts/outputs, evaluation judges, and result dataclasses receive no example state.
- Add telemetry/logging regressions proving configured example values are never emitted, while exported detection builders intentionally contain the prompt examples required for remote execution.
- Add telemetry regressions proving:
  - explicit non-default labels and exclusions remain recorded;
  - `entity_label_count` retains its existing pre-exclusion value;
  - `effective_entity_labels` contains the post-exclusion default or explicit label set;
  - `effective_entity_label_count == len(effective_entity_labels)`;
  - configured example values never appear in telemetry;
  - historical records without the effective fields remain valid and reporting falls back to `entity_label_count`.

## Documentation and agent skill

- Update `[docs/concepts/detection.md](docs/concepts/detection.md)`, `[docs/concepts/choosing-a-strategy.md](docs/concepts/choosing-a-strategy.md)`, and `[docs/troubleshooting.md](docs/troubleshooting.md)` with separate examples for default labels, explicit defaults plus non-default labels, and explicit non-default-only detection.
- Document positive-only semantics, explicit membership for non-default labels, exclusion precedence and warnings, empty-set errors, typo rejection, prompt cost, synthetic-data guidance, and deterministic alternatives for guaranteed negatives.
- Explain the configured-examples-only augmenter divergence and its prompt-size rationale.
- Warn that exported builders and provider requests contain configured examples, and recommend synthetic values rather than real secrets or PII.
- Update `[skills/anonymizer/SKILL.md](skills/anonymizer/SKILL.md)` and its `Detect(...)` template to require explicit `entity_labels` membership for every non-default configured example key.


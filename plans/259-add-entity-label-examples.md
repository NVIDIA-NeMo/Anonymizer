---
name: entity label examples
overview: "Combine built-in and custom detection examples into one detection-only implementation for issue #259. Built-in examples merge additively; custom example keys automatically extend defaults when `entity_labels=None`, while explicit label lists remain strict."
todos:
  - id: configure-label-examples
    content: Add normalized, isolated built-in/custom example configuration and cross-field validation.
    status: pending
  - id: resolve-effective-ontology
    content: Resolve additive built-ins and automatic defaults-plus-custom labels without global mutation.
    status: pending
  - id: propagate-detection-examples
    content: Pass effective labels to GLiNER, full resolved examples to validator, and configured examples only to augmenter.
    status: pending
  - id: test-example-behavior
    content: Cover validation, activation, prompts, workflow parity, isolation, prompt growth, and detection-only boundaries.
    status: pending
  - id: document-example-api
    content: Document built-in/custom semantics, issue divergence, costs, risks, examples, and update the agent skill.
    status: pending
isProject: false
---

# Entity Label Examples

## Goal and boundaries

- Add `Detect.entity_label_examples: dict[str, list[str]]` as per-run positive detection guidance.
- Pass the full resolved example ontology to the validator and only user-configured examples to the augmenter. Do not change substitution, evaluation, or `AnonymizerResult`/`PreviewResult`.
  - Example: a detected custom `vendor_api_key` is still substituted using the existing generic custom-label fallback; its configured detection examples do not guide replacement generation.
  - Evaluation example: suppose detection uses `{"vendor_api_key": ["acme_live_abc123"]}`, finds `acme_live_xyz789`, and `Substitute()` produces `acme_live_qrs456`. Existing evaluation may report `entity_coverage=1.0` when the original value is covered, `detection_valid=True` when the optional detection judge accepts the value/label from context, and `type_fidelity_valid=True` when the synthetic value preserves a plausible class and structure. These outcomes are model-dependent, and none of these judges receive `acme_live_abc123`; a custom label falls back to generic/contextual judgment rather than configured-example comparison.
- Examples improve model interpretation but are not format allowlists or guaranteed exclusions.

## Intentional divergences and known limitations

- Document two intentional differences from issue #259:
  - custom example keys automatically activate labels alongside defaults when `entity_labels=None`;
  - the augmenter receives only user-configured examples rather than the full resolved built-in mapping.
- Keep evaluation out of scope and record the consequence: configured examples are not persisted on `AnonymizerResult`/`PreviewResult`, and automatically activated custom labels are not reproduced as an explicit evaluation allowlist. Entity coverage remains permissive when the originating `entity_labels` was `None` and does not receive the configured examples.
- Treat example values as potentially sensitive configuration. They are embedded in validator/augmenter prompts, included in exported detection builders, and sent to configured model providers.
  - Example: use synthetic `acme_live_abc123`, never a real production credential or customer identifier.
- Do not include example values in telemetry, logs, warning text, or measurement attributes.

## Built-in entity labels

### Behavior

- When an example key exists in `DEFAULT_ENTITY_LABELS`, keep the label’s built-in examples and append the configured additions.
  - Example: `{"api_key": ["sk-ant-api03-abc123"]}` retains built-in formats such as `sk-abc123def456`.
- Stable-deduplicate the merged list without mutating `[ENTITY_LABEL_EXAMPLES](src/anonymizer/engine/constants.py)`.
- Examples for a built-in label are valid with `entity_labels=None` because that label is already active through defaults.
- If `entity_labels` is explicit, require the built-in example key to be present in that list.
  - Example: `entity_labels=["email"]` with examples for `api_key` is an error.
- Let `excluded_entity_labels` take precedence: warn and ignore examples for an excluded built-in label; error only when no effective labels remain.
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

### Built-in label flow

```mermaid
flowchart TB
    subgraph builtConfig [NEW OR MODIFIED configuration behavior]
        BuiltStart["Detect configuration"] --> BuiltExamples{"Built-in examples configured?"}
        BuiltExamples -->|"No"| BuiltCurrent["Resolve labels as today"]
        BuiltExamples -->|"Yes"| BuiltNormalize["Normalize keys and values"]
        BuiltNormalize --> BuiltExplicit{"entity_labels explicitly set?"}
        BuiltExplicit -->|"No"| BuiltDefaults["Use default labels"]
        BuiltExplicit -->|"Yes"| BuiltListed{"Example key in explicit list?"}
        BuiltListed -->|"No"| BuiltMismatch["Raise allowlist mismatch"]
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
- With built-in additions, GLiNER still receives only active label names. The validator receives built-in plus configured examples, while the augmenter receives only configured additions.
- An explicit `entity_labels` list remains authoritative: a configured built-in example key outside that list is an error.
- Exclusions remove the label and its examples. Processing continues with a warning when labels remain and fails only when the effective set becomes empty.

## Custom entity labels

### Behavior

- When an example key is absent from `DEFAULT_ENTITY_LABELS` and `entity_labels=None`, automatically activate it alongside all defaults.
  - Example: `{"vendor_api_key": ["acme_live_abc123"]}` resolves to `[*DEFAULT_ENTITY_LABELS, "vendor_api_key"]`.
- This intentionally differs from issue #259, which says examples should not activate labels. The divergence avoids requiring users to inspect, import, and unpack `DEFAULT_ENTITY_LABELS` for the common defaults-plus-custom case.
- When `entity_labels` is explicit, keep it strict: every custom example key must already be listed.
  - Example: `entity_labels=["vendor_api_key"]` with matching examples detects only that custom label.
  - Example: `entity_labels=["email"]` with examples for `vendor_api_key` is an error.
- Let exclusions take precedence under automatic and explicit activation. Warn, omit the custom examples, and do not auto-activate an excluded custom key; error only if all effective default and custom labels are excluded.
  - Example: excluded `vendor_api_key` is ignored while defaults continue; `entity_labels=["vendor_api_key"]` plus the same exclusion is an empty-set error.
- Treat unknown normalized keys as intentional custom labels when `entity_labels=None`; spelling intent cannot be inferred.
  - Example: `vendor_api_ky` is activated as written. With explicit `entity_labels=["vendor_api_key"]`, the mismatch is caught.
- Preserve existing augmenter strictness: automatic defaults-plus-custom mode remains permissive; an explicit label list remains strict.

### Resolution examples

Defaults plus custom:

```python
Detect(
    entity_label_examples={
        "vendor_api_key": ["acme_live_abc123"],
    },
)
```

Custom only:

```python
Detect(
    entity_labels=["vendor_api_key"],
    entity_label_examples={
        "vendor_api_key": ["acme_live_abc123"],
    },
)
```

### Custom label flow

```mermaid
flowchart TB
    subgraph customConfig [NEW OR MODIFIED custom-label behavior]
        CustomStart["Custom example key"] --> CustomNormalize["Normalize key and values"]
        CustomNormalize --> CustomKnown{"Key is built in?"}
        CustomKnown -->|"Yes"| CustomBuiltIn["Use built-in flow"]
        CustomKnown -->|"No"| CustomExplicit{"entity_labels explicitly set?"}
        CustomExplicit -->|"No"| CustomAuto["Activate defaults plus custom key"]
        CustomExplicit -->|"Yes"| CustomListed{"Custom key in explicit list?"}
        CustomListed -->|"No"| CustomMismatch["Raise allowlist mismatch"]
        CustomListed -->|"Yes"| CustomSelected["Use explicit labels exactly"]
        CustomAuto --> CustomExcluded{"Custom label excluded?"}
        CustomSelected --> CustomExcluded
        CustomExcluded -->|"Yes"| CustomWarn["Warn, omit examples, and do not activate"]
        CustomWarn --> CustomRemaining{"Any labels remain?"}
        CustomRemaining -->|"No"| CustomEmpty["Raise empty-set error"]
        CustomRemaining -->|"Yes"| CustomResolve["Resolve remaining labels"]
        CustomExcluded -->|"No"| CustomExamples["Create run-local custom examples"]
        CustomExamples --> CustomResolve
        CustomResolve --> CustomOntology["Effective labels and resolved examples"]
    end

    subgraph customPipeline [Detection pipeline]
        CustomOntology --> CustomGliner(["UNCHANGED STAGE: GLiNER gets all effective label names"])
        CustomGliner --> CustomCandidates(["UNCHANGED STAGE: seed candidates"])
        CustomOntology -->|"Labels and existing built-in examples"| CustomValidator(["UNCHANGED STAGE: validator"])
        CustomOntology --> CustomConfigured["NEW INPUT: configured custom examples"]
        CustomConfigured --> CustomValidator
        CustomCandidates --> CustomValidator
        CustomValidator --> CustomDecisions(["UNCHANGED: keep, drop, or reclass"])
        CustomOntology -->|"All effective label names"| CustomAugmenter(["UNCHANGED STAGE: augmenter"])
        CustomConfigured --> CustomAugmenter
        CustomDecisions --> CustomAugmenter
        CustomAugmenter --> CustomFinal(["UNCHANGED: recover misses and finalize"])
        CustomFinal --> CustomDownstream(["UNCHANGED: generic substitution and evaluation"])
    end
```

- Diagram key: rectangles and decision diamonds are new or modified behavior; rounded nodes are unchanged stages from `main`.
- With `entity_labels=None`, a normalized custom key is intentionally treated as defaults-plus-custom activation. This is a documented usability divergence from issue #259.
- With explicit `entity_labels`, the list remains strict. It can select custom-only detection or defaults plus custom, but every example key must already be listed.
- A misspelled unknown key is treated as an intentional custom label in automatic mode; explicit mode can catch a mismatch.
- Custom and built-in labels use the same detection sequence. GLiNER-found candidates receive validator review; the augmenter then searches for misses using all active label names and only user-configured examples.
- Augmented findings are not independently revalidated. If validator cost later justifies augmenter-only custom labels, that should be a separate feature with explicit scopes and documented quality tradeoffs.

## Validation shared by both cases

- Extend `[src/anonymizer/config/anonymizer_config.py](src/anonymizer/config/anonymizer_config.py)` with an isolated `default_factory=dict`.
- Normalize keys with `strip().lower()`, trim values without changing case, and copy nested input state.
- Reject blank/non-string keys, empty or non-list collections, and blank/non-string examples.
- Merge normalized duplicate keys in input order and stable-deduplicate their values, emitting a warning.
  - Example: `{" Email ": ["alice@example.test"], "email": ["bob@example.test"]}` becomes one ordered `email` list.
- Compute the effective label set after automatic activation and exclusions, and reject only when that final set is empty.
  - Example: excluding every default remains valid when a non-excluded custom key is automatically activated; excluding that custom key too raises.
- Ensure separate configs, caller-owned lists, and sequential runs cannot contaminate one another.

## Shared detection pipeline

- Add a pure run-local resolver near `[src/anonymizer/engine/detection/detection_workflow.py](src/anonymizer/engine/detection/detection_workflow.py)` that resolves effective labels and examples once without modifying module defaults.
- Thread the resolved state through `[src/anonymizer/interface/anonymizer.py](src/anonymizer/interface/anonymizer.py)` for normal runs, previews, rewrite-mode detection, dataframe exports, and seed-builder exports.
- Use the same pipeline for built-in and custom labels:
  - GLiNER receives effective label names only; it has no per-label examples input.
  - Validator receives every effective label with its full resolved examples: built-ins plus configured additions for built-in labels, and configured examples for custom labels.
  - Augmenter receives every effective label name as it does today, plus examples only for labels explicitly present in `entity_label_examples`.

## Prompt construction and growth

- Update `_format_label_examples()` and `_get_validation_prompt()` to consume the full resolved mapping. Update `_get_augment_prompt()` separately to receive only normalized user-configured examples, filtered to effective active labels.
- Encode configured values safely as prompt data so punctuation and template-like strings cannot alter rendering.
- Add no runtime size limits in this phase. Document that the full ontology repeats per validator chunk, while only the smaller configured-example block is added once per augmenter row.
  - Example: prefer two representative synthetic vendor-key formats over dozens of real credentials.
- Keep all active labels/examples in validator prompts because candidate-only examples weaken reclassification into labels absent from a chunk.
- Add non-blocking prompt-size regression measurements to inform future limits or retrieval-based designs.

## Tests

- In `[tests/config/test_anonymizer_config.py](tests/config/test_anonymizer_config.py)`, cover shared normalization/isolation plus built-in merging, automatic custom activation, explicit custom-only and mismatch cases, typo behavior, excluded-example warnings, remaining-label continuation, empty effective sets, and serialization.
- In `[tests/engine/test_detection_workflow.py](tests/engine/test_detection_workflow.py)`, separately cover:
  - built-in additions merging without global mutation;
  - custom names reaching GLiNER, full resolved examples reaching validator, and configured-only examples reaching augmenter;
  - built-in defaults remaining absent from augmenter unless the user explicitly configured additions for that label;
  - validator keep/drop/reclass and augmenter recovery;
  - effective-label filtering, strict/permissive behavior, safe special characters, and cross-run isolation.
- In `[tests/engine/test_detection_config_serialization.py](tests/engine/test_detection_config_serialization.py)` and interface tests, verify parity across run, preview, rewrite detection, dataframe export, seed export, and round-trip reconstruction.
- Add regressions proving replacement prompts/outputs, evaluation judges, and result dataclasses receive no example state.
- Add telemetry/logging regressions proving configured example values are never emitted, while exported detection builders intentionally contain the prompt examples required for remote execution.

## Documentation and agent skill

- Update `[docs/concepts/detection.md](docs/concepts/detection.md)`, `[docs/concepts/choosing-a-strategy.md](docs/concepts/choosing-a-strategy.md)`, and `[docs/troubleshooting.md](docs/troubleshooting.md)` with separate built-in, defaults-plus-custom, and explicit custom-only examples.
- Document positive-only semantics, exclusion precedence and warnings, empty-set errors, typo activation, prompt cost, synthetic-data guidance, and deterministic alternatives for guaranteed negatives.
- Explicitly explain both intentional issue #259 divergences, their user-experience/prompt-size rationale, and the evaluation reproducibility limitation.
- Warn that exported builders and provider requests contain configured examples, and recommend synthetic values rather than real secrets or PII.
- Update `[skills/anonymizer/SKILL.md](skills/anonymizer/SKILL.md)` and its `Detect(...)` template to use automatic defaults-plus-custom behavior unless the user requests strict/custom-only detection.


# Default Model Configs

This directory contains the default model configurations used by the Anonymizer when no custom `model_configs` are provided to `run()` or `preview()`.

## Files

- **`models.yaml`** — Defines the pool of available models (alias, provider, inference parameters). Each entry becomes a `ModelConfig` that NeMo Data Designer can route requests to.
- **`entity_detection.yaml`** — Maps detection workflow roles (e.g. `entity_detector`, `entity_validator`) to model aliases from `models.yaml`.
- **`replace_workflow.yaml`** — Maps replacement workflow roles (e.g. `replacement_generator`) to model aliases from `models.yaml`.

## Overriding

Users can override these defaults by passing `model_configs` (as a `list[ModelConfig]`, file path, or YAML string) to `Anonymizer.run()` or `Anonymizer.preview()`. When provided, the user-supplied configs replace these defaults entirely.

## Adding a New Workflow

To add default model selections for a new workflow:

1. Create `<workflow_name>.yaml` with a top-level `selected_models` mapping.
2. Each key should be a role name; each value should be an alias defined in `models.yaml`.

Alias references are validated at runtime when `load_workflow_config()` is called. A future `Anonymizer.validate_config()` step will surface these errors earlier.

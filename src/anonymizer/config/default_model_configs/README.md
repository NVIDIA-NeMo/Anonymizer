# Default Model Configs

This directory contains the default model configurations used by the Anonymizer when no custom `model_configs` are provided to `Anonymizer.__init__`.

## Files

- **`models.yaml`** — Defines the pool of available models (alias, provider, inference parameters). Each entry becomes a `ModelConfig` that NeMo Data Designer can route requests to.
- **`detection.yaml`** — Maps detection workflow roles (e.g. `entity_detector`, `entity_validator`) to model aliases from `models.yaml`.
- **`replace.yaml`** — Maps replacement workflow roles (e.g. `replacement_generator`) to model aliases from `models.yaml`.
- **`rewrite.yaml`** — Maps rewrite workflow roles (e.g. `rewriter`, `evaluator`) to model aliases from `models.yaml`. Placeholder for now until we add rewriting.

## Relationship to `config/models.py`

These YAML files are the single source of truth for default role-to-alias mappings. The pydantic classes in `config/models.py` (`DetectionModelSelection`, `ReplaceModelSelection`, etc.) define the schema (which roles exist and their types) but contain no hardcoded defaults.

At startup, `load_default_model_selection()` reads these YAML files and constructs the default `ModelSelection`. Users override roles via the unified YAML passed to `Anonymizer(model_configs=...)`.

## Unified YAML Format

Users provide a single YAML with the model pool and optional role overrides:

```yaml
selected_models:          # optional — omitted roles use defaults
  detection:
    entity_detector: my-custom-detector
  replace:
    replacement_generator: my-fast-model

model_configs:
  - alias: my-custom-detector
    model: some/model
    provider: nvidia
  # ...
```

## Adding a New Workflow

1. Create `<workflow_name>.yaml` with a top-level `selected_models` mapping.
2. Each key should be a role name; each value should be an alias defined in `models.yaml`.
3. Add a corresponding entry to `WorkflowName` enum in `model_loader.py`.

Alias references are validated at runtime when `load_workflow_config()` is called. A future `Anonymizer.validate_config()` step will surface these errors earlier.

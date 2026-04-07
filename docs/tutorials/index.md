<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorials Overview

The Anonymizer tutorial series covers everything from basic replace workflows to advanced rewrite workflows.

## 🚀 Setting Up Your Environment

### Local Setup Best Practices

First, download the tutorial `anonymizer_tutorial.zip` from the
[Anonymizer release assets](https://github.com/NVIDIA-NeMo/Anonymizer/releases) page. 
<!-- TODO: update with direct link after cutting a release -->

```bash
# Extract tutorial notebooks
unzip anonymizer_tutorial.zip
cd anonymizer_tutorial
```
We recommend using a virtual environment to manage dependencies.
=== "uv (Recommended)"

    ```bash
    # Create and activate a virtual environment
    uv venv .venv
    source .venv/bin/activate

    # Install Anonymizer + notebook tooling
    uv pip install nemo-anonymizer jupyter ipykernel

    # Register this env as a notebook kernel (one-time)
    python -m ipykernel install --user --name anonymizer-venv

    # Set API key before launching Jupyter
    export NVIDIA_API_KEY="your-nvidia-api-key"

    # Launch Jupyter
    uv run jupyter notebook
    ```

=== "pip + venv"

    ```bash
    # Create and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate

    # Install Anonymizer + notebook tooling
    pip install nemo-anonymizer jupyter ipykernel

    # Optional: register this env as a named kernel for Jupyter/VS Code
    python -m ipykernel install --user --name anonymizer-venv

    # Set API key before launching Jupyter
    export NVIDIA_API_KEY="your-nvidia-api-key"

    # Launch Jupyter
    jupyter notebook
    ```

Set your `NVIDIA_API_KEY` in the same shell where you launch Jupyter. If `NVIDIA_API_KEY` appears to be missing in a notebook:

1. Launch Jupyter from the same shell where you exported it.
2. Confirm the selected kernel is your intended environment.
3. Restart the notebook kernel after setting or changing environment variables.

## 📚 Tutorial Series

### 1. [Your First Anonymization](../notebooks/01_your_first_anonymization/)

Learn the basic end-to-end flow: load data, configure replace mode, preview, inspect, and run.

### 2. [Inspecting Detected Entities](../notebooks/02_inspecting_detected_entities/)

Understand what detection produced, how entities are labeled, and where they came from.

### 3. [Choosing a Replacement Strategy](../notebooks/03_choosing_a_replacement_strategy/)

Compare replace mode options -- Redact, Annotate, Hash, and Substitute -- side by side, including custom templates.

### 4. [Rewriting Biographies](../notebooks/04_rewriting_biographies/)

Use rewrite mode with privacy goals, evaluation criteria, and review-flag triage.

### 5. [Rewriting Legal Documents](../notebooks/05_rewriting_legal_documents/)

Apply rewrite mode to legal-domain text with custom entity labels and stricter protection goals.
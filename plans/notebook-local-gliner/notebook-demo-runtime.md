<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Notebook Demo Runtime: Local GLiNER2 and Externally Hosted LLMs

## Status

This work should be developed on a separate branch from current `main`. It is
related to, but should not be added to, the already-large
[PR #212](https://github.com/NVIDIA-NeMo/Anonymizer/pull/212).
This notebook-runtime PR is intended to land before PR #212 and must not depend
on code, files, or documentation introduced there.

## Executive Summary

Make every tutorial notebook runnable without a repository checkout or a separate
terminal process. A notebook user should install one optional package extra,
provide the API keys required by the bundled default providers, and receive a
configured `Anonymizer` instance. The defaults may use multiple external hosts,
such as OpenAI for one model and OpenRouter or NVIDIA Build for another:

```python
%pip install "nemo-anonymizer[notebooks]"
```

```python
from anonymizer.notebooks import create_anonymizer

anonymizer = create_anonymizer()
```

The runtime must:

- run `fastino/gliner2-privacy-filter-PII-multi` locally through native PyTorch;
- select CUDA on a GPU-enabled Colab runtime, MPS on Apple Silicon, and CPU
  otherwise;
- use the bundled default aliases, model IDs, inference parameters, provider
  assignments, and role selections for every non-GLiNER2 role;
- start, health-check, reuse, and stop the local GLiNER2 service from Python;
- work from an installed wheel, without importing anything from `tools/`;
- keep the production vLLM Factory path from PR #212 separate.

## Product Decision

The desired topology is:

```text
Notebook kernel
  |
  |-- local loopback HTTP --> native PyTorch GLiNER2
  |
  `-- HTTPS ---------------> one or more external LLM hosts
```

“Local GLiNER2” means local to the Python environment running the notebook. On
Colab, this is the Colab VM; on a Mac, it is the user's machine; in notebook CI,
it is the GitHub Actions runner.

Installing the package must not itself start a persistent process. Package
installers are not process supervisors, and their processes end when installation
finishes. The explicit `create_anonymizer()` call starts and internally owns the
service from within the notebook kernel.

## Goals

1. Provide a real published extra installable as
   `nemo-anonymizer[notebooks]`.
2. Package a portable native GLiNER2 server inside the wheel, using
   `fastino/gliner2-privacy-filter-PII-multi` by default.
3. Provide a small synchronous notebook API that hides ports, subprocesses,
   health checks, provider YAML, model YAML, and cleanup.
4. Automatically accelerate local GLiNER2 with CUDA or MPS when available.
5. Preserve the bundled provider assignment for every tutorial LLM model,
   including configurations that use multiple external hosts.
6. Keep the bundled default model configuration as the source of truth for the
   no-argument path, while allowing callers to supply the existing
   `model_configs` and `model_providers` overrides.
7. Execute all five generated tutorial notebooks successfully in CI.
8. Change the repository's bundled default entity detector from
   `nvidia/gliner-pii` to `fastino/gliner2-privacy-filter-PII-multi`, while
   preserving the existing non-detector defaults.

## Non-Goals

- Do not make the full anonymization pipeline local. The LLM validation,
  augmentation, substitution, rewrite, repair, and evaluation stages use the
  configured external hosts.
- Do not use vLLM or vLLM Factory for the notebook runtime.
- Do not turn the notebook helper into a production service manager.
- Do not invent notebook-specific model-selection arguments such as
  `external_model`, `fast_model`, or `strong_model`. Reuse the existing
  `model_configs` and `model_providers` interfaces.
- Do not promise that input text remains on the user's machine.
- Do not add notebook-specific imports to `anonymizer/__init__.py`; users without
  the optional extra must retain a lightweight, working core import.

## Relationship to PR #212

At the time of this handoff, PR #212:

- deletes `tools/serve_gliner.py`;
- adds a Linux/NVIDIA-GPU vLLM Factory deployment tool;
- adds a `local-models` dependency group restricted to Linux and Python 3.12+;
- does not add an installable package extra or a Mac/CPU/Colab notebook runtime.

The two efforts serve different users:

| Path | Intended environment | Runtime |
| --- | --- | --- |
| Notebook runtime | Mac, Colab, CPU notebook CI | Native GLiNER2/PyTorch |
| PR #212 deployment tool | Linux with NVIDIA GPUs | vLLM Factory |

The notebook runtime is the supported convenience path for tutorials and local
interactive demos. It is not the recommended production serving path. Production
and non-notebook self-hosting guidance in this PR should explain that users must
supply a compatible self-hosted GLiNER2 endpoint, identify PR #212 as the planned
vLLM Factory production path, and avoid implying that the notebook-native server
is production infrastructure. After PR #212 lands, its documentation should
become the canonical production setup and show how to configure Anonymizer with
that deployment's detector endpoint. Bare `Anonymizer()` remains a client and
must not implicitly start or supervise a model service.

Start this work from current `main`. Refactor the reusable implementation of
`tools/serve_gliner.py` into the nested package
`anonymizer.notebooks.local_inference.gliner2`, and leave the tools entrypoint as
a thin wrapper while it exists. If PR #212 lands first and deletes that wrapper,
the installed implementation must remain unaffected.

Do not copy implementation details from PR #212's vLLM compatibility adapter into
the notebook runtime unless they are part of the external request/response
contract. The native implementation already on `main` is the closer starting
point.

## Target Public API

Create a deliberately small optional namespace:

```python
from anonymizer.notebooks import create_anonymizer, stop_local_runtime
```

Recommended API shape:

```python
anonymizer = create_anonymizer(gliner_device="auto")

# Optional explicit cleanup. Kernel exit also cleans up automatically.
stop_local_runtime()
```

The normal tutorial path takes no provider or model arguments. API keys are read
from the environment variables named by the resolved provider configuration.
For customization, `create_anonymizer()` accepts the same `model_configs` and
`model_providers` shapes as `Anonymizer`, including multiple named providers, but
it must not introduce a single-endpoint shortcut that collapses the provider
topology.

```python
anonymizer = create_anonymizer(
    model_configs="models.yaml",
    model_providers="providers.yaml",
    gliner_device="auto",
)
```

This still guarantees local GLiNER2 detection. The helper injects its local
detector model/provider and overrides only `detection.entity_detector`; all other
caller-supplied model definitions, provider assignments, inference parameters,
and role selections remain intact.

`create_anonymizer()` returns an ordinary `Anonymizer`, not a session wrapper or
proxy. Notebook code after setup should be identical to normal Anonymizer code:

```python
preview = anonymizer.preview(config=config, data=input_data, num_records=3)
```

The notebook module should keep its local service manager private. It should:

- return a fully configured core `Anonymizer` from `create_anonymizer()`;
- emit the GLiNER2 model, selected device, and local endpoint at startup without
  exposing credentials;
- provide an idempotent `stop_local_runtime()` function for explicit cleanup;
- register best-effort `atexit` cleanup for notebooks;
- retain enough private diagnostic information to explain startup failures
  without exposing API keys;
- keep the local process alive independently of the lifetime of one returned
  `Anonymizer` object.

`create_anonymizer()` and `stop_local_runtime()` should be synchronous. Notebook
users should not have to manage an event loop merely to initialize or clean up
the tutorials.

The module owns at most one notebook runtime per Python process. Repeated calls
with equivalent runtime configuration reuse it. A call that requests an
incompatible runtime configuration may replace it, after which previously
returned `Anonymizer` instances are no longer supported because they retain the
old endpoint. Document this latest-runtime-only behavior and emit a warning when
replacement occurs.

### Device Selection

For `gliner_device="auto"`, select in this order:

```text
torch.cuda.is_available()          -> cuda
torch.backends.mps.is_available() -> mps
otherwise                         -> cpu
```

An explicit unavailable device must fail with an actionable error. For example,
requesting CUDA in a CPU-only Colab session should tell the user how to enable a
GPU runtime rather than silently falling back.

Do not initialize CUDA in the notebook parent before starting a child process.
Launch the packaged server file with `sys.executable` and prepend only the private
server dependency directory to the child's `PYTHONPATH`; load PyTorch/GLiNER2 in
that child. Direct-file execution avoids importing Anonymizer and Data Designer
through the incompatible server dependency graph.

## Proposed Package Layout

Keep local inference nested under the optional notebook package because this plan
introduces it specifically to support self-contained notebook demos. This avoids
creating a new top-level Anonymizer subsystem or implying that the native FastAPI
server is the production serving path:

```text
src/anonymizer/notebooks/
  __init__.py              # optional public API only
  _runtime.py              # create_anonymizer and private lifecycle manager
  _model_config.py         # tutorial model/provider construction
  local_inference/
    __init__.py            # installed notebook-local inference namespace
    gliner2/
      __init__.py
      backend.py           # model loading, inference, result conversion
      server.py            # FastAPI app and module entrypoint

tools/serve_gliner.py      # temporary thin wrapper around packaged server
```

Launch the child internally as the equivalent of:

```text
PYTHONPATH=<private-server-site-packages> python <installed-package>/notebooks/local_inference/gliner2/server.py
```

Do not re-export local-inference implementation symbols from
`anonymizer.__init__` or `anonymizer.notebooks.__init__`. The supported notebook
API remains `create_anonymizer()` and `stop_local_runtime()`. The nested
`local_inference` namespace provides clear code ownership without presenting the
FastAPI internals as the primary API.
Document that it is a lightweight notebook-serving path, not the production
vLLM deployment path. The module entrypoint may be documented for development and
debugging through `tools/serve_gliner.py`. Until PR #212 lands, production and non-notebook users should be told
to supply a compatible self-hosted endpoint and linked to PR #212 as the planned
production path; this PR must remain independently usable and testable.

Keep imports of `torch`, `gliner2`, `fastapi`, and `uvicorn` inside the optional
namespace or runtime functions. Importing `anonymizer` without the extra must not
fail. If optional dependencies are missing, raise an error that includes the
exact remediation:

```text
Install notebook support with: pip install "nemo-anonymizer[notebooks]"
```

## Packaging

The existing `pyproject.toml` has a development dependency group named
`notebooks`; that is not a published package extra. Add a PEP 621 optional
dependency section:

```toml
[project.optional-dependencies]
notebooks = [
    "datasets>=4.0.0,<6",
    "gliner2==2.0.0",
    "ipykernel>=6.29.0,<8",
    "jupyter>=1.0.0,<2",
    "jupytext>=1.16.0,<2",
    "pip>=25,<27",
    "pillow>=12.0.0,<13",
]
```

Resolve and test actual bounds rather than copying the placeholders. Local
inference dependencies must be installed: GLiNER2 2.0 puts PyTorch, Transformers,
PEFT, NumPy, and Safetensors behind the `local` extra. Its local stack requires
`huggingface-hub<1`, while the current Data Designer release requires
`huggingface-hub>=1.0.1`; therefore the helper creates a cached, private server
environment and installs `gliner2[local]==2.0.0` there. The notebook extra keeps
the thin GLiNER2 package for discoverability without forcing those incompatible
dependency graphs into the kernel. Avoid a strict PyTorch pin that replaces
Colab's compatible CUDA-enabled installation, and allow the private environment
to see compatible system packages where possible.

Keep FastAPI and Uvicorn in the private server dependency set rather than the
published notebook extra. Include them in the development dependency group so
the HTTP contract tests run in the normal unit-test suite.

The pinned checkpoint serializes `extra_special_tokens` using the Transformers v5
list format, while GLiNER2 2.0 requires Transformers v4. Build a deterministic
local adapter copy that renames this field to v4's `additional_special_tokens`;
do not mutate the Hugging Face cache or model weights. Include SentencePiece and
Protobuf in the private server stack and cover the adapter with a unit test.

Prefer making the published optional dependency the single source of truth:

```make
install-dev-notebooks:
	uv sync --group dev --extra notebooks

convert-notebooks:
	uv run --extra notebooks --group docs ...
```

Remove or rename the duplicate development dependency group if that can be done
without breaking repository automation. If both must remain temporarily, add a
comment and a test/check that prevents silent divergence.

Build a wheel and inspect its contents. The acceptance test must install the wheel
into a clean environment and successfully import `anonymizer.notebooks`; testing
only from an editable checkout is insufficient.

## Local GLiNER2 Service

Use the native implementation currently in `tools/serve_gliner.py` as a contract
and lifecycle reference, but replace its GLiNER inference backend with GLiNER2.
The new model is not a drop-in `gliner.GLiner` replacement. Load it with:

```python
from gliner2 import GLiNER2

model = GLiNER2.from_pretrained(
    resolved_snapshot_path,
    map_location=device,
)
```

Use the pinned revision already recorded by PR #212's GLiNER2 profile unless
validation requires a newer revision:

```text
59894c087cb2923b01f337d4ee72f6ff84d5bdd6
```

The commit must be enforced, not merely recorded in documentation or a CI cache
key. Resolve or download the Hugging Face snapshot at that exact revision (or use
a verified revision-aware GLiNER2 loader), load from the resolved snapshot, and
include the resolved model ID and revision in readiness diagnostics. Tests must
fail if startup silently resolves a moving repository head.

Preserve the Anonymizer-facing HTTP contract:

- `GET /v1/models` for readiness and model discovery;
- `POST /v1/chat/completions`;
- request fields for dynamic `labels`, `threshold`, `chunk_length`, `overlap`,
  `flat_ner`, and inference batch size where currently supported;
- an assistant message whose `content` is a JSON string containing
  `{"entities": [...]}`;
- exact document-relative offsets across overlapped chunks;
- conversion of GLiNER2's label-keyed entity output into Anonymizer's flat
  `{text, label, start, end, score}` dictionaries, mapping `confidence` to
  `score`;
- bounded batching and concurrency.

Call GLiNER2 with `include_spans=True` and `include_confidence=True`. Prefer its
document and batch APIs when they preserve the existing chunk-size, overlap,
offset, and concurrency contract; otherwise retain the server's explicit bounded
chunking and adapt the GLiNER2 result shape. Do not recover offsets with
`text.find()`, because repeated values require the exact spans returned by the
model.

GLiNER2 expresses overlap behavior with `overlap_policy` rather than the existing
`flat_ner` request field. Preserve the Anonymizer-facing field and define its
translation in one adapter: `flat_ner=True` selects GLiNER2's flat/disallow policy;
`flat_ner=False` selects GLiNER2's `longest` policy, matching the current server's
subset-span removal before score-based deduplication. Add contract tests for disjoint, nested, crossing, same-span, and
duplicate detections so the backend swap cannot silently change overlap semantics.

Runtime requirements:

1. Bind to `127.0.0.1`, never `0.0.0.0`, from the notebook helper.
2. Choose an available port and retry a bounded number of times if another
   process wins the port-selection race.
3. Use a per-runtime random bearer token if DataDesigner's provider contract can
   pass it cleanly. Otherwise document why loopback binding is the security
   boundary.
4. Poll readiness with a bounded timeout. A successful TCP connection alone is
   insufficient; verify the expected model response.
5. Capture child output in a bounded log or buffer. On failure, show the useful
   tail without printing secrets.
6. Track the exact child process started by the private manager. Do not kill
   unrelated processes merely because they occupy the same port.
7. Terminate and await the child from `stop_local_runtime()`, escalating to a
   forced stop after a bounded grace period.
8. Treat repeated execution of the setup cell as normal. Reuse a healthy runtime
   with equivalent configuration, or cleanly replace the runtime without leaking
   processes or GPU memory.
9. Do not download the model during package installation. Download/cache it on
   first runtime startup using the normal Hugging Face cache.

The model download and initial load can take time. Emit concise progress such as
the chosen model, device, and readiness state so the notebook does not appear
hung.

## Default Model and External Hosts Configuration

When its configuration arguments are omitted, `create_anonymizer()` should load
the same bundled model defaults used by the current notebooks. When they are
provided, it should parse them with the same semantics as `Anonymizer`. In both
cases, transform a private copy in memory and pass it to:

```python
Anonymizer(
    model_configs=...,
    model_providers=...,
)
```

Do not write API keys to YAML, logs, notebook outputs, artifacts, or command-line
arguments. Load the bundled named providers and resolve their credentials from
environment variables, following the existing Anonymizer provider contract. A
single default configuration may use multiple OpenAI-compatible hosts; for
example, one model may use OpenAI while another uses OpenRouter. The runtime API
and tutorial language must not require one hosting service specifically.

The configuration must map:

- `entity_detector` to local GLiNER2;
- every other detection role to its bundled default provider;
- replacement generation to its bundled default provider;
- every rewrite role to its bundled default provider;
- every opt-in evaluation role to its bundled default provider.

The helper must make exactly two configuration transformations:

1. Route the bundled default `entity_detector`, which must now be
   `fastino/gliner2-privacy-filter-PII-multi`, through the local GLiNER2
   provider.
2. Preserve every other resolved model alias's provider assignment, model ID,
   inference parameters, and role selections unchanged, whether it came from the
   bundled defaults or caller-supplied configuration.

Update `src/anonymizer/config/default_model_configs/models.yaml` so the bundled
detector entry uses `fastino/gliner2-privacy-filter-PII-multi`. Retain the existing
`gliner-pii-detector` alias for configuration compatibility: callers may supply a
model pool while relying on bundled role selections, so changing the selected
alias would break otherwise valid partial configurations. Documentation should
identify the underlying detector as GLiNER2 even though the compatibility alias
is unchanged.

This is an intentional product-default change, not merely a notebook override.
`nvidia/gliner-pii` is retired on NVIDIA Build and must not remain as a nominal
default or fallback. Bare `Anonymizer()` is a client and does not start the
notebook service. It must never route the Fastino model ID to NVIDIA Build or any
other provider that does not serve it. Outside the notebook helper, users must
configure a reachable GLiNER2 provider; production and non-notebook users should
be told to supply a compatible self-hosted endpoint. PR #212 is the planned vLLM
Factory production path, but this PR must land and function without it.

Do not add a model-specific network preflight to the core `Anonymizer` interface.
Externally managed providers are responsible for serving their configured model,
and connection or response errors surface through the normal Data Designer model
request. The notebook helper owns its local process and therefore performs the
stronger authenticated readiness check itself, including exact model ID and
pinned revision validation. Documentation should distinguish the self-hosted and
notebook paths and may identify PR #212 as the forthcoming recommended production
deployment.

Accept `model_configs` and `model_providers` using the existing `Anonymizer`
types and replacement/selection semantics. Do not accept `external_model`,
`fast_model`, or `strong_model`, and do not create a second configuration format.
Do not duplicate default LLM aliases or role mappings in notebook source files;
load them through the existing default-model configuration helpers so future
default changes automatically flow into the tutorials.

Use a reserved, truthful alias such as `local-gliner2-pii` for the injected
detector. Reject a caller collision with that reserved alias using an actionable
configuration error rather than silently overwriting it. A caller-supplied
`detection.entity_detector` selection is intentionally replaced because this
helper's contract is local GLiNER2; all other supplied selections are preserved.

The implementation must test every default detection, replacement, rewrite, and
evaluation role. This protects against a future role being added to the defaults
but accidentally left on a different provider.

The GLiNER2 privacy model documents 42 trained PII labels, while Anonymizer's
default entity-label vocabulary is broader and uses some different names. Run a
label-coverage comparison and representative notebook evaluation before landing.
Do not silently rename Anonymizer's public labels in the server. Record any known
unsupported or weakly supported default labels and rely on the existing LLM
augmentation path where appropriate.

## Notebook Changes

The source of truth is the five Jupytext files in `docs/notebook_source/`, not the
generated `.ipynb` files:

```text
01_your_first_anonymization.py
02_inspecting_detected_entities.py
03_choosing_a_replacement_strategy.py
04_rewriting_biographies.py
05_rewriting_legal_documents.py
```

Update each notebook to:

1. show the `nemo-anonymizer[notebooks]` install experience;
2. request each API key required by the bundled default providers with `getpass`
   only when its environment variable is absent;
3. call `create_anonymizer()` and use the returned `Anonymizer` directly;
4. show which device GLiNER2 selected through the helper's startup output;
5. remove hard-coded host setup and single-vendor API-key guidance;
6. explain that GLiNER2 is local while LLM stages use one or more external hosts;
7. avoid exposing the API key in generated outputs;
8. close the runtime explicitly at the end when practical, while retaining
   `atexit` cleanup for interrupted notebooks.

### Installing During Repository Builds

A published notebook should default to installing from PyPI, but repository CI
must execute the current checkout rather than accidentally installing the last
released package. Use one explicit mechanism, such as an environment-controlled
package specification:

```python
import os

package_spec = os.getenv(
    "ANONYMIZER_NOTEBOOK_PACKAGE",
    "nemo-anonymizer[notebooks]",
)
```

The visible notebook setup can install `package_spec`. The repository build can
set it to the checkout or a wheel built from the checkout. Prefer building and
installing the wheel because this validates the actual user artifact. Verify that
the chosen IPython `%pip` or subprocess form handles both the PyPI specification
and local wheel specification correctly.

Do not let a notebook build silently pass because imports resolve from the source
tree while the wheel is missing modules or resources.

## CI and Notebook Build

Update `.github/workflows/build-notebooks.yml`:

- configure the API-key secrets required by the bundled default providers; use
  their existing endpoints, model IDs, provider assignments, and role mappings;
- install the current wheel with the notebooks extra;
- execute GLiNER2 on CPU on `ubuntu-latest`;
- cache the Hugging Face model files using a key that includes the pinned model
  revision and relevant dependency lock state;
- retain artifact upload for generated notebooks;
- ensure child GLiNER2 processes are stopped even when notebook execution fails;
- set conservative timeouts for model download, startup, and notebook execution;
- avoid printing secret-bearing environment values.

The current notebook job is scheduled and manually dispatchable, and the docs
workflow may skip it in some PR contexts. Add a fast unit/wheel smoke test to
ordinary CI so packaging regressions do not wait for the scheduled notebook job.

Do not assume that GitHub-hosted pull requests from forks can access external-host
secrets. Keep secret-requiring end-to-end execution in an appropriate trusted
context and run non-secret local-runtime tests on ordinary PRs.

## Privacy and Security Language

Every tutorial setup section must include an accurate statement equivalent to:

```text
GLiNER2 detection runs locally in this notebook environment. LLM-assisted
validation, augmentation, replacement, rewriting, repair, and evaluation use
the configured external hosts and may send them original or tagged input text.
Do not treat this configuration as an all-local privacy boundary.
```

Also link to each default external host's current terms and privacy information
where the tutorial provides third-party service guidance.

The local server must bind only to loopback. Never include an external-host key in
the local server command line, process receipt, URL, or logs.

## Implementation Sequence

### Phase 1: Establish the Packaged Boundary

- Create the feature branch from freshly fetched `main`.
- Change the bundled default detector model to
  `fastino/gliner2-privacy-filter-PII-multi` while retaining the compatibility
  alias `gliner-pii-detector`.
- Remove the retired NVIDIA Build detector route and tell production users to
  supply a compatible self-hosted endpoint while identifying PR #212 as the
  planned production path. Leave externally managed endpoint errors to the normal
  Data Designer request path rather than adding a model-specific core preflight.
- Add the optional package extra.
- Refactor native serving code into the nested package
  `src/anonymizer/notebooks/local_inference/gliner2/` and replace its inference
  backend with GLiNER2.
- Convert `tools/serve_gliner.py` into a thin wrapper without changing its CLI
  behavior.
- Add unit tests for inference helpers and the HTTP contract.
- Build and inspect the wheel.

### Phase 2: Add Runtime Ownership

- Implement `create_anonymizer()` and the private local service manager.
- Implement idempotent `stop_local_runtime()` and `atexit` cleanup.
- Add device selection and explicit-device validation.
- Enforce the pinned Hugging Face model revision and report it in readiness
  diagnostics.
- Add free-port selection, readiness checks, diagnostic capture, idempotent
  cleanup, and repeated-cell handling.
- Test lifecycle behavior without downloading the real model by injecting or
  replacing the child command and readiness client.
- Add one optional integration test with a real small GLiNER2 request.

### Phase 3: Preserve External Host Model Wiring

- Load the bundled default providers and model selections in memory.
- Ensure every non-detector role preserves its bundled default alias, model ID,
  inference parameters, provider assignment, and selection.
- Repeat the same preservation checks with caller-supplied `model_configs` and
  multiple caller-supplied `model_providers`.
- Validate structured-output support and parameter routing.
- Add tests that fail if any non-detector tutorial role changes its bundled
  provider assignment.
- Run small live smoke tests without logging prompts or outputs containing PII.

### Phase 4: Migrate Tutorials and CI

- Update all five Jupytext sources.
- Update notebook setup documentation and self-hosted GLiNER2 documentation.
- Update the Makefile and notebook workflow to install/test the wheel extra.
- Regenerate notebooks only through `make convert-notebooks`.
- Review generated outputs for secrets, transient paths, noisy server logs, and
  unstable content.

### Phase 5: Platform Validation

- macOS Apple Silicon: assert MPS selection and run at least one preview.
- Colab with GPU: assert CUDA selection and run at least one preview.
- Linux CPU: execute the CI-sized notebook path.
- Explicit CPU override: verify it works even when an accelerator is present.
- Re-run a setup cell: verify no orphan server or duplicate GPU allocation.

## Test Plan

### Unit Tests

- auto device resolution for mocked CUDA, MPS, and CPU states;
- explicit unavailable-device errors;
- optional dependency error message;
- provider/model configuration contains local GLiNER2 plus the bundled external
  providers, including configurations with more than one external host;
- bundled default model configuration selects
  `fastino/gliner2-privacy-filter-PII-multi` for `entity_detector`;
- every non-detector alias, model ID, inference parameter, provider assignment,
  and role selection matches the bundled defaults;
- caller-supplied model/provider configuration is preserved for every role except
  `detection.entity_detector`;
- a collision with the reserved local-detector alias fails clearly;
- every workflow role resolves to a defined alias;
- no external-host key appears in serialized configs or log text;
- `stop_local_runtime()` is idempotent;
- startup timeout terminates the owned child;
- notebook-runtime readiness rejects the wrong model or revision;
- repeated startup reuses or safely replaces the runtime;
- incompatible repeated startup warns that earlier returned `Anonymizer`
  instances are no longer supported;
- chunk offsets, overlap deduplication, batching, and empty-input behavior;
- `flat_ner` maps to verified GLiNER2 overlap policies for disjoint, nested,
  crossing, same-span, and duplicate detections;
- API request and response compatibility with Anonymizer's parser.

### Packaging Tests

- build the wheel;
- inspect that `anonymizer/notebooks/*`, including
  `anonymizer/notebooks/local_inference/gliner2/*`, is included;
- create a clean virtual environment;
- install the wheel with `[notebooks]`;
- import the optional API;
- verify the extra installs GLiNER2's complete local-inference dependency set;
- verify core installation without the extra still imports `anonymizer` and gives
  an actionable error only when notebook runtime functionality is requested.

### Integration Tests

- start the packaged server as a child and query `/v1/models`;
- send one known-name detector request and validate entity value and offsets;
- instantiate `Anonymizer` from the runtime-generated configuration;
- exercise a minimal preview with mocked responses for every bundled external
  provider in ordinary CI;
- exercise a minimal live preview across the bundled external hosts only in
  secret-enabled CI;
- verify shutdown leaves no owned child process.

### Repository Validation

Run the standard commands required by `AGENTS.md` and the contributor guide:

```bash
make format
make check
make test
make docs-build
make convert-notebooks
```

Also run the clean-wheel and platform smoke tests added by this work.

## Acceptance Criteria

The work is complete only when all of the following are true:

- A clean Python environment can install
  `nemo-anonymizer[notebooks]` from the built wheel.
- No repository checkout or `tools/` file is required at runtime.
- A notebook starts GLiNER2 using only Python executed in notebook cells.
- `create_anonymizer()` can be called without model IDs or model configuration
  for the default tutorial path.
- `create_anonymizer(model_configs=..., model_providers=...)` supports custom
  external models and multiple hosts while retaining local GLiNER2 detection.
- The local detector default is
  `fastino/gliner2-privacy-filter-PII-multi` at the validated pinned revision.
- `src/anonymizer/config/default_model_configs/models.yaml` also uses that model
  as the repository's default entity detector.
- The compatibility alias `gliner-pii-detector` remains valid for existing
  partial model configurations.
- Ordinary `Anonymizer()` never routes GLiNER2 to the retired NVIDIA Build model
  service. It remains a client for a separately managed endpoint and relies on
  the normal Data Designer request path for endpoint errors; documentation points
  notebook users to the managed helper and identifies PR #212 as the planned
  production implementation.
- This PR lands and passes independently before PR #212. Interim production and
  non-notebook guidance requires a compatible self-hosted GLiNER2 endpoint and
  identifies PR #212 as the planned vLLM Factory path; the native PyTorch server
  is labeled notebook/development only.
- Colab CUDA, Apple MPS, and CPU paths are supported through `device="auto"`.
- Every non-GLiNER2 tutorial role preserves its bundled default provider
  assignment, including when the defaults span multiple external hosts.
- All five notebook sources use the new runtime and request the environment keys
  required by the bundled providers rather than assuming one hard-coded key.
- The generated notebooks complete successfully in the trusted notebook build.
- Re-running setup and shutting down the kernel do not leave an owned server
  process behind.
- Core users who do not install `[notebooks]` do not acquire the heavyweight
  optional dependencies.
- Documentation clearly states which stages are local and which may send text to
  one or more external hosts.
- The implementation remains independent of PR #212's Linux/vLLM Factory path.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Changing the bundled detector model leaves ordinary `Anonymizer()` pointing at the retired NVIDIA Build route | Remove that route, keep bare `Anonymizer()` as a client for a compatible self-hosted endpoint, and identify PR #212 as the forthcoming recommended production path without depending on it. |
| GLiNER2 is not API-compatible with the existing GLiNER backend | Isolate conversion in `gliner2/backend.py`; test label-keyed results, `confidence` to `score`, empty results, repeated values, exact spans, and malformed output. |
| GLiNER2's 42 trained labels do not cover or exactly name Anonymizer's broader default vocabulary | Produce a label-coverage report, run representative notebook and benchmark data, preserve public Anonymizer labels, and document weak or unsupported labels rather than silently remapping them. |
| The detector swap changes tutorial precision, recall, or threshold behavior | Compare the old and new detector on the tutorial datasets using the configured default threshold. Review missed sensitive values and false positives before accepting the new default. |
| The notebook extra installs plain GLiNER2 without its local-inference stack, or dependency resolution replaces Colab's working CUDA build | Depend on `gliner2[local]`, verify Torch/Transformers/PEFT/NumPy/Safetensors in a clean wheel environment, avoid stricter Torch pins, and assert CUDA remains available after installation. |
| GLiNER2 contains operations unsupported or unstable on MPS | Run a real Apple Silicon smoke test. In `auto` mode, fall back to CPU with a clear warning only when MPS initialization/inference fails; an explicitly requested `mps` device must fail clearly. |
| The model exceeds available accelerator or system memory | Keep batching conservative by device, expose bounded batch controls only if needed, catch out-of-memory errors with actionable guidance, and test representative CPU and Colab memory limits. |
| First-run model download is slow, unavailable, or changes upstream | Resolve the exact pinned Hugging Face revision, load from that snapshot, report the resolved revision, download only at runtime, emit progress, use bounded timeouts/retries, cache by revision in CI, and surface offline/cache guidance. |
| A custom `model_configs` injection loses or mutates caller settings | Parse a private copy with existing Anonymizer semantics and assert that every non-detector alias, parameter, provider assignment, pool, and role selection is byte-equivalent or semantically equivalent after injection. |
| Caller configuration collides with the reserved local-detector alias | Reserve one documented alias and reject collisions before starting the child process; never silently replace a caller-defined model entry. |
| Multiple external providers have missing or misnamed credentials | Preflight every resolved provider before model execution and report all missing environment-variable names together, without printing their values. |
| An external model/provider pair lacks required structured-output parameters | Validate every resolved pair in trusted smoke tests, require supported request parameters where the host supports such routing controls, and identify the failing alias and provider in errors. |
| External rate limits, cost, or transient failures make generated notebooks flaky | Keep executed demo datasets small, use bounded retries already supported by the model layer, record expected call volume, and avoid free or rolling routes with unstable availability in CI defaults. |
| Port selection races with another local process | Bind only to loopback, retry a bounded number of candidate ports, authenticate with a per-runtime token, and verify readiness returns the expected model identity. |
| Cleanup terminates the wrong process or leaves a child holding GPU memory | Track the exact child PID/process handle, verify identity before signaling, use bounded TERM/KILL cleanup, await process exit, and test both normal and failed startup paths. |
| Re-running a notebook setup cell creates duplicate servers or invalidates an earlier returned client | Make startup idempotent for equivalent configuration; otherwise stop the owned runtime, warn that earlier clients are unsupported, and document latest-runtime-only ownership. |
| Notebook or server logs expose external API keys | Keep keys in provider-named environment variables, redact child commands and diagnostics, and scan generated notebooks and uploaded artifacts for secret values. |
| Tutorial wording implies that the entire pipeline is local | Include the data-flow disclosure in every notebook: GLiNER2 is local, while LLM stages may send original or tagged text to one or more external hosts. |
| `%pip` installs the released package instead of the current checkout during source CI | Build the current wheel, pass its path through the notebook package-spec override, install it in a clean kernel environment, and assert the imported version/commit. |
| Optional imports make core `import anonymizer` require heavyweight notebook dependencies | Keep Torch, GLiNER2, FastAPI, and Uvicorn imports lazy below `anonymizer.notebooks`; test the core wheel without the extra. |
| PR #212 deletes or conflicts with `tools/serve_gliner.py` | Keep reusable code in `anonymizer.notebooks.local_inference.gliner2`; treat the tools file as a thin disposable wrapper and test the installed module entrypoint directly. |

## Expected Files

Likely additions or modifications include:

```text
pyproject.toml
uv.lock
Makefile
.github/workflows/build-notebooks.yml
src/anonymizer/config/default_model_configs/models.yaml
src/anonymizer/config/default_model_configs/detection.yaml
src/anonymizer/notebooks/__init__.py
src/anonymizer/notebooks/_runtime.py
src/anonymizer/notebooks/_model_config.py
src/anonymizer/notebooks/local_inference/__init__.py
src/anonymizer/notebooks/local_inference/gliner2/__init__.py
src/anonymizer/notebooks/local_inference/gliner2/backend.py
src/anonymizer/notebooks/local_inference/gliner2/server.py
tools/serve_gliner.py
tests/notebooks/test_runtime.py
tests/notebooks/test_model_config.py
tests/notebooks/test_wheel.py
tests/local_inference/test_gliner2_backend.py
tests/local_inference/test_gliner2_server.py
docs/notebook_source/01_your_first_anonymization.py
docs/notebook_source/02_inspecting_detected_entities.py
docs/notebook_source/03_choosing_a_replacement_strategy.py
docs/notebook_source/04_rewriting_biographies.py
docs/notebook_source/05_rewriting_legal_documents.py
docs/concepts/self-hosting-gliner.md
```

Do not edit generated notebook JSON manually. Regenerate `docs/notebooks/*.ipynb`
from the Jupytext sources after the source changes are complete.

Because `create_anonymizer()` and `stop_local_runtime()` are new public surfaces,
check whether the bundled
Anonymizer skill needs corresponding setup guidance before shipping, as required
by `AGENTS.md` for public API changes.

## Branch and PR Guidance

Follow the repository branch convention. With the current contributor identity,
an appropriate branch name would be:

```text
lipikaramaswamy/feature/notebook-local-gliner2
```

If a tracking issue is created, include its number in the branch name. Use a
conventional PR title such as:

```text
feat(notebooks): add portable local GLiNER2 runtime
```

The PR body should link the tracking issue and this plan. Commits must include DCO
signoff.

## Handoff Prompt for a New Agent

The following prompt can be given directly to the implementation agent:

```text
Implement the notebook demo runtime described in
plans/notebook-local-gliner/notebook-demo-runtime.md.

Read AGENTS.md, STYLEGUIDE.md, CONTRIBUTING.md, and the complete plan before
editing. Work from a new branch based on the latest main and preserve unrelated
working-tree changes. Do not add this work to PR #212.

The required user experience is a published
`pip install "nemo-anonymizer[notebooks]"` extra followed by a Python
`create_anonymizer()` call inside the notebook. That function must return an
ordinary configured `Anonymizer`, not a runtime wrapper. GLiNER2 must run locally
through native PyTorch with automatic CUDA/MPS/CPU selection, using the default
`fastino/gliner2-privacy-filter-PII-multi` model. Every other tutorial model role
must preserve the repository's bundled default alias, model ID, inference
parameters, provider assignment, and role selection. The bundled defaults may use
multiple external hosts; do not collapse them onto one endpoint. The runtime must
work from the built wheel without a repository checkout, safely own and clean up
its child server, and update all five Jupytext notebook sources plus notebook CI.

Change the repository's bundled default entity detector in
`src/anonymizer/config/default_model_configs/models.yaml` from
`nvidia/gliner-pii` to `fastino/gliner2-privacy-filter-PII-multi`, but retain the
existing `gliner-pii-detector` alias for partial-configuration compatibility.
Remove the retired NVIDIA Build detector route. Bare `Anonymizer()` remains a
client and must not start a model service or point the Fastino model ID at an
incompatible host. Do not add a model-specific endpoint preflight to core
`Anonymizer`; externally managed endpoint errors should surface through the
normal Data Designer request path. Tell production and non-notebook users to
configure a compatible self-hosted GLiNER2 endpoint and notebook users to call
`create_anonymizer()`. This PR must land and pass independently before PR #212;
identify PR #212 only as the planned vLLM Factory production path.

Follow the implementation phases and acceptance criteria in the plan. Validate
with focused tests, clean-wheel installation, standard repository checks, and
the feasible platform matrix. Do not log API keys or claim the whole pipeline is
local. The no-argument `create_anonymizer()` path must load the bundled external
provider topology and credentials from its named environment variables. Also
accept the existing `model_configs` and `model_providers` overrides, including
multiple hosts, while overriding only `detection.entity_detector` with local
GLiNER2. Do not expose ad-hoc model arguments or a single-endpoint shortcut.
Verify every resolved model/provider pair and report any incompatible
configuration clearly. Install
GLiNER2's complete local stack through `gliner2[local]`, enforce and report the
pinned model revision, define the `flat_ner` to `overlap_policy` mapping, and
document that replacing an incompatible notebook runtime invalidates earlier
returned clients. The native PyTorch server is notebook/development-only;
production serving belongs to PR #212.
```

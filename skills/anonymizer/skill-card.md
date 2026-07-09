<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Description

Use NeMo Anonymizer through an interactive agent workflow: inspect text data,
choose Replace or Rewrite, select a replacement strategy, draft a runnable
Python script, preview before full execution, diagnose failed records first, and
configure self-hosted GLiNER when detection must stay local.

This skill package is prepared for NVSkills publication review. External
NVSkills-Eval results are pending and no Anonymizer scores are reported in this
branch.

## Owner

NVIDIA

### License/Terms of Use

Apache 2.0

## Use Case

Developers, privacy engineers, and data practitioners using NeMo Anonymizer to
detect, replace, redact, hash, annotate, or rewrite sensitive entities in text
datasets while keeping a durable script for review and reruns.

### Deployment Geography for Use

Global

## Known Risks and Mitigations

Risk: Users may overinterpret anonymized output as a privacy guarantee.

Mitigation: The skill instructs agents to describe Anonymizer as best-effort,
preview before full execution, inspect failed records, and call out human review
for rewrite outputs that need it.

Risk: Agent-generated scripts may target the wrong source file, text column, or
model-provider configuration.

Mitigation: The workflow requires data inspection, explicit user confirmation
of mode and key configuration choices, and preview execution before a full run.

Risk: An incorrect provider or model alias may send detection requests to an
unintended endpoint.

Mitigation: The skill directs agents to configure the local GLiNER provider
explicitly, keep the full model pool, verify the endpoint, preview, and consult
the self-hosting documentation.

## Reference(s)

- [Interactive workflow](references/interactive.md)
- [Choosing a Strategy](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/choosing-a-strategy/)
- [Detection](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/detection/)
- [Evaluation](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/evaluation/)
- [Models](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/models/)
- [Self-hosting GLiNER](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/self-hosting-gliner/)
- [Troubleshooting](https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/)

## Skill Output

**Output Type(s):** Python scripts, shell commands, configuration guidance,
diagnostic guidance

**Output Format:** A runnable Python script plus concise Markdown guidance for
previewing, diagnosing failures, and running the full pipeline

**Output Parameters:** Dataset path, text column, data summary, mode
(`Replace` or `Rewrite`), replacement strategy when applicable, privacy goal,
risk tolerance, entity labels, and optional model-provider paths

**Other Properties Related to Output:** The generated script previews by
default, exits on failed records, optionally evaluates output with
LLM-as-judge, and leaves full dataset execution under explicit user control.

## Evaluation Agents Used

The external NVSkills-Eval run is pending. Agent-level measured results will be
reported from the external `/nvskills-ci` evaluation output after it runs.

## Evaluation Tasks

The prepared evaluation dataset contains 6 NVSkills-Eval tasks: 4 positive
activation cases and 2 negative activation cases. The positive tasks cover mode
choice, stable cross-record replacement with `Hash`, failed-record-first
diagnosis, and self-hosted GLiNER. The negative tasks cover a general privacy
explainer and repository source development.

## Evaluation Metrics Used

Metrics will be reported by the external NVSkills-Eval run. Expected benchmark
dimensions are:

- Security: Checks whether skill-assisted execution avoids unsafe behavior such
  as secret leakage, destructive commands, or unauthorized access.
- Correctness: Checks whether the agent follows the expected workflow and
  produces the correct final output.
- Discoverability: Checks whether the agent loads the skill when relevant and
  avoids using it when irrelevant.
- Effectiveness: Checks whether the agent performs measurably better with the
  skill than without it.
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant
  work.

## Evaluation Results

External NVSkills-Eval execution is pending. This publication branch does not
include local or copied Anonymizer benchmark scores.

| Dimension | Tasks | Result |
|---|---:|---|
| Security | 6 | Pending external NVSkills-Eval run |
| Correctness | 6 | Pending external NVSkills-Eval run |
| Discoverability | 6 | Pending external NVSkills-Eval run |
| Effectiveness | 6 | Pending external NVSkills-Eval run |
| Efficiency | 6 | Pending external NVSkills-Eval run |

## Skill Version(s)

Publication candidate from this repository branch. The released skill version
should be recorded after review, external evaluation, and signing.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and has established
policies and practices to enable development for a wide array of AI
applications. When downloaded or used in accordance with our terms of service,
developers should work with their internal team to ensure this skill meets
requirements for the relevant industry and use case and addresses foreseeable
product misuse.

(For Release on NVIDIA Platforms Only)

Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns
[here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).

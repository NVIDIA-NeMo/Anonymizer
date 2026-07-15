<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluation Report

Evaluation report for the `anonymizer` skill before publication through
NVSkills-Eval.

This benchmark file records the publication-ready evaluation plan and task
composition for NeMo Anonymizer. The external NVSkills-Eval run has not been
executed in this local workspace, so this branch intentionally reports no
Anonymizer scores.

## Evaluation Summary

- Skill: `anonymizer`
- Evaluation date: pending external `/nvskills-ci` run
- NVSkills-Eval profile: external
- Environment: external NVSkills-Eval runner
- Dataset: 6 evaluation tasks
- Attempts per task: recorded by external NVSkills-Eval after execution
- Pass threshold: recorded by external NVSkills-Eval after execution
- Overall verdict: pending external NVSkills-Eval run

## Agents Used

Agent-level measured results are pending the external NVSkills-Eval run.

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such
  as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and
  produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and
  avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the
  skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant
  work.

Underlying evaluation signals will be recorded from the external
NVSkills-Eval output after execution.

## Test Tasks

The benchmark dataset contains 6 evaluation tasks:

- Positive tasks: 4 tasks where the skill is expected to activate.
- Negative tasks: 2 tasks where no skill is expected.
- Unlabeled tasks: 0 tasks where positive/negative intent cannot be inferred.

Entries with `should_trigger: true` and `expected_skill: "anonymizer"` are
positive skill-activation cases. Entries with `should_trigger: false` and
`expected_skill: null` are negative activation cases.

## Results

External NVSkills-Eval execution is pending. No copied or locally inferred
Anonymizer results are reported here.

| Dimension | Tasks | Result |
|---|---:|---|
| Security | 6 | Pending external NVSkills-Eval run |
| Correctness | 6 | Pending external NVSkills-Eval run |
| Discoverability | 6 | Pending external NVSkills-Eval run |
| Effectiveness | 6 | Pending external NVSkills-Eval run |
| Efficiency | 6 | Pending external NVSkills-Eval run |

## Tier 1: Static Validation Summary

Local static validation is covered by this branch's validation evidence. The
external NVSkills-Eval Tier 1 result is pending the `/nvskills-ci` run.

## Tier 2: Deduplication Summary

External NVSkills-Eval deduplication results are pending.

## Publication Recommendation

Proceed to external NVSkills-Eval and signing. Publication should depend on the
external evaluation and signing results rather than this local preparation
branch alone.

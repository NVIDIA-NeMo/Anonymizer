# Skill Benchmark: anonymizer

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `anonymizer`
- Evaluation date: 2026-09-10
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:c2c13b2d794c6117dac0402f1261bd2d80972085c5e716426bedff6b3d59b8ae` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.8% — baseline ran, but no comparable score was available; uplift unavailable | 89.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 95.0% → 100.0% (+5.0 points) | 100.0% → 83.3% (-16.7 points) |
| Correctness | 46.0% → 86.7% (+40.7 points) | 75.0% → 100.0% (+25.0 points) |
| Discoverability | 97.5% — baseline ran, but no comparable score was available; uplift unavailable | 93.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 37.6% → 80.3% (+42.7 points) | 51.3% → 87.8% (+36.5 points) |
| Efficiency | 84.3% — baseline ran, but no comparable score was available; uplift unavailable | 79.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,699,362 | 2,863,530 | N/A | N/A | skill 6/6; base 10/10 |
| claude-code | anonymizer-negative-general-privacy-explainer | 30,472 | 30,464 | +8 | +0.03% | skill 1/1; base 1/1 |
| claude-code | anonymizer-negative-repository-source-development | 261,745 | 150,049 | +111,696 | +74.44% | skill 1/1; base 1/1 |
| claude-code | anonymizer-positive-failed-records-first | 185,130 | 431,085 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | anonymizer-positive-hash-cross-record-consistency | 456,894 | 31,819 | +425,075 | +1335.92% | skill 1/1; base 1/1 |
| claude-code | anonymizer-positive-mode-choice | 361,402 | 32,805 | +328,597 | +1001.67% | skill 1/1; base 1/1 |
| claude-code | anonymizer-positive-self-hosted-gliner | 403,719 | 2,187,308 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 1,045,950 | 846,940 | N/A | N/A | skill 6/6; base 8/8 |
| codex | anonymizer-negative-general-privacy-explainer | 13,778 | 13,555 | +223 | +1.65% | skill 1/1; base 1/1 |
| codex | anonymizer-negative-repository-source-development | 853,740 | 578,858 | +274,882 | +47.49% | skill 1/1; base 1/1 |
| codex | anonymizer-positive-failed-records-first | 66,683 | 151,908 | N/A | N/A | skill 1/1; base 3/3 |
| codex | anonymizer-positive-hash-cross-record-consistency | 29,917 | 24,789 | +5,128 | +20.69% | skill 1/1; base 1/1 |
| codex | anonymizer-positive-mode-choice | 34,490 | 25,537 | +8,953 | +35.06% | skill 1/1; base 1/1 |
| codex | anonymizer-positive-self-hosted-gliner | 47,342 | 52,293 | -4,951 | -9.47% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,745,312 | 3,710,470 | N/A | N/A | skill 12/12; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 1 validator(s); 2 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/anonymizer/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/anonymizer/SKILL.md`)

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
